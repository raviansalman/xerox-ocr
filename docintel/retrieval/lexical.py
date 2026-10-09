"""Exact, lexical and fuzzy retrieval over the term-postings index. None of these use embeddings.

* ExactRetriever: quoted phrases, identifiers in any spelling, the whole question as a phrase or exact token, file
  names. Tier 1 (tier 2 for an exact single word).
* LexicalRetriever: all terms (English stems), joined/split word forms, most terms for longer questions, ranked
  with BM25.
* FuzzyRetriever: OCR-folded matching, partial identifiers (prefix and suffix of identifier tokens), and typo
  correction against the tenant's own vocabulary.

Query terms are produced by PostgreSQL's own text-search configurations, the same ones that built the index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from docintel import text as T
from docintel.retrieval.contracts import Budget, Candidate, RetrievalContext, Scope
from docintel.retrieval.postings import Postings, UnitHit
from docintel.search.plan import QueryPlan

_PHRASE = "tsv @@ phraseto_tsquery('simple', %s)"
_FOLD_PHRASE = "tsv_fold @@ phraseto_tsquery('simple', %s)"
_YEAR = re.compile(r"(?:19|20)\d{2}")


@dataclass
class QueryTerms:
    words: list[str]          # exact lexemes in order ('simple' configuration)
    stems: list[str]          # English stems and Arabic light stems, stop words removed
    folded: list[str]         # OCR-folded lexemes in order


def analyze(rc: RetrievalContext, text: str) -> QueryTerms:
    st, folded = T.search_text(text), T.fold(text)
    if not st.strip():
        return QueryTerms([], [], [])
    row = rc.conn.execute(
        "SELECT (SELECT array_agg(lexeme ORDER BY pos) FROM (SELECT lexeme, unnest(positions) AS pos "
        "        FROM unnest(to_tsvector('simple', %(s)s))) a) AS words, "
        "       tsvector_to_array(to_tsvector('english', %(st)s)) AS stems, "      # as chunks.tsv_en is built
        "       (SELECT array_agg(lexeme ORDER BY pos) FROM (SELECT lexeme, unnest(positions) AS pos "
        "        FROM unnest(to_tsvector('simple', %(f)s))) b) AS folded",
        {"s": st, "st": T.stem_text(text), "f": folded}).fetchone()
    return QueryTerms(list(row["words"] or []), list(row["stems"] or []), list(row["folded"] or []))


def _scope(scope: Scope) -> tuple[str, ...] | None:
    return scope.document_ids


def _candidates(hits: dict[str, UnitHit], keep: set[str] | None, retriever: str, match_type: str,
                scores: dict[str, float] | None, matched: tuple[str, ...]) -> list[Candidate]:
    out = []
    for uid, h in hits.items():
        if keep is not None and uid not in keep:
            continue
        out.append(Candidate(h.document_id, uid, retriever, match_type, (scores or {}).get(uid, 1.0), [], matched))
    out.sort(key=lambda c: -c.raw_score)
    return out


class ExactRetriever:
    name = "exact"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        p = Postings(rc.conn, rc.auth.tenant_id)
        out: list[Candidate] = []
        cap = budget.max_candidates * 25
        for phrase in plan.phrases:                                   # quoted phrases must appear as written
            q = analyze(rc, phrase)
            if not q.words:
                continue
            hits = p.all_of([f"w:{w}" for w in q.words], _scope(scope), cap)
            keep = p.verify(list(hits), _PHRASE, T.search_text(phrase)) if len(q.words) > 1 else set(hits)
            out += _candidates(hits, keep, self.name, "exact_phrase", p.bm25(hits, [f"w:{w}" for w in q.words]), (phrase,))
        if plan.identifiers:                                          # INV-2026-1 == inv20261 == INV 2026 1
            hits = p.any_of([f"i:{i}" for i in plan.identifiers], _scope(scope), cap)
            out += _candidates(hits, None, self.name, "exact_identifier", None, tuple(plan.identifiers))
        q = analyze(rc, plan.text)
        quoted = {T.search_text(p) for p in plan.phrases}
        if q.words and T.search_text(plan.text) not in quoted:      # the whole question as a phrase / exact word
            terms = [f"w:{w}" for w in q.words]
            hits = p.all_of(terms, _scope(scope), cap)
            if len(q.words) > 1:
                keep = p.verify(list(hits), _PHRASE, T.search_text(plan.text))
                out += _candidates(hits, keep, self.name, "exact_phrase", p.bm25(hits, terms), (plan.text,))
            else:
                out += _candidates(hits, None, self.name, "exact_term", p.bm25(hits, terms), (plan.text,))
        out += self._filenames(rc, p, plan, scope)
        return out

    def _filenames(self, rc: RetrievalContext, p: Postings, plan: QueryPlan, scope: Scope) -> list[Candidate]:
        text = " ".join([*plan.phrases, plan.text])
        q = analyze(rc, text)
        if not q.words or len("".join(q.words)) < 3:
            return []
        content = [w for w in q.words if w not in T.STOPWORDS] or q.words
        hits = p.all_of([f"n:{w}" for w in content], _scope(scope), 200)
        if not hits:
            return []
        docs = {h.document_id for h in hits.values()}
        whole = T.search_text(text)
        rows = rc.conn.execute("SELECT id FROM documents WHERE id = ANY(%s::uuid[]) AND filename_search LIKE %s",
                               (list(docs), f"%{whole}%")).fetchall()
        exact_docs = {str(r["id"]) for r in rows}
        out = []
        for uid, h in hits.items():
            mt = "filename" if h.document_id in exact_docs else "filename_terms"
            if mt == "filename_terms" and len(content) < 2:
                continue
            out.append(Candidate(h.document_id, uid, self.name, mt, 1.0, [], (text,)))
        return out


class LexicalRetriever:
    name = "lexical"

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        q = analyze(rc, plan.text)
        if not q.words:
            return []
        p = Postings(rc.conn, rc.auth.tenant_id)
        out: list[Candidate] = []
        cap = budget.max_candidates * 25
        # every content word, any inflection; a single Arabic word too, since Arabic attaches articles, prepositions
        # and plural endings to the word ("طابعة" should find "الطابعات")
        if q.stems and (len(q.words) > 1 or T.is_arabic_word(q.words[0])):
            terms = [f"s:{s}" for s in q.stems]
            hits = p.all_of(terms, _scope(scope), cap)
            out += _candidates(hits, None, self.name, "all_terms", p.bm25(hits, terms), tuple(q.stems))
            if len(q.stems) >= 3 and not plan.phrases:               # most words, for longer questions
                need = max(2, -(-len(q.stems) * 3 // 5))
                anyhits = p.any_of(terms, _scope(scope))
                most = {u: h for u, h in anyhits.items() if len(h.tf) >= need and u not in hits}
                out += _candidates(most, None, self.name, "most_terms", p.bm25(most, terms), tuple(q.stems))
        for variant in T.join_variants(q.words):                     # "Data Vault" ~ "DataVault"
            vq = variant.split()
            hits = p.all_of([f"w:{w}" for w in vq], _scope(scope), cap)
            keep = p.verify(list(hits), _PHRASE, variant) if len(vq) > 1 else set(hits)
            out += _candidates(hits, keep, self.name, "joined_form", None, (variant,))
        return out


def edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insertions, deletions, substitutions and adjacent transpositions."""
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


class FuzzyRetriever:
    name = "fuzzy"
    SHORTLIST_SIMILARITY = 0.3

    def retrieve(self, rc: RetrievalContext, plan: QueryPlan, scope: Scope, budget: Budget) -> list[Candidate]:
        q = analyze(rc, plan.text)
        p = Postings(rc.conn, rc.auth.tenant_id)
        out: list[Candidate] = []
        cap = budget.max_candidates * 25
        if q.folded:                                                  # OCR-folded on both sides
            terms = [f"w:{w}|f:{w}" for w in q.folded]
            hits = p.all_of(terms, _scope(scope), cap)
            keep = p.verify(list(hits), _FOLD_PHRASE, T.fold(plan.text)) if len(q.folded) > 1 else set(hits)
            out += _candidates(hits, keep, self.name, "ocr_tolerant", None, (plan.text,))
        for w in q.words:                                             # partial identifiers: "00481", "0048", "481"
            if len(w) >= 4 and any(c.isdigit() for c in w) and not _YEAR.fullmatch(w):
                hits = p.prefix(f"w:{w}", _scope(scope)) | p.prefix(f"r:{w[::-1]}", _scope(scope))
                out += _candidates(hits, None, self.name, "partial_identifier", None, (w,))
        if not out and not budget.expired():
            out += self._typos(rc, p, q, scope, cap)
        return out

    def _typos(self, rc: RetrievalContext, p: Postings, q: QueryTerms, scope: Scope, cap: int) -> list[Candidate]:
        """Correct words the tenant's documents do not contain to the closest words they do contain."""
        words = [w for w in q.words if w not in T.STOPWORDS]
        if not words or len(words) > 6:
            return []
        df = p.df(f"w:{w}" for w in words)
        groups, corrected = [], False
        for w in words:
            if df.get(f"w:{w}", 0) > 0 or len(w) < 5 or w.isdigit():
                groups.append(f"w:{w}")
                continue
            # trigrams only shortlist; a typo is accepted by edit distance (one dropped letter in a short word can score
            # well below any trigram threshold that still rejects unrelated words)
            rows = rc.conn.execute(
                "SELECT term FROM vocabulary WHERE length BETWEEN %s AND %s AND term >= %s AND term < %s "
                "AND similarity(term, %s) >= %s ORDER BY similarity(term, %s) DESC LIMIT 50",
                (len(w) - 2, len(w) + 2, w[0], chr(ord(w[0]) + 1), w, self.SHORTLIST_SIMILARITY, w)).fetchall()
            allowed = 1 if len(w) < 8 else 2
            close = sorted((d, r["term"]) for r in rows if (d := edit_distance(w, r["term"])) <= allowed)
            if not close:
                return []
            best = close[0][0]
            groups.append("|".join(f"w:{t}" for d, t in close[:5] if d == best))
            corrected = True
        if not corrected:
            return []
        hits = p.all_of(groups, _scope(scope), cap)
        return _candidates(hits, None, self.name, "fuzzy", None, tuple(words))
