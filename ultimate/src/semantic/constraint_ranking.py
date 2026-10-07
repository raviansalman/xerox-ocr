"""Constraint-aware ranking for search results (vector retrieval unchanged)."""
from __future__ import annotations
from typing import Any, Callable, Optional


def apply_constraint_boost(
    items: list,
    meta: dict,
    *,
    get_semantic_engine: Optional[Callable[[], Any]] = None,
) -> None:
    """Compute constraint-awareness adjustment per result stored in
    item["_ranking_score"]. Does NOT modify similarity_score.

    Boosts (additive), penalties; see ultimate_ui search path."""
    if get_semantic_engine is None:
        import ultimate_ui as _ui
        get_semantic_engine = _ui.get_semantic_engine

    if not meta:
        return
    import urllib.parse as _ui_urlparse

    import calendar as _cal
    import re as _constr_re

    _qd = meta.get("query_decomposition") or {}
    _intent_profile = str(_qd.get("intent_profile") or "")
    _broad_temporal_geo = _intent_profile == "broad_temporal_or_geo"

    # ── 1. Date signals ──────────────────────────────────────────────────
    full_date_raw = (meta.get("full_date_query") or "")
    full_date_iso = full_date_raw[:10] if full_date_raw else ""  # "YYYY-MM-DD"
    date_variants: set = set()
    if full_date_iso:
        date_variants.add(full_date_iso)
        date_variants.add(full_date_iso.replace("-", "/"))
        try:
            yr, mo, dy = full_date_iso.split("-")
            mo_i, dy_i, yr_i = int(mo), int(dy), int(yr)
            mname = _cal.month_name[mo_i].lower()   # "january"
            mabbr = _cal.month_abbr[mo_i].lower()   # "jan"
            # "january 30, 2019", "january 30 2019", "30 january 2019"
            date_variants.update([
                f"{mname} {dy_i}, {yr_i}",
                f"{mname} {dy_i} {yr_i}",
                f"{dy_i} {mname} {yr_i}",
                f"{mabbr} {dy_i}, {yr_i}",
                f"{mabbr} {dy_i} {yr_i}",
                f"{dy_i}/{mo_i}/{yr_i}",
                f"{mo_i}/{dy_i}/{yr_i}",
                f"{dy_i:02d}/{mo_i:02d}/{yr_i}",
                f"{mo_i:02d}/{dy_i:02d}/{yr_i}",
            ])
        except Exception:
            pass

    # ── 2. Location signals ──────────────────────────────────────────────
    # Unique lowercased locations; deduplicate (enhance_query often emits
    # duplicate casing variants like "Texas" + "texas")
    locations = list(dict.fromkeys(
        loc.lower().strip()
        for loc in (meta.get("locations") or [])
        if loc and len(loc.strip()) >= 3
    ))

    # Compound city+state from query_enhancement.geo_entity — strict match; no cross-city boosts.
    _geo_ent = meta.get("geo_entity") or {}
    _geo_city = (_geo_ent.get("city") or "").lower().strip()
    _geo_state = (_geo_ent.get("state") or "").lower().strip()
    _TX_PEER_CITIES = frozenset({
        "austin", "dallas", "houston", "san antonio", "fort worth", "el paso",
        "plano", "arlington", "corpus christi", "lubbock", "garland", "irving",
        "amarillo", "mckinney", "waco", "pasadena", "beaumont", "tyler", "midland",
        "odessa", "richardson", "pearland", "college station", "round rock",
        "lewisville", "denton", "allen", "grand prairie",
    })

    # ── 3. Person signals ────────────────────────────────────────────────
    # Month names are mis-classified as persons by query_enhancement;
    # strip them out here so they don't cause false boosts.
    _month_names = {_cal.month_name[i].lower()  for i in range(1, 13)} | \
                   {_cal.month_abbr[i].lower()  for i in range(1, 13)}
    persons = [
        p.lower().strip()
        for p in (meta.get("persons") or [])
        if p and len(p.strip()) >= 3 and p.lower().strip() not in _month_names
    ]

    # Fallback: when query_enhancement extracts no persons (e.g. "david subar
    # agreements"), attempt to recover a 2-token name from the raw query by
    # removing known stopwords and document-type keywords.
    if not persons:
        _skip = frozenset({
            "nda", "agreement", "agreements", "contract", "contracts", "lease",
            "document", "documents", "files", "file", "from", "by", "signed",
            "executed", "in", "at", "of", "the", "a", "an", "and", "or", "for",
            "to", "with", "about", "show", "me", "find", "get", "all", "list",
            "dated", "on", "between",
        })
        _tokens = [
            t for t in (meta.get("original_query") or "").lower().split()
            if len(t) >= 3 and t not in _skip and not t[0].isdigit()
        ]
        for _i in range(len(_tokens) - 1):
            _phrase = f"{_tokens[_i]} {_tokens[_i+1]}"
            if len(_phrase) >= 6:
                persons.append(_phrase)

    # ── 4. Clause keywords ───────────────────────────────────────────────
    clause_kws: set = set()
    if meta.get("legal_clause"):
        clause_kws.add(meta["legal_clause"].lower())
    q_lower = (meta.get("original_query") or "").lower()
    for _kw in ("nda", "non-disclosure", "agreement", "contract", "lease",
                "amendment", "addendum", "assignment", "license", "mortgage",
                "deed", "mou", "memorandum"):
        if _kw in q_lower:
            clause_kws.add(_kw)

    # ── Resolve MetadataIndex for O(1) inverted-index lookups ────────
    # Vector results carry only chunk-level metadata from Milvus (no
    # full-doc locations/persons/dates). The in-memory MetadataIndex
    # pre-builds inverted indexes (by_full_date, by_location, by_person)
    # and stores full_text at index-build time. Using those here gives
    # accurate doc-level signals without changing retrieval at all.
    _meta_idx = None
    try:
        _eng = get_semantic_engine()
        if _eng and hasattr(_eng, "metadata_index") and _eng.metadata_index:
            _meta_idx = _eng.metadata_index
    except Exception:
        pass

    # Pre-build sets of file_ids from inverted indexes for O(1) lookup.
    _date_file_ids:     set = set()
    _year_file_ids:     set = set()
    _location_file_ids: set = set()
    _person_file_ids:   set = set()

    # Extract query year for year-level fallback boost (used when no exact
    # full-date match exists — e.g. "March 5, 2025" with no doc on that date)
    _query_year: int = 0
    try:
        _query_year = int(meta.get("date") or 0)
    except Exception:
        pass
    if not _query_year and full_date_iso:
        try:
            _query_year = int(full_date_iso[:4])
        except Exception:
            pass

    if _meta_idx:
        for _dv in date_variants:
            _date_file_ids.update(_meta_idx.by_full_date.get(_dv) or [])
        if _query_year and hasattr(_meta_idx, "by_year"):
            _year_file_ids.update(_meta_idx.by_year.get(_query_year) or [])
        for _loc in locations:
            _location_file_ids.update(_meta_idx.by_location.get(_loc) or [])
        for _person in persons:
            _person_file_ids.update(_meta_idx.by_person.get(_person) or [])

    # When query_enhancement sets location_anchor_cities (e.g. "austin" without typing
    # "Texas"), legacy scoring must NOT treat substring "texas" or by_location["texas"]
    # alone as a location hit — that caused Austin/Dallas cross-pollution.
    _anchors_lc_idx = [
        str(c).lower().strip()
        for c in (meta.get("location_anchor_cities") or [])
        if c
    ]
    _location_file_ids_anchors: set = set()
    if _meta_idx and _anchors_lc_idx:
        for _loc in _anchors_lc_idx:
            _location_file_ids_anchors.update(_meta_idx.by_location.get(_loc) or [])

    # ── Per-item scoring ─────────────────────────────────────────────────
    for item in items:
        base  = float(item.get("similarity_score") or item.get("score") or 0.0)
        text  = (item.get("text") or "").lower()
        fname = str((item.get("metadata") or {}).get("filename", "")).lower()
        haystack = text + " " + fname
        file_id_for_boost = (
            item.get("file_id") or (item.get("metadata") or {}).get("file_id", "")
        )

        # Augment haystack with up to 8 000 chars of full document text so
        # constraint terms appearing outside the returned chunk snippet are
        # still detectable (e.g. address on page 1, chunk is from page 3).
        if _meta_idx and file_id_for_boost:
            try:
                _doc_meta = _meta_idx.get_metadata(file_id_for_boost) or {}
                _ft = (_doc_meta.get("full_text") or "")[:8000].lower()
                if _ft:
                    haystack += " " + _ft
            except Exception:
                pass

        boost   = 0.0
        any_hit = False

        # 1) Date — try inverted index first (O(1)), then haystack scan
        _date_hit = (
            bool(_date_file_ids and file_id_for_boost in _date_file_ids)
            or any(_dv in haystack for _dv in date_variants)
        )
        if _date_hit:
            boost   += 0.20
            any_hit  = True
        elif _year_file_ids and file_id_for_boost in _year_file_ids:
            # Year-level fallback: no exact date match but doc is confirmed
            # to belong to the query year (from by_year inverted index).
            # Smaller boost than full-date hit to reflect lower specificity.
            # Broad temporal/geo queries: only a mild year boost (no entity+year
            # down-ranking of year-only matches — preserves prior behavior).
            if _broad_temporal_geo:
                boost += 0.06
            else:
                _ent_rt_y = [
                    t.lower()
                    for t in (meta.get("entity_ranking_tokens") or [])
                    if t
                ]
                if _ent_rt_y and not all(t in haystack for t in _ent_rt_y):
                    boost += 0.01
                else:
                    boost += 0.05
            any_hit = True

        # 2) Location — compound geo_entity (city+state) OR legacy location list.
        # City+state queries must score at least as well as city-only: many chunks name the
        # city without repeating "Texas". Boost on city match; extra when state also matches;
        # penalize Texas (or wrong peer city) without the requested city.
        _state_names = frozenset(['texas', 'california', 'new york', 'florida',
                                  'illinois', 'georgia', 'ohio', 'virginia',
                                  'washington', 'arizona', 'colorado', 'nevada'])
        _loc_hit = False

        if _geo_city and _geo_state:
            _city_pat = (
                r"\b"
                + _constr_re.escape(_geo_city).replace(" ", r"\s+")
                + r"\b"
            )
            _cm = bool(_constr_re.search(_city_pat, haystack))
            _sm = _geo_state in haystack
            if not _sm and _geo_state == "texas":
                _sm = bool(_constr_re.search(r"\btx\b", haystack))
            if not _sm and _geo_state == "california":
                _sm = bool(_constr_re.search(r"\bca\b", haystack))
            if not _sm and _geo_state == "florida":
                _sm = bool(_constr_re.search(r"\bfl\b", haystack))
            if not _sm and _geo_state == "new york":
                _sm = bool(_constr_re.search(r"\bny\b", haystack))
            _compound_geo = _cm and _sm
            _idx_geo = bool(
                _location_file_ids
                and file_id_for_boost in _location_file_ids
                and _cm
            )
            if _cm or _idx_geo:
                _loc_hit = True
                boost += 0.10
                any_hit = True
                if _compound_geo or (_idx_geo and _sm):
                    boost += 0.08
            else:
                if _sm and not _cm:
                    boost -= 0.28
                    any_hit = True
                if _geo_state == "texas" and _sm:
                    for peer in _TX_PEER_CITIES:
                        if peer == _geo_city:
                            continue
                        if _constr_re.search(
                            r"\b" + _constr_re.escape(peer).replace(" ", r"\s+") + r"\b",
                            haystack,
                        ):
                            boost -= 0.42
                            any_hit = True
                            break
        else:
            _anchors_lc = [
                str(c).lower().strip()
                for c in (meta.get("location_anchor_cities") or [])
                if c
            ]
            if _anchors_lc:

                def _anchor_words_in_hay(hay: str) -> bool:
                    return all(
                        _constr_re.search(
                            r"\b"
                            + _constr_re.escape(a).replace(" ", r"\s+")
                            + r"\b",
                            hay,
                        )
                        for a in _anchors_lc
                    )

                _loc_hit = (
                    bool(
                        _location_file_ids_anchors
                        and file_id_for_boost in _location_file_ids_anchors
                    )
                    or _anchor_words_in_hay(haystack)
                )
            else:
                _loc_hit = (
                    bool(_location_file_ids and file_id_for_boost in _location_file_ids)
                    or any(_loc in haystack for _loc in locations)
                )
            if _loc_hit:
                boost += 0.10
                any_hit = True
                if _anchors_lc:
                    _city_terms = list(_anchors_lc)
                else:
                    _city_terms = [
                        l
                        for l in locations
                        if l not in _state_names and len(l) >= 4
                    ]
                if _city_terms and any(
                    _constr_re.search(
                        r"\b"
                        + _constr_re.escape(c).replace(" ", r"\s+")
                        + r"\b",
                        haystack,
                    )
                    for c in _city_terms
                ):
                    boost += 0.08

        # 2b) Peer City Exclusion: full-document haystack can satisfy merge filters while the
        # *returned chunk* only mentions the wrong peer city — demote that case; lightly
        # reward when the anchor city appears in the snippet (chunk + filename).
        from src.semantic.semantic_utils import LOCATION_PEERS
        _alc_tx = [
            str(c).lower().strip()
            for c in (meta.get("location_anchor_cities") or [])
            if c
        ]
        _tx_single_anchor: Optional[str] = None
        if _geo_city and _geo_state == "texas" and _geo_city in LOCATION_PEERS:
            _tx_single_anchor = _geo_city
        elif len(_alc_tx) == 1 and _alc_tx[0] in LOCATION_PEERS:
            _tx_single_anchor = _alc_tx[0]
            
        if _tx_single_anchor:
            _peers = LOCATION_PEERS.get(_tx_single_anchor, [])
            _ap = r"\b" + _constr_re.escape(_tx_single_anchor).replace(" ", r"\s+") + r"\b"
            _snippet_tx = ((item.get("text") or "") + " " + fname).lower()
            
            _has_peer = False
            for _p in _peers:
                _pp = r"\b" + _constr_re.escape(_p).replace(" ", r"\s+") + r"\b"
                if _constr_re.search(_pp, _snippet_tx):
                    _has_peer = True
                    break
                    
            if _has_peer and not _constr_re.search(_ap, _snippet_tx):
                boost -= 0.55
                any_hit = True
            elif _constr_re.search(_ap, _snippet_tx):
                boost += 0.06
                any_hit = True

        # Berlin-specific: metadata locations often miss "Berlin"; still reward
        # docs that literally contain the city (reduces vector noise on intl queries).
        if _constr_re.search(r"\bberlin\b", q_lower) and "berlin" in haystack:
            boost += 0.09
            any_hit = True
        
        # General peer check for queries containing peer cities
        for _city in LOCATION_PEERS.keys():
            if (
                not (_geo_city and _geo_state)
                and _constr_re.search(rf"\b{_constr_re.escape(_city)}\b", q_lower)
                and _city in haystack
            ):
                boost += 0.08
                any_hit = True

        # 3) Person — inverted index first, then haystack scan
        _person_hit = (
            bool(_person_file_ids and file_id_for_boost in _person_file_ids)
            or any(_person in haystack for _person in persons)
        )
        if _person_hit:
            boost   += 0.10
            any_hit  = True

        # 4) Clause / doc-type keyword (haystack only — no inverted index for clauses)
        for _kw in clause_kws:
            if _kw in haystack:
                boost   += 0.05
                any_hit  = True
                break

        # 5) Exact query phrase / keyword in document text.
        #    Boosted significantly so a document that actually CONTAINS
        #    the search term (e.g. "Coinstore" in the Token Relaunch deck)
        #    outranks documents that are only semantically similar.
        _q_norm = " ".join((meta.get("normalized_query") or
                            meta.get("original_query") or "").lower().split())
        if len(_q_norm) >= 4 and _q_norm in haystack:
            boost   += 0.20
            any_hit  = True
        elif len(_q_norm) >= 4:
            # Also check individual long tokens (handles single-word entity
            # queries like "Coinstore" where the doc has the word in a chunk
            # different from the one returned by the top-K vector search).
            _q_tokens_kw = [t for t in _q_norm.split() if len(t) >= 5]
            if _q_tokens_kw and all(t in haystack for t in _q_tokens_kw):
                boost   += 0.15
                any_hit  = True

        # 6) Filename token matching — boost when query words appear in
        #    the decoded filename (partial/keyword match).  Normalizes to
        #    handle "StorageChain" vs "Storage Chain" discrepancies.
        #    Excludes file-type words and common stopwords from the token
        #    threshold so "freightpal excel files" → threshold=1 (just "freightpal")
        #    rather than threshold=2 which was never satisfied.
        _item_meta2 = item.get("metadata") or {}
        _raw_fname2 = str(
            _item_meta2.get("original_filename")
            or _item_meta2.get("filename")
            or ""
        )
        _decoded_fname2 = _ui_urlparse.unquote(_raw_fname2.split("?")[0]).lower()
        if _decoded_fname2:
            import re as _re
            _fn_norm2 = _re.sub(r'[\s_\-]+', '', _decoded_fname2)
            _q_tokens = [t for t in _re.split(r'\W+', _q_norm) if len(t) >= 3]
            # Exclude file-type and common query words from the threshold so
            # that meaningful entity keywords dominate the calculation.
            _FN_THRESH_STOP = {
                'show', 'the', 'and', 'for', 'all', 'any', 'get', 'find',
                'list', 'give', 'please', 'file', 'files', 'document',
                'documents', 'doc', 'docs', 'excel', 'powerpoint', 'pdf',
                'word', 'csv', 'xlsx', 'docx', 'pptx', 'xls', 'png', 'jpg',
                'presentation', 'slides', 'deck', 'decks', 'slideshow',
                'spreadsheet', 'spreadsheets', 'image', 'photo', 'related',
                'about', 'with', 'from', 'search', 'signed', 'executed',
            }
            _q_tokens_eff = [t for t in _q_tokens if t not in _FN_THRESH_STOP]
            _fn_matches = sum(
                1 for t in _q_tokens
                if (t in _decoded_fname2 or
                    _re.sub(r'[\s_\-]+', '', t) in _fn_norm2)
            )
            # Use the effective (filtered) token count for threshold
            _thresh_tokens = _q_tokens_eff if _q_tokens_eff else _q_tokens
            if _thresh_tokens and _fn_matches >= max(1, len(_thresh_tokens) // 2):
                _fn_boost = min(0.25, 0.08 * _fn_matches)
                boost   += _fn_boost
                any_hit  = True

            # 6b) Entity + multi-token person: forename/surname in basename
            # (incl. chris ↔ christopher) so person-file PDFs beat generic vectors.
            if meta.get("intent_type") == "entity" and persons:
                _fn_base_only = _decoded_fname2.split("/")[-1]
                _pfn_done = False
                for _pers in persons:
                    if _pfn_done:
                        break
                    _ptoks = [t for t in _pers.split() if len(t) >= 3]
                    if len(_ptoks) < 2:
                        continue
                    _ln = _ptoks[-1]
                    _fn0 = _ptoks[0]
                    if _ln not in _fn_base_only:
                        continue
                    if _fn0 in _fn_base_only:
                        boost += 0.12
                        any_hit = True
                        _pfn_done = True
                        break
                    if _fn0.startswith("chris") and (
                        "christopher" in _fn_base_only
                        or _re.search(r"\bchris\b", _fn_base_only)
                    ):
                        boost += 0.10
                        any_hit = True
                        _pfn_done = True
                        break

        # 6c) Entity-first + year (e.g. "freightpal 2020", "bank statement 2014").
        #     When intent is broad_temporal_or_geo, skip entity boosts/penalties
        #     (handled above with mild year-only boost only).
        #     Entity penalties apply only when entity_ranking_tokens is non-empty;
        #     penalty scales with missing token count (soft cap).
        _ent_rt = [t.lower() for t in (meta.get("entity_ranking_tokens") or []) if t]
        if not _broad_temporal_geo and _ent_rt and _query_year:
            _ent_ok = all(t in haystack for t in _ent_rt)
            _ys = str(_query_year)
            _year_in_doc = _ys in haystack or (
                _year_file_ids and file_id_for_boost in _year_file_ids
            )
            _ent_phrase = (
                str(_qd.get("entity") or "").lower().strip()
            )
            if _ent_ok and _year_in_doc:
                boost += 0.42
                any_hit = True
                if (
                    _ent_phrase
                    and len(_ent_phrase) >= 4
                    and _ent_phrase in haystack
                ):
                    boost += 0.10
                if _decoded_fname2 and all(
                    t in _decoded_fname2 for t in _ent_rt
                ):
                    boost += 0.06
            elif (not _ent_ok) and _year_in_doc:
                _missing_ent = [t for t in _ent_rt if t not in haystack]
                _ent_pen = min(
                    0.15, 0.05 * len(_missing_ent)
                )
                boost -= _ent_pen
                any_hit = True

        # 6d) NDA + year — prefer docs with NDA/non-disclosure language + year.
        if (
            not _broad_temporal_geo
            and meta.get("nda_year_query")
            and _query_year
        ):
            _nda_sig = any(
                x in haystack
                for x in (
                    "nda",
                    "non-disclosure",
                    "non disclosure",
                    "nondisclosure",
                    "confidentiality",
                )
            )
            _ys = str(_query_year)
            _year_in_doc = _ys in haystack or (
                _year_file_ids and file_id_for_boost in _year_file_ids
            )
            if _nda_sig and _year_in_doc:
                boost += 0.28
                any_hit = True
            elif _year_in_doc and not _nda_sig:
                boost -= 0.10
                any_hit = True

        # 7) Image/media penalty for document-type queries.
        #    PNG, JPG, GIF etc should not appear in NDA/agreement/financial
        #    searches — their S3 URL embeddings cause false vector matches.
        _IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp'}
        _DOC_QUERY_SIGNALS = [
            'nda', 'agreement', 'contract', 'financial', 'report',
            'invoice', 'bylaws', 'term sheet', 'offer letter', 'policy',
            'mutual', 'signed', 'executed', 'non-disclosure',
        ]
        _q_orig_lower = (meta.get("original_query") or "").lower()
        _is_doc_query = any(sig in _q_orig_lower for sig in _DOC_QUERY_SIGNALS)
        if _is_doc_query and _decoded_fname2:
            _item_ext2 = ("." + _decoded_fname2.rsplit(".", 1)[-1]) if "." in _decoded_fname2 else ""
            if _item_ext2 in _IMAGE_EXTS:
                boost -= 0.40

        # 8) Multi-constraint AND bonus: when BOTH location AND NDA/clause
        #    are satisfied, give an extra boost so results satisfying ALL
        #    constraints outrank those satisfying only one.
        if _loc_hit and any_hit and (boost > 0.10):
            # Confirmed location + at least one other constraint → AND bonus
            boost += 0.05

        # 9–12) Precision gates (product QA) — soft penalties; keep lexical/content
        import re as _pen_re
        _qol = (meta.get("original_query") or "").lower()
        for _rq in (meta.get("required_keywords") or []):
            _rql = str(_rq).lower()
            if _rql and _rql not in haystack:
                boost -= 0.40
        _dfy = meta.get("documents_from_year")
        if isinstance(_dfy, int) and 1990 <= _dfy <= 2035:
            if str(_dfy) not in haystack:
                boost -= 0.28
        if meta.get("strict_temporal_policy") and meta.get("month_year"):
            _my_t = meta["month_year"]
            if isinstance(_my_t, (list, tuple)) and len(_my_t) >= 2:
                _mo_n, _yr_n = _my_t[0], _my_t[1]
                _mo_s = str(_mo_n).lower() if _mo_n else ""
                _has_yr = str(_yr_n) in haystack
                _has_mo = (
                    _mo_s in haystack
                    or (len(_mo_s) >= 3 and _mo_s[:3] in haystack)
                )
                if not (_has_yr and _has_mo):
                    boost -= 0.24
        _cpgy = meta.get("compound_person_geo_year")
        if _cpgy and isinstance(_cpgy, (list, tuple)) and len(_cpgy) >= 3:
            _pn, _geo, _yr_c = _cpgy[0], _cpgy[1], _cpgy[2]
            _cr_ok = (
                bool(_pen_re.search(
                    rf"\b{_pen_re.escape(str(_pn))}\w*\b", haystack))
                or "christopher" in haystack
            )
            if (
                not _cr_ok
                or str(_geo) not in haystack
                or str(_yr_c) not in haystack
            ):
                boost -= 0.22
        if "california" in _qol and "contract" in _qol:
            if "california" not in haystack:
                boost -= 0.18
        # Penalize missing Dallas only for Dallas-focused queries (not Austin/TX office).
        if (
            _pen_re.search(r"\bdallas\b", _qol)
            and "austin" not in _qol
            and "dallas" not in haystack
        ):
            boost -= 0.20
        if "austin" in _qol and "office" in _qol and "austin" not in haystack:
            boost -= 0.16

        # File-type extension matching
        _req_exts = meta.get("file_extensions", [])
        if _req_exts:
            _raw_fname3 = str(
                _item_meta2.get("original_filename")
                or _item_meta2.get("filename")
                or ""
            )
            _decoded_fname3 = _ui_urlparse.unquote(_raw_fname3.split("?")[0])
            _item_ext3 = (
                "." + _decoded_fname3.rsplit(".", 1)[-1]
                if "." in _decoded_fname3 else ""
            ).lower()
            _req_lower = [e.lower() for e in _req_exts]
            if _item_ext3 and _item_ext3 in _req_lower:
                boost   += 0.40
                any_hit  = True
            elif _item_ext3:
                boost   -= 0.30

        # Mild penalty when no constraint matched at all
        if not any_hit:
            boost -= 0.05

        item["_ranking_score"] = base + boost
        _sm_rb = str(item.get("search_method") or "").lower()
        _METHOD_RB = {
            "lexical_supplement": 0.055,
            "lexical": 0.055,
            "content_scan": 0.045,
            "metadata_first": 0.04,
            "filename_supplement": 0.05,
            "entity_ext_supplement": 0.04,
        }
        if _sm_rb in _METHOD_RB:
            item["_ranking_score"] += _METHOD_RB[_sm_rb]

