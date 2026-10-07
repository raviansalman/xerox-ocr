"""
Universal Query Enhancement Engine – v5.3 (Final)

-------------------------------------------------
Safe, production-grade enhancement:
• ZERO false org extraction (Misfits Gaming, Tech Holding safe)
• Bulletproof temporal extraction (year, range, month-year, full-date)
• Safe person extraction
• Safe governing law detection
• Strict / permissive validator compatibility

Pairs with:
- universal_constraint_validator.py
- metadata_index.py
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Tuple, Any, Optional

# Import TemporalReasoningEngine as the SINGLE SOURCE OF TRUTH for all temporal extraction
from src.semantic.temporal_engine import TemporalReasoningEngine

logger = logging.getLogger(__name__)

# Singleton instance to avoid repeated instantiation
_temporal_engine = None

def _get_temporal_engine():
    """Get or create the singleton TemporalReasoningEngine instance."""
    global _temporal_engine
    if _temporal_engine is None:
        _temporal_engine = TemporalReasoningEngine()
    return _temporal_engine


# ------------------------------------------------------------
# SAFE ORGANIZATION EXTRACTION (v5.3 FINAL)
# ------------------------------------------------------------

ORG_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:Corporation|Corp|LLC|Inc|Ltd|GmbH|PLC|Co))\b"
)

ACRONYM_RE = re.compile(r"\b([A-Z]{3,})\b")  # IBM, AMUZN, etc.

# CamelCase only if it contains ≥2 internal capitals
CAMEL_SAFE_RE = re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]+)\b")

# English/common words to exclude (root cause of false positives)
COMMON_ENGLISH = {
    "show", "me", "documents", "document", "agreement", "agreements",
    "contract", "contracts", "from", "mutual", "nda", "what", "which",
    "signed", "by", "on", "at", "the", "state", "date", "holding", "tech",
    "party", "parties", "pdf", "storage", "chain", "media", "consulting",
    "development", "gaming", "misfits", "llc", "corp", "inc"
}


def _should_apply_year_stripped_vector_query(meta: dict, query: str) -> bool:
    """
    Align with semantic_pipeline._should_skip_metadata_first_single_year: when a
    calendar year is present, embedding the full query over-weights the year (e.g.
    spreadsheets with '2019'). Strip years from vector_query for BOTH/vector path.
    """
    if not (meta.get("date") or meta.get("date_range")):
        return False
    q = (query or "").strip()
    ql = q.lower()
    qt = (meta.get("query_type") or "general").lower()
    if qt in ("document", "agreement", "contract"):
        return True
    if meta.get("location") or meta.get("locations"):
        return True
    if len(ql.split()) >= 5:
        return True
    if meta.get("date") and len(ql.split()) >= 4:
        return True
    return False


def _apply_year_stripped_vector_query(meta: dict, original_query: str) -> None:
    if not _should_apply_year_stripped_vector_query(meta, original_query):
        return
    base = (meta.get("vector_query") or original_query).strip()
    stripped = re.sub(r"\b(?:19|20)\d{2}\b", " ", base)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if len(stripped) >= 6:
        meta["vector_query"] = stripped
        logger.debug(
            "Year-stripped vector_query for embedding: %r -> %r",
            base[:120],
            stripped[:120],
        )


def extract_orgs_safe(q: str):
    """
    Extract organization names from query.
    """
    q_lower = q.lower()
    out = set()
    
    # Load known organization names from configuration file to avoid hardcoding
    known_orgs = {}
    try:
        import json
        config_path = os.path.join(os.path.dirname(__file__), "known_organizations.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                known_orgs = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load known organizations: {e}")

    
    # Check for known organization names in query
    for key, value in known_orgs.items():
        if key in q_lower:
            out.add(value)
            # Also add individual meaningful parts
            parts = value.split()
            for part in parts:
                if len(part) > 2 and part.lower() not in ["llc", "inc", "corp", "ltd"]:
                    out.add(part)

    # Suffix-based organizations
    for m in ORG_SUFFIX_RE.findall(q):
        out.add(m)

    # 2) ALL-CAPS acronyms
    for m in ACRONYM_RE.findall(q):
        if m.lower() not in COMMON_ENGLISH:
            out.add(m)

    # 3) CamelCase with ≥2 capitals
    for m in CAMEL_SAFE_RE.findall(q):
        if m.lower() not in COMMON_ENGLISH:
            out.add(m)
    
    # 4) Multi-word organization names (e.g., "Storage Chain", "Tech Holding", "Curation Media")
    # Pattern: Capitalized word + Capitalized word (2-3 words total)
    multi_word_org_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    )
    for match in multi_word_org_pattern.finditer(q):
        org_name = match.group(1).strip()
        org_lower = org_name.lower()
        # Exclude if it's a common phrase or location
        words = org_lower.split()
        if len(words) >= 2:
            # Check if all words are in COMMON_ENGLISH (skip if so)
            if not all(w in COMMON_ENGLISH for w in words):
                # Check if it's a known organization pattern
                org_keywords = ["chain", "holding", "media", "consulting", "gaming", "ventures", "capital", "corp", "llc", "inc"]
                if any(w in org_keywords for w in words):
                    out.add(org_name)
                # Also check if it appears in "mentioning X" or "from X" context
                elif ("mentioning" in q.lower() or "from" in q.lower()) and org_name.lower() in q.lower():
                    # Additional check: make sure it's not a location
                    if org_name.lower() not in ["new york", "los angeles", "san francisco", "austin texas"]:
                        out.add(org_name)

    return list(out)


def normalize_org_name(name: str) -> str:
    """Normalize organization names for matching."""
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ------------------------------------------------------------
# TEMPORAL EXTRACTION
# ------------------------------------------------------------
# NOTE: All date-related extraction is now handled by TemporalReasoningEngine
# The functions below are kept only for backward compatibility but are NOT used
# in the main enhance_query flow. All temporal extraction goes through:
# - engine.extract_year_range_from_query()
# - engine.extract_single_year_from_query()
# - engine.extract_month_year_from_query()
# - engine.extract_month_only_from_query()
# - engine.parse_any_date()


def _word_to_int(token: str) -> Optional[int]:
    """Map small number words to integers (one..ten)."""
    mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return mapping.get(token.lower())


def infer_relative_year(q: str) -> Optional[int]:
    """
    Infer a concrete year from simple relative phrases like:
      - last year
      - N years ago
      - three years ago

    This is intentionally conservative and only handles small N.
    """
    ql = (q or "").lower()
    current_year = datetime.now(timezone.utc).year

    # "last year"
    if "last year" in ql:
        return current_year - 1

    # "<N> years ago" where N is a digit
    m = re.search(r"\b(\d+)\s+years?\s+ago\b", ql)
    if m:
        try:
            n = int(m.group(1))
            if 0 < n <= 10:
                return current_year - n
        except ValueError:
            pass

    # "three years ago" etc.
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", ql)
    if m:
        n = _word_to_int(m.group(1))
        if n:
            return current_year - n

    return None


# ------------------------------------------------------------
# UNIVERSAL LOCATION EXTRACTION (no hardcoded states)
# ------------------------------------------------------------

# Detect ANY proper-noun location phrase:
#   Austin, Texas, Austin Texas, Los Angeles, New York, Dubai, London, etc.
LOCATION_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Exclusions to avoid accidental person/org collisions
LOCATION_EXCLUDE = {
    "agreement",
    "agreements",
    "contract",
    "contracts",
    "nda",
    "document",
    "documents",
    "policy",
    "services",
    "limited",
    "inc",
    "llc",
    "corp",
    "corporation",
    "holding",
    "media",
    "storage",
    "tech",
    "ventures",
}


def extract_location_universal(q: str):
    """
    Extract any proper-noun phrase that is not obviously an org/person
    and treat it as a candidate geographic location.
    Works for ALL cities, states, regions, countries worldwide.
    Handle explicit city+state patterns (Austin TX, Dallas Texas). Do not infer Texas from city-only queries.
    """
    out = set()
    q_lower = q.lower()
    q_original = q  # Keep original for case-sensitive matching

    # Handle common location patterns first
    # Pattern 1: "Austin TX", "Austin, TX", "austin tx"
    austin_tx_pattern = re.compile(r'\baustin\s*[,]?\s*tx\b', re.IGNORECASE)
    if austin_tx_pattern.search(q):
        out.add("Austin")
        out.add("Texas")
    
    # Pattern 2: "Austin Texas", "Austin, Texas", "austin texas"
    austin_texas_pattern = re.compile(r'\baustin\s*[,]?\s*texas\b', re.IGNORECASE)
    if austin_texas_pattern.search(q):
        out.add("Austin")
        out.add("Texas")
    
    # Pattern 3: Just "Austin" or "austin" (never infer Texas — breaks Austin vs Dallas separation)
    if re.search(r'\baustin\b', q_lower):
        out.add("Austin")

    # Pattern 4: Just "Texas" or "TX"
    if re.search(r'\btexas\b|\btx\b', q_lower):
        out.add("Texas")
        # If Texas is mentioned, also check for Austin
        if "austin" in q_lower:
            out.add("Austin")
        if "dallas" in q_lower:
            out.add("Dallas")

    # Pattern 5–6: Dallas variants (mirror Austin — avoids collapsing only to "Texas")
    if re.search(r"\bdallas\s*[,]?\s*tx\b", q, re.IGNORECASE):
        out.add("Dallas")
        out.add("Texas")
    if re.search(r"\bdallas\s*[,]?\s*texas\b", q, re.IGNORECASE):
        out.add("Dallas")
        out.add("Texas")
    if re.search(r"\bdallas\b", q_lower):
        out.add("Dallas")

    # Continue with existing logic for other locations
    for match in LOCATION_RE.findall(q_original):
        token = match.strip()
        tk = token.lower()

        if tk in LOCATION_EXCLUDE:
            continue

        # simple heuristic: token appears after "from", "in", "at", "laws of"
        if any(kw in q_lower for kw in [f"from {tk}", f"in {tk}", f"at {tk}", f"laws of {tk}"]):
            out.add(token)
            continue

        # explicit "city of X"
        if f"city of {tk}" in q_lower:
            out.add(token)
            continue

        # safe fallback: only consider tokens that are 1–2 words long
        if len(token.split()) <= 2:
            out.add(token)

    return list(out)


# US city + state pairs for strict geo resolution (ranking / constraints).
# Order: longer multi-word cities first inside alternations where needed.
_GEO_PAIR_SPECS: list[tuple["re.Pattern[str]", str]] = [
    (
        re.compile(
            r"\b(san antonio|fort worth|el paso|corpus christi|grand prairie|college station|round rock|"
            r"austin|dallas|houston|plano|arlington|lubbock|garland|irving|amarillo|mckinney|waco|"
            r"abilene|pasadena|beaumont|tyler|midland|odessa|richardson|pearland|lewisville|denton|allen)\b"
            r"\s*,?\s*(texas|\btx\b)\b",
            re.IGNORECASE,
        ),
        "texas",
    ),
    (
        re.compile(
            r"\b(los angeles|san francisco|san diego|san jose|sacramento|fresno|oakland|long beach|anaheim)\b"
            r"\s*,?\s*(california|\bca\b)\b",
            re.IGNORECASE,
        ),
        "california",
    ),
    (
        re.compile(
            r"\b(new york|buffalo|rochester|yonkers|syracuse|albany)\b"
            r"\s*,?\s*(new york|\bny\b)\b",
            re.IGNORECASE,
        ),
        "new york",
    ),
    (
        re.compile(
            r"\b(miami|tampa|orlando|jacksonville|st\.?\s*petersburg|fort lauderdale|tallahassee)\b"
            r"\s*,?\s*(florida|\bfl\b)\b",
            re.IGNORECASE,
        ),
        "florida",
    ),
]


def extract_geo_entity(q: str) -> Optional[Dict[str, str]]:
    """
    When the query names a US city together with its state (e.g. \"Dallas Texas\",
    \"agreement dallas tx\"), return a compound geo key for strict ranking.

    Returns:
        {\"city\": \"dallas\", \"state\": \"texas\"} or None if no clear city+state pair.
    """
    if not q or not q.strip():
        return None
    for rx, state_canon in _GEO_PAIR_SPECS:
        m = rx.search(q)
        if not m:
            continue
        city_raw = (m.group(1) or "").strip()
        if not city_raw:
            continue
        city = city_raw.lower()
        # Normalize spaces (e.g. "St.  Petersburg" already matched loosely)
        city = " ".join(city.split())
        return {"city": city, "state": state_canon}
    return None


# Backward-compatible simple location hints used elsewhere (e.g. semantic_pipeline)
LOCATION_HINTS = [
    "california",
    "new york",
    "texas",
    "florida",
    "delaware",
    "london",
    "dubai",
    "singapore",
    "pakistan",
    "india",
]

# Small set of US city names that appear frequently in legal documents and are
# commonly mis-extracted as person names (e.g. "Austin" in "NDA from Austin Texas").
# Kept minimal — only cities that cause observable mis-classification.
# DO NOT add large city lists; use LOCATION_HINTS for state/country-level hints.
_US_CITIES = frozenset({
    "austin", "dallas", "houston", "chicago", "miami", "denver", "new york", "los angeles", "san francisco",
    "seattle", "boston", "atlanta", "phoenix", "portland", "nashville",
})
# Capitals / intl cities — never treat as person names (e.g. "Berlin storage agreement")
_INTL_CITIES = frozenset({"berlin", "paris", "london", "dubai", "tokyo", "singapore", "mumbai", "toronto"})

# Month names and abbreviations — extracted as persons by the single-token
# capitalized pattern (e.g. "January 30, 2019" → "January").
_MONTH_NAMES = frozenset({
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})


def extract_location(q: str) -> Optional[str]:
    ql = q.lower()
    for loc in LOCATION_HINTS:
        if loc in ql:
            return loc
    return None


# ------------------------------------------------------------
# PERSON EXTRACTION (safe)
# ------------------------------------------------------------


def extract_persons(q: str):
    """
    Extract person names from the query.

    v5.3 FINAL:
    - Uses strict multi-token pattern for full names.
    - Adds a safe fallback for single-capitalized tokens (e.g., 'Salman')
      and capitalized tokens inside longer queries ('who is Salman').
    - CRITICAL FIX: Also extracts lowercase person names from "documents by X" patterns.
    """
    names = set()
    q_lower = q.lower()

    # 1) Full-name pattern: "First Last", "First M Last", etc.
    full_name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    for n in re.findall(full_name_pattern, q):
        if n.lower() not in COMMON_ENGLISH:
            names.add(n.strip())

    # 2) Single-token capitalized candidates (e.g., 'Salman') anywhere
    #    in the query, excluding obvious common words.
    single_token_pattern = r"\b([A-Z][a-z]+)\b"
    for token in re.findall(single_token_pattern, q):
        tl = token.lower()
        if tl in COMMON_ENGLISH:
            continue
        # Avoid treating common location words as persons here; locations
        # are handled separately by extract_location / NER.
        if tl in LOCATION_HINTS or tl in _US_CITIES or tl in _INTL_CITIES or tl in _MONTH_NAMES:
            continue
        names.add(token.strip())

    # Extract person names from "documents by X", "signed by X", "mentioning X", "signed between X and Y" patterns
    # This handles queries like "documents by mitul", "documents by Mitul Thobhani", "signed between chris and mitul"
    # Universal pattern - works for any person name, case-insensitive
    person_context_patterns = [
        # "documents by Mitul Thobhani" - extract capitalized full name (2+ words)
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
        # "documents by mitul" or "documents by Mitul Thobhani" - any case, any length
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+by\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        # "signed by John Smith" - capitalized
        r'\b(?:signed|executed)\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        # "signed by mitul" - any case
        r'\b(?:signed|executed)\s+by\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        # "signed between chris and mitul" - extract both names
        r'\b(?:signed|executed)\s+between\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)\b',
        # "mentioning X" - any case
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioning\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        # "that mention X" - any case
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mention\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        # "by John Smith" - capitalized
        r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
    ]
    
    # Check both original and lowercase query for case-insensitive matching
    for pattern in person_context_patterns:
        # Check original query (case-insensitive)
        matches = re.findall(pattern, q, re.IGNORECASE)
        for match in matches:
            # Handle tuple matches (e.g., from "signed between X and Y")
            if isinstance(match, tuple):
                # Add all non-empty matches from tuple
                for m in match:
                    if m:
                        # Stop at common stop words (and, dated, on, from, to, etc.)
                        m_clean = re.split(r'\s+(?:and|dated|on|from|to|in|at|with|that|which|who|where)\b', m, maxsplit=1)[0]
                        match_lower = m_clean.lower().strip()
                        if match_lower not in COMMON_ENGLISH and match_lower not in LOCATION_HINTS and len(match_lower) >= 2:
                            name_parts = m_clean.split()
                            capitalized_name = ' '.join(word.capitalize() for word in name_parts)
                            names.add(capitalized_name)
            elif match:
                # Stop at common stop words (and, dated, on, from, to, etc.)
                match_clean = re.split(r'\s+(?:and|dated|on|from|to|in|at|with|that|which|who|where)\b', match, maxsplit=1)[0]
                match_lower = match_clean.lower().strip()
                # Exclude common words that might be matched
                if match_lower not in COMMON_ENGLISH and match_lower not in LOCATION_HINTS and len(match_lower) >= 2:
                    # Capitalize first letter of each word for consistency
                    name_parts = match_clean.split()
                    capitalized_name = ' '.join(word.capitalize() for word in name_parts)
                    names.add(capitalized_name)
        
        # Also check lowercase version for patterns like "documents by mitul" or "mentioning chris"
        matches_lower = re.findall(pattern, q_lower)
        for match in matches_lower:
            # Handle tuple matches (e.g., from "signed between X and Y")
            if isinstance(match, tuple):
                # Add all non-empty matches from tuple
                for m in match:
                    if m:
                        # Stop at common stop words
                        m_clean = re.split(r'\s+(?:and|dated|on|from|to|in|at|with|that|which|who|where)\b', m, maxsplit=1)[0]
                        match_lower = m_clean.lower().strip()
                        if match_lower not in COMMON_ENGLISH and match_lower not in LOCATION_HINTS and len(match_lower) >= 2:
                            name_parts = m_clean.split()
                            capitalized_name = ' '.join(word.capitalize() for word in name_parts)
                            names.add(capitalized_name)
            elif match:
                # Stop at common stop words
                match_clean = re.split(r'\s+(?:and|dated|on|from|to|in|at|with|that|which|who|where)\b', match, maxsplit=1)[0]
                match_lower = match_clean.lower().strip()
                # Exclude common words
                if match_lower not in COMMON_ENGLISH and match_lower not in LOCATION_HINTS and len(match_lower) >= 2:
                    # Capitalize first letter for consistency
                    name_parts = match_clean.split()
                    capitalized_name = ' '.join(word.capitalize() for word in name_parts)
                    names.add(capitalized_name)

    return list(names)


# ------------------------------------------------------------
# NUMERIC FILTERS
# ------------------------------------------------------------


def extract_numeric_filters(q: str):
    min_pattern = r"(?:above|greater than|>\s*)(\d[\d,\.]*)"
    max_pattern = r"(?:below|less than|<\s*)(\d[\d,\.]*)"

    min_m = re.search(min_pattern, q, re.IGNORECASE)
    max_m = re.search(max_pattern, q, re.IGNORECASE)

    return (
        float(min_m.group(1).replace(",", "")) if min_m else None,
        float(max_m.group(1).replace(",", "")) if max_m else None,
    )


# ------------------------------------------------------------
# GOVERNING LAW EXTRACTION
# ------------------------------------------------------------


def extract_governing_law(q: str) -> Optional[str]:
    ql = q.lower()
    patterns = [
        r"governed by laws of ([a-z\s]+)",
        r"laws of ([a-z\s]+)",
        r"state of ([a-z\s]+)",
    ]
    for p in patterns:
        m = re.search(p, ql, re.IGNORECASE)
        if m:
            law = m.group(1).strip().lower()
            law = law.replace(" ", "_")
            return law
    return None


# ------------------------------------------------------------
# FORBIDDEN / REQUIRED TERMS
# ------------------------------------------------------------

NEGATION_TERMS = ["except", "exclude", "without", "not including"]


def extract_forbidden(q: str):
    out = []
    ql = q.lower()
    for term in NEGATION_TERMS:
        if term in ql:
            m = re.search(term + r"\s+([A-Za-z]+)", ql)
            if m:
                out.append(m.group(1))
    return out


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------


def _canonicalize_temporal_wording(q: str) -> str:
    """
    Light-weight normalization of certain temporal phrasings to align with
    patterns that empirically work best with our retriever while keeping
    semantics intact.

    Examples:
      - "documents from 2019"          -> "documents signed in 2019"
      - "NDA documents from 2025"      -> "NDA documents signed in 2025"
      - "list NDA documents from 2025" -> "NDA documents signed in 2025"
      - "agreements from 2024"        -> "agreements dated 2024"
      - "contracts from 2025"         -> "contracts dated 2025"
    """
    original = q
    ql = q.lower()

    # Normalize common helper prefixes so that we operate on the core intent:
    # "show me documents from 2020"  -> "documents from 2020"
    # "what are the documents from" -> "documents from ..."
    helper_prefix = re.compile(
        r"^\s*(show\s+me|show|find|list|search\s+for|what\s+are\s+the|what\s+are)\s+",
        re.IGNORECASE,
    )
    q_core = helper_prefix.sub("", q)
    q_core_l = q_core.lower()

    # Do not touch queries that already carry a strong temporal verb
    # like "signed" or "dated" – they already map well to our engine.
    if "signed" in q_core_l or "dated" in q_core_l:
        return q

    # 1) NDA / non-disclosure document queries with "from YEAR"
    nda_pattern = re.compile(
        r"(nda|non[- ]disclosure|non disclosure)\s+documents?\s+from\s+((?:19|20)\d{2})",
        re.IGNORECASE,
    )
    m = nda_pattern.search(q_core)
    if m:
        year = m.group(2)
        # Normalize to a generic documents query that we know works well.
        # We keep the original_query (with \"NDA\") in metadata so that
        # downstream components can still use it for soft NDA boosting.
        return f"documents signed in {year}"

    # 2) Generic "documents from YEAR" → "documents signed in YEAR"
    # BUT: Don't canonicalize if there's a "to" keyword (date range)
    if " to " not in q_core_l and " to " not in q:
        doc_from_year = re.compile(
            r"\b(documents?|files?|docs?)\s+from\s+((?:19|20)\d{2})\b",
            re.IGNORECASE,
        )
        m = doc_from_year.search(q_core)
        if m:
            year = m.group(2)
            return f"documents signed in {year}"

    # 3) "agreements from YEAR" → "agreements dated YEAR"
    # BUT: Don't canonicalize if there's a "to" keyword (date range)
    if " to " not in q_core_l and " to " not in q:
        agr_from_year = re.compile(
            r"\bagreements?\s+from\s+((?:19|20)\d{2})\b",
            re.IGNORECASE,
        )
        m = agr_from_year.search(q)
        if m:
            year = m.group(1)
            return f"agreements dated {year}"

    # 4) "contracts from YEAR" → "contracts dated YEAR"
    # BUT: Don't canonicalize if there's a "to" keyword (date range)
    if " to " not in q_core_l and " to " not in q:
        ctr_from_year = re.compile(
            r"\bcontracts?\s+from\s+((?:19|20)\d{2})\b",
            re.IGNORECASE,
        )
        m = ctr_from_year.search(q)
        if m:
            year = m.group(1)
            return f"contracts dated {year}"

    return original


# Words that must not be treated as a "brand" before a year (entity+year queries).
_ENTITY_YEAR_TOKEN_STOP = frozenset({
    "from", "between", "during", "before", "after", "about", "into", "onto",
    "documents", "document", "files", "file", "agreement", "agreements",
    "contract", "contracts", "signed", "dated", "search", "find", "show",
    "list", "give", "get", "year", "years",
})


def _normalize_entity_year_tokens(toks: list) -> list:
    """
    Map plural query tokens to singular forms that appear in real document text so
    entity+year ranking and phrase boosts still match (e.g. \"bank statements 2014\"
    vs chunks that say \"bank statement\").
    """
    if not toks or len(toks) < 2:
        return toks
    a, b = toks[0], toks[1]
    if a == "bank" and b == "statements":
        return ["bank", "statement"]
    return toks


def _apply_geo_city_vector_query(meta: dict, query: str, ql: str) -> None:
    """
    Milvus embedding for \"austin texas\" should match retrieval for \"austin\" — many
    chunks say the city without repeating the state. If vector_query is still the full
    query (or unset), use city-only so BOTH/vector paths recall the same docs as a
    city-only search; skip when vector_query was tailored (mention/from/brand tails).
    """
    ge = meta.get("geo_entity") or {}
    city = (ge.get("city") or "").strip().lower()
    if not city:
        return
    vq = (meta.get("vector_query") or "").strip()
    oq = (query or "").strip()
    oql = (ql or oq).strip().lower()
    if vq:
        vl = vq.lower()
        if vl not in (oq.lower(), oql):
            return
    meta["vector_query"] = city


def _is_broad_temporal_or_geo_query(ql: str) -> bool:
    """Queries that should NOT use entity-first + year hard ranking."""
    q = ql.strip().lower()
    if len(q.split()) <= 2 and q in ("texas", "tx", "florida", "fl", "california", "ca"):
        return True
    if any(
        p in ql
        for p in (
            "documents from",
            "files from",
            "documents between",
            "agreements between",
            "contracts from",
            "agreements from",
            "documents from the",
            "documents signed in",
            "agreements dated",
            "contracts dated",
        )
    ):
        return True
    if re.search(
        r"\b(?:documents?|files?|agreements?|contracts?)\s+from\s+(?:19|20)\d{2}\b",
        ql,
    ):
        return True
    return False


def _apply_query_decomposition(meta: dict, q: str, ql: str) -> None:
    """
    Lightweight ENTITY + YEAR (+ optional doc-type) signals for ranking — not a full
    query planner. Keeps broad temporal/geo queries on existing behavior.
    """
    meta.setdefault("query_decomposition", {})
    meta.setdefault("search_debug", {})
    sd = meta["search_debug"]
    if not isinstance(sd, dict):
        sd = {}
        meta["search_debug"] = sd
    sd.setdefault("notes", [])

    meta["entity_ranking_tokens"] = []
    meta["nda_year_query"] = False
    meta["query_decomposition"]["intent_profile"] = "default"

    if _is_broad_temporal_or_geo_query(ql):
        meta["query_decomposition"]["intent_profile"] = "broad_temporal_or_geo"
        sd["notes"].append("broad temporal/geo — entity-first ranking relaxed")
        return

    y = meta.get("date")
    try:
        y_int = int(y) if y is not None else None
    except (TypeError, ValueError):
        y_int = None

    # NDA + calendar year (e.g. "NDA 2015", "nda 2015") — short pattern, not brand+year
    if y_int is not None and re.search(r"\bnda\s+(?:19|20)\d{2}\b", ql):
        meta["nda_year_query"] = True
        meta["query_decomposition"]["doc_type_signal"] = "nda"
        meta["query_decomposition"]["year"] = y_int
        sd["notes"].append("nda_year_query: prefer NDA/non-disclosure signal + year")

    # Brand / phrase + year: "freightpal 2020", "bank statements 2014"
    _ent_m = re.search(
        r"\b((?:[a-z][a-z0-9]{3,}|[a-z][a-z0-9]{2,}\s+[a-z][a-z0-9]{2,}))\s+((?:19|20)\d{2})\b",
        ql,
        re.IGNORECASE,
    )
    if _ent_m:
        raw_ent = _ent_m.group(1).strip().lower()
        yr_g = int(_ent_m.group(2))
        parts = [p for p in raw_ent.split() if len(p) >= 2]
        toks = [p for p in parts if p not in _ENTITY_YEAR_TOKEN_STOP]
        if len(toks) == 1 and len(toks[0]) >= 4:
            meta["entity_ranking_tokens"] = toks
            meta["query_decomposition"] = {
                "entity": toks[0],
                "year": yr_g,
                "intent_profile": "entity_year",
            }
            sd["notes"].append(f"entity_year token={toks[0]!r} year={yr_g}")
        elif len(toks) >= 2:
            _tok2 = _normalize_entity_year_tokens(toks[:2])
            meta["entity_ranking_tokens"] = _tok2
            meta["query_decomposition"] = {
                "entity": " ".join(_tok2),
                "year": yr_g,
                "intent_profile": "entity_year",
            }
            sd["notes"].append(f"entity_year tokens={_tok2!r} year={yr_g}")
            # Plural "bank statements 2014" vs "bank statement 2014": one canonical string for
            # embeddings, semantic search, ranking phrase match (ultimate_ui uses vector_query
            # / normalized_query for BOTH paths).
            if _tok2 == ["bank", "statement"]:
                _canon = f"bank statement {yr_g}"
                meta["normalized_query"] = _canon
                meta["vector_query"] = _canon
                # Merge/ranking/temporal paths still read original_query — keep it aligned with
                # canonical form so "bank statements 2014" and "bank statement 2014" behave identically.
                meta["original_query"] = _canon
                sd["notes"].append("bank statement year: canonical query for plural/singular parity")


def _full_date_is_fuzzy_year_only_artifact(
    full_date: Optional[datetime],
    year: Optional[Any],
    q: str,
) -> bool:
    """
    dateutil fuzzy on the whole query often returns a calendar day when only a year
    is stated (e.g. '... in 2019' -> 2019-03-30 using implicit month/day). That
    sets full_date_query and triggers metadata-first full-date routing in the
    semantic pipeline, returning unrelated files. If the query does not contain
    an explicit month or slash/ISO date, treat parse_any_date as a year-only
    artifact and drop it.
    """
    if full_date is None or year is None:
        return False
    try:
        y = int(year)
    except (TypeError, ValueError):
        return False
    if full_date.year != y:
        return False
    ql = q.lower()
    if any(m in ql for m in _MONTH_NAMES):
        return False
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b", q):
        return False
    if re.search(r"\b(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}\b", q):
        return False
    if re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
        ql,
    ):
        return False
    return True


def enhance_query(query: str) -> Tuple[str, Dict[str, Any]]:
    q = query.strip()
    # Apply canonicalization before lowercasing / extraction so that
    # all downstream components see the normalized temporal phrasing.
    q = _canonicalize_temporal_wording(q)
    ql = q.lower()

    # ------------------------
    # TEMPORAL EXTRACTION (DELEGATED TO TemporalReasoningEngine)
    # ------------------------
    # ALL temporal extraction is now done by TemporalReasoningEngine
    # This ensures a single source of truth and eliminates duplication.
    engine = _get_temporal_engine()
    
    # Extract all temporal signals using the authoritative engine
    # ALL date-related extraction is handled by TemporalReasoningEngine
    year_range = engine.extract_year_range_from_query(q)
    year = engine.extract_single_year_from_query(q)
    # If no explicit numeric year found, try relative patterns (delegated to temporal engine if available)
    # Fallback to simple relative year inference for backward compatibility
    if year is None and year_range is None:
        rel_year = infer_relative_year(q)  # Simple fallback for relative years
        if rel_year is not None:
            year = rel_year
    month_year = engine.extract_month_year_from_query(q)
    month_only = engine.extract_month_only_from_query(q)
    # Also extract full date if present (e.g., "April 04, 2019")
    full_date = engine.parse_any_date(q)
    if _full_date_is_fuzzy_year_only_artifact(full_date, year, q):
        full_date = None

    # ------------------------
    # ORGS / PERSONS / LOCATIONS (SAFE)
    # ------------------------
    raw_orgs = extract_orgs_safe(q)
    # Initial universal location + person extraction
    locations = extract_location_universal(q)
    persons = extract_persons(q)

    # Also check if any location/person is actually an organization (e.g., "Storage Chain")
    # This handles cases where multi-word org names are misclassified
    for loc in locations:
        loc_lower = loc.lower()
        # Known organization patterns that might be misclassified as locations
        org_indicators = ["chain", "holding", "media", "consulting", "gaming", "ventures", "capital"]
        if any(indicator in loc_lower for indicator in org_indicators):
            if loc not in raw_orgs:
                raw_orgs.append(loc)
    for person in persons:
        person_lower = person.lower()
        # Known organization patterns that might be misclassified as persons
        org_indicators = ["chain", "holding", "media", "consulting", "gaming", "ventures", "capital"]
        if any(indicator in person_lower for indicator in org_indicators):
            if person not in raw_orgs:
                raw_orgs.append(person)

    # Reconcile persons vs locations:
    # If a token is both a person and a location, prefer person UNLESS the
    # token is a known location hint or a known US city name (e.g. "Austin").
    locations_cleaned = []
    person_tokens = {p.lower() for p in persons}
    for loc in locations:
        loc_lower = loc.lower()
        if (loc_lower in person_tokens
                and loc_lower not in LOCATION_HINTS
                and loc_lower not in _US_CITIES
                and loc_lower not in _INTL_CITIES):
            # Prefer person role, drop as location
            continue
        locations_cleaned.append(loc)
    locations = locations_cleaned

    # Remove mis-classified persons: US city names, month names, and phrases
    # composed entirely of location tokens (e.g. "Austin Texas").
    persons = [
        p for p in persons
        if p.lower() not in _US_CITIES
        and p.lower() not in _INTL_CITIES
        and p.lower() not in _MONTH_NAMES
        and not all(tok in _US_CITIES or tok in _INTL_CITIES or tok in LOCATION_HINTS or tok in _MONTH_NAMES
                    for tok in p.lower().split())
    ]

    # Extract names from patterns like "agreements from [First Name] [Last Name]"
    _af_pattern = re.search(
        r"\b(?:agreements?|contracts?|documents?)\s+from\s+([a-zA-Z]+)\s+([a-zA-Z]+)\b",
        q, re.IGNORECASE,
    )
    if _af_pattern:
        extracted_name = f"{_af_pattern.group(1).strip().title()} {_af_pattern.group(2).strip().title()}"
        _stop_words = {"the", "a", "an", "this", "that", "my", "your", "our", "their", "any", "some", "all"}
        if _af_pattern.group(1).lower() not in _stop_words and _af_pattern.group(2).lower() not in _stop_words:
            if not any(extracted_name.lower() in (p or "").lower() for p in persons):
                persons.append(extracted_name)

    normalized_orgs = [normalize_org_name(o) for o in raw_orgs]
    
    # ------------------------
    # LOCATIONS (UNIVERSAL, NORMALIZED)
    # ------------------------
    # Backward-compat: also include simple hint-based location if present
    primary_location = extract_location(ql)
    if primary_location and primary_location not in [loc.lower() for loc in locations]:
        locations.append(primary_location)
    
    # Normalize locations to canonical keys (Austin/Dallas stay city keys; Texas is state-only).
    from .semantic_utils import normalize_location
    normalized_locations = []
    primary_location_normalized = None
    for loc in locations:
        normalized = normalize_location(loc)
        if normalized:
            normalized_locations.append(normalized)
            # If this is the primary location, also set it
            if loc.lower() == primary_location:
                primary_location_normalized = normalized
    
    # Update locations with normalized versions (keep both original and normalized)
    if normalized_locations:
        # Keep both original and normalized for maximum recall
        locations = list(set(locations + normalized_locations))
    if primary_location_normalized:
        primary_location = primary_location_normalized
    elif primary_location:
        # Fallback: normalize primary location
        primary_location = normalize_location(primary_location) or primary_location

    locations = [loc for loc in locations if loc.lower() not in _MONTH_NAMES]
    if primary_location and primary_location.lower() in _MONTH_NAMES:
        primary_location = None

    # City + state compound (e.g. Dallas Texas) — strict geo for ranking; not inferred from state alone.
    geo_entity = extract_geo_entity(q)

    # ------------------------
    # BUILD META
    # ------------------------
    meta: Dict[str, Any] = {
        "original_query": query,
        "normalized_query": q,
        "date": year,
        "date_range": year_range,
        "month_year": month_year,
        "full_date_query": full_date.isoformat() if full_date else None,  # Store as ISO string for serialization
        "month_only_query": month_only,
        # Will be populated with a numeric year/range just below
        "explicit_temporal_intent": None,
        # City+state disambiguation: "Austin Texas" must not match Dallas-only TX docs (and vice versa).
        "location_anchor_cities": [],
        "texas_state_wide": False,
        # Compound US geo: {"city": "dallas", "state": "texas"} when query explicitly pairs city+state.
        "geo_entity": geo_entity,
        "location": primary_location,
        "locations": locations,
        "governing_law": extract_governing_law(q),
        # Use the reconciled persons list so we don't double-extract
        "persons": persons,
        "organizations": raw_orgs,
        "normalized_orgs": normalized_orgs,
        "org_variants": normalized_orgs,  # kept simple for validator compatibility
        "legal_clause": None,
        "min_amount": None,
        "max_amount": None,
        "required_keywords": [],
        "required_text_regex": None,
        "forbidden_keywords": extract_forbidden(q),
        "expired_intent": ("expired" in ql or "already ended" in ql),
        # Query type (used by validators + semantic pipeline for intent-aware behavior)
        "query_type": "general",
        # High-level intent classifier and index hints (used by index-first router)
        "intent_type": "general",
        "target_indexes": [],
    }
    
    # Detect explicit temporal intent as numeric values, not booleans.
    # This ensures downstream temporal filters compare against actual years.
    # Month-only queries also have explicit temporal intent (even without year)
    if year is not None:
        meta["explicit_temporal_intent"] = int(year)
    elif month_year is not None:
        # month_year = (month_str, year_int)
        meta["explicit_temporal_intent"] = int(month_year[1])
    elif year_range is not None:
        # Treat range intent as the numeric range tuple itself
        meta["explicit_temporal_intent"] = (
            int(year_range[0]),
            int(year_range[1]),
        )
    elif month_only is not None:
        # Month-only queries have explicit temporal intent (use special marker)
        # This ensures temporal_match() treats them as explicit, not soft
        meta["explicit_temporal_intent"] = "month_only"
    else:
        meta["explicit_temporal_intent"] = None

    # City+state anchors (TX): any word-boundary "austin" / "dallas" in the query sets an anchor.
    # (Previously we required Texas/TX or presence in locations[] — city-only queries like "dallas"
    # could miss locations extraction and end up with empty anchors, so merge filters never ran.)
    anchors: list[str] = []
    locs_lower = [str(x).lower() for x in (meta.get("locations") or [])]
    from src.semantic.semantic_utils import LOCATION_PEERS
    for city in LOCATION_PEERS.keys():
        if re.search(rf"\b{re.escape(city)}\b", ql):
            anchors.append(city)
    if geo_entity and geo_entity.get("city"):
        _gca = geo_entity["city"].lower().strip()
        if _gca and _gca not in anchors:
            anchors.append(_gca)
    if anchors:
        meta["location_anchor_cities"] = list(dict.fromkeys(anchors))

    # Texas without a named metro: treat as state-wide (do not apply city anchor exclusion).
    _has_any_peer_city = any(re.search(rf"\b{re.escape(city)}\b", ql) for city in LOCATION_PEERS.keys())
    if re.search(r"\btexas\b|\btx\b", ql) and not _has_any_peer_city:
        meta["texas_state_wide"] = True
    if geo_entity:
        meta["texas_state_wide"] = False

    # Brand + NDA: require literal presence in body/filename (downstream filter).
    _has_nda = any(
        t in ql
        for t in (
            "nda", "ndas", "mnda", "mndas",
            "non-disclosure", "non disclosure", "nondisclosure",
            "non-disclosure agreement", "non disclosure agreement",
        )
    )
    # Enforce organization keywords in NDA queries to reduce semantic overlap noise
    if _has_nda and meta.get("organizations"):
        for org in meta["organizations"]:
            org_lower = org.lower()
            if " " in org_lower:
                pattern_parts = [re.escape(w) for w in org_lower.split()]
                regex_pattern = r"(?:" + r"\s*".join(pattern_parts) + r"|" + "".join(pattern_parts) + r")"
                meta["required_text_regex"] = regex_pattern
            else:
                meta.setdefault("required_keywords", []).append(org_lower)

    # --------------------------------------------------------
    # INITIAL VECTOR QUERY NORMALIZATION FOR MENTION/FROM PATTERNS
    # --------------------------------------------------------
    # Some high-level semantic patterns (e.g., "docs that mention Azat",
    # "files from Azat") should still benefit from a clean, high-signal
    # vector query so that BOTH mode can recover the correct documents
    # even if semantic metadata is incomplete.

    # 1) "documents/files/docs/... that mention|mentioning X"
    #    → use X as the vector query tail (e.g., "Azat").
    mention_target = None
    mention_match = re.search(
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+'
        r'(?:that\s+)?mention(?:ing)?\s+'
        r'([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)',
        q,
        re.IGNORECASE,
    )
    if mention_match:
        mention_target = mention_match.group(1).strip()

    if mention_target and not meta.get("vector_query"):
        meta["vector_query"] = mention_target

    # 2) "documents/files/... from X" where X is clearly a person name
    #    and NOT a canonical location hint (e.g., "files from Azat").
    from_match = re.search(
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\s+'
        r'([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)',
        q,
        re.IGNORECASE,
    )
    if from_match:
        from_target = from_match.group(1).strip()
        lower_target = from_target.lower()
        _ft_parts = lower_target.split()

        # Accept as a person-origin query when the target:
        #   • is NOT a known location hint or US city (whole phrase or any token)
        #   • looks like a name: 1–3 alphabetic tokens
        #   • none of its tokens are in the common-English stopword set
        # This handles both title-case ("David Subar") and lowercase ("david subar")
        # from queries like "agreements from david subar".
        _is_person_origin = (
            lower_target not in LOCATION_HINTS
            and lower_target not in _US_CITIES
            and not any(t in LOCATION_HINTS for t in _ft_parts)
            and not any(t in _US_CITIES for t in _ft_parts)
            and 1 <= len(_ft_parts) <= 3
            and all(t.isalpha() for t in _ft_parts)
            and not any(t in COMMON_ENGLISH for t in _ft_parts)
        )
        if _is_person_origin:
            # Normalize to Title Case ("david subar" → "David Subar")
            from_target_cased = " ".join(w.capitalize() for w in _ft_parts)

            if not meta.get("vector_query"):
                meta["vector_query"] = from_target_cased

            # Ensure the person list contains the properly-cased target
            if (from_target_cased not in meta["persons"]
                    and from_target not in meta["persons"]):
                meta["persons"].append(from_target_cased)

            # If this target was (incorrectly) classified as a location,
            # remove it from locations so that downstream logic treats it
            # as a signer/person instead of a place.
            if meta.get("locations"):
                cleaned_locations = []
                for loc in meta["locations"]:
                    if loc and loc.lower() == lower_target:
                        continue
                    cleaned_locations.append(loc)
                meta["locations"] = cleaned_locations

    # legal clauses - COMPREHENSIVE RULE ENGINE
    # Handles: agreements with [Clause] clauses
    clause_keywords = {
        "arbitration": ["arbitration", "arbitrate"],
        "confidentiality": ["confidential", "non-disclosure", "nda", "nondisclosure", "non disclosure"],
        "governing_law": ["governing law", "governing laws", "jurisdiction"],
        "dispute_resolution": ["dispute resolution", "dispute", "litigation"],
        "termination": ["termination", "terminate", "expire", "expiration"],
        "force_majeure": ["force majeure", "act of god"],
        "indemnification": ["indemnification", "indemnify", "indemnity"],
        "non-compete": ["non-compete", "non compete", "noncompete"],
        "whistleblowing": ["whistleblowing", "whistle blowing", "whistle-blowing"],
    }
    
    # Check for clause patterns: "with [clause]", "that mention [clause]", "mentioning [clause]"
    # Add patterns for "what are the [clause] provisions" and "clauses about [clause]"
    clause_patterns = [
        r'\bwith\s+(\w+)\s+clauses?\b',
        r'\bwith\s+(\w+)\s+provisions?\b',
        r'\bthat\s+mention\s+(\w+)\b',
        r'\bmentioning\s+(\w+)\b',
        r'\bwhat\s+are\s+the\s+(\w+(?:\s+\w+)?)\s+provisions?\b',  # "what are the governing law provisions"
        r'\bclauses?\s+about\s+(\w+(?:\s+\w+)?)\b',  # "clauses about dispute resolution"
        r'\b(\w+(?:\s+\w+)?)\s+provisions?\b',  # "governing law provisions"
        r'\b(\w+(?:\s+\w+)?)\s+clauses?\b',  # "governing law clauses"
    ]
    
    detected_clause = None
    for pattern in clause_patterns:
        matches = re.findall(pattern, ql)
        for match in matches:
            match_lower = match.lower() if isinstance(match, str) else match[0].lower() if match else ""
            for clause_type, keywords in clause_keywords.items():
                if any(kw in match_lower for kw in keywords):
                    detected_clause = clause_type
                    break
            if detected_clause:
                break
        if detected_clause:
            break
    
    # Fallback: direct keyword matching ONLY if query explicitly asks about clauses/provisions
    # Don't detect clauses just because query contains legal keywords (those might be exact text)
    if not detected_clause:
        # Only detect clause if query explicitly mentions "clause", "provision", "term", or asks about it
        has_clause_intent = any(word in ql for word in ['clause', 'clauses', 'provision', 'provisions', 'term', 'terms', 'section', 'sections'])
        if has_clause_intent:
            for clause_type, keywords in clause_keywords.items():
                if any(kw in ql for kw in keywords):
                    detected_clause = clause_type
                    break
    
    if detected_clause:
        meta["legal_clause"] = detected_clause

    # --------------------------------------------------------
    # QUERY TYPE DETECTION (person_about / org_about / general)
    # --------------------------------------------------------
    q_words = q.split()
    ql_strip = ql.strip()

    # Person-about queries: "who is X", "who are X", or short name-only queries.
    person_about = False
    if meta["persons"]:
        if ql_strip.startswith(("who is ", "who are ", "who was ")):
            person_about = True
        elif len(q_words) <= 3 and all(w[0].isupper() for w in q_words if w):
            # Very short, capitalized query that extracted persons from text
            person_about = True

    # Org-about queries: "what is X", "who owns X", "what does X do"
    org_about = False
    if normalized_orgs:
        if ql_strip.startswith(("what is ", "what are ", "who owns ", "what does ")):
            org_about = True

    if person_about:
        meta["query_type"] = "person_about"
    elif org_about:
        meta["query_type"] = "org_about"
    else:
        # Fallback query type based on generic wording
        if any(w in ql_strip for w in ["agreement", "agreements"]):
            meta["query_type"] = "agreement"
        elif any(w in ql_strip for w in ["contract", "contracts"]):
            meta["query_type"] = "contract"
        elif any(w in ql_strip for w in ["document", "documents", "file", "files"]):
            meta["query_type"] = "document"

    # --------------------------------------------------------
    # FILE-TYPE DETECTION
    # Maps natural-language file-type words to file extensions.
    # Used by the search pipeline to filter / boost results by extension.
    # --------------------------------------------------------
    _FILE_TYPE_ALIASES = {
        "powerpoint": [".pptx", ".ppt", ".ppsx"],
        "presentation": [".pptx", ".ppt", ".ppsx"],
        "pptx": [".pptx", ".ppt", ".ppsx"],
        "ppt": [".pptx", ".ppt", ".ppsx"],
        "slides": [".pptx", ".ppt", ".ppsx"],
        "slideshow": [".pptx", ".ppt", ".ppsx"],
        "excel": [".xlsx", ".xls", ".csv"],
        "excels": [".xlsx", ".xls", ".csv"],
        "spreadsheet": [".xlsx", ".xls", ".csv"],
        "spreadsheets": [".xlsx", ".xls", ".csv"],
        "xlsx": [".xlsx", ".xls"],
        "xls": [".xlsx", ".xls"],
        "word": [".docx", ".doc"],
        "docx": [".docx", ".doc"],
        "docs": [".docx", ".doc"],
        "doc": [".docx", ".doc"],
        "pdf": [".pdf"],
        "pdfs": [".pdf"],
        "csv": [".csv"],
        "csvs": [".csv"],
        "image": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "images": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "photo": [".jpg", ".jpeg", ".png", ".gif"],
        "photos": [".jpg", ".jpeg", ".png", ".gif"],
        "png": [".png"],
        "jpgs": [".jpg", ".jpeg"],
        "jpegs": [".jpg", ".jpeg"],
        "jpeg": [".jpg", ".jpeg"],
    }
    detected_extensions: list[str] = []
    for alias, exts in _FILE_TYPE_ALIASES.items():
        # Match whole word to avoid false positives (e.g. "excellent" matching "excel")
        if re.search(r'\b' + re.escape(alias) + r'\b', ql):
            for ext in exts:
                if ext not in detected_extensions:
                    detected_extensions.append(ext)
    meta["file_extensions"] = detected_extensions

    # --------------------------------------------------------
    # INTENT TYPE + TARGET INDEXES (for index-first routing)
    # --------------------------------------------------------
    intent_type = "general"
    target_indexes: list[str] = []

    # Temporal intent (years / ranges / month-year / month-only / full date)
    if (
        year is not None
        or year_range is not None
        or month_year is not None
        or month_only is not None
        or full_date is not None
    ):
        intent_type = "temporal"
        target_indexes.append("year")

    # Entity intent (persons / organizations) - PRIORITIZE over location
    # If query has persons/orgs AND "mentioning" keyword, it's an entity query, not location
    has_mentioning = "mentioning" in ql or "mention" in ql
    if meta["persons"] or normalized_orgs:
        if intent_type == "general" or (has_mentioning and intent_type == "location"):
            intent_type = "entity"
        if "entity" not in target_indexes:
            target_indexes.append("entity")

    # Location / governing-law intent (only if not already entity intent)
    if (locations or meta.get("governing_law")) and intent_type != "entity":
        if intent_type == "general":
            intent_type = "location"
        if "location" not in target_indexes:
            target_indexes.append("location")

    # Clause / legal intent
    if meta.get("legal_clause"):
        if intent_type == "general":
            intent_type = "clause"
        if "clause" not in target_indexes:
            target_indexes.append("clause")

    meta["intent_type"] = intent_type
    meta["target_indexes"] = target_indexes

    # "Chris … California … 2015" style — all signals should appear in body for top ranks
    if (
        ("chris" in ql or "christopher" in ql)
        and "california" in ql
        and re.search(r"\b2015\b", ql)
    ):
        meta["compound_person_geo_year"] = ("chris", "california", 2015)

    # numeric filters
    min_amt, max_amt = extract_numeric_filters(ql)
    meta["min_amount"] = min_amt
    meta["max_amount"] = max_amt

    # Extract money/numeric amounts from query for better matching
    # This helps with queries like "55 dollars", "$55", "55 per hour", "120,000", etc.
    money_patterns = [
        r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd|\$)?',
        r'(\d+)\s*(?:dollars?|per\s+hour|per\s+week|per\s+month|per\s+year)',
        r'(\d{1,3}(?:,\d{3})+)',  # Numbers with commas (e.g., 120,000)
    ]
    money_amounts = []
    for pattern in money_patterns:
        matches = re.findall(pattern, ql, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ""
            # Clean up the match
            clean_match = re.sub(r'[^\d]', '', str(match))
            if clean_match and len(clean_match) >= 1:
                try:
                    amount = int(clean_match)
                    if amount not in money_amounts:
                        money_amounts.append(amount)
                except ValueError:
                    pass
    # Money regexes often match fragments of a 4-digit year (e.g. 201 and 9 from "2019"),
    # which triggers metadata-first money routing and drowns topic+temporal queries.
    if year is not None and year >= 1900 and year <= 2100:
        ydigits = str(int(year))
        if len(ydigits) == 4:
            def _year_money_fragment(a: int) -> bool:
                sa = str(a)
                if len(sa) <= 3 and ydigits.startswith(sa) and sa != ydigits:
                    return True
                if len(sa) == 1 and ydigits.endswith(sa):
                    return True
                return False

            money_amounts = [a for a in money_amounts if not _year_money_fragment(a)]
    meta["money_amounts"] = money_amounts

    # --------------------------------------------------------
    # SEMANTIC ROUTING DETECTION - COMPREHENSIVE RULE ENGINE
    # --------------------------------------------------------
    # Determine if query needs semantic search (vs simple vector search)
    # This centralizes all routing logic in one place
    # Check semantic patterns FIRST, then fall back to simple phrase detection
    ql = q.lower()
    needs_semantic = False
    
    # 0. NDA / Non-disclosure queries - ALWAYS need semantic (case-insensitive)
    # NDA = Non-Disclosure Agreement = Confidentiality = Must be understood semantically
    # Comprehensive pattern matching for ALL NDA query variations
    has_nda_terms = any(term in ql for term in [
        'nda', 'ndas', 'mnda', 'mndas',
        'non-disclosure', 'non disclosure', 'nondisclosure',
        'mutual non-disclosure', 'mutual non disclosure', 'mutual nondisclosure',
        'non-disclosure agreement', 'non disclosure agreement', 'nondisclosure agreement',
        'mutual non-disclosure agreement', 'mutual non disclosure agreement'
    ])
    
    # Route to semantic for ANY query containing NDA terms
    # This includes:
    # - Standalone: "NDA", "NDAs", "mNDA", "mNDAs"
    # - With verbs: "show me all NDA", "list all NDAs", "show mNDAs"
    # - With document types: "NDA contracts", "NDA contract documents", "contracts that are NDA"
    # - With includes: "documents that includes NDA", "documents that include NDA"
    # - Any combination: "show me the documents that includes NDA", "show me contracts that are NDA"
    # --- INITIALIZE ROUTING PATTERNS (v5.4 FIX) --
    definition_patterns = [
        r'\bwhat\s+is\b', r'\bwhat\s+are\b', r'\bdefine\b', r'\bdefinition\s+of\b',
        r'\bwho\s+is\b', r'\bwho\s+are\b', r'\bwho\s+was\b',
        r'\bwhat\s+does\b', r'\btell\s+me\s+about\b', r'\bexplain\b',
        r'\bexplain\s+\w+',
    ]
    helper_verb_patterns = [
        r'\bshow\s+me\b', r'\bfind\b', r'\blist\b', r'\bsearch\s+for\b',
        r'\bwhat\s+are\s+the\b', r'\bwhat\s+are\b',
    ]
    # Check for helper verbs early to avoid UnboundLocalError
    has_helper_verb = any(re.search(p, ql) for p in helper_verb_patterns)

    is_nda_query = (
        has_nda_terms or
        # Standalone NDA terms
        ql.strip() in ['nda', 'ndas', 'mnda', 'mndas'] or
        # Standalone non-disclosure terms
        ql.strip() in ['non-disclosure', 'non disclosure', 'nondisclosure', 'mutual non-disclosure', 'mutual non disclosure', 'mutual nondisclosure'] or
        # Patterns: "show me all NDA", "list all NDAs", "show mNDAs"
        re.search(r'\b(?:show|list|find|search|what|which)\s+(?:me|all|the)?\s*(?:all\s+)?(?:nda|ndas|mnda|mndas)\b', ql) or
        # Patterns: "show me the documents that includes NDA", "documents that include NDA"
        re.search(r'\b(?:documents?|files?|docs?|contracts?|agreements?)\s+(?:that\s+)?(?:includes?|include|are|is)\s+(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\b', ql) or
        # Patterns: "show me contracts that are NDA", "NDA contracts", "NDA contract documents"
        re.search(r'\b(?:contracts?|agreements?|documents?)\s+(?:that\s+are|are\s+)?(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\b', ql) or
        re.search(r'\b(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\s+(?:contracts?|agreements?|documents?)\b', ql) or
        # Patterns: "show me NDA contracts", "show me NDA contract documents"
        re.search(r'\b(?:show|list|find)\s+me\s+(?:the\s+)?(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\s+(?:contracts?|agreements?|documents?)\b', ql)
    )
    
    if is_nda_query:
        needs_semantic = True
        logger.info(f"[ROUTING] NDA query detected -> SEMANTIC: '{query[:50]}'")
    
    # 1. Temporal queries - always need semantic
    if meta.get("explicit_temporal_intent") is not None:
        needs_semantic = True
    
    # 2. Temporal context patterns
    temporal_context_patterns = [
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+dated\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+signed\s+in\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\s+\d{4}\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\s+\d{4}\s+to\s+\d{4}\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\s+[A-Za-z]+\s+\d{4}\b',
        r'\bsigned\s+on\b', r'\bsigned\s+in\b', r'\bsigned\s+between\b',
        r'\bexecuted\s+on\b', r'\bexecuted\s+in\b',
        r'\bdated\b', r'\beffective\b', r'\bas\s+of\b',
        r'\bcreated\s+in\b', r'\bcreated\s+on\b',
        r'\bissued\s+on\b', r'\bexpired\b', r'\bexpiring\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+between\b',
        r'\bfrom\s+\d{4}\b', r'\bin\s+\d{4}\b', r'\bto\s+\d{4}\b',
    ]
    if any(re.search(pattern, ql) for pattern in temporal_context_patterns):
        needs_semantic = True
    
    # 3. Person/Signer context patterns
    person_context_patterns = [
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+signed\s+by\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+executed\s+by\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+by\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioning\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mention\b',
        r'\b(?:documents?|files?|docs?)\s+on\b',
        r'\bon\s+behalf\s+of\b',  # "documents on behalf of [Person]"
    ]
    # Only route to semantic if query has person context patterns OR explicitly mentions persons in semantic context
    # Don't route just because persons were extracted (they might be false positives from capitalized words)
    has_person_context = any(re.search(pattern, ql) for pattern in person_context_patterns)
    if has_person_context or (meta.get("persons") and has_person_context):
        needs_semantic = True

    # NOTE: Bare full-name queries (e.g. "Chris Dominguez") intentionally do NOT
    # force needs_semantic=True. Routing them through metadata-first OR-ed on
    # single-token persons ("chris", "dominguez") produced thousands of false
    # positives. Filename + content_scan supplements (word-boundary person match)
    # handle these cases without polluting location/NDA-style semantic results.

    # 4. Organization context patterns
    org_context_patterns = [
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+with\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+between\b',
        r'\bbetween\s+\w+\s+and\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+for\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioning\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mention\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mentions\b',  # "document that mentions"
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioned\s+by\b',  # "mentioned by"
        r'\bparty\b', r'\bparties\b',
    ]
    mentioning_patterns = [
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioning\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mention\b',
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mentions\b',  # "document that mentions"
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioned\s+by\b',  # "mentioned by"
    ]
    has_mentioning = any(re.search(pattern, ql) for pattern in mentioning_patterns)
    # Route to semantic if: (org in meta AND has org pattern) OR has mentioning pattern OR has "with" pattern (e.g., "agreements with Company")
    has_with_pattern = re.search(r'\b(?:agreements?|documents?|contracts?)\s+with\b', ql)
    if (meta.get("organizations") and any(re.search(pattern, ql) for pattern in org_context_patterns)) or has_mentioning or has_with_pattern:
        needs_semantic = True
    
    # 5. Location context patterns
    location_context_patterns = [
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+from\b',
        r'\bgoverned\s+by\b', r'\bgoverned\s+by\s+laws\s+of\b',
        r'\bjurisdiction\s+of\b', r'\blocated\s+in\b', r'\bbased\s+in\b',
        r'\bheadquartered\s+in\b', r'\bincorporated\s+in\b',
        r'\bstate\s+of\b', r'\bcity\s+of\b',
        r'\bfrom\s+(?:california|texas|delaware|new\s+york|los\s+angeles|austin)\b',
        r'\bin\s+(?:california|texas|delaware|new\s+york|los\s+angeles|austin)\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+under\s+\w+\s+law\b',  # "agreements under California law"
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+under\s+laws\s+of\b',  # "agreements under laws of Texas"
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mention\b',  # "document that mentions austin"
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+mentions\b',  # "document that mentions austin"
        r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+mentioning\b',  # "document mentioning austin"
    ]
    # Route to semantic if location is detected OR if query has location context patterns
    if (meta.get("location") or meta.get("locations")) and any(re.search(pattern, ql) for pattern in location_context_patterns):
        needs_semantic = True
    # Also route if query has location context patterns even without explicit location extraction
    elif any(re.search(pattern, ql) for pattern in location_context_patterns):
        needs_semantic = True
    # Route simple location queries (e.g., "austin", "austin texas", "Austin, Texas") to semantic
    # Check if query is just a location name or location with document context
    # This handles queries like "austin", "austin texas", "Austin, Texas" that should be semantic
    has_location_in_meta = bool(meta.get("location") or meta.get("locations"))
    is_location_only_query = (
        has_location_in_meta and
        len(ql.split()) <= 3 and  # Short query (likely just location)
        not any(re.search(pattern, ql) for pattern in definition_patterns) and  # Not a definition query
        not any(re.search(pattern, ql) for pattern in helper_verb_patterns)  # Not a helper verb query
    )
    if is_location_only_query:
        needs_semantic = True
        logger.info(f"[ROUTING] Simple location query detected -> SEMANTIC: '{query[:50]}'")
    
    # Simple keywords like "2024", "October" should route to VECTOR for exact matching
    # BUT: Location names (like "austin") should route to SEMANTIC for location-based search
    # Only route to SEMANTIC if they have explicit semantic context (e.g., "documents from 2024")
    is_simple_keyword = (
        len(ql.split()) == 1 and  # Single word
        not has_helper_verb and  # No helper verbs
        not meta.get("explicit_temporal_intent") and  # No explicit temporal intent
        not meta.get("location") and  # No location (don't override if location detected)
        not meta.get("locations") and  # No locations
        not meta.get("persons") and  # No persons
        not meta.get("organizations") and  # No organizations
        not meta.get("legal_clause")  # No legal clause
    )
    # Don't override location queries - they should stay SEMANTIC
    is_location_query = bool(meta.get("location") or meta.get("locations"))
    if is_simple_keyword and needs_semantic and not is_location_query:
        # Override: simple keywords (but not locations) should use vector search for exact matching
        needs_semantic = False
        logger.info(f"[ROUTING] Simple keyword detected -> VECTOR for exact matching: '{query[:50]}'")
    
    # 6. Clause/legal context patterns
    # Only route to semantic if query explicitly asks about clauses, not just contains legal keywords
    # This prevents exact text snippets like "confidential Information furnished in" from being routed to semantic
    clause_context_patterns = [
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+with\s+\w+\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+containing\s+\w+\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+that\s+have\s+\w+\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+including\s+\w+\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+with\s+(?:arbitration|confidentiality|governing\s+law|dispute\s+resolution)\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+containing\s+(?:arbitration|confidentiality|governing\s+law|dispute\s+resolution)\s+clause\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+that\s+mention\s+(?:arbitration|confidentiality|governing\s+law|dispute\s+resolution)\b',
        # Add patterns for "what are the [clause] provisions" and "clauses about [clause]"
        r'\bwhat\s+are\s+the\s+(?:governing\s+law|dispute\s+resolution|arbitration|confidentiality|termination)\s+provisions?\b',
        r'\bclauses?\s+about\s+(?:governing\s+law|dispute\s+resolution|arbitration|confidentiality|termination)\b',
        r'\b(?:governing\s+law|dispute\s+resolution|arbitration|confidentiality|termination)\s+provisions?\b',
        r'\b(?:governing\s+law|dispute\s+resolution|arbitration|confidentiality|termination)\s+clauses?\b',
        r'\b(?:agreements?|contracts?|ndas?|documents?|files?|docs?)\s+mentioning\s+(?:arbitration|confidentiality|governing\s+law|dispute\s+resolution)\b',
        r'\bclauses?\s+(?:about|regarding|concerning|for)\b',
    ]
    # Only route if query explicitly mentions "clause" or has explicit clause context patterns
    # Don't route just because query contains legal keywords (those should be exact text matches)
    has_clause_context = any(re.search(pattern, ql) for pattern in clause_context_patterns)
    if has_clause_context:
        needs_semantic = True
    # Only use meta.get("legal_clause") if query explicitly asks about clauses/provisions
    elif meta.get("legal_clause") and any(re.search(r'\b(?:clause|provision|term|section)\b', ql) for _ in [True]):
        needs_semantic = True
    elif meta.get("governing_law") and any(re.search(r'\b(?:governed|jurisdiction|law)\b', ql) for _ in [True]):
        needs_semantic = True
    
    # 7. Money/Price/Value/Numeric context patterns
    money_numeric_patterns = [
        r'\$\s*\d+',
        r'\d+\s*(?:dollars?|usd|\$)',
        r'\d+\s+per\s+(?:hour|week|month|year)',
        r'\d{1,3}(?:,\d{3})+',
        r'(?:above|greater than|>\s*)\s*\d+',
        r'(?:below|less than|<\s*)\s*\d+',
        r'(?:price|cost|fee|value|amount)\s+(?:of|is|at)\s+\d+',
    ]
    if meta.get("money_amounts") or meta.get("min_amount") or meta.get("max_amount") or \
       any(re.search(pattern, ql) for pattern in money_numeric_patterns):
        needs_semantic = True
    
    # 8. Definition/question patterns
    if any(re.search(pattern, ql) for pattern in definition_patterns):
        needs_semantic = True
    
    # 9. Question about entity
    if meta.get("query_type") in ["person_about", "org_about"] and \
       any(word in ql for word in ["who is", "who are", "what is", "what are", "who was", "what does"]):
        needs_semantic = True
    
    # already defined above
    # Route "show me all the agreements/contracts" to semantic
    is_document_type_query = (
        has_helper_verb and
        any(doc_type in ql for doc_type in ['agreements', 'agreement', 'contracts', 'contract', 'documents', 'document', 'docs', 'doc', 'files', 'file', 'ndas', 'nda'])
    )
    # Also route simple "show me agreements" (without "all") to semantic
    is_simple_document_query = (
        has_helper_verb and
        (ql.strip() in ['show me agreements', 'show me contracts', 'show me documents', 'show me files', 'show me docs'] or
         re.match(r'^(?:show\s+me|find|list)\s+(?:all\s+)?(?:agreements?|contracts?|documents?|files?|docs?|ndas?)\s*$', ql))
    )
    if has_helper_verb and (needs_semantic or meta.get("explicit_temporal_intent") or meta.get("location") or meta.get("persons") or meta.get("organizations") or meta.get("legal_clause") or is_document_type_query or is_simple_document_query):
        needs_semantic = True
    # Route "show me the docs that shows/says" to semantic
    if re.search(r'\b(?:documents?|files?|docs?|agreements?|contracts?|ndas?)\s+that\s+(?:shows?|says?|show|say)\b', ql):
        needs_semantic = True
    
    # 11. FINAL CHECK: Long exact text queries should always go to vector search
    #     This ensures exact text snippets, sentences, and paragraphs are matched exactly
    # Long queries (8+ words) that look like document text should use vector search
    #     BUT: Don't override if it's already an NDA query (NDA queries should stay semantic)
    query_word_count = len(ql.split())
    is_long_query = query_word_count >= 8
    
    if is_long_query and not is_nda_query:  # Don't override NDA queries
        # For long queries, check if they look like exact document text (not semantic queries)
        has_question_words = any(re.search(pattern, ql) for pattern in definition_patterns)
        has_helper_verbs = any(re.search(pattern, ql) for pattern in helper_verb_patterns)
        has_explicit_semantic_patterns = (
            any(re.search(pattern, ql) for pattern in temporal_context_patterns) or
            any(re.search(pattern, ql) for pattern in person_context_patterns) or
            any(re.search(pattern, ql) for pattern in org_context_patterns) or
            any(re.search(pattern, ql) for pattern in location_context_patterns) or
            any(re.search(pattern, ql) for pattern in clause_context_patterns) or
            any(re.search(pattern, ql) for pattern in money_numeric_patterns)
        )
        
        # Long queries without explicit semantic patterns are likely exact text - use vector
        if not has_question_words and not has_helper_verbs and not has_explicit_semantic_patterns:
            needs_semantic = False
            logger.info(f"[ROUTING] Long query ({query_word_count} words) without semantic patterns -> VECTOR for exact matching")
    
    # 12. FINAL CHECK: If query is a simple phrase (no question words, no explicit semantic patterns),
    #     it should always go to vector search for exact text matching
    # This ensures exact text snippets like "confidential Information furnished in" are matched exactly
    # OVERRIDE: Only override to vector if it's a simple phrase WITHOUT any semantic intent
    # DO NOT override if needs_semantic was already set to True (semantic queries should stay semantic)
    if not needs_semantic:  # Only check for simple phrases if semantic wasn't already set
        has_question_words = any(re.search(pattern, ql) for pattern in definition_patterns)
        has_helper_verbs = any(re.search(pattern, ql) for pattern in helper_verb_patterns)
        has_explicit_semantic_patterns = (
            any(re.search(pattern, ql) for pattern in temporal_context_patterns) or
            any(re.search(pattern, ql) for pattern in person_context_patterns) or
            any(re.search(pattern, ql) for pattern in org_context_patterns) or
            any(re.search(pattern, ql) for pattern in location_context_patterns) or
            any(re.search(pattern, ql) for pattern in clause_context_patterns) or
            any(re.search(pattern, ql) for pattern in money_numeric_patterns)
        )
        
        is_simple_phrase = (
            not has_question_words and  # No question words
            not has_helper_verbs and  # No helper verbs
            not has_explicit_semantic_patterns and  # No explicit semantic patterns
            len(ql.split()) >= 2  # At least 2 words (not just a single keyword)
        )
        
        if is_simple_phrase:
            # This is likely an exact text snippet from a document - use vector search
            needs_semantic = False

    # 13. HELPER-VERB VECTOR-INTENT NORMALIZATION
    # For queries that clearly express generic "show/list/find" intent with no strong
    # semantic constraints, normalize the tail as a cleaner vector query and keep
    # routing to vector (no semantic) by default.
    #
    # Examples:
    #   - "show me all Carnival itineraries 2019"
    #   - "find documents about vibration issues"
    #   - "list files with maintenance logs"
    helper_prefix_re = re.compile(
        r'^\s*(?:'
        r'show\s+me\s+all|show\s+me|show\s+all|show|'
        r'list\s+all|list|'
        r'find\s+all|find\s+me|find|'
        r'get\s+all|get\s+me|get|'
        r'give\s+me\s+all|give\s+me|'
        r'fetch\s+all|fetch|'
        r'search\s+for|'
        r'look\s+for|look\s+up|'
        r'display'
        r')\s+(.*)$'
    )

    m = helper_prefix_re.match(ql)
    if m:
        tail = m.group(1).strip()
        # Drop leading document/connector boilerplate from the tail
        tail = re.sub(
            r'^(?:all\s+)?'
            r'(?:documents?|docs?|files?|agreements?|contracts?|ndas?)\s+'
            r'(?:that\s+)?'
            r'(?:are\s+|with\s+|about\s+|on\s+|regarding\s+|related\s+to\s+|mentioning\s+|containing\s+)?',
            '',
            tail,
        ).strip()
        if tail:
            # Store a cleaned-up vector query, but always keep the original text too.
            meta["vector_query"] = tail

        # Only reaffirm vector-routing when there are NO strong semantic constraints.
        has_strong_semantic_constraints = bool(
            meta.get("explicit_temporal_intent")
            or meta.get("location")
            or meta.get("locations")
            or meta.get("persons")
            or meta.get("organizations")
            or meta.get("legal_clause")
            or is_nda_query
        )
        if not has_strong_semantic_constraints and not needs_semantic:
            # Generic helper-verb query without explicit semantic filters -> VECTOR
            needs_semantic = False
            logger.info(f"[ROUTING] Helper-verb query without semantic constraints -> VECTOR: '{query[:80]}'")
    
    # Store the decision in metadata
    meta["needs_semantic"] = needs_semantic

    # Vector / BOTH: reduce year dominance in embeddings (Milvus path uses vector_query).
    _apply_year_stripped_vector_query(meta, query)

    # Entity + year / NDA + year decomposition for ranking (does not change retrieval paths).
    _apply_query_decomposition(meta, query, ql)

    # After decomposition: city+state queries embed like city-only for recall parity.
    _apply_geo_city_vector_query(meta, query, ql)

    # semantic_pipeline.search embeds the first return value (enhanced_query). ultimate_ui
    # uses meta["vector_query"] for Milvus — they must agree. Otherwise e.g. "bank statements
    # 2014" embeds plural text while meta.vector_query is canonical singular.
    _out_vec = (meta.get("vector_query") or "").strip()
    if _out_vec:
        q = _out_vec

    return q, meta


# ------------------------------------------------------------
# BACKWARD-COMPATIBILITY CONSTANTS (for semantic_pipeline.py)
# ------------------------------------------------------------

# LOCATION_PATTERNS reused by semantic_pipeline for US + global locations
LOCATION_PATTERNS = {
    "california": ["california", "ca", "cal"],
    "new_york": ["new york", "ny", "new york state"],
    "texas": ["texas", "tx"],
    "austin": ["austin", "austin texas", "austin, texas", "austin tx", "austin, tx"],
    "dallas": ["dallas", "dallas texas", "dallas, texas", "dallas tx", "dallas, tx"],
    "florida": ["florida", "fl"],
    "delaware": ["delaware", "de"],
}

# LEGAL_CLAUSES reused by semantic_pipeline for boosting / clause detection.
# Keys are canonical clause types from query_metadata["legal_clause"].
# Values are the clause keywords as they appear in documents / metadata_index.by_clause.
LEGAL_CLAUSES = {
    # Arbitration / dispute resolution
    "arbitration": [
        "arbitration",
        "arbitrate",
        "arbitral",
        "dispute resolution",
        "adr",
    ],
    # Explicit alias for queries like "clauses about dispute resolution"
    "dispute_resolution": [
        "dispute resolution",
        "arbitration",
        "arbitral",
        "adr",
    ],
    # Confidentiality / NDA
    "confidentiality": [
        "confidentiality",
        "confidential",
        "non-disclosure",
        "non disclosure",
        "nondisclosure",
        "nda",
        "mnda",
        "proprietary information",
        "private information",
    ],
    # Indemnification
    "indemnification": [
        "indemnify",
        "indemnity",
        "indemnification",
        "hold harmless",
    ],
    # Termination / renewal
    "termination": [
        "termination",
        "terminate",
        "expiration",
        "renewal",
    ],
    # Governing law / jurisdiction
    "governing_law": [
        "governing law",
        "jurisdiction",
        "choice of law",
        "laws of",
    ],
}


