#!/usr/bin/env python3
"""
TemporalReasoningEngine (HYBRID MODE – PRODUCTION VERSION)
==========================================================

This engine unifies all date parsing, extraction, reasoning, and validation:

✓ Full date parsing (dateutil, ISO, regex)
✓ Month, day, year extraction
✓ Year-range extraction
✓ Month-year detection
✓ Temporal intent classification
✓ Expired contract detection
✓ Hybrid strict + semantic soft reasoning
✓ Zero filename-based false positives
✓ Suitable for 1000-query batch testing

Used by:
- semantic_pipeline.py
- universal_constraint_validator.py
- metadata_index (month-year indexing)
- ingestion pipeline
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    date_parser = None
    DATEUTIL_AVAILABLE = False


# -----------------------------------------------
# REGEX DEFINITIONS
# -----------------------------------------------
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MONTH_RE = re.compile(
    r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)",
    re.IGNORECASE
)
MONTH_YEAR_RE = re.compile(
    r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+((?:19|20)\d{2})",
    re.IGNORECASE
)
DATE_RE = re.compile(
    r"(\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2})|"                           # 12/03/2024
    r"((?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2})|"                           # 2024/03/12
    r"((?:January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sep|Sept|October|Oct|November|Nov|December|Dec)"
    r"\s+\d{1,2},?\s+(?:19|20)\d{2})",                                   # March 5 2024
    re.IGNORECASE
)

# FINAL UNIVERSAL CORE REGEXES (Option B)
# Updated to handle space separators: 09 08 2025
FULL_DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-\s]\d{1,2}[/\-\s](?:19|20)\d{2})\b|"
    r"\b((?:19|20)\d{2}[/\-\s]\d{1,2}[/\-\s]\d{1,2})\b|"
    r"\b((jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2})\b",
    re.IGNORECASE,
)

RANGE_RE = re.compile(
    r"(?:from|between)\s+(?:19|20)\d{2}\s*(?:to|and)\s*(?:19|20)\d{2}",
    re.IGNORECASE,
)

# 2-digit year patterns inside date-like strings (e.g. 09/26/25, 26-09-25, 02.03.25)
TWO_DIGIT_TRAILING_DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-](\d{2})\b"
)
TWO_DIGIT_LEADING_DATE_RE = re.compile(
    r"\b(\d{2})[/-]\d{1,2}[/-]\d{1,2}\b"
)
TWO_DIGIT_DOTTED_DATE_RE = re.compile(
    r"\b\d{1,2}\.\d{1,2}\.(\d{2})\b"
)

# Generic 4-digit date-like structures (machine timestamps, ISO-like, etc.)
# FINAL PATTERN SET (per production spec)
GENERIC_DATE_STRUCTURES = [
    r"\b(19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}\b",   # 2025-09-26 or 2025.09.26
    r"\b\d{1,2}[./-]\d{1,2}[./-](19|20)\d{2}\b",   # 09-26-2025
    r"\b(19|20)\d{2}\.\d{1,2}\.\d{1,2}\b",         # 2025.09.26
    r"\b(19|20)\d{2}-\d{2}-\d{2}T",                # 2025-09-26T...
    r"\b(19|20)\d{2}T\d{2}:\d{2}\b",               # 2025T21:09 (ISO-like timestamps)
]
GENERIC_DATE_RE = re.compile("|".join(GENERIC_DATE_STRUCTURES))

# Unambiguous full-date patterns extracted from document TEXT at index-build time.
# Only matches clear "month-name day year" and "day month-name year" forms —
# deliberately excludes numeric-only patterns (e.g. 30/01/2019) which are
# ambiguous (day-first vs month-first) and would create false positives.
_FULL_DATE_FROM_TEXT_RE = re.compile(
    # "January 30, 2019"  /  "Jan 30 2019"
    r"\b((?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\s+\d{1,2},?\s+(?:19|20)\d{2})\b"
    r"|"
    # "30 January 2019"  /  "30 Jan 2019"
    r"\b(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\s+(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


SAFE_CONTEXTS = [
    r"(dated|signed|executed|effective|as of|on|issued|created)\s*:?\s*"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)?"
    r"\s*[,\s]*\d{1,2}?[,\s]*(?:19|20)\d{2}",
    r"(dated|signed|executed|effective|as of|on|issued|created)\s*:?\s*(?:19|20)\d{2}",
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:19|20)\d{2}",
    r"\d{1,2}[/\-]\d{1,2}[/\-](?:19|20)\d{2}",
    r"(?:19|20)\d{2}[/\-]\d{1,2}[/\-]\d{1,2}",
    # European formats
    r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:19|20)\d{2}",
    r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(?:19|20)\d{2}",
    r"\d{1,2}\.\d{1,2}\.(?:19|20)\d{2}",  # 02.03.2024
    # US narrative formats
    r"this\s+\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:19|20)\d{2}",
    # Encoded dates
    r"effective\s+date\s*:?\s*\d{1,2}\.\d{4}",  # Effective Date: 04.2024
]


# ================================================================
# TEMPORAL ENGINE CLASS
# ================================================================
class TemporalReasoningEngine:
    """
    Central engine used by semantic + validator + ingestion.

    Hybrid mode behavior:
    - Hard match when query expresses explicit temporal intent
    - Soft match when temporal reasoning is implied
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 2-digit year normalization (e.g. '25' -> 2025)
    # ------------------------------------------------------------
    @staticmethod
    def _normalize_two_digit_year(y: str) -> Optional[int]:
        """Map 2-digit years to 4-digit, biased to 2000s for 00–29."""
        try:
            val = int(y)
        except (TypeError, ValueError):
            return None

        if 0 <= val <= 29:
            return 2000 + val
        if 30 <= val <= 99:
            return 1900 + val
        return None

    # ============================================================
    # UNIVERSAL DATE PARSING
    # ============================================================
    def parse_date(self, value):
        """Backward-compatible alias for parse_any_date."""
        return self.parse_any_date(value)

    # ============================================================
    # UNIVERSAL DATE PARSING (Option B core)
    # ============================================================
    def parse_any_date(self, text: str) -> Optional[datetime]:
        """
        Universal date parser - handles all formats:
        - April 04, 2019
        - April 4, 2019
        - April 04 2019
        - April 4 2019
        - 2019 April 04
        - 2019 April 4
        - 2019 April
        - 04/04/2019
        - 2019-04-04
        """
        if not text:
            return None

        s = str(text).strip()

        # Guard: do NOT parse street addresses as dates.
        # "1100 W Town and Country Rd", "228 Main St Venice", etc.
        # Pattern: leading 3-5 digit number followed by a direction word or
        # common street-type word.  dateutil(fuzzy=True) would happily
        # parse "1100 W Town ..." as year=1100, which is wrong.
        _STREET_DIRS = {'w', 'e', 'n', 's', 'north', 'south', 'east', 'west',
                        'ne', 'nw', 'se', 'sw'}
        _STREET_TYPES = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd',
                         'boulevard', 'lane', 'ln', 'drive', 'dr', 'court', 'ct',
                         'place', 'pl', 'way', 'circle', 'cir', 'terrace', 'ter',
                         'highway', 'hwy', 'pkwy', 'parkway'}
        _addr_tokens = s.lower().split()
        if (len(_addr_tokens) >= 3 and _addr_tokens[0].isdigit()
                and 2 <= len(_addr_tokens[0]) <= 5
                and (_addr_tokens[1] in _STREET_DIRS or
                     any(_addr_tokens[i] in _STREET_TYPES
                         for i in range(1, min(5, len(_addr_tokens)))))):
            return None  # street address, not a date

        # 1) Try python-dateutil (best - handles fuzzy dates)
        if DATEUTIL_AVAILABLE and date_parser:
            try:
                return date_parser.parse(s, fuzzy=True)
            except Exception:
                pass

        # 2) Try ISO format
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass

        # 3) Try explicit patterns for common formats
        # Pattern: "April 04, 2019" or "April 4, 2019" or "April 04 2019" or "09 08 2025"
        month_day_year = re.compile(
            r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2}),?\s+((?:19|20)\d{2})",
            re.IGNORECASE
        )
        m = month_day_year.search(s)
        if m:
            try:
                month_name = m.group(1).lower()
                day = int(m.group(2))
                year = int(m.group(3))
                month_map = {
                    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
                    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
                    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
                }
                month_num = month_map.get(month_name)
                if month_num:
                    return datetime(year, month_num, day)
            except Exception:
                pass
        
        # Pattern: Numeric spaces: "09 08 2025"
        numeric_spaces = re.compile(r"\b(\d{1,2})\s+(\d{1,2})\s+((?:19|20)\d{2})\b")
        m = numeric_spaces.search(s)
        if m:
            try:
                # Assume US format MM DD YYYY if first group <= 12
                v1, v2, v3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= v1 <= 12 and 1 <= v2 <= 31:
                    return datetime(v3, v1, v2)
                elif 1 <= v2 <= 12 and 1 <= v1 <= 31: # Try DD MM YYYY
                    return datetime(v3, v2, v1)
            except Exception:
                pass

        # 4) Pattern: "2019 April 04" or "2019 April 4" or "2019 April"
        year_month_day = re.compile(
            r"((?:19|20)\d{2})\s+(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)(?:\s+(\d{1,2}))?",
            re.IGNORECASE
        )
        m = year_month_day.search(s)
        if m:
            try:
                year = int(m.group(1))
                month_name = m.group(2).lower()
                day = int(m.group(3)) if m.group(3) else 1
                month_map = {
                    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
                    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
                    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
                }
                month_num = month_map.get(month_name)
                if month_num:
                    return datetime(year, month_num, day)
            except Exception:
                pass

        # 5) Year-only fallback
        m = YEAR_RE.search(s)
        if m:
            try:
                year_val = int(m.group(0))
                return datetime(year_val, 1, 1)
            except Exception:
                return None

        return None

    # ============================================================
    # QUERY INTENT
    # ============================================================
    def extract_year_range_from_query(self, q: str):
        """Detect ranges: between X and Y, from X to Y, X-Y."""
        q = (q or "").lower()
        # Prefer unified RANGE_RE for robustness
        for m in RANGE_RE.finditer(q):
            nums = [int(x.group(0)) for x in YEAR_RE.finditer(m.group(0))]
            if len(nums) >= 2:
                a, b = sorted(nums[:2])
                return (a, b)
        # Fallback to simple patterns if needed
        patterns = [
            r"between\s+(\d{4})\s+and\s+(\d{4})",
            r"from\s+(\d{4})\s+to\s+(\d{4})",
            r"(\d{4})\s*[-–]\s*(\d{4})",
        ]
        for p in patterns:
            m = re.search(p, q)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                return (min(a, b), max(a, b))
        return None

    def extract_single_year_from_query(self, q: str):
        """Extract single year from query."""
        if not q:
            return None
        for m in YEAR_RE.finditer(q):
            try:
                return int(m.group(0))
            except Exception:
                continue
        return None

    def extract_month_year_from_query(self, q: str):
        """Extract (month_name, year) from query."""
        m = MONTH_YEAR_RE.search((q or "").lower())
        if not m:
            return None
        month_name = m.group(1).lower()
        year_str = m.group(2)  # Now captures full 4-digit year (e.g., "2019")
        year = int(year_str)
        return (month_name, year)
    
    def extract_month_only_from_query(self, q: str) -> Optional[str]:
        """
        Extract month-only from query (e.g., 'document from April', 'agreement from april').
        Only extracts if NO year is present in the query.
        Handles: "from April", "signed in April", "created in April", "document from April 4"
        """
        if not q:
            return None
        q_lower = q.lower()
        
        # First check: if there's a month-year pattern, don't extract month-only
        if MONTH_YEAR_RE.search(q_lower):
            return None
        
        # Second check: if there's a full date (month + day + year), don't extract month-only
        full_date_pattern = re.compile(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+(?:19|20)\d{2}",
            re.IGNORECASE
        )
        if full_date_pattern.search(q_lower):
            return None
        
        # Third check: if there's a year-first full date, don't extract month-only
        year_first_date = re.compile(
            r"(?:19|20)\d{2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
            re.IGNORECASE
        )
        if year_first_date.search(q_lower):
            return None
        
        # Now extract month-only (handles "from April", "signed in April", "created in April", "April 4" without year)
        # Updated pattern to be more permissive for "document from April" and "document signed in April"
        month_pattern = re.compile(
            r"(?:from|in|during|of|for|signed\s+in|signed\s+on|created\s+in|created\s+on|dated|expired\s+on|expiring\s+in)\s+(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)(?:\s+\d{1,2})?(?!\s*[,.]?\s*(?:19|20)\d{2})(?:\s|$|,|\.|$)",
            re.IGNORECASE
        )
        match = month_pattern.search(q_lower)
        if match:
            month = match.group(1).lower()
            # Normalize abbreviations
            month_map = {
                "jan": "january", "feb": "february", "mar": "march",
                "apr": "april", "may": "may", "jun": "june",
                "jul": "july", "aug": "august", "sep": "september",
                "sept": "september", "oct": "october", "nov": "november",
                "dec": "december"
            }
            return month_map.get(month, month)
        
        # Fallback: standalone month in temporal context (but not if year is nearby)
        if any(kw in q_lower for kw in ["from", "signed", "created", "dated", "document", "agreement", "contract", "file", "expired", "expiring"]):
            standalone_month = re.compile(
                r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)\b",
                re.IGNORECASE
            )
            for match in standalone_month.finditer(q_lower):
                month_pos = match.start()
                # Check if there's a year pattern nearby (within 30 chars)
                nearby_text = q_lower[max(0, month_pos-15):min(len(q_lower), month_pos+30)]
                if not re.search(r"(?:19|20)\d{2}", nearby_text):
                    month = match.group(1).lower()
                    month_map = {
                        "jan": "january", "feb": "february", "mar": "march",
                        "apr": "april", "may": "may", "jun": "june",
                        "jul": "july", "aug": "august", "sep": "september",
                        "sept": "september", "oct": "october", "nov": "november",
                        "dec": "december"
                    }
                    return month_map.get(month, month)
        
        return None

    def detect_explicit_temporal_intent(self, q: str) -> bool:
        """
        Detect explicit temporal intent in query.
        Handles: from, between, signed in, dated, expired, expiring, etc.
        """
        q = q.lower()
        hard_signals = [
            "from ", "between", "signed in", "signed on", "signed between",
            "created in", "created on", "dated", "executed in", "executed on",
            "effective", "effective on", "year ", "month ", "day ", "as of",
            "period", "range", "term", "agreement dated", "document from",
            "agreement from", "contract from", "file from", "files from",
            "documents from", "agreements from", "contracts from",
            "expired on", "expiring in", "expired in", "expiring on"
        ]

        if any(x in q for x in hard_signals):
            return True
            
        # Also check for explicit 4-digit years
        if YEAR_RE.search(q):
            return True
            
        return False

    def detect_expired_intent(self, q: str) -> bool:
        """Detect if query asks for expired documents."""
        q = q.lower()
        explicit = [
            "expired now",
            "already expired",
            "contract expired",
            "agreement expired",
            "expired contract",
            "expired agreement",
            "now expired",
            "past contract",
            "past agreement",
            "expiry passed"
        ]
        if any(x in q for x in explicit):
            return True
        if "expired" in q:
            return True
        return False

    # ============================================================
    # STRICT YEAR NORMALIZATION HELPERS (Option B refinement)
    # ============================================================
    def _normalize_year_candidate(self, candidate):
        """
        Accept a candidate that may be:
          - int year (2- or 4-digit)
          - string containing digits
        Return a clean 4-digit int in [1900,2100] or None.
        """
        if candidate is None:
            return None

        try:
            yi = int(candidate)
        except Exception:
            m = re.search(r"\d{1,4}", str(candidate))
            if not m:
                return None
            try:
                yi = int(m.group(0))
            except Exception:
                return None

        # Already a 4-digit year in a realistic range
        if 1900 <= yi <= 2100:
            return yi

        # Handle 1–2 digit years via two-digit normalization
        if 0 <= yi <= 99:
            return self._normalize_two_digit_year(str(yi))

        # Anything else (e.g. 3-digit) is treated as garbage
        return None

    # ============================================================
    # UNIVERSAL DOC TEMPORAL PROFILE (Option B, normalized)
    # ============================================================
    def extract_doc_dates(self, metadata: dict, text: str):
        """
        Extract all temporal signals from a document (text + metadata),
        with strict normalization:
          - years: only real 4-digit years in [1900,2100] or
                   well-mapped 2-digit → 4-digit values
          - dates: datetime objects parsed from any reasonable format
        """
        years = set()
        dates = set()

        # -------------------------------
        # 1) Metadata (authoritative)
        # -------------------------------
        for _, v in (metadata or {}).items():
            if v is None:
                continue
            sv = str(v)

            # Prefer explicit 4-digit years where present
            for m in re.finditer(r"\b(?:19|20)\d{2}\b", sv):
                y = self._normalize_year_candidate(m.group(0))
                if y:
                    years.add(y)

            # If we didn't get any years yet from this doc, try parsing as date
            if not years:
                dt = self.parse_any_date(sv)
                if dt:
                    dates.add(dt)
                    years.add(dt.year)
                else:
                    # As a last resort, normalize a short numeric token (e.g. '20' -> 2020)
                    m2 = re.search(r"\b(\d{1,2})\b", sv)
                    if m2:
                        cand = self._normalize_year_candidate(m2.group(1))
                        if cand:
                            years.add(cand)

        # -------------------------------
        # 2) Full dates from text
        # -------------------------------
        for m in FULL_DATE_RE.finditer(text or ""):
            for g in m.groups():
                if not g:
                    continue
                dt = self.parse_any_date(g)
                if dt:
                    dates.add(dt)
                    years.add(dt.year)

        # -------------------------------
        # 3) Any 2–4 digit year-like tokens in text
        # -------------------------------
        for yraw in re.findall(r"\b\d{2,4}\b", text or ""):
            ynorm = self._normalize_year_candidate(yraw)
            if ynorm:
                years.add(ynorm)

        dates_list = sorted(list(dates))
        return {
            "years": sorted(list(years)),
            "dates": dates_list,
            "earliest": dates_list[0] if dates_list else None,
            "latest": dates_list[-1] if dates_list else None,
        }

    # ============================================================
    # UNIVERSAL QUERY INTERVAL (Option B)
    # ============================================================
    def extract_query_interval(self, query: str):
        """
        Map a free-text query to a concrete time interval [start, end]
        plus a mode flag: 'year', 'range', 'month-year', 'relative',
        'before', 'after', or None.
        """
        q = (query or "").lower().strip()

        # 1) Explicit year range
        m = RANGE_RE.search(q)
        if m:
            nums = [int(mm.group(0)) for mm in YEAR_RE.finditer(m.group(0))]
            if len(nums) >= 2:
                y1, y2 = sorted(nums[:2])
                return datetime(y1, 1, 1), datetime(y2, 12, 31), "range"

        # 2) Month-year (e.g. January 2023)
        m = MONTH_YEAR_RE.search(q)
        if m:
            month = m.group(1).lower()
            year = int(m.group(2))
            months = [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
            ]
            month_idx = months.index(month) + 1
            start = datetime(year, month_idx, 1)
            end = start.replace(day=28) + timedelta(days=4)
            end = end.replace(day=1) - timedelta(days=1)
            return start, end, "month-year"

        # 3) Single years
        years = [int(m.group(0)) for m in YEAR_RE.finditer(q)]
        if years:
            y = years[0]
            return datetime(y, 1, 1), datetime(y, 12, 31), "year"

        # 4) Relative patterns (e.g. last year, past month)
        relative_patterns = {
            "last year": -365,
            "past year": -365,
            "previous year": -365,
            "last month": -30,
            "past month": -30,
            "previous month": -30,
        }
        for phrase, offset in relative_patterns.items():
            if phrase in q:
                end = datetime.now()
                start = end + timedelta(days=offset)
                return start, end, "relative"

        # 5) "before YYYY"
        if "before" in q:
            m = YEAR_RE.search(q)
            if m:
                yr = int(m.group(0))
                return None, datetime(yr, 1, 1), "before"

        # 6) "after YYYY"
        if "after" in q:
            m = YEAR_RE.search(q)
            if m:
                yr = int(m.group(0))
                return datetime(yr, 12, 31), None, "after"

        return None, None, None

    # ============================================================
    # MASTER MATCH FUNCTION (authoritative core, Option B)
    # ============================================================
    def matches(self, query: str, metadata: dict, text: str) -> bool:
        """
        Single authoritative temporal rule:
          - If the query expresses a concrete temporal intent
            (years, ranges, month-year, before/after, relative),
            the document must have at least one compatible year/date.
          - If no temporal intent, always pass.
        """
        start, end, mode = self.extract_query_interval(query)

        # Always build a temporal profile once – reused for all modes
        profile = self.extract_doc_dates(metadata or {}, text or "")
        years = profile["years"]
        earliest = profile["earliest"]
        latest = profile["latest"]

        # -----------------------------------------------
        # Month-only queries (e.g. "documents from April")
        # -----------------------------------------------
        # extract_query_interval() only handles year / range / month-year /
        # relative / before / after. Pure month-only intent is handled here
        # so that all logic still flows through this central engine.
        if mode is None:
            month_only = self.extract_month_only_from_query(query)
            if month_only:
                # Normalize month name to index (1–12)
                months = [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                ]
                m_lower = month_only.lower()
                if m_lower not in months:
                    # Unknown month token – fail safe
                    return False
                month_idx = months.index(m_lower) + 1

                # Consider all concrete dates we know for this document.
                dates_list = list(profile.get("dates") or [])
                # If dates list is empty but we inferred earliest/latest, add them.
                if not dates_list:
                    if earliest:
                        dates_list.append(earliest)
                    if latest and latest is not earliest:
                        dates_list.append(latest)

                if not dates_list:
                    # No concrete dates at all → no evidence for month-only query
                    return False

                return any(dt.month == month_idx for dt in dates_list)

            # No temporal intent at all → do not filter
            return True

        # No temporal evidence in document → reject for explicit queries
        if not years and not earliest:
            return False

        # YEAR / RANGE: check intersection of years
        if mode in ("year", "range"):
            if not years:
                return False
            if start and end:
                return any(start.year <= y <= end.year for y in years)

        # MONTH-YEAR: require any date within [start, end]
        if mode == "month-year":
            if not earliest and not latest:
                return False
            if earliest and start <= earliest <= end:
                return True
            if latest and start <= latest <= end:
                return True
            return False

        # BEFORE: any earliest date before the cutoff
        if mode == "before":
            if not earliest or not end:
                return False
            return earliest <= end

        # AFTER: any latest date after the cutoff
        if mode == "after":
            if not latest or not start:
                return False
            return latest >= start

        # RELATIVE: any date inside the relative window
        if mode == "relative":
            if not earliest and not latest:
                return False
            if earliest and start <= earliest <= end:
                return True
            if latest and start <= latest <= end:
                return True
            return False

        # Fallback: if we can't interpret, do not block
        return True

    # ============================================================
    # SAFE EXTRACTION FROM DOCUMENT TEXT
    # ============================================================
    def _extract_safe_years_from_text(self, text: str):
        """Internal: Extract years from safe contexts only."""
        if not text:
            return []

        years = set()
        low = text.lower()

        for pattern in SAFE_CONTEXTS:
            for match in re.finditer(pattern, low, re.IGNORECASE):
                y = YEAR_RE.search(match.group(0))
                if y:
                    year_str = y.group(0) if y.group(0) else y.group(1)
                    try:
                        year = int(year_str)
                        if 1900 <= year <= 2100:
                            years.add(year)
                    except:
                        continue

        return sorted(list(years))

    def safe_extract_years_from_text(self, text: str):
        """Public API: Extract years from safe contexts only."""
        return self._extract_safe_years_from_text(text)

    # ============================================================
    # DOCUMENT TEMPORAL PROFILE
    # ============================================================
    def extract_doc_temporal_profile(self, file_id: str, metadata: dict, text: str):
        """
        Produce a full temporal profile:
        - explicit_dates: dates parsed with full precision
        - safe_years: safe context years
        - years: combined final set
        - month_years: month-year tuples
        """
        explicit_dates = []
        month_years = []
        years = set()

        # -------------------------------
        # Metadata fields (authoritative)
        # -------------------------------
        meta_fields = [
            "signed_on", "date", "year", "created_at",
            "issued_on", "effective_on",
            "expiry_date", "end_date", "terminates_on",
            "source_date"
        ]

        for field in meta_fields:
            val = metadata.get(field)
            if val:
                dt = self.parse_date(val)
                if dt:
                    explicit_dates.append(dt)
                    years.add(dt.year)

        # ---------------------------------------------------------------
        # Full-date extraction from text (month-name patterns ONLY)
        # Populates explicit_dates → by_full_date inverted index so that
        # date-based metadata-first routing can find documents like the
        # METAFESTO agreements which are "dated as of January 30, 2019".
        # Only unambiguous "Month Day, Year" / "Day Month Year" patterns
        # are matched — numeric-only dates are intentionally excluded to
        # avoid day/month ambiguity false positives.
        # ---------------------------------------------------------------
        if text:
            for _m in _FULL_DATE_FROM_TEXT_RE.finditer(text):
                _raw = (_m.group(1) or _m.group(2) or "").strip()
                if _raw:
                    try:
                        _dt = self.parse_any_date(_raw)
                        # Only keep dates with both month AND day resolved
                        if _dt and _dt.month and _dt.day:
                            explicit_dates.append(_dt)
                            years.add(_dt.year)
                    except Exception:
                        pass

        # -------------------------------
        # Month-year extraction (text)
        # -------------------------------
        if text:
            for match in MONTH_YEAR_RE.finditer(text.lower()):
                month = match.group(1).lower()
                year = int(match.group(2))
                month_years.append((month, year))
                years.add(year)

        # -------------------------------
        # Safe-text year extraction (4-digit, narrative contexts)
        # -------------------------------
        safe = self._extract_safe_years_from_text(text)
        for y in safe:
            years.add(y)

        # -------------------------------
        # Standalone 4-digit year extraction
        # Extract years that aren't in safe contexts but are valid 4-digit years
        # This ensures "2019" in text is found even if not in a date pattern
        # -------------------------------
        if text:
            # Extract all 4-digit years (1900-2100) from text
            # Capture the FULL 4-digit year, not just the first 2 digits
            for yraw in re.findall(r"\b((?:19|20)\d{2})\b", text):
                ynorm = self._normalize_year_candidate(yraw)
                if ynorm and 1900 <= ynorm <= 2100:
                    years.add(ynorm)

        # -------------------------------
        # 2-digit year inference inside date-like patterns
        # -------------------------------
        if text:
            low = text.lower()

            for regex in (
                TWO_DIGIT_TRAILING_DATE_RE,
                TWO_DIGIT_LEADING_DATE_RE,
                TWO_DIGIT_DOTTED_DATE_RE,
            ):
                for m in regex.finditer(low):
                    # Each pattern has a single capturing group for the 2-digit year
                    y2 = m.group(1)
                    norm = self._normalize_two_digit_year(y2)
                    if norm is not None and 1900 <= norm <= 2100:
                        years.add(norm)

            # -------------------------------
            # Generic 4-digit dates (machine timestamps, ISO-like)
            # -------------------------------
            for m in GENERIC_DATE_RE.finditer(low):
                y = YEAR_RE.search(m.group(0))
                if y:
                    try:
                        year_val = int(y.group(0))
                        if 1900 <= year_val <= 2100:
                            years.add(year_val)
                    except ValueError:
                        continue

        # -------------------------------
        # Deduplicate and sort
        # -------------------------------
        explicit_dates = sorted(list(set(explicit_dates)), key=lambda d: d.year)
        month_years = sorted(list(set(month_years)), key=lambda x: (x[1], x[0]))
        years = sorted(list(years))

        return {
            "explicit_dates": explicit_dates,
            "safe_years": safe,
            "month_years": month_years,
            "years": years,
        }

    # ============================================================
    # BACKWARD COMPATIBILITY METHODS
    # ============================================================
    def extract_doc_years(self, file_id: str, metadata: dict, text: str):
        """Backward compatibility: Returns list of years only."""
        profile = self.extract_doc_temporal_profile(file_id, metadata, text)
        return profile["years"]

    def extract_year(self, value):
        """Extract year from a value."""
        if not value:
            return None
        if isinstance(value, int) and 1900 <= value <= 2100:
            return value
        dt = self.parse_date(value)
        if dt:
            return dt.year
        return None

    def extract_full_dates(self, text: str) -> List[datetime]:
        """Extract full dates (datetime objects) from document text."""
        if not text:
            return []
        profile = self.extract_doc_temporal_profile("", {}, text)
        return profile.get("explicit_dates", [])

    def extract_month_only(self, text: str) -> Optional[str]:
        """Extract month-only from document text (e.g., 'in March', 'during October')."""
        if not text:
            return None
        # Look for month patterns in safe contexts
        month_pattern = re.compile(
            r"(?:in|during|of|for)\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
            re.IGNORECASE
        )
        match = month_pattern.search(text.lower())
        if match:
            month = match.group(1).lower()
            # Normalize abbreviations
            month_map = {
                "jan": "january", "feb": "february", "mar": "march",
                "apr": "april", "may": "may", "jun": "june",
                "jul": "july", "aug": "august", "sep": "september",
                "sept": "september", "oct": "october", "nov": "november",
                "dec": "december"
            }
            return month_map.get(month, month)
        return None

    # ============================================================
    # HARD FILTERS (for explicit temporal queries)
    # ============================================================
    def doc_matches_year_range(self, profile, start, end, strict=True):
        """
        Check if document matches year range.
        strict=True: Only accept if explicit year present
        strict=False: Accept documents with missing dates if semantic score is high
        """
        if strict:
            # explicit full dates
            for dt in profile.get("explicit_dates", []):
                if start <= dt.year <= end:
                    return True
            # month-year
            for _, y in profile.get("month_years", []):
                if start <= y <= end:
                    return True
            # safe years
            for y in profile.get("years", []):
                if start <= y <= end:
                    return True
            return False
        else:
            # Soft mode: Accept if document has any temporal signal
            return len(profile.get("years", [])) > 0 or len(profile.get("explicit_dates", [])) > 0

    def doc_matches_single_year(self, profile, year, strict=True):
        """Check if document matches single year."""
        return self.doc_matches_year_range(profile, year, year, strict=strict)

    def is_doc_in_range(self, doc_years, start, end, strict=True):
        """Backward compatibility: Check if doc_years list contains year in range."""
        if strict:
            if not doc_years:
                return False
            for y in doc_years:
                if start <= y <= end:
                    return True
            return False
        else:
            # Soft mode: Accept if any years present
            return len(doc_years) > 0

    # ============================================================
    # HYBRID MODE (MAIN ENTRYPOINT)
    # ============================================================
    def temporal_match(self, query: str, metadata: dict, text: str, strict: bool = True) -> bool:
        """
        Hybrid logic used by semantic pipeline and validator.

        1. If query has explicit temporal intent → use STRICT HARD filters.
        2. If query is vague/implicit → use SOFT reasoning.
        """
        profile = self.extract_doc_temporal_profile(
            file_id="",
            metadata=metadata or {},
            text=text or ""
        )

        # ----------------------------------------
        # 1) Extract query-side units
        # ----------------------------------------
        yr_range = self.extract_year_range_from_query(query)
        single_year = self.extract_single_year_from_query(query)
        month_year = self.extract_month_year_from_query(query)
        month_only = self.extract_month_only_from_query(query)
        # Extract full date from query (e.g., "April 04, 2019", "April 4, 2019")
        query_full_date = self.parse_any_date(query)
        explicit = self.detect_explicit_temporal_intent(query)

        # ----------------------------------------
        # 2) HARD MODE (explicit query intent)
        # ----------------------------------------
        if explicit:
            # Priority order: year_range > month_year > single_year > full_date > month_only
            # This ensures year-only queries are handled correctly
            
            if yr_range:
                a, b = yr_range
                res = self.doc_matches_year_range(profile, a, b, strict=strict)
                logger.info(f"[TEMPORAL-TRACE] yr_range={yr_range} res={res}")
                return res

            if month_year:
                m, y = month_year
                # Check month_years first
                month_years = profile.get("month_years", [])
                if any((m.lower() == mm.lower() and y == yy) for (mm, yy) in month_years):
                    logger.info(f"[TEMPORAL-TRACE] month_year match: {m} {y}")
                    return True
                # Fallback: check if year matches and month appears in text/metadata
                doc_years = profile.get("years", [])
                if y in doc_years:
                    text_lower = (text or "").lower()
                    month_pattern = re.compile(rf"\b{re.escape(m.lower())}\b", re.IGNORECASE)
                    if month_pattern.search(text_lower):
                        logger.info(f"[TEMPORAL-TRACE] month_year text fallback match: {m} {y}")
                        return True
                    # Also check metadata
                    meta_str = str(metadata or {}).lower()
                    if month_pattern.search(meta_str):
                        logger.info(f"[TEMPORAL-TRACE] month_year meta fallback match: {m} {y}")
                        return True
                logger.info(f"[TEMPORAL-TRACE] month_year no match: {m} {y}")
                return False

            if single_year:
                res = self.doc_matches_single_year(profile, single_year, strict=strict)
                logger.info(f"[TEMPORAL-TRACE] single_year={single_year} strict={strict} res={res} doc_years={profile.get('years')}")
                return res

            # Full date query: match exact date or year+month
            # Only treat as full date if query actually contains month/day patterns
            if query_full_date:
                query_lower = query.lower()
                has_month_day = bool(
                    re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}", query_lower) or
                    re.search(r"\d{1,2}[/-]\d{1,2}", query_lower)
                )
                
                if has_month_day:
                    # This is a real full date query
                    doc_dates = profile.get("explicit_dates", [])
                    query_date_str = query_full_date.date().isoformat()
                    query_year = query_full_date.year
                    query_month = query_full_date.strftime("%B").lower()
                    
                    # 1) Exact date match
                    for dt in doc_dates:
                        if hasattr(dt, 'date'):
                            if dt.date().isoformat() == query_date_str:
                                return True
                    
                    # 2) Year + month match (more permissive)
                    doc_years = profile.get("years", [])
                    if query_year in doc_years:
                        # Check month_years
                        month_years = profile.get("month_years", [])
                        for (mm, yy) in month_years:
                            if mm.lower() == query_month and yy == query_year:
                                return True
                        # Check if month appears in text (even if not in month_years)
                        text_lower = (text or "").lower()
                        month_pattern = re.compile(rf"\b{re.escape(query_month)}\b", re.IGNORECASE)
                        if month_pattern.search(text_lower):
                            return True
                        # Also check metadata
                        meta_str = str(metadata or {}).lower()
                        if month_pattern.search(meta_str):
                            return True
                    
                    # 3) If strict=False, accept if year matches (soft mode)
                    if not strict and query_year in doc_years:
                        return True
                        
                    return False
            
            if yr_range:
                a, b = yr_range
                return self.doc_matches_year_range(profile, a, b, strict=strict)

            if month_year:
                m, y = month_year
                # Check month_years first
                month_years = profile.get("month_years", [])
                if any((m.lower() == mm.lower() and y == yy) for (mm, yy) in month_years):
                    return True
                # Fallback: check if year matches and month appears in text/metadata
                doc_years = profile.get("years", [])
                if y in doc_years:
                    text_lower = (text or "").lower()
                    month_pattern = re.compile(rf"\b{re.escape(m.lower())}\b", re.IGNORECASE)
                    if month_pattern.search(text_lower):
                        return True
                    # Also check metadata
                    meta_str = str(metadata or {}).lower()
                    if month_pattern.search(meta_str):
                        return True
                return False

            if single_year:
                return self.doc_matches_single_year(profile, single_year, strict=strict)

            # Month-only query: match if document has that month in any year
            if month_only:
                month_lower = month_only.lower()
                text_lower = (text or "").lower()
                doc_month_years = profile.get("month_years", [])
                doc_dates = profile.get("explicit_dates", [])
                
                # 1) Check metadata month_only field FIRST (most reliable)
                if metadata:
                    meta_month_only = metadata.get("month_only", "").lower()
                    if meta_month_only == month_lower:
                        return True  # Direct metadata match - accept immediately
                
                # 2) Check if month appears in document text (very permissive)
                # This is the most reliable check since text extraction works well
                month_pattern = re.compile(rf"\b{re.escape(month_lower)}\b", re.IGNORECASE)
                if month_pattern.search(text_lower):
                    return True
                
                # 3) Check month_years
                for (mm, yy) in doc_month_years:
                    if mm.lower() == month_lower:
                        return True
                
                # 4) Check explicit dates
                for dt in doc_dates:
                    if hasattr(dt, 'strftime'):
                        doc_month = dt.strftime("%B").lower()
                        if doc_month == month_lower:
                            return True
                
                # 5) Check metadata string representation
                if metadata:
                    meta_str = str(metadata).lower()
                    if month_pattern.search(meta_str):
                        return True
                    # Check all metadata values
                    for key, value in metadata.items():
                        if isinstance(value, str) and month_lower in value.lower():
                            return True
                
                # 6) Very permissive: if document has years, check text more broadly
                if profile.get("years"):
                    # Simple substring check (more permissive than regex)
                    if month_lower in text_lower:
                        return True
                    # Check filename/source_file
                    if metadata:
                        source_file = str(metadata.get("source_file", "")).lower()
                        if month_lower in source_file:
                            return True
                
                # 7) Final fallback: if month appears anywhere in text or metadata, accept
                # This ensures month-only queries work even with minimal metadata
                if month_lower in text_lower:
                    return True
                if metadata:
                    meta_full_str = str(metadata).lower()
                    if month_lower in meta_full_str:
                        return True
                    
                return False

            # explicit intent but no parseable units: reject
            return False
        
        # ----------------------------------------
        # 2.5) Month-only queries (even if not detected as explicit by keywords)
        # ----------------------------------------
        # Handle month-only queries that might not trigger explicit detection
        if month_only:
            month_lower = month_only.lower()
            text_lower = (text or "").lower()
            doc_month_years = profile.get("month_years", [])
            doc_dates = profile.get("explicit_dates", [])
            
            # 1) Check metadata month_only field FIRST (most reliable)
            if metadata and metadata.get("month_only", "").lower() == month_lower:
                return True
            
            # 2) Check if month appears in document text (most permissive and reliable)
            month_pattern = re.compile(rf"\b{re.escape(month_lower)}\b", re.IGNORECASE)
            if month_pattern.search(text_lower):
                return True
            
            # 3) Check month_years
            for (mm, yy) in doc_month_years:
                if mm.lower() == month_lower:
                    return True
            
            # 4) Check explicit dates
            for dt in doc_dates:
                if hasattr(dt, 'strftime'):
                    doc_month = dt.strftime("%B").lower()
                    if doc_month == month_lower:
                        return True
            
            # 5) Check metadata string representation
            if metadata:
                meta_str = str(metadata).lower()
                if month_pattern.search(meta_str):
                    return True
                # Check all metadata values
                for key, value in metadata.items():
                    if isinstance(value, str) and month_lower in value.lower():
                        return True
            
            # 6) Very permissive: simple substring check in text
            if month_lower in text_lower:
                return True
            
            # 7) Check filename/source_file
            if metadata:
                source_file = str(metadata.get("source_file", "")).lower()
                if month_lower in source_file:
                    return True
                
            return False

        # ----------------------------------------
        # 3) SOFT MODE (implicit intent)
        # ----------------------------------------
        # If document has ANY date at all, it is considered temporally relevant
        if profile.get("explicit_dates") or profile.get("years") or profile.get("month_years"):
                return True

        return False

    # ============================================================
    # EXPIRED DETECTION
    # ============================================================
    def is_expired(self, metadata: dict) -> bool:
        """Check if document is expired based on metadata."""
        raw = (
            metadata.get("expiry_date")
            or metadata.get("end_date")
            or metadata.get("terminates_on")
        )

        if not raw:
            return False

        dt = self.parse_date(raw)
        if not dt:
                return False

        from datetime import timezone
        now = datetime.now(timezone.utc).date()
        return dt.date() < now

    # ============================================================
    # SOFT TEMPORAL SIGNALS (for ranking)
    # ============================================================
    def compute_temporal_soft_features(self, text: str, metadata: dict):
        """
        Extracts temporal soft signals useful for ranking.
        """
        out = {
            "has_any_date": False,
            "dates": [],
            "tenure_years": None,
            "start_year": None,
            "end_year": None
        }

        # 1) extract any date in document
        matches = DATE_RE.findall(text or "")
        for groups in matches:
            for g in groups:
                if not g:
                    continue
                dt = self.parse_date(g)
                if dt:
                    out["has_any_date"] = True
                    out["dates"].append(dt)

        # 2) employment/tenure reasoning (CVs, contracts, letters)
        start = metadata.get("start_date")
        end = metadata.get("end_date")

        start_dt = self.parse_date(start) if start else None
        end_dt = self.parse_date(end) if end else None

        if start_dt and end_dt:
            out["start_year"] = start_dt.year
            out["end_year"] = end_dt.year
            out["tenure_years"] = max(0, out["end_year"] - out["start_year"])

        return out


# ============================================================================
# Ingestion Temporal Normalizer (merged from ingestion_temporal_normalizer.py)
# ============================================================================

try:
    from dateutil import parser as _dateutil_parser  # type: ignore
    DATEUTIL_AVAILABLE = True
except Exception:
    _dateutil_parser = None
    DATEUTIL_AVAILABLE = False

# Common month mapping to support spelled-out months and abbreviations
MONTHS_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Regex patterns for explicit date contexts and common date formats
SAFE_DATE_CONTEXTS = [
    r"(?:dated|dated on|dated:|dated:|dated\s+on|signed on|signed|signed:|executed|executed on|effective|effective on|effective date|issued on|issued|as of|as at)\s*[,:]?\s*(?:[A-Za-z]{3,9}\s*\d{1,2}(?:st|nd|rd|th)?[,]?\s*(?:19|20)\d{2}|\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}|(?:19|20)\d{2})",
    # Standalone month-year contexts: "February 2024", "Feb 2024"
    r"(?:effective|dated|issued|signed)\s*(?:[:\-])?\s*(?:[A-Za-z]{3,9})\s+\d{4}",
]

# Generic date patterns (order matters: ISO/Timestamp, Month day, day/month/year, mm/dd/yyyy)
DATE_PATTERNS = [
    # ISO-like: 2024-02-05, 2024/02/05, 2024-02-05T00:00:00Z
    r"(?P<iso>(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}(?:[T\s]\d{1,2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?)",
    # Month day, year: February 5, 2024 / Feb 5 2024 / February 05 2024
    r"(?P<mdy>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s*(?:19|20)\d{2})",
    # Month Year: February 2024, Feb 2024
    r"(?P<my>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2})",
    # Day/Month/Year or Month/Day/Year variants like 05/02/2024 or 2-5-2024 or 09 08 2025
    r"(?P<dmy>\d{1,2}[/\-\s]\d{1,2}[/\-\s](?:19|20)\d{2})",
    # Year only
    r"(?P<y>(?:19|20)\d{2})",
]

# Precompiled regex objects
SAFE_CONTEXT_RE = [re.compile(p, re.IGNORECASE) for p in SAFE_DATE_CONTEXTS]
DATE_REGEX = re.compile("|".join("(" + p + ")" for p in DATE_PATTERNS), re.IGNORECASE)


def _try_dateutil_parse(s: str) -> Optional[datetime]:
    if not DATEUTIL_AVAILABLE or not _dateutil_parser:
        return None
    try:
        # fuzzy to accept things like "Signed: February 5, 2024"
        dt = _dateutil_parser.parse(s, fuzzy=True, default=datetime(1900, 1, 1))
        # Reject improbable years
        if dt.year < 1900 or dt.year > 2100:
            return None
        return dt
    except Exception:
        return None


def _iso_safe_parse(s: str) -> Optional[datetime]:
    # Try ISO-like formats using fromisoformat (Python 3.7+)
    try:
        s2 = s.strip()
        # Handle trailing Z
        if s2.endswith("Z"):
            s2 = s2.replace("Z", "+00:00")
        return datetime.fromisoformat(s2)
    except Exception:
        return None


def _parse_using_regex_token(token: str) -> Optional[datetime]:
    """
    Heuristic parse for matched token groups when dateutil is unavailable or fuzzy failed.
    Supports:
      - 'February 5, 2024'
      - 'Feb 2024' (month-year)
      - '05/02/2024' (dmy or mdy ambiguous)
      - '2024-02-05'
    """
    token = token.strip()
    
    # 1) ISO attempt
    dt = _iso_safe_parse(token)
    if dt:
        return dt

    # 2) Month day, year e.g., February 5, 2024
    m = re.match(r"(?P<month>[A-Za-z]{3,9})\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s*(?P<year>(?:19|20)\d{2})", token, re.IGNORECASE)
    if m:
        mo = MONTHS_MAP.get(m.group("month").lower()[:3])
        if mo:
            day = int(m.group("day"))
            year = int(m.group("year"))
            try:
                return datetime(year, mo, day)
            except Exception:
                pass

    # 3) Month Year only (Feb 2024)
    m2 = re.match(r"(?P<month>[A-Za-z]{3,9})\s+(?P<year>(?:19|20)\d{2})", token, re.IGNORECASE)
    if m2:
        mo = MONTHS_MAP.get(m2.group("month").lower()[:3])
        if mo:
            year = int(m2.group("year"))
            # default to 1st of month
            try:
                return datetime(year, mo, 1)
            except Exception:
                pass

    # 4) Numeric d/m/y or m/d/y (handles / - and space)
    m3 = re.match(r"(?P<p1>\d{1,2})[/\-\s](?P<p2>\d{1,2})[/\-\s](?P<year>(?:19|20)\d{2})", token)
    if m3:
        p1 = int(m3.group("p1"))
        p2 = int(m3.group("p2"))
        year = int(m3.group("year"))
        # Ambiguity: assume d/m/y if p1 > 12 or common locale heuristics.
        # We'll attempt both safely: prefer day<=31 and month<=12 and a valid date.
        candidates = []
        # Try day=p1, month=p2 (DMY)
        try:
            candidates.append(datetime(year, p2, p1))
        except Exception:
            pass
        # Try day=p2, month=p1 (MDY - common in US)
        try:
            candidates.append(datetime(year, p1, p2))
        except Exception:
            pass
        return candidates[0] if candidates else None

    # 5) Year only
    m4 = re.match(r"^(?:19|20)\d{2}$", token)
    if m4:
        y = int(token)
        try:
            return datetime(y, 1, 1)
        except Exception:
            pass

    return None


class IngestionTemporalNormalizer:
    """
    Class for ingestion-time normalization of temporal metadata.

    Usage:
        normalizer = IngestionTemporalNormalizer()
        normalized = normalizer.normalize(metadata_dict, full_text, file_id)

    Returns a dict with fields:
      - years: List[int] (safe extracted years)
      - primary_dates: List[str] (ISO yyyy-mm-dd strings)
      - month_years: List[Tuple[str,int]] (e.g., [("february", 2024)])
      - day_month_years: List[Tuple[int,int,int]] (day, month, year)
      - is_expired: bool
      - earliest_date / latest_date: ISO strings or None
      - normalized_metadata: original metadata merged with normalized temporal fields
    """

    def __init__(self, allow_filename_dates: bool = False):
        # By default, filenames are NOT used as date sources to prevent false positives.
        self.allow_filename_dates = allow_filename_dates

    def normalize(self, metadata: Dict[str, Any], full_text: str = "", file_id: str = "") -> Dict[str, Any]:
        meta = dict(metadata or {})  # copy to avoid mutation
        text = (full_text or "").strip()
        file_id = (file_id or "").strip()

        # 1. Collect candidate tokens: explicit safe contexts + global date regex
        candidate_tokens = self._collect_candidate_tokens(text)

        # Optionally include filename tokens if enabled (disabled by default)
        if self.allow_filename_dates and file_id:
            candidate_tokens.extend(self._collect_candidate_tokens_from_filename(file_id))

        # 2. Parse tokens into datetime objects
        parsed_datetimes = []
        for tok in candidate_tokens:
            dt = None
            # Prefer dateutil parse when available for best coverage
            if DATEUTIL_AVAILABLE:
                dt = _try_dateutil_parse(tok)
            if not dt:
                dt = _parse_using_regex_token(tok)
            if dt:
                # Accept only sensible years
                if 1900 <= dt.year <= 2100:
                    parsed_datetimes.append(dt)

        # 3. Also consider explicit metadata date fields (trusted)
        explicit_dates = self._parse_metadata_date_fields(meta)
        parsed_datetimes.extend(explicit_dates)

        # Deduplicate by date (use date() to reduce timezone noise)
        unique_dates = {}
        for dt in parsed_datetimes:
            try:
                key = dt.date().isoformat()
            except Exception:
                continue
            if key not in unique_dates:
                unique_dates[key] = dt

        parsed_list = sorted(unique_dates.values())

        # 4. Build normalized outputs
        years = sorted({d.year for d in parsed_list})
        primary_dates_iso = [d.date().isoformat() for d in parsed_list]  # ISO yyyy-mm-dd
        month_years = []
        day_month_years = []
        for d in parsed_list:
            month_years.append((d.strftime("%B").lower(), d.year))
            day_month_years.append((d.day, d.month, d.year))

        earliest_date = primary_dates_iso[0] if primary_dates_iso else None
        latest_date = primary_dates_iso[-1] if primary_dates_iso else None

        # 5. Expired flag: consult explicit expiry metadata first, else infer from latest_date
        explicit_expiry = meta.get("expiry_date") or meta.get("end_date") or meta.get("terminates_on")
        is_expired = None
        if explicit_expiry:
            try:
                # Try parse explicit expiry using dateutil or regex
                if DATEUTIL_AVAILABLE:
                    dt_e = _try_dateutil_parse(str(explicit_expiry))
                else:
                    dt_e = _parse_using_regex_token(str(explicit_expiry))
                if dt_e:
                    is_expired = dt_e.date() < datetime.now(timezone.utc).date()
            except Exception:
                is_expired = None

        if is_expired is None:
            # Fallback: if latest_date exists and it's in the past, consider expired (conservative)
            if latest_date:
                try:
                    latest_dt = datetime.fromisoformat(latest_date)
                    is_expired = latest_dt.date() < datetime.now(timezone.utc).date()
                except Exception:
                    is_expired = False
            else:
                is_expired = False

        # 6. Populate normalized metadata keys
        meta_normalized = dict(meta)
        meta_normalized["years"] = years
        meta_normalized["primary_dates"] = primary_dates_iso
        # unique month-year tuples
        unique_my = []
        seen_my = set()
        for m, y in month_years:
            key = f"{m}|{y}"
            if key not in seen_my:
                unique_my.append((m, y))
                seen_my.add(key)
        meta_normalized["month_years"] = unique_my
        meta_normalized["day_month_years"] = day_month_years
        meta_normalized["earliest_date"] = earliest_date
        meta_normalized["latest_date"] = latest_date
        meta_normalized["is_expired"] = bool(is_expired)

        # For compatibility with older code expecting single month_year or date fields:
        if unique_my:
            meta_normalized.setdefault("month_year", unique_my[0])
        if primary_dates_iso:
            meta_normalized.setdefault("date", primary_dates_iso[0])
            meta_normalized.setdefault("signed_on", primary_dates_iso[0])

        return meta_normalized

    def _collect_candidate_tokens(self, text: str) -> List[str]:
        """
        Extracts tokens likely to contain dates.
        Prioritizes tokens found in SAFE_CONTEXTS, then falls back to global DATE_REGEX matches.
        """
        tokens: List[str] = []

        # 1) Safe context matches (high-confidence)
        for rx in SAFE_CONTEXT_RE:
            for m in rx.finditer(text):
                tok = m.group(0).strip()
                if tok and tok not in tokens:
                    tokens.append(tok)

        # 2) Generic date regex matches across the text
        for m in DATE_REGEX.finditer(text):
            # m.group(0) is entire match
            tok = m.group(0).strip()
            if tok and tok not in tokens:
                tokens.append(tok)

        # 3) If still nothing, attempt to find month-year tokens loosely (e.g., "February 2024")
        if not tokens:
            loose_my = re.findall(r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2}", text, re.IGNORECASE)
            for m in loose_my:
                s = m.strip()
                if s and s not in tokens:
                    tokens.append(s)

        return tokens

    def _collect_candidate_tokens_from_filename(self, file_id: str) -> List[str]:
        """
        Extract tokens from filename - used only when allow_filename_dates is True.
        We still try to be conservative: use the core DATE_REGEX for detection.
        """
        if not file_id:
            return []
        tokens: List[str] = []
        fname = file_id.replace("-", " ").replace("_", " ").replace(".", " ")
        
        # 1. Use the full DATE_REGEX on filename tokens
        for m in DATE_REGEX.finditer(fname):
            tok = m.group(0).strip()
            if tok and tok not in tokens:
                tokens.append(tok)
                
        # 2. find month-year like "february 2024" (fallback)
        for m in re.findall(r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2}", fname, re.IGNORECASE):
            if m not in tokens:
                tokens.append(m)
        # 3. find 4-digit groups that look like years (fallback)
        for y in re.findall(r"(?:19|20)\d{2}", fname):
            if y not in tokens:
                tokens.append(y)
        return tokens

    def _parse_metadata_date_fields(self, meta: Dict[str, Any]) -> List[datetime]:
        """
        Parse explicit metadata date fields such as 'signed_on', 'date', 'issued_on', 'created_at', etc.
        These are considered authoritative and get highest priority.
        """
        out: List[datetime] = []
        candidate_keys = ["signed_on", "signed", "date", "issued_on", "created_at", "effective_on", "start_date", "end_date", "expiry_date", "terminates_on"]
        for k in candidate_keys:
            if k in meta and meta[k]:
                val = meta[k]
                dt = None
                if DATEUTIL_AVAILABLE:
                    dt = _try_dateutil_parse(str(val))
                if not dt:
                    dt = _parse_using_regex_token(str(val))
                if dt and 1900 <= dt.year <= 2100:
                    out.append(dt)
        return out
