"""
Semantic Search Utilities

Consolidated utility functions for semantic search:
- Filename normalization and matching
- Location normalization
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Filename Utilities (merged from filename_utils.py)
# ============================================================================

# remove typical chunk and UUID suffix patterns and extensions, return basename (no ext)
_CHUNK_RE = re.compile(r'(?:(?:::chunk::\d+(?:::\d+)?)$)|(?:[_-][0-9a-f]{8,}(?:-[0-9a-f]{4,})*$)', flags=re.IGNORECASE)

def normalize_filename(file_id: str) -> str:
    """
    Normalize a file_id into a stable basename (without extension and chunk/uuid suffix).
    If file_id is empty -> returns empty string.
    Examples:
      "/path/to/file.pdf::chunk::1::123abc" -> "file"
      "resume_john_doe_abcdef12.pdf" -> "resume_john_doe"
    """
    if not file_id:
        return ""
    filename_only = os.path.basename(file_id)
    # strip chunk suffix if present
    filename_base = _CHUNK_RE.sub("", filename_only)
    # remove extension
    base, _ext = os.path.splitext(filename_base)
    return base

def normalize_file_id(file_id: str) -> str:
    """
    Normalize file_id by stripping Milvus chunk suffixes like ::chunk::1.
    Returns lowercased file_id without chunk suffix.
    """
    if not file_id:
        return ""
    # remove ::chunk::N
    file_id = re.sub(r"::chunk::\d+", "", file_id)
    # strip directory paths
    file_id = os.path.basename(file_id)
    return file_id.lower()


# ============================================================================
# Location Normalization (merged from location_normalizer.py)
# ============================================================================

# Load location map from configuration file with default fallback
GLOBAL_LOCATION_MAP: Dict[str, List[str]] = {}
try:
    import json
    config_path = os.path.join(os.path.dirname(__file__), "location_map.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            GLOBAL_LOCATION_MAP = json.load(f)
except Exception:
    pass

if not GLOBAL_LOCATION_MAP:
    GLOBAL_LOCATION_MAP = {
        "united_kingdom": ["united kingdom", "uk", "u.k.", "britain", "great britain", "gb", "england", "scotland", "wales", "england and wales"],
        "england": ["england"],
        "scotland": ["scotland"],
        "wales": ["wales"],
        "london": ["london", "greater london"],
        "united_arab_emirates": ["united arab emirates", "uae", "u.a.e."],
        "dubai": ["dubai", "dubayy"],
        "abu_dhabi": ["abu dhabi"],
        "pakistan": ["pakistan", "pk"],
        "karachi": ["karachi"],
        "islamabad": ["islamabad"],
        "lahore": ["lahore"],
        "south_africa": ["south africa", "za", "rsa"],
        "johannesburg": ["johannesburg", "joburg"],
        "cape_town": ["cape town", "capetown"],
        "india": ["india", "in"],
        "delhi": ["delhi", "new delhi"],
        "mumbai": ["mumbai", "bombay"],
        "canada": ["canada", "ca"],
        "toronto": ["toronto"],
        "australia": ["australia", "au"],
        "sydney": ["sydney"],
        "new_york": ["new york", "ny", "nyc", "new york city", "manhattan"],
        "california": ["california", "ca", "los angeles", "la", "san francisco", "sf"],
        "berlin": ["berlin"],
        "dallas": ["dallas", "dallas tx", "dallas, tx", "dallas texas", "dallas, texas"],
        "austin": ["austin", "austin tx", "austin, tx", "austin texas", "austin, texas"],
        "texas": ["texas", "tx"],
        "florida": ["florida", "fl", "miami"],
        "illinois": ["illinois", "il", "chicago"],
    }


# Build alias -> canonical mapping for direct lookup
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in GLOBAL_LOCATION_MAP.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical

_word_boundary_cache = {}

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    # Normalize unicode and lower-case
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    return text.strip().lower()

def normalize_location(name: str) -> Optional[str]:
    """
    Normalize a location name (free text) into a canonical key if possible.
    Returns canonical key (e.g., 'dubai') or fallback normalized string (spaces -> underscores),
    or None if input empty.
    """
    if not name:
        return None
    name_norm = _normalize_text(name)
    # direct alias lookup
    if name_norm in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[name_norm]
    # try word-boundary partial match against aliases
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        # prebuild regex in cache
        if alias not in _word_boundary_cache:
            escaped_alias = re.escape(alias)
            pattern = rf"\b{escaped_alias}\b"
            _word_boundary_cache[alias] = re.compile(pattern, flags=re.IGNORECASE)
        if _word_boundary_cache[alias].search(name_norm):
            return canonical
    # not found: return normalized fallback (replace spaces with underscores)
    return name_norm.replace(" ", "_")

def get_location_aliases(canonical_key: str) -> List[str]:
    """
    Return alias list for canonical key, or empty list if not present.
    """
    return GLOBAL_LOCATION_MAP.get(canonical_key, [])


def tx_metro_snippet_has_wrong_peer_only(
    anchor_cities: List[str],
    chunk_text: str,
    decoded_filename: str,
    file_id: str = "",
) -> bool:
    """
    Single-anchor Austin or Dallas query: drop the hit if the *peer* metro name appears
    anywhere in the visible surface (chunk + decoded basename + file_id basename).

    This matches strict /search verification (no peer city in top hits) and removes
    mixed-city chunks, Austin-only chunks that passed full-doc anchor checks, and peer
    tokens embedded in basenames or IDs.
    """
    if not anchor_cities or len(anchor_cities) != 1:
        return False
    ac = (anchor_cities[0] or "").strip().lower()
    if ac not in ("austin", "dallas"):
        return False
    peer = "dallas" if ac == "austin" else "austin"
    fid_tail = os.path.basename((file_id or "").split("?")[0]).lower()
    surface = f"{chunk_text or ''} {decoded_filename or ''} {fid_tail}".lower()
    return bool(re.search(rf"\b{re.escape(peer)}\b", surface))

