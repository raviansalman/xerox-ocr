"""
Semantic Components - Consolidated Module

This module consolidates:
- CrossEncoderReranker: Cross-encoder reranking for search results
- MetadataIndex: Universal metadata indexing system
- Image Processing: Image captioning and analysis utilities
- EntityExtractor: Dynamic entity extraction system

All components are production-ready and fully backward compatible.
"""

import re
import os
import logging
import time
import uuid
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

import numpy as np
# Heavy imports moved inside classes/methods to save baseline RAM
# import torch
# import tiktoken
# from PIL import Image
# from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ============================================================================
# Cross-Encoder Reranker (merged from reranker.py)
# ============================================================================

# Lazy imports used inside CrossEncoderReranker
TRANSFORMERS_AVAILABLE = True


class CrossEncoderReranker:
    """Cross-encoder reranker for document search results."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or "/app/cache"
            offline = os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            
            logger.info(f"Loading CrossEncoderReranker (offline={offline}, cache={cache_dir})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/bge-reranker-base",
                cache_dir=cache_dir,
                local_files_only=offline,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-base",
                cache_dir=cache_dir,
                local_files_only=offline,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info("CrossEncoderReranker loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoderReranker: {e}. Falling back to no-op reranking.")
            self.tokenizer = None
            self.model = None

    def rerank(self, query, docs):
        if not docs:
            return []
            
        # Fallback if model failed to load
        if self.model is None or self.tokenizer is None:
            return [1.0] * len(docs)
        """
        Rerank documents for a given query using a cross-encoder.

        Processes documents in batches, applies sigmoid to logits to obtain
        scores in [0,1], and falls back to 0.0 scores if a batch fails.
        """
        if not docs:
            return []

        batch_size = 16  # Process 16 documents at a time
        all_scores = []
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            pairs = [(query, d) for d in batch_docs]

            try:
                enc = self.tokenizer.batch_encode_plus(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512,
                )

                # Log when truncation is likely happening
                max_seq = enc["input_ids"].shape[1]
                if max_seq >= 512:
                    logger.warning(
                        f"Truncation: max seq length {max_seq} (limit 512) "
                        f"in batch {i // batch_size + 1}"
                    )
                
                # Move tensors to device
                import torch
                enc = {k: v.to(self.device) for k, v in enc.items()}
                
                # Model inference
                with torch.no_grad():
                    logits = self.model(**enc).logits

                    # Normalize logit dimension
                    if logits.dim() == 2:
                        logits = logits[:, -1]

                    scores = logits.squeeze(-1)

                # Normalize scores using sigmoid to [0,1] range
                normalized_scores = torch.sigmoid(scores)

                # Move to CPU before .tolist() for GPU tensors
                # Handle both tensor and scalar cases
                if isinstance(normalized_scores, torch.Tensor):
                    scores_list = normalized_scores.detach().cpu().tolist()
                    # If it's a single-element tensor, tolist() returns a scalar, so wrap it
                    if isinstance(scores_list, (int, float)):
                        all_scores.append(float(scores_list))
                    else:
                        all_scores.extend(scores_list)
                else:
                    # Already a scalar
                    all_scores.append(float(normalized_scores))

            except Exception as e:
                logger.warning(
                    f"Reranking failed for batch {i // batch_size + 1}: {e}"
                )
                # Ensure we add exactly one zero score per document in the failed batch
                all_scores.extend([0.0] * len(batch_docs))
        
        return all_scores


# ============================================================================
# Universal Metadata Index (merged from metadata_index.py)
# ============================================================================

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
FILENAME_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
MONEY_RE = re.compile(r"\$?([0-9]{2,9}(?:\.[0-9]{1,2})?)")

# Import temporal engine for safe extraction
try:
    from src.semantic.temporal_engine import TemporalReasoningEngine
except ImportError:
    from .temporal_engine import TemporalReasoningEngine


def _thread_safe(method):
    """Decorator to ensure thread-safe execution of MetadataIndex methods."""
    import functools
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                return method(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    return wrapper


class MetadataIndex:
    """
    UNIVERSAL METADATA INDEX v3.0 (Temporal-Ready)
    
    Extends metadata index with:
    - organizations, persons, clauses, skills, amounts
    - semantic tags, roles, departments, categories
    - Hybrid temporal reasoning (via TemporalReasoningEngine)
    - Month-year indexing, safe year indexing, expired flag storage
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.by_year = defaultdict(list)
        self.by_year_range = defaultdict(list)
        self.by_month_year = defaultdict(list)
        self.by_location = defaultdict(list)
        self.by_expiry = defaultdict(list)
        
        # Temporal engine for safe extraction
        self.temporal = TemporalReasoningEngine()
        
        # Enhanced temporal indices (v5.0 Full Integration)
        self.by_full_date = defaultdict(list)          # full_date (YYYY-MM-DD) → [file_id]
        self.by_month_only = defaultdict(list)         # month_name → [file_id]
        self.by_expired = {"yes": [], "no": []}
        self.by_temporal_hash = {}  # unique per-doc temporal signature

        # NEW universal indices
        self.by_person = defaultdict(list)
        self.by_org = defaultdict(list)
        self.by_clause = defaultdict(list)
        self.by_skill = defaultdict(list)
        self.by_amount = defaultdict(list)
        self.by_category = defaultdict(list)
        self.by_signer = defaultdict(list)  # Signer index for lexical-first retrieval
        self.by_document_type = defaultdict(list)  # document_type -> [file_id]

        self.docs = {}  # file_id -> metadata

    @_thread_safe
    def index_document(self, file_id: str, meta: dict):
        """
        Store and invert all metadata into fast indices.
        
        LEXICAL-FIRST ARCHITECTURE:
        - Extracts ALL tokens from full_text at ingestion time
        - Years, month-years, locations, persons, organizations, clauses, signers
        - Guarantees 100% recall for lexical matches
        """
        self.docs[file_id] = meta

        # ============================================================
        # TEMPORAL INDEXING (HYBRID ENGINE) - v5.0 Full Integration
        # ============================================================
        # Extract from full_text FIRST, then fall back to metadata
        full_text = meta.get("full_text", "") or ""
        
        # Extract safe doc years (prefer metadata-provided values from ingestion)
        raw_years = meta.get("years")
        doc_years: list[int] = []
        if raw_years:
            if isinstance(raw_years, (list, tuple, set)):
                for y in raw_years:
                    try:
                        doc_years.append(int(y))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(raw_years, (int, float)):
                doc_years.append(int(raw_years))
        
        # ALWAYS extract from full_text to ensure complete coverage
        if full_text:
            text_years = self.temporal.extract_doc_years(
                file_id,
                meta,
                full_text
            )
            doc_years = list(set(doc_years + text_years))
        
        # Final fallback if still no years
        if not doc_years:
            doc_years = self.temporal.extract_doc_years(
                file_id,
                meta,
                full_text
            )
        filename = meta.get("source_file") or file_id or ""
        filename_years: list[int] = []
        for match in FILENAME_YEAR_RE.findall(filename):
            try:
                filename_years.append(int(match))
            except (TypeError, ValueError):
                continue
        if filename_years:
            doc_years = list({*doc_years, *filename_years})
        meta["years"] = sorted(doc_years)
        
        # Index by each year (using set for deduplication)
        if doc_years:
            for y in doc_years:
                if y not in self.by_year:
                    self.by_year[y] = []
                if file_id not in self.by_year[y]:
                    self.by_year[y].append(file_id)

            # Index by overall year range for the document (min_year, max_year)
            try:
                start_y = int(min(doc_years))
                end_y = int(max(doc_years))
                key = (start_y, end_y)
                if file_id not in self.by_year_range[key]:
                    self.by_year_range[key].append(file_id)
            except (TypeError, ValueError):
                # If doc_years contain non-numeric entries, skip range indexing
                pass

        # Full dates indexing (v5.0)
        full_dates = meta.get("full_dates") or meta.get("primary_dates") or []
        if not full_dates:
            # Extract full dates from text if not in metadata
            full_text = meta.get("full_text", "") or ""
            if full_text:
                full_dates = self.temporal.extract_full_dates(full_text)
                if full_dates:
                    meta["full_dates"] = full_dates
        
        if full_dates:
            for dt in full_dates:
                # Convert datetime to string key for indexing
                if hasattr(dt, 'isoformat'):
                    dt_key = dt.isoformat()[:10]  # YYYY-MM-DD format
                elif hasattr(dt, 'date'):
                    dt_key = dt.date().isoformat()
                else:
                    dt_key = str(dt)[:10]
                if dt_key not in self.by_full_date:
                    self.by_full_date[dt_key] = []
                if file_id not in self.by_full_date[dt_key]:
                    self.by_full_date[dt_key].append(file_id)

        # Month + Year extraction
        # Accepts: "February 2024", "Feb 2024", "2024-02-05", etc.
        month_year = meta.get("month_year")
        if not month_year:
            # Infer month-year from doc text using temporal profile
            full_text = meta.get("full_text", "") or ""
            if full_text:
                profile = self.temporal.extract_doc_temporal_profile(file_id, meta, full_text)
                month_years_list = profile.get("month_years", [])
                if month_years_list:
                    month_year = month_years_list[0]
                    meta["month_year"] = month_year
        
        # Index month-year
        if month_year:
            m, y = month_year
            key = (m.lower(), int(y))
            if key not in self.by_month_year:
                self.by_month_year[key] = []
            if file_id not in self.by_month_year[key]:
                self.by_month_year[key].append(file_id)

        # Month-only indexing (v5.0)
        month_only = meta.get("month_only")
        if not month_only:
            # Extract month-only from text if not in metadata
            full_text = meta.get("full_text", "") or ""
            if full_text:
                month_only = self.temporal.extract_month_only(full_text)
                if month_only:
                    meta["month_only"] = month_only
        
        if month_only:
            month_key = month_only.lower()
            if month_key not in self.by_month_only:
                self.by_month_only[month_key] = []
            if file_id not in self.by_month_only[month_key]:
                self.by_month_only[month_key].append(file_id)

        # Expired status (explicit or inferred)
        expired = meta.get("is_expired")
        if expired is None:
            expired = self.temporal.is_expired(meta)
        meta["is_expired"] = bool(expired)
        
        if expired:
            self.by_expired["yes"].append(file_id)
            self.by_expiry["expired"].append(file_id)  # Backward compatibility
        else:
            self.by_expired["no"].append(file_id)
            self.by_expiry["active"].append(file_id)  # Backward compatibility
        
        # Temporal signature — used for debugging, caching, validation sync
        temporal_hash = f"{sorted(doc_years)}|{month_year}|{expired}"
        self.by_temporal_hash[file_id] = temporal_hash
        
        logger.debug(
            f"[TEMPORAL-INDEX] file={file_id} | years={doc_years} | month_year={month_year} | expired={expired}"
        )

        # ----- LOCATIONS -----
        # LEXICAL-FIRST: Extract from full_text + metadata
        # Location can come from multiple metadata fields and may contain
        # composite phrases such as "Austin, Texas" or "State of California".
        # We index both the full phrase and individual tokens so that:
        #   "austin"  → "Austin, Texas"
        #   "texas"   → "Austin, Texas"
        #   "california" → "State of California"
        locations: list[str] = []
        # Primary single-value fields
        for key in ("location", "jurisdiction", "governing_law"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                locations.append(val)
        # Multi-value field produced by entity extractor
        for loc_val in meta.get("locations", []) or []:
            if isinstance(loc_val, str) and loc_val.strip():
                locations.append(loc_val)
        
        # EXTRACT FROM FULL_TEXT: City/state/country patterns (Title Case or ALL CAPS)
        if full_text:
            # Matches "Austin", "AUSTIN", "Austin Texas", "AUSTIN TEXAS", "State of California"
            # Allows for commas and common bridge words like 'of'.
            location_pattern = re.compile(
                r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s*,\s*[A-Z][A-Za-z]+)*)\b'
            )
            for match in location_pattern.finditer(full_text):
                loc_phrase = match.group(1).strip()
                # Additional: handle ALL CAPS sequences like "AUSTIN"
                if len(loc_phrase) > 2 and loc_phrase not in locations:
                    locations.append(loc_phrase)
            
            # Catch all-caps sequences separately if they were missed
            all_caps_pattern = re.compile(r'\b([A-Z]{3,})\b')
            for match in all_caps_pattern.finditer(full_text):
                loc_caps = match.group(1).strip()
                if loc_caps not in locations:
                    # Generic filter: if it's a known common word, skip
                    if loc_caps not in ("AND", "THE", "DATE", "CITY", "STATE", "NAME"):
                        locations.append(loc_caps)

        def _index_location_phrase(phrase: str):
            pl = phrase.lower().strip()
            if not pl:
                return
            # Index full phrase
            self.by_location[pl].append(file_id)
            # Index individual meaningful tokens
            for token in re.split(r"[^a-zA-Z0-9]+", pl):
                tok = token.strip()
                if len(tok) < 3:
                    continue
                self.by_location[tok].append(file_id)

        for loc in locations:
            _index_location_phrase(str(loc))

        # ------------------------------------------------------------------
        # UNIVERSAL INDEX FIELDS
        # ------------------------------------------------------------------

        # ----- PERSONS -----
        # LEXICAL-FIRST: Extract from full_text + metadata
        # Index both full names and individual name tokens
        persons = meta.get("persons", []) or []
        persons_set = set(p.strip() for p in persons if isinstance(p, str) and p.strip())
        
        # EXTRACT FROM FULL_TEXT: Person name patterns
        # Pattern: "Signed by John Smith", "Lisa Riordan", "C. Dominguez", "Mitul", etc.
        if full_text:
            # Match capitalized sequences that appear in signature contexts
            signature_contexts = [
                r'(?:signed|executed|by|authorized)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*[,;]\s*(?:signed|executed|authorized))',
                r'([A-Z]\.\s*[A-Z][a-z]+)',  # Initials: C. Dominguez
            ]
            for pattern in signature_contexts:
                for match in re.finditer(pattern, full_text, re.IGNORECASE):
                    name = match.group(1).strip()
                    if len(name) > 2 and name not in persons_set:
                        persons_set.add(name)
            
            # Also extract single capitalized words that appear in person contexts
            # This catches names like "Mitul" that appear in filenames or document text
            # Pattern: Look for capitalized words that appear near signature/author contexts
            single_name_pattern = re.compile(
                r'\b([A-Z][a-z]+)\b(?=.*(?:signed|executed|authorized|by|agreement|contract))',
                re.IGNORECASE
            )
            for match in single_name_pattern.finditer(full_text):
                name = match.group(1).strip()
                # Exclude common words
                if (len(name) > 2 and 
                    name.lower() not in ["the", "and", "for", "with", "from", "this", "that", "agreement", "contract"] and
                    name not in persons_set):
                    persons_set.add(name)
            
            # Extract from filename patterns (e.g., "MitulStorageChainLLC")
            # This ensures "Mitul" is extracted from filenames like "MitulStorageChainLLC_mutual_NDA"
            if file_id:
                # Pattern: Capitalized word at start of filename (likely a person name)
                filename_name_pattern = re.compile(r'^([A-Z][a-z]+)')
                match = filename_name_pattern.match(file_id)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2 and name.lower() not in ["the", "and", "for", "with"]:
                        persons_set.add(name)
        
        # Index all persons (from metadata + extracted from text)
        for name in persons_set:
            key_full = name.lower()
            self.by_person[key_full].append(file_id)
            # Token-level index (e.g. "Lisa", "Riordan")
            for token in re.split(r"[^a-zA-Z0-9]+", name):
                tok = token.strip()
                if len(tok) < 2:
                    continue
                self.by_person[tok.lower()].append(file_id)

        # ----- ORGANIZATIONS -----
        # LEXICAL-FIRST: Extract from full_text + metadata
        # Normalize organization names for consistent matching
        try:
            from .query_enhancement import normalize_org_name
        except ImportError:
            # Fallback if import fails
            def normalize_org_name(name: str) -> str:
                if not name:
                    return ""
                import re
                n = name.lower()
                n = re.sub(r"[^a-z0-9]", "", n)
                return n
        
        orgs = meta.get("organizations", []) or []
        orgs_set = set(o.strip() for o in orgs if isinstance(o, str) and o.strip())
        
        # EXTRACT FROM FULL_TEXT: Organization patterns
        # Pattern: "Storage Chain LLC", "CURATION MEDIA INC", "Tech Holding Corp"
        if full_text:
            # Match org suffixes (Corp, LLC, Inc, Ltd, etc.)
            org_suffix_pattern = re.compile(
                r'\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\s+(?:Corporation|Corp|LLC|Inc|Ltd|GmbH|PLC|Co\.?))\b'
            )
            for match in org_suffix_pattern.finditer(full_text):
                org_name = match.group(1).strip()
                if len(org_name) > 2 and org_name not in orgs_set:
                    orgs_set.add(org_name)
            
            # Match ALL-CAPS acronyms (3+ letters)
            acronym_pattern = re.compile(r'\b([A-Z]{3,})\b')
            for match in acronym_pattern.finditer(full_text):
                acronym = match.group(1).strip()
                if len(acronym) >= 3 and acronym not in orgs_set:
                    orgs_set.add(acronym)
        
        # Index all organizations (from metadata + extracted from text)
        for raw in orgs_set:
            # Store both original and normalized for backward compatibility
            key_original = raw.lower()
            key_normalized = normalize_org_name(raw)
            self.by_org[key_original].append(file_id)
            if key_normalized and key_normalized != key_original:
                self.by_org[key_normalized].append(file_id)
            # Token-level index for partial queries (e.g. "Curation", "Media")
            for token in re.split(r"[^a-zA-Z0-9]+", raw):
                tok = token.strip()
                if len(tok) < 2:
                    continue
                self.by_org[tok.lower()].append(file_id)

        # ----- CLAUSES -----
        # LEXICAL-FIRST: Extract from full_text + metadata
        clauses = meta.get("clauses", []) or []
        clauses_set = set(c.strip().lower() for c in clauses if isinstance(c, str) and c.strip())
        
        # EXTRACT FROM FULL_TEXT: Legal clause keywords
        if full_text:
            clause_keywords = [
                "confidentiality", "non-disclosure", "non disclosure", "nda", "mnda",
                "arbitration", "arbitral", "dispute resolution",
                "indemnification", "indemnify", "indemnity", "hold harmless",
                "governing law", "jurisdiction", "choice of law",
                "termination", "expiration", "renewal",
                "warranty", "warranties", "representation",
                "limitation of liability", "liability cap",
            ]
            text_lower = full_text.lower()
            for keyword in clause_keywords:
                if keyword in text_lower and keyword not in clauses_set:
                    clauses_set.add(keyword)
        
        # Index all clauses (from metadata + extracted from text)
        for c in clauses_set:
            self.by_clause[c].append(file_id)

        # ----- SIGNERS -----
        # LEXICAL-FIRST: Extract signers from full_text + metadata
        signers = meta.get("signers", []) or []
        signers_set = set(s.strip() for s in signers if isinstance(s, str) and s.strip())
        
        # EXTRACT FROM FULL_TEXT: Signer patterns
        if full_text:
            # Pattern: "Signed by John Smith", "Executed by Lisa Riordan"
            signer_patterns = [
                r'(?:signed|executed|authorized)\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*[,;]\s*(?:signed|executed|authorized))',
                r'([A-Z]\.\s*[A-Z][a-z]+)',  # Initials: C. Dominguez
            ]
            for pattern in signer_patterns:
                for match in re.finditer(pattern, full_text, re.IGNORECASE):
                    signer_name = match.group(1).strip()
                    if len(signer_name) > 2 and signer_name not in signers_set:
                        signers_set.add(signer_name)
        
        # Index all signers (from metadata + extracted from text)
        for signer in signers_set:
            key_full = signer.lower()
            self.by_signer[key_full].append(file_id)
            # Token-level index (e.g. "Lisa", "Riordan")
            for token in re.split(r"[^a-zA-Z0-9]+", signer):
                tok = token.strip()
                if len(tok) < 2:
                    continue
                self.by_signer[tok.lower()].append(file_id)

        # ----- DOCUMENT TYPE -----
        # Index document_type for fast lookup (e.g., "nda", "agreement", "contract")
        doc_type = meta.get("document_type", "").lower().strip()
        if doc_type:
            self.by_document_type[doc_type].append(file_id)
        # Also check filename for NDA indicators
        if file_id:
            file_lower = file_id.lower()
            if "nda" in file_lower or "mnda" in file_lower or "non-disclosure" in file_lower or "nondisclosure" in file_lower:
                if "nda" not in self.by_document_type:
                    self.by_document_type["nda"].append(file_id)
                elif file_id not in self.by_document_type["nda"]:
                    self.by_document_type["nda"].append(file_id)
        
        # ----- SKILLS / KEYWORDS -----
        for skill in meta.get("skills", []):
            key = skill.lower().strip()
            self.by_skill[key].append(file_id)

        # ----- NUMERIC VALUES / AMOUNTS -----
        # Extract money amounts from metadata AND full_text
        amounts_set = set()
        
        # From metadata
        for amt in meta.get("amounts", []):
            try:
                value = float(amt)
                amounts_set.add(value)
            except Exception:
                pass
        
        # Extract money amounts from full_text
        # Patterns: "$55", "55 dollars", "55 per hour", "55/hour", "$55.00", "55,000", etc.
        if full_text:
            money_patterns = [
                r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|usd|\$)?',
                r'(\d+)\s*(?:dollars?|per\s+hour|per\s+week|per\s+month|per\s+year|/hour|/week)',
                r'(\d{1,3}(?:,\d{3})+)',  # Numbers with commas (e.g., 120,000)
            ]
            for pattern in money_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    # Clean up the match
                    clean_match = re.sub(r'[^\d]', '', str(match))
                    if clean_match and len(clean_match) >= 1:
                        try:
                            amount = float(clean_match)
                            # Only index reasonable amounts (avoid indexing years as amounts)
                            if 1 <= amount <= 1000000000:  # $1 to $1 billion
                                amounts_set.add(amount)
                        except ValueError:
                            pass
        
        # Index all amounts
        for amount in amounts_set:
            try:
                bucket = int(amount // 100)  # bucket by $100 for fast filtering
                self.by_amount[bucket].append(file_id)
                # Also index exact amount (for precise matching)
                amount_key = f"exact_{int(amount)}"
                if not hasattr(self, 'by_amount_exact'):
                    self.by_amount_exact = defaultdict(list)
                self.by_amount_exact[amount_key].append(file_id)
            except Exception:
                pass

        # ----- DOCUMENT CATEGORY -----
        if meta.get("category"):
            cat = str(meta["category"]).lower()
            self.by_category[cat].append(file_id)

    # ------------------------------------------------------------------
    # TEMPORAL HELPER METHODS
    # ------------------------------------------------------------------
    
    def safe_extract_month_year(self, text: str):
        """
        Extracts a single month-year from text using safe legal contexts.
        """
        if not text:
            return None
        
        MONTHS = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        
        pattern = re.compile(
            r"(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"[ ,\-]+(\d{4})",
            re.IGNORECASE,
        )
        
        m = pattern.search(text)
        if not m:
            return None
        
        month = m.group(1).lower()
        year = int(m.group(2))
        return (month, year)

    def find_by_full_date(self, date_str: str):
        """Find documents by full date string (YYYY-MM-DD)."""
        return list(set(self.by_full_date.get(date_str[:10], [])))

    # ------------------------------------------------------------------

    def find_by_person(self, name: str):
        """Find documents by person name (supports variations)."""
        name_lower = name.lower().strip()
        file_ids = set()
        
        # Direct lookup
        file_ids.update(self.by_person.get(name_lower, []))
        
        # Also try individual tokens (e.g., "Mitul" should match "Mitul StorageChain")
        name_tokens = re.split(r"[^a-zA-Z0-9]+", name_lower)
        for token in name_tokens:
            if len(token) >= 2:
                file_ids.update(self.by_person.get(token, []))
        
        return list(file_ids)

    def find_by_org(self, org: str):
        """Find documents by organization name (supports normalized matching and variations)."""
        try:
            from .query_enhancement import normalize_org_name
        except ImportError:
            def normalize_org_name(name: str) -> str:
                if not name:
                    return ""
                import re
                n = name.lower()
                n = re.sub(r"[^a-z0-9]", "", n)
                return n
        
        file_ids = set()
        org_lower = org.lower().strip()
        
        # Try both original and normalized
        org_original = org.lower().strip()
        org_normalized = normalize_org_name(org)
        
        results = set()
        results.update(self.by_org.get(org_original, []))
        if org_normalized != org_original:
            results.update(self.by_org.get(org_normalized, []))
        
        # Also try individual tokens (e.g., "RH" or "Associates" should match "RH Associates")
        org_tokens = re.split(r"[^a-zA-Z0-9]+", org_lower)
        for token in org_tokens:
            if len(token) >= 2:
                token_normalized = normalize_org_name(token)
                results.update(self.by_org.get(token_normalized, []))
                results.update(self.by_org.get(token, []))
        
        return list(results)

    def find_by_clause(self, clause: str):
        return list(set(self.by_clause.get(clause.lower().strip(), [])))
    
    def find_by_document_type(self, doc_type: str):
        """Find documents by document_type (e.g., 'nda', 'agreement', 'contract')."""
        doc_type_lower = doc_type.lower().strip()
        file_ids = set()
        
        # Direct lookup
        file_ids.update(self.by_document_type.get(doc_type_lower, []))
        
        # Also check for variations (e.g., "nda" should match "NDA", "mnda", "non-disclosure")
        if doc_type_lower == "nda":
            # Check for all NDA variations
            for variant in ["nda", "mnda", "non-disclosure", "nondisclosure", "mutual nda", "mutual non-disclosure"]:
                file_ids.update(self.by_document_type.get(variant, []))
        
        return list(file_ids)
    
    def find_by_signer(self, signer: str):
        """Find documents by signer name (supports token-level matching)."""
        signer_lower = signer.lower().strip()
        results = set()
        results.update(self.by_signer.get(signer_lower, []))
        # Also try token-level matching
        for token in re.split(r"[^a-zA-Z0-9]+", signer_lower):
            tok = token.strip()
            if len(tok) >= 2:
                results.update(self.by_signer.get(tok, []))
        return list(results)

    def find_by_skill(self, skill: str):
        return list(set(self.by_skill.get(skill.lower().strip(), [])))

    def find_by_amount_min(self, min_amt: float):
        """Find documents with amounts >= min_amt."""
        bucket = int(min_amt // 100)
        return list(set(self.by_amount.get(bucket, [])))
    
    def find_by_amount_exact(self, amount: float):
        """Find documents with exact amount (within $100 tolerance)."""
        amount_int = int(amount)
        file_ids = set()
        
        # Check exact match
        if hasattr(self, 'by_amount_exact'):
            exact_key = f"exact_{amount_int}"
            file_ids.update(self.by_amount_exact.get(exact_key, []))
        
        # Check bucket match (within $100)
        bucket = int(amount // 100)
        file_ids.update(self.by_amount.get(bucket, []))
        
        # Also check adjacent buckets for tolerance
        if bucket > 0:
            file_ids.update(self.by_amount.get(bucket - 1, []))
        file_ids.update(self.by_amount.get(bucket + 1, []))
        
        return list(file_ids)

    def find_by_category(self, category: str):
        return list(set(self.by_category.get(category.lower(), [])))

    # EXISTING LOOKUP METHODS REMAIN PERFECT
    def find_by_year(self, y: int):
        return list(set(self.by_year.get(y, [])))

    def find_by_year_range(self, s: int, e: int):
        out = []
        for yr in range(s, e + 1):
            out.extend(self.by_year.get(yr, []))
        return list(set(out))

    def find_by_location(self, loc: str) -> List[str]:
        """Find documents by location using semantic_utils aliases (city ≠ state).

        Previously, any query containing \"texas\" used an Austin-only variant list
        and skipped Dallas — and merged Austin + Texas keys for unrelated metros.
        """
        if not loc:
            return []

        loc_lower = loc.lower().strip()
        file_ids: set = set()

        try:
            from .semantic_utils import normalize_location, get_location_aliases

            canonical = normalize_location(loc)
            terms: set[str] = {loc_lower}
            if canonical:
                terms.add(str(canonical).lower())
                for a in get_location_aliases(canonical) or []:
                    if a:
                        terms.add(str(a).lower())
            for t in terms:
                if t in self.by_location:
                    file_ids.update(self.by_location[t])
        except ImportError:
            if loc_lower in self.by_location:
                file_ids.update(self.by_location[loc_lower])

        return list(file_ids)

    def find_by_month_year(self, m: str, y: int):
        """Find documents by month-year. Also includes documents with full dates in that month."""
        import logging
        logger = logging.getLogger(__name__)
        
        hits = list(set(self.by_month_year.get((m.lower(), y), [])))
        logger.debug(f"[FIND_MONTH_YEAR] Direct month_year lookup for ({m.lower()}, {y}): {len(hits)} hits")
        
        # Fallback: Also check full_date index for dates in that month-year
        month_num = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }.get(m.lower(), None)
        
        if month_num:
            # Check all dates in that month-year
            full_date_hits_count = 0
            for day in range(1, 32):
                date_key = f"{y}-{month_num:02d}-{day:02d}"
                day_hits = self.by_full_date.get(date_key, [])
                if day_hits:
                    full_date_hits_count += len(day_hits)
                    hits.extend(day_hits)
            logger.debug(f"[FIND_MONTH_YEAR] Full_date fallback for {m.lower()} {y}: found {full_date_hits_count} additional hits across {month_num:02d} days")
        
        final_hits = list(set(hits))
        logger.debug(f"[FIND_MONTH_YEAR] Total hits for {m.lower()} {y}: {len(final_hits)}")
        return final_hits

    def get_metadata(self, file_id: str) -> Optional[Dict]:
        doc = self.docs.get(file_id)
        if doc is None:
            return None
        # Inject a clean `filename` if it's missing.
        # The full_text field contains a [FILE: name?query_params] prefix;
        # extract the basename and URL-decode it so the API client shows a
        # human-readable filename instead of the raw file_id.
        if "filename" not in doc:
            import re, urllib.parse as _up
            full_text = doc.get("full_text", "") or ""
            clean = ""
            for raw in re.findall(r'\[FILE:\s*([^\]]+)\]', full_text):
                raw = raw.strip()
                if raw.startswith(file_id):
                    continue  # skip the [FILE: file_id] marker, use only the name
                base = raw.split("?")[0].split("/")[-1]
                decoded = _up.unquote(base)
                if decoded and not decoded.startswith(("69c", "http")):
                    clean = decoded
                    break
            doc = {**doc, "filename": clean}
        return doc

    def find_expired(self) -> List[str]:
        """Find all expired documents (backward compatibility)."""
        return list(set(self.by_expiry.get("expired", [])))

    def find_active(self) -> List[str]:
        """Find all active (non-expired) documents (backward compatibility)."""
        return list(set(self.by_expiry.get("active", [])))

    def clear(self):
        """Clear all indices (useful for testing or reset)."""
        with self._lock:
            self.by_year.clear()
            self.by_year_range.clear()
            self.by_location.clear()
            self.by_expiry.clear()
            self.by_month_year.clear()
            self.by_person.clear()
            self.by_org.clear()
            self.by_clause.clear()
            self.by_skill.clear()
            self.by_amount.clear()
            self.by_category.clear()
            self.by_signer.clear()
            self.by_full_date.clear()
            self.by_month_only.clear()
            self.by_expired.clear()
            self.by_temporal_hash.clear()
            self.docs.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Disk persistence — pure I/O, does not change index content/logic
    # ------------------------------------------------------------------

    # ── Redis key helpers ─────────────────────────────────────────────────────
    _REDIS_BLOB_KEY   = "metadata_index_blob:{user_id}"    # compressed pickle bytes
    _REDIS_VER_KEY    = "metadata_index_version:{user_id}" # integer version counter
    _REDIS_TTL        = 60 * 60 * 24 * 30                  # 30 days

    @staticmethod
    def _get_redis_client():
        """Return a Redis client using REDIS_URL env var, or None if unavailable."""
        try:
            import redis as _redis
            url = os.getenv("REDIS_URL", "")
            if not url:
                return None
            client = _redis.from_url(url, socket_connect_timeout=3, socket_timeout=5)
            client.ping()
            return client
        except Exception:
            return None

    @_thread_safe
    def save_to_disk(self, user_id: str, cache_dir: str = None) -> bool:
        """Persist the built metadata index to disk and push to Redis so the
        dedicated search server picks it up automatically (no file transfer needed)."""
        import os, pickle, time, zlib
        if not cache_dir:
            cache_dir = os.getenv("METADATA_CACHE_DIR", "data/metadata_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"metadata_index_{user_id}.pkl")
        t0 = time.time()

        # Never persist an empty index — it would corrupt both disk and Redis
        if not self.docs:
            logger.warning(f"[CACHE] Skipping save for user={user_id}: metadata index is empty (no docs)")
            return False

        try:
            state = {
                "version": "5.0.8",
                "timestamp": t0,
                "docs": self.docs,
                "by_year": dict(self.by_year),
                "by_year_range": dict(self.by_year_range),
                "by_month_year": dict(self.by_month_year),
                "by_full_date": dict(self.by_full_date),
                "by_month_only": dict(self.by_month_only),
                "by_location": dict(self.by_location),
                "by_person": dict(self.by_person),
                "by_org": dict(self.by_org),
                "by_clause": dict(self.by_clause),
                "by_skill": dict(self.by_skill),
                "by_amount": dict(self.by_amount),
                "by_category": dict(self.by_category),
                "by_signer": dict(self.by_signer),
                "by_document_type": dict(self.by_document_type),
                "by_expiry": dict(self.by_expiry),
                "by_expired": self.by_expired,
                "by_temporal_hash": self.by_temporal_hash,
            }
            raw = pickle.dumps(state)
            with open(cache_file, "wb") as f:
                f.write(raw)
            elapsed_ms = (time.time() - t0) * 1000
            logger.info(
                f"[CACHE] Saved metadata index for user={user_id} → {cache_file} "
                f"({len(self.docs)} docs, {elapsed_ms:.0f}ms)"
            )

            # ── Push compressed bytes to Redis so the search server syncs ──────
            # Only the processing/publisher node writes to Redis.  The dedicated
            # search server is a subscriber (read-only from Redis).  Set
            # METADATA_INDEX_PUBLISHER=true in docker.env on the processing server.
            _is_publisher = os.getenv("METADATA_INDEX_PUBLISHER", "false").lower() in ("true", "1", "yes")
            try:
                rc = MetadataIndex._get_redis_client() if _is_publisher else None
                if rc:
                    import time as _time
                    compressed = zlib.compress(raw, level=1)
                    blob_key = MetadataIndex._REDIS_BLOB_KEY.format(user_id=user_id)
                    # Use a sorted set scored by timestamp so the single global
                    # watcher can detect exactly which users changed (scales to
                    # any number of users — no per-user version counters).
                    changed_set = "metadata_index_changed_users"
                    pipe = rc.pipeline()
                    pipe.set(blob_key, compressed, ex=MetadataIndex._REDIS_TTL)
                    pipe.zadd(changed_set, {user_id: _time.time()})
                    pipe.execute()
                    logger.info(
                        f"[CACHE] Pushed metadata index to Redis for user={user_id} "
                        f"({len(compressed)//1024}KB compressed)"
                    )
            except Exception as re:
                logger.warning(f"[CACHE] Redis push skipped for user={user_id}: {re}")
            # ────────────────────────────────────────────────────────────────────

            return True
        except Exception as e:
            logger.warning(f"[CACHE] Failed to save metadata index for user={user_id}: {e}")
            return False

    @classmethod
    def _load_state_from_redis(cls, user_id: str):
        """Try to load the raw state dict from Redis. Returns (state_dict, score)
        or (None, -1) on any miss/error."""
        try:
            import zlib, pickle as _pickle
            rc = cls._get_redis_client()
            if not rc:
                return None, -1
            blob_key = cls._REDIS_BLOB_KEY.format(user_id=user_id)
            compressed = rc.get(blob_key)
            if not compressed:
                return None, -1
            state = _pickle.loads(zlib.decompress(compressed))
            # Score = timestamp stored in the changed-users sorted set
            score = rc.zscore("metadata_index_changed_users", user_id) or 0
            return state, score
        except Exception as e:
            logger.debug(f"[CACHE] Redis load failed for user={user_id}: {e}")
            return None, -1

    @classmethod
    def _build_instance_from_state(cls, state: dict):
        inst = cls()
        inst.docs = state["docs"]
        inst.by_year = defaultdict(list, state["by_year"])
        inst.by_year_range = defaultdict(list, state.get("by_year_range", {}))
        inst.by_month_year = defaultdict(list, state["by_month_year"])
        inst.by_full_date = defaultdict(list, state["by_full_date"])
        inst.by_month_only = defaultdict(list, state["by_month_only"])
        inst.by_location = defaultdict(list, state["by_location"])
        inst.by_person = defaultdict(list, state["by_person"])
        inst.by_org = defaultdict(list, state["by_org"])
        inst.by_clause = defaultdict(list, state["by_clause"])
        inst.by_skill = defaultdict(list, state["by_skill"])
        inst.by_amount = defaultdict(list, state["by_amount"])
        inst.by_category = defaultdict(list, state["by_category"])
        inst.by_signer = defaultdict(list, state["by_signer"])
        inst.by_document_type = defaultdict(list, state["by_document_type"])
        inst.by_expiry = defaultdict(list, state.get("by_expiry", {}))
        inst.by_expired = state.get("by_expired", {"yes": [], "no": []})
        inst.by_temporal_hash = state.get("by_temporal_hash", {})
        return inst

    @classmethod
    def load_from_disk(cls, user_id: str, cache_dir: str = None):
        """Restore a previously saved metadata index.
        Priority: Redis (always freshest) → disk cache → None.
        When Redis has a newer version than disk, it is used and the local
        disk cache is refreshed so future restarts remain sub-second."""
        import os, pickle, time
        if not cache_dir:
            cache_dir = os.getenv("METADATA_CACHE_DIR", "data/metadata_cache")
        cache_file = os.path.join(cache_dir, f"metadata_index_{user_id}.pkl")
        t0 = time.time()

        # ── 1. Try Redis first (shared by processing + search server) ──────────
        try:
                redis_state, redis_score = cls._load_state_from_redis(user_id)
                if redis_state and redis_state.get("version") == "5.0.8":
                    # Check if Redis is newer than the disk file we already have
                    disk_ts = 0.0
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, "rb") as _f:
                                _disk = pickle.load(_f)
                            disk_ts = _disk.get("timestamp", 0.0)
                        except Exception:
                            disk_ts = 0.0

                    redis_ts = redis_state.get("timestamp", 0.0)
                    if redis_ts > disk_ts or not os.path.exists(cache_file):
                        inst = cls._build_instance_from_state(redis_state)
                        logger.info(
                            f"[CACHE] Loaded metadata index from Redis for user={user_id} "
                            f"({len(inst.docs)} docs, {(time.time()-t0)*1000:.0f}ms)"
                        )
                    # Write back to local disk so next restart is instant
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, "wb") as _f:
                            pickle.dump(redis_state, _f)
                        logger.debug(f"[CACHE] Refreshed disk cache from Redis for user={user_id}")
                    except Exception:
                        pass
                    return inst
        except Exception as re:
            logger.debug(f"[CACHE] Redis check skipped for user={user_id}: {re}")
        # ────────────────────────────────────────────────────────────────────────

        # ── 2. Fall back to local disk cache ────────────────────────────────────
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, "rb") as f:
                state = pickle.load(f)
            if state.get("version") != "5.0.8":
                logger.info(f"[CACHE] Stale version for user={user_id}, will rebuild")
                return None
            inst = cls._build_instance_from_state(state)
            logger.info(
                f"[CACHE] Loaded metadata index for user={user_id} from {cache_file} "
                f"({len(inst.docs)} docs, {(time.time()-t0)*1000:.0f}ms)"
            )
            return inst
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load metadata index for user={user_id}: {e}")
            return None


# ============================================================================
# Image Processing Utilities (merged from clip_utils.py)
# ============================================================================

# Lazy initialization to avoid import errors
_image_captioner = None
_semantic_pipeline = None

# Global instance for lazy initialization
_captioner_instance = None

def generate_caption(image_path: str) -> str:
    """Generate caption for an image using BLIP model."""
    skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
    if skip_captioning:
        return "image content"  
    
    global _captioner_instance
    if _captioner_instance is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_name = "Salesforce/blip-image-captioning-base"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
            _captioner_instance = {
                'processor': processor,
                'model': model,
                'device': device
            }
        except Exception as e:
            logger.warning(f"BLIP model not available: {e}")
            return "image content"
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _captioner_instance['processor'](image, return_tensors="pt").to(_captioner_instance['device'])
        output = _captioner_instance['model'].generate(**inputs, max_new_tokens=30)
        caption = _captioner_instance['processor'].decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.error(f"[CAPTION ERROR] {e}")
        return "image content"

class ImageCaptioner:
    """Image captioner using BLIP model."""
    
    def __init__(self):
        skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
        if skip_captioning:
            self.processor = None
            self.model = None
            self.device = None
            return
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_name = "Salesforce/blip-image-captioning-base"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize ImageCaptioner: {e}")
            self.processor = None
            self.model = None
            self.device = None

    def caption(self, image_path: str) -> str:
        if self.processor is None or self.model is None:
            return "image content"
        return generate_caption(image_path)

def _get_captioner():
    """Lazy initialization of image captioner."""
    # Check skip flag FIRST to prevent BLIP initialization
    skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
    if skip_captioning:
        logger.debug("Image captioning skipped (SKIP_IMAGE_CAPTIONING_IN_PROCESSOR=true)")
        return None
    
    global _image_captioner
    if _image_captioner is None:
        try:
            _image_captioner = ImageCaptioner()
            logger.info("Image captioner initialized (BLIP)")
        except Exception as e:
            logger.warning(f"Image captioner not available: {e}")
            _image_captioner = False
    return _image_captioner if _image_captioner is not False else None

def _get_semantic_pipeline():
    """Lazy initialization of semantic pipeline."""
    global _semantic_pipeline
    if _semantic_pipeline is None:
        try:
            from .semantic_pipeline import SemanticPipeline
            _semantic_pipeline = SemanticPipeline()
            logger.info("Semantic pipeline initialized")
        except Exception as e:
            logger.warning(f"Semantic pipeline not available: {e}")
            _semantic_pipeline = False
    return _semantic_pipeline if _semantic_pipeline is not False else None

def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image and return metadata including caption, objects, scene, etc.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with analysis results including:
        - description: Image caption
        - objects: List of detected objects (placeholder for now)
        - scene: Scene type (placeholder for now)
        - dominant_colors: List of dominant colors (placeholder for now)
    """
    try:
        captioner = _get_captioner()
        if not captioner:
            logger.warning("Image captioner not available, returning default analysis")
            return {
                "description": "image content",
                "objects": [],
                "scene": "unknown",
                "dominant_colors": []
            }
        
        # Generate caption
        caption = captioner.caption(image_path)
        
        # For now, return basic analysis
        return {
            "description": caption,
            "objects": [],  # Placeholder - can be enhanced with object detection
            "scene": "unknown",  # Placeholder - can be enhanced with scene classification
            "dominant_colors": []  # Placeholder - can be enhanced with color analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {
            "description": "image content",
            "objects": [],
            "scene": "unknown",
            "dominant_colors": []
        }

def get_image_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Generate CLIP-like embedding for an image.
    
    This function generates a text embedding from the image caption,
    which can be used for text-to-image search.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        NumPy array of shape (512,) with the image embedding, or None if failed
    """
    try:
        # First, get the caption
        captioner = _get_captioner()
        if not captioner:
            logger.warning("Image captioner not available, cannot generate embedding")
            return None
        
        caption = captioner.caption(image_path)
        if not caption or not caption.strip():
            logger.warning("Empty caption generated, cannot create embedding")
            return None
        
        # Get semantic pipeline for embedding
        pipeline = _get_semantic_pipeline()
        if not pipeline:
            logger.warning("Semantic pipeline not available, cannot generate embedding")
            return None
        
        # Generate embedding from caption (for text-to-image search)
        # Use for_images=True to get image-compatible embedding
        emb = pipeline.embed_text(caption, for_images=True)
        
        # Ensure it's the right shape (512 for image vectors)
        if emb is None:
            return None
        
        emb = np.asarray(emb, dtype=np.float32)
        
        # Pad or trim to 512 dimensions (image vector dimension)
        target_dim = 512
        if emb.shape[0] < target_dim:
            pad = np.zeros(target_dim - emb.shape[0], dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)
        elif emb.shape[0] > target_dim:
            emb = emb[:target_dim]
        
        return emb
        
    except Exception as e:
        logger.error(f"Error generating image embedding: {e}")
        return None


# ============================================================================
# Entity Extractor (merged from entity_extractor.py)
# ============================================================================

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class EntityExtractor:
    """
    Generic entity extractor using NER + pattern matching + OCR correction.
    No hardcoded company names, document types, or aliases.
    """
    
    def __init__(self, ner_model_name: Optional[str] = None):
        self.ner = None
        if SPACY_AVAILABLE:
            try:
                # Force small model to avoid Celery Worker OOMs
                model_name = "en_core_web_sm"
                self.ner = spacy.load(model_name)
            except OSError:
                pass
    
    def correct_ocr_noise(self, text: str) -> str:
        """
        Correct common OCR errors:
        - 2O24 -> 2024 (letter O instead of zero)
        - C. Dominguez -> Chris Dominguez (if context suggests)
        - Common character confusions
        """
        # Fix year OCR errors: letter O instead of zero (e.g., 2O24 -> 2024)
        try:
            text = re.sub(
                r"\b([0-9])O([0-9]{2})\b",
                lambda m: f"{m.group(1)}0{m.group(2)}",
                text,
                flags=re.IGNORECASE,
            )
        except re.error:
            # Fail-open: if regex engine complains, just return original text
            return text

        # Additional OCR character confusions can be added here if needed,
        # but keep this lightweight and robust.
        return text
    
    def extract_organizations(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Extract organization names.

        v5.3: Delegate to the global safe query/org extractor to ensure
        consistent organization handling across the system.
        """
        from .query_enhancement import extract_orgs_safe
        
        # Combine metadata hints and text into a single string for pattern-based extraction
        meta_parts: List[str] = []
        if metadata:
            for key in ("company", "vendor", "client", "party", "organization"):
                val = metadata.get(key)
                if isinstance(val, str):
                    meta_parts.append(val)

        combined = " ".join(meta_parts + [text or ""])
        return sorted(list(set(extract_orgs_safe(combined))))
    
    def extract_persons(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Extract person names using NER + metadata + signature patterns.
        Handles initials (C. Dominguez) and full names.
        """
        persons = set()
        
        # 1. From metadata
        if metadata:
            for key in ("signatories", "author", "executed_by", "signed_by"):
                val = metadata.get(key)
                if val:
                    if isinstance(val, list):
                        persons.update(val)
                    elif isinstance(val, str):
                        persons.add(val.strip())
        
        # 2. From NER
        if self.ner:
            try:
                doc = self.ner(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        person_text = ent.text.strip()
                        if len(person_text) > 1:
                            persons.add(person_text)
            except Exception:
                pass
        
        # 3. Signature patterns
        signature_patterns = [
            r'(?:signed|executed|by)\s+([A-Z][a-zA-Z.-]+(?:\s+[A-Z][a-zA-Z.-]+)+)',
            r'([A-Z]\.\s+[A-Z][a-zA-Z]+)',  # Initials: C. Dominguez
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\s+(?:signed|executed)',
        ]
        for pattern in signature_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match and len(match.strip()) > 1:
                    # Skip if looks like email
                    if "@" not in match:
                        persons.add(match.strip())
        
        return sorted(list(persons))
    
    def extract_locations(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Extract location names using NER + metadata.
        Returns list of unique location names.
        """
        locations = set()
        
        # 1. From metadata
        if metadata:
            for key in ("jurisdiction", "location", "venue", "governing_law", "state", "country"):
                val = metadata.get(key)
                if val and isinstance(val, str):
                    locations.add(val.strip())
        
        # 2. From NER
        if self.ner:
            try:
                doc = self.ner(text)
                for ent in doc.ents:
                    if ent.label_ in {"GPE", "LOC"}:
                        loc_text = ent.text.strip()
                        if len(loc_text) > 1:
                            locations.add(loc_text)
            except Exception:
                pass
        
        return sorted(list(locations))
    
    def generate_aliases(self, entity: str) -> List[str]:
        """
        Generate aliases for an entity name dynamically.
        Handles:
        - CamelCase splitting (StorageChain -> storage chain, storagechain)
        - Abbreviations (United Kingdom -> UK, U.K.)
        - Common variations
        """
        aliases = [entity.lower()]
        
        # Split CamelCase
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', entity)
        if camel_split != entity:
            aliases.append(camel_split.lower())
            aliases.append(camel_split.replace(" ", "").lower())
        
        # Extract first letters for abbreviations
        words = entity.split()
        if len(words) > 1:
            abbrev = "".join(w[0].upper() for w in words if w)
            aliases.append(abbrev.lower())
            aliases.append(abbrev)
        
        # Common variations
        aliases.append(entity.replace("-", " ").lower())
        aliases.append(entity.replace(" ", "-").lower())
        aliases.append(entity.replace(" ", "").lower())
        
        return sorted(list(set(aliases)))
    
    def extract_all_entities(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, List[str]]:
        """
        Extract all entities (organizations, persons, locations) from text and metadata.
        Returns dict with keys: "organizations", "persons", "locations"
        """
        # Correct OCR noise first
        text_corrected = self.correct_ocr_noise(text)
        
        return {
            "organizations": self.extract_organizations(text_corrected, metadata),
            "persons": self.extract_persons(text_corrected, metadata),
            "locations": self.extract_locations(text_corrected, metadata),
        }


# ============================================================================
# Document Normalization (merged from normalize.py)
# ============================================================================

@dataclass
class DocumentElement:
    """Represents a single element extracted from a PDF."""
    text: str
    element_type: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExtractedDocument:
    """Represents a complete extracted document with metadata."""
    file_path: str
    file_hash: str
    elements: List[DocumentElement]
    total_pages: int
    extraction_method: str
    processing_time: float
    
    @property
    def full_text(self) -> str:
        """Get the full text content of the document."""
        return "\n".join([elem.text for elem in self.elements if elem.text.strip()])
    
    @property
    def text_by_page(self) -> Dict[int, str]:
        """Get text content organized by page number."""
        page_text = {}
        for elem in self.elements:
            if elem.page_number is not None:
                if elem.page_number not in page_text:
                    page_text[elem.page_number] = ""
                page_text[elem.page_number] += elem.text + "\n"
        return page_text


# ============================================================================
# Text Chunking (merged from chunking.py)
# ============================================================================

import tiktoken
import time
import uuid
import hashlib
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    """Represents a single text chunk with metadata."""
    text: str
    chunk_id: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = None
    object_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChunkingResult:
    """Represents the result of text chunking."""
    chunks: List[TextChunk]
    total_chunks: int
    total_tokens: int
    average_chunk_size: float
    processing_time: float
    chunking_strategy: str
    
    @property
    def chunk_size_distribution(self) -> Dict[str, int]:
        """Get distribution of chunk sizes."""
        distribution = {}
        for chunk in self.chunks:
            size_range = f"{(chunk.token_count // 100) * 100}-{(chunk.token_count // 100) * 100 + 99}"
            distribution[size_range] = distribution.get(size_range, 0) + 1
        return distribution


class TextChunker:
    """Handles text chunking with token awareness and overlap."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        model_name: str = "gpt-3.5-turbo",
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Model name for tiktoken tokenizer
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize tiktoken tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            logger.info(f"Initialized tokenizer for model: {model_name}")
        except KeyError:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
        
        # Initialize text splitter
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                " ",     # Spaces
                ".",     # Sentences
                "!",     # Exclamations
                "?",     # Questions
                ";",     # Semicolons
                ":",     # Colons
                ",",     # Commas
                ")",     # Closing parentheses
                "]",     # Closing brackets
                "}",     # Closing braces
                "\"",    # Quotes
                "'",     # Apostrophes
                "—",     # Em dashes
                "–",     # En dashes
                "…",     # Ellipses
                " ",     # Spaces
                ""       # Character level
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=separators,
            is_separator_regex=False
        )
        
        logger.info(f"Initialized text chunker: {chunk_size} tokens, {chunk_overlap} overlap")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int, text_preview: str = "") -> str:
        """
        Generate a globally unique chunk ID.
        
        Args:
            document_id: Unique identifier for the document
            chunk_index: Index of the chunk within the document
            text_preview: First few characters of the chunk text for uniqueness
            
        Returns:
            Globally unique chunk ID
        """
        # Create a hash of document ID + chunk index + text preview for uniqueness
        content_hash = hashlib.md5(f"{document_id}_{chunk_index}_{text_preview[:50]}".encode()).hexdigest()[:8]
        return f"{document_id}_{chunk_index:04d}_{content_hash}"
    
    def _get_document_id(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Generate a unique document ID from file path and metadata.
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata
            
        Returns:
            Unique document identifier
        """
        # Use file path as base
        file_path = Path(file_path)
        base_name = file_path.stem
        
        # Add timestamp for uniqueness
        timestamp = int(time.time() * 1000)  # milliseconds
        
        # Add metadata hash if available
        if metadata:
            metadata_str = str(sorted(metadata.items()))
            metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:4]
            return f"{base_name}_{timestamp}_{metadata_hash}"
        
        return f"{base_name}_{timestamp}"
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, document_id: str = None) -> List[TextChunk]:
        """
        Chunk a single text string.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks
            document_id: Unique document identifier for chunk IDs
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"text_{int(time.time() * 1000)}"
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Convert to TextChunk objects
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            token_count = self._count_tokens(chunk_text)
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(document_id, i, chunk_text)
            
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                chunk_index=i,
                token_count=token_count,
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_document_elements(
        self, 
        elements: List[DocumentElement],
        preserve_page_boundaries: bool = True,
        document_id: str = None
    ) -> List[TextChunk]:
        """
        Chunk a list of document elements.
        
        Args:
            elements: List of document elements to chunk
            preserve_page_boundaries: Whether to respect page boundaries
            document_id: Unique document identifier for chunk IDs
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        chunk_index = 0
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = f"doc_{int(time.time() * 1000)}"
        
        for element in elements:
            if not element.text.strip():
                continue
            
            # Prepare metadata
            metadata = {
                "element_type": element.element_type,
                "page_number": element.page_number,
                "original_metadata": element.metadata or {}
            }
            
            # Extract object_id from element metadata if available
            object_id = None
            if element.metadata and 'object_id' in element.metadata:
                object_id = element.metadata['object_id']
            
            if preserve_page_boundaries and element.page_number:
                # Chunk within page boundaries
                element_chunks = self.chunk_text(element.text, metadata, document_id)
                
                # Update chunk indices, page numbers, and object_id
                for chunk in element_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.page_number = element.page_number
                    chunk.object_id = object_id
                    chunk_index += 1
                
                all_chunks.extend(element_chunks)
            else:
                # Chunk without page boundary constraints
                element_chunks = self.chunk_text(element.text, metadata, document_id)
                
                # Update chunk indices and object_id
                for chunk in element_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.object_id = object_id
                    chunk_index += 1
                
                all_chunks.extend(element_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(elements)} elements")
        return all_chunks
    
    def chunk_document(
        self, 
        document: ExtractedDocument,
        preserve_page_boundaries: bool = True
    ) -> ChunkingResult:
        """
        Chunk an entire document.
        
        Args:
            document: ExtractedDocument to chunk
            preserve_page_boundaries: Whether to respect page boundaries
            
        Returns:
            ChunkingResult with chunking statistics
        """
        start_time = time.time()
        
        logger.info(f"Chunking document: {document.file_path}")
        logger.info(f"Total elements: {len(document.elements)}")
        logger.info(f"Preserve page boundaries: {preserve_page_boundaries}")
        
        # Generate unique document ID
        document_id = self._get_document_id(document.file_path, document.metadata)
        
        # Chunk document elements
        chunks = self.chunk_document_elements(
            document.elements, 
            preserve_page_boundaries,
            document_id
        )
        
        # Calculate statistics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        average_chunk_size = total_tokens / len(chunks) if chunks else 0
        processing_time = time.time() - start_time
        
        result = ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            average_chunk_size=average_chunk_size,
            processing_time=processing_time,
            chunking_strategy="recursive_character"
        )
        
        logger.info(f"Chunking completed: {len(chunks)} chunks, {total_tokens} tokens")
        logger.info(f"Average chunk size: {average_chunk_size:.1f} tokens")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Document ID: {document_id}")
        
        return result


# ============================================================================
# Constraint Validator (merged from validators/universal_constraint_validator.py)
# ============================================================================

def normalize_str(s: str) -> str:
    """Normalize string for comparison."""
    return (s or "").lower().strip()


def word_boundary(pattern: str) -> str:
    """Wrap pattern in word boundaries."""
    return rf"\b{re.escape(pattern)}\b"


def any_match(patterns, text):
    """Check if any pattern matches in text."""
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def all_words_in_filename(words, filename):
    """Check if all words appear in filename."""
    return all(w in filename.lower() for w in words if len(w) > 3)


def normalize_org_name_validator(name: str) -> str:
    """
    Normalize organization names for matching.
    Removes spaces, dashes, underscores, and converts to lowercase.
    
    Examples:
    - "Storage Chain" -> "storagechain"
    - "Storage_Chain" -> "storagechain"
    - "STORAGE-CHAIN" -> "storagechain"
    """
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"[^a-z0-9]", "", n)  # remove spaces, dashes, underscores
    return n


def loc_match(query_loc, doc_text, filename, metadata):
    """
    UNIVERSAL location matcher.
    Fuzzy, compositional, no hardcoding of specific cities/states.
    Matches:
      - 'Austin'       → 'Austin, Texas'
      - 'Austin Texas' → 'Austin, Texas'
      - 'Texas'        → 'Austin, TX'
      - any city/state/country without predefined lists.
    """
    blob = f"{(doc_text or '').lower()} {(filename or '').lower()} {str(metadata or {}).lower()}"
    q = (query_loc or "").lower().strip()

    if not q:
        return True

    # exact substring
    if q in blob:
        return True

    # all components present somewhere (order-agnostic)
    parts = [p for p in q.split() if p]
    if parts and all(p in blob for p in parts):
        return True

    return False


def validate_temporal_constraints(query_meta, doc_meta, doc_text, from_lexical_index=False):
    """
    FINAL v5.x patch:
    - Build full temporal profile from text + metadata.
    - If profile has no years but metadata has a parseable date,
      inject that year into the profile.
    - Then delegate to TemporalReasoningEngine.temporal_match.
    - Fail-open on any exception.
    
    LEXICAL-FIRST: If from_lexical_index=True, only reject explicit contradictions.
    """
    logger.debug(f"[TEMPORAL-VALIDATOR] Checking doc against query. query_meta={query_meta}")
    try:
        from .temporal_engine import TemporalReasoningEngine
        temporal_engine = TemporalReasoningEngine()
        
        query_meta = query_meta or {}
        doc_meta = doc_meta or {}
        
        if from_lexical_index:
            # Extract query years from metadata AND query text
            query_years = set()
            if query_meta.get("date"):
                try:
                    query_years.add(int(query_meta["date"]))
                except (TypeError, ValueError):
                    pass
            if query_meta.get("date_range"):
                try:
                    start, end = map(int, query_meta["date_range"])
                    query_years.update(range(start, end + 1))
                except (TypeError, ValueError):
                    pass
            
            # Also extract years from query text (e.g., "document from 2025", "document dated 2024")
            query_text = (
                query_meta.get("original_query") or 
                query_meta.get("normalized_query") or 
                ""
            ).lower()
            query_text_years = set(int(y) for y in re.findall(r'\b(19|20)\d{2}\b', query_text))
            query_years.update(query_text_years)
            
            # Extract document years from metadata, text, and filename
            doc_years = set()
            if doc_meta.get("years"):
                for y in doc_meta.get("years", []):
                    try:
                        doc_years.add(int(y))
                    except (TypeError, ValueError):
                        pass
            
            # Also extract years from document text and filename
            doc_text_full = (doc_text or "").lower() + " " + str(doc_meta).lower()
            doc_text_years = set(int(y) for y in re.findall(r'\b(19|20)\d{2}\b', doc_text_full))
            doc_years.update(doc_text_years)
            
            # If query has explicit years, document MUST have matching years
            if query_years:
                if not doc_years:
                    # Query asks for specific year but document has no years - reject
                    return False
                if not query_years.intersection(doc_years):
                    # Query year doesn't match document year - reject
                    return False
            elif query_text_years:
                # Query text has years but query_meta doesn't - extract from text and validate
                if not doc_years:
                    return False
                if not query_text_years.intersection(doc_years):
                    return False
            
            return True

        # Reconstruct query
        query = (
            query_meta.get("original_query")
            or query_meta.get("normalized_query")
            or ""
        )

        explicit_intent_val = query_meta.get("explicit_temporal_intent")
        explicit = explicit_intent_val is not None

        if not (
            query_meta.get("date")
            or query_meta.get("date_range")
            or query_meta.get("month_year")
            or query_meta.get("full_date_query")
            or query_meta.get("month_only_query")
            or explicit
        ):
            return True

        # Full profile from text + metadata
        profile = temporal_engine.extract_doc_temporal_profile(
            file_id="",
            metadata=doc_meta,
            text=doc_text or "",
        )

        # ---- FINAL FIX ---
        # If document text does NOT contain a year but metadata DOES,
        # force metadata year into profile.
        if not profile.get("years"):
            meta_date = (
                doc_meta.get("date")
                or doc_meta.get("signed_on")
                or doc_meta.get("effective_on")
                or doc_meta.get("created_at")
                or doc_meta.get("issued_on")
                or doc_meta.get("source_date")
                or doc_meta.get("expiry_date")
            )
            if meta_date:
                dt = temporal_engine.parse_date(meta_date)
                if dt:
                    profile["years"] = [dt.year]

        # Normalize years to realistic 4-digit values only
        years = profile.get("years") or []
        clean_years = []
        for y in years:
            try:
                yi = int(y)
            except Exception:
                continue
            if 1900 <= yi <= 2100:
                clean_years.append(yi)
        profile["years"] = clean_years

        # SPECIAL CASE: explicit month-year queries must match month+year exactly.
        q_month_year = query_meta.get("month_year")
        if q_month_year:
            q_month, q_year = q_month_year
            month_years = profile.get("month_years") or []
            # Check month_years first
            if any((m == q_month and int(y) == int(q_year)) for (m, y) in month_years):
                return True
            # Fallback: check if document has matching month_only + year
            doc_month_only = doc_meta.get("month_only")
            doc_years = profile.get("years") or []
            if doc_month_only and doc_month_only.lower() == q_month.lower() and int(q_year) in doc_years:
                return True
            # Also check explicit dates
            explicit_dates = profile.get("explicit_dates", [])
            for dt in explicit_dates:
                if hasattr(dt, 'strftime'):
                    if dt.year == int(q_year) and dt.strftime("%B").lower() == q_month.lower():
                        return True
                return False
        
        # SPECIAL CASE: month-only queries (e.g., "document from April")
        q_month_only = query_meta.get("month_only_query")
        if q_month_only:
            # Match if document has that month in any year
            month_years = profile.get("month_years", [])
            explicit_dates = profile.get("explicit_dates", [])
            doc_month_only = doc_meta.get("month_only")  # Also check document's month_only field
            
            # Check month_years
            for (mm, yy) in month_years:
                if mm.lower() == q_month_only.lower():
                    return True
            # Check explicit dates
            for dt in explicit_dates:
                if hasattr(dt, 'strftime'):
                    doc_month = dt.strftime("%B").lower()
                    if doc_month == q_month_only.lower():
                        return True
            # Check document's month_only field FIRST (most reliable)
            if doc_month_only and doc_month_only.lower() == q_month_only.lower():
                return True
            # Also check if month_only is in any metadata field
            for key, value in doc_meta.items():
                if isinstance(value, str) and q_month_only.lower() in value.lower():
                    return True
            # Final fallback: check if month appears in document text
            text_lower = (doc_text or "").lower()
            month_pattern = re.compile(rf"\b{re.escape(q_month_only.lower())}\b", re.IGNORECASE)
            if month_pattern.search(text_lower):
                return True
            # Very permissive: if document has years and month appears anywhere, accept
            doc_years = profile.get("years", [])
            if doc_years and q_month_only.lower() in text_lower:
                return True
            return False
        
        # Now run main temporal match (hybrid strict/soft) for all other cases.
        explicit_intent_val = query_meta.get("explicit_temporal_intent")
        explicit = explicit_intent_val is not None
        
        res = temporal_engine.temporal_match(
            query=query,
            metadata=doc_meta,
            text=doc_text or "",
            strict=explicit
        )
        logger.info(f"[TEMPORAL-VALIDATOR] Decision: {res} for doc_id={doc_meta.get('file_id')} query={query} doc_years={profile.get('years')}")
        return res

    except Exception as e:
        logger.warning(f"[TEMPORAL-VALIDATOR] Error: {e}")
        # fail-open
        return True


CLAUSE_SYNONYMS = {
    # Arbitration-style clauses
    "arbitration": [
        "arbitration",
        "arbitral",
        "dispute resolution",
        "arbitration clause",
    ],
    # Confidentiality / NDA-style clauses
    "confidentiality": [
        "confidential",
        "confidentiality",
        "non-disclosure",
        "non disclosure",
        "non-disclosure agreement",
        "nda",
        "mutual nda",
        "mnda",
    ],
    # Indemnification
    "indemnification": [
        "indemnify",
        "indemnity",
        "indemnification",
        "hold harmless",
    ],
}


def validate_constraints(query_meta, doc_meta, doc_text=None, file_id=None) -> bool:
    """
    FINAL v5.2 — SOFT MODE CONSTRAINT VALIDATOR

    Soft mode:
    - Reject only when metadata CONTRADICTS the query.
    - If metadata is missing → allow.
    - If text doesn't explicitly confirm → allow.
    """
    try:
        from .temporal_engine import TemporalReasoningEngine
        temporal_engine = TemporalReasoningEngine()
    except Exception:
        temporal_engine = None
    
    text = (doc_text or "").lower()
    file_l = (file_id or "").lower()
    q_text = str(
        (query_meta or {}).get("normalized_query")
        or (query_meta or {}).get("original_query")
        or ""
    ).lower()
    
    if "that mention" in q_text or "mentioning" in q_text:
        return True

    q_persons = [p.lower() for p in (query_meta.get("persons") or [])]
    q_orgs_norm = [o.lower() for o in (query_meta.get("normalized_orgs") or [])]
    q_locs = (query_meta or {}).get("locations") or []  # Define early for "that mention" check
    text_full = (doc_text or "").lower() + " " + str(doc_meta).lower() + " " + str(file_id or "").lower()

    is_mention_query = "that mention" in q_text or "mentioning" in q_text
    if is_mention_query:
        return True

    # STRICT "signed by <person>" OR "documents by <person>" OR "agreements signed between X and Y"
    has_person_query = (
        "signed by" in q_text or 
        "documents by" in q_text or 
        "files by" in q_text or 
        "docs by" in q_text or
        "agreements by" in q_text or
        "contracts by" in q_text or
        "signed between" in q_text or
        "executed by" in q_text
    )
    if has_person_query and q_persons:
        all_persons_found = True
        for p in q_persons:
            if p:
                # Check if person name (or parts) appears in text, metadata, or filename
                person_lower = p.lower()
                person_parts = person_lower.split()
                
                if len(person_parts) >= 2:
                    # Check if full name appears (with word boundaries) in text or filename
                    full_name_pattern = rf'\b{re.escape(person_lower)}\b'
                    full_name_match = re.search(full_name_pattern, text_full)
                    # Or check if first name appears (with word boundaries) in text or filename
                    first_name_pattern = rf'\b{re.escape(person_parts[0])}\b' if person_parts else None
                    first_name_match = re.search(first_name_pattern, text_full) if first_name_pattern else False
                    # Also check if last name appears in filename (for cases like "MitulStorageChainLLC_mutual_NDA")
                    last_name_match = False
                    if len(person_parts) >= 2 and file_id:
                        last_name_pattern = rf'\b{re.escape(person_parts[-1])}\b'
                        last_name_match = re.search(last_name_pattern, file_id.lower())
                    # Check if BOTH first and last name appear (even separately)
                    # This handles cases like "MitulStorageChainLLC" where "Mitul" and "Thobhani" appear separately
                    both_parts_found = False
                    if len(person_parts) >= 2:
                        first_found = person_parts[0] in text_full
                        last_found = person_parts[-1] in text_full
                        both_parts_found = first_found and last_found
                    # Also check metadata for person names (more lenient)
                    meta_persons = doc_meta.get("persons", []) or []
                    meta_persons_str = " ".join(str(p).lower() for p in meta_persons)
                    meta_match = person_lower in meta_persons_str or person_parts[0] in meta_persons_str
                    if not (full_name_match or first_name_match or last_name_match or both_parts_found or meta_match):
                        all_persons_found = False
                        break
                else:
                    # Single name - require exact word boundary match (not partial)
                    if not re.search(rf'\b{re.escape(person_lower)}\b', text_full):
                        all_persons_found = False
                        break
        
        # If any person is not found, reject the document
        if not all_persons_found:
            return False

    if (
        not query_meta.get("legal_clause")
        and not query_meta.get("governing_law")
        and not query_meta.get("explicit_temporal_intent")
        and not any(w in q_text for w in ["agreement", "agreements", "contract", "contracts", "nda", "mnda"])
    ):
        # If a person is mentioned anywhere → allow
        for p in q_persons:
            if p and p in text_full:
                return True

        # If a normalized org is mentioned anywhere → allow
        for org in q_orgs_norm:
            if org and org in text_full:
                return True

    is_mention_query = "that mention" in q_text or "mentioning" in q_text
    mention_location_found = False
    mention_person_found = False
    
    if is_mention_query:
        # Check if any mentioned location is in the document
        if q_locs:
            for q_loc in q_locs:
                if loc_match(q_loc, doc_text or "", file_id or "", doc_meta or {}):
                    # Location found - mark it so we skip strict location validation below
                    mention_location_found = True
                    break
        # Check if any mentioned person is in the document
        if q_persons:
            for p in q_persons:
                if p and p in text_full:
                    mention_person_found = True
                    break

        if mention_location_found or mention_person_found:
            return True
        else:
            return True

    # For legal/contract-style queries that explicitly mention an organization,
    # require that organization to appear in the document text/metadata/filename.
    has_legal_or_contract_intent = any(
        w in q_text
        for w in [
            "agreement",
            "agreements",
            "contract",
            "contracts",
            "nda",
            "policy",
            "clause",
            "clauses",
        ]
    ) or bool(query_meta.get("legal_clause") or query_meta.get("governing_law"))

    if q_orgs_norm and has_legal_or_contract_intent:
        if not any(org and org in text_full for org in q_orgs_norm):
            # Org-specific legal/contract query, but org name never appears
            # in the document → treat as false positive.
            return False

    # -------------------------
    # UNIVERSAL LOCATION VALIDATION
    # -------------------------
    if q_locs:
        if is_mention_query:
            pass
        else:
            # Strict location validation for non-mention queries
            for q_loc in q_locs:
                if not loc_match(q_loc, doc_text or "", file_id or "", doc_meta or {}):
                    return False

    # -------------------------
    # LEGAL VALIDATION
    # UNIVERSAL LEGAL CORE v1.0
    # -------------------------
    q_gov = (query_meta.get("governing_law") or "").lower().strip()
    if q_gov:
        if q_gov not in text_full:
            return False

    q_clause = (query_meta.get("legal_clause") or "").lower().strip()
    if q_clause:
        # Allow clause synonyms, especially for NDA/confidentiality,
        # so that queries containing just "NDA" still match
        # non-disclosure / confidentiality language in the document.
        synonyms = CLAUSE_SYNONYMS.get(q_clause, [q_clause])
        if not any(term in text_full for term in synonyms):
            return False

    # -------------------------
    # TEMPORAL (authoritative engine)
    # -------------------------
    if is_mention_query:
        pass
    else:
        if temporal_engine and not temporal_engine.matches(
            query_meta.get("original_query") or query_meta.get("normalized_query") or "",
            doc_meta,
            doc_text or "",
        ):
            return False
    
    # -------------------------
    # NDA VALIDATION (must come BEFORE soft intent fallback)
    # -------------------------
    boilerplate = [
        "confidential information",
        "mutual nda",
        "terms and conditions",
        "receiving party",
        "disclosing party",
        "whereas",
        "representations and warranties",
    ]

    # Comprehensive NDA query detection - matches ALL variations
    # NDA = Non-Disclosure Agreement = Confidentiality = Must be understood semantically
    has_nda_terms = any(term in q_text for term in [
        'nda', 'ndas', 'mnda', 'mndas',
        'non-disclosure', 'non disclosure', 'nondisclosure',
        'mutual non-disclosure', 'mutual non disclosure', 'mutual nondisclosure'
    ])
    
    is_nda_query = (
        has_nda_terms or
        # Standalone NDA terms
        q_text.strip() in ["nda", "ndas", "mnda", "mndas"] or
        # Standalone non-disclosure terms
        q_text.strip() in ["non-disclosure", "non disclosure", "nondisclosure", "mutual non-disclosure", "mutual non disclosure", "mutual nondisclosure"] or
        # Patterns: "show me all NDA", "list all NDAs", "show mNDAs"
        re.search(r'\b(?:show|list|find|search|what|which)\s+(?:me|all|the)?\s*(?:all\s+)?(?:nda|ndas|mnda|mndas)\b', q_text) or
        # Patterns: "show me the documents that includes NDA", "documents that include NDA"
        re.search(r'\b(?:documents?|files?|docs?|contracts?|agreements?)\s+(?:that\s+)?(?:includes?|include|are|is)\s+(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\b', q_text) or
        # Patterns: "show me contracts that are NDA", "NDA contracts", "NDA contract documents"
        re.search(r'\b(?:contracts?|agreements?|documents?)\s+(?:that\s+are|are\s+)?(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\b', q_text) or
        re.search(r'\b(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\s+(?:contracts?|agreements?|documents?)\b', q_text) or
        # Patterns: "show me NDA contracts", "show me NDA contract documents"
        re.search(r'\b(?:show|list|find)\s+me\s+(?:the\s+)?(?:nda|ndas|mnda|mndas|non[- ]?disclosure)\s+(?:contracts?|agreements?|documents?)\b', q_text)
    )
    
    if is_nda_query:
        q_is_contract_intent = True

        file_lower = (file_id or "").lower()
        filename_has_nda = (
            re.search(r'\bnda\b', file_lower) or
            re.search(r'\bmnda\b', file_lower) or
            file_lower.endswith('_nda') or
            file_lower.endswith('_mnda') or
            file_lower.endswith('nda') or
            file_lower.endswith('mnda') or
            "non-disclosure" in file_lower or
            "nondisclosure" in file_lower or
            "mutual nda" in file_lower or
            "mutual non-disclosure" in file_lower
        )
        
        # Check filename first (most reliable), then metadata, then text
        # This ensures NDA documents are found even if filename doesn't explicitly say "NDA"
        meta_has_nda = (
            doc_meta and (
                doc_meta.get("document_type", "").lower() == "nda" or
                "nda" in str(doc_meta.get("tags", [])).lower()
            )
        )
        
        # Check text for explicit NDA document title/header (first 500 chars only - title area)
        text_sample = (doc_text or "")[:500].lower()
        text_has_nda_title = (
            # Document title/header explicitly says it's an NDA (must be at start or after newline)
            re.search(r'^(?:mutual\s+)?(?:non[- ]?disclosure|nondisclosure)\s+agreement', text_sample, re.IGNORECASE | re.MULTILINE) or
            re.search(r'^(?:nda|mnda)\s+(?:agreement|document)', text_sample, re.IGNORECASE | re.MULTILINE) or
            re.search(r'^(?:agreement|document)\s+(?:nda|mnda)', text_sample, re.IGNORECASE | re.MULTILINE) or
            # "This Non-Disclosure Agreement" or "The Mutual Non-Disclosure Agreement" in first 300 chars (title area)
            re.search(r'\b(?:this|the)\s+(?:mutual\s+)?(?:non[- ]?disclosure|nondisclosure)\s+agreement\b', text_sample[:300], re.IGNORECASE)
        )
        
        has_nda_terms = filename_has_nda or meta_has_nda or text_has_nda_title
        if not has_nda_terms:
            return False
    else:
        q_is_contract_intent = False
        
    # ---------------------------------------------------------
    # SOFT INTENT FALLBACK (Option 2)
    # This comes AFTER NDA validation to avoid bypassing it
    # ---------------------------------------------------------
    try:
        qtype = (query_meta or {}).get("query_type", "") or ""
        qtype = qtype.lower()

        # Fallback: infer query_type from query text if missing
        if not qtype:
            q_txt = (query_meta or {}).get("normalized_query") or (query_meta or {}).get("original_query") or ""
            q_txt_l = str(q_txt).lower()
            if any(w in q_txt_l for w in ["document", "documents", "file", "files"]):
                qtype = "document"
            elif any(w in q_txt_l for w in ["agreement", "agreements", "contract", "contracts", "nda"]):
                qtype = "agreement"

        doctype = (doc_meta.get("element_type") or doc_meta.get("doc_type") or "").lower()

        is_document_query = qtype in ["document", "documents", "file", "files"]
        is_contract = (
            "agreement" in doctype
            or "contract" in doctype
            or "nda" in doctype
            or "agreement" in file_l
            or "contract" in file_l
            or "nda" in file_l
        )

        # Skip soft intent fallback for NDA queries (already validated above)
        # Only activate fallback when:
        # 1. It's a document/file query
        # 2. Document type is actually a contract/agreement
        # 3. It's NOT an NDA query (NDA queries are handled above)
        if is_document_query and is_contract and not is_nda_query:
            return True
    except Exception:
        pass

    if any(bp in text_full for bp in boilerplate):
        if q_is_contract_intent:
            pass
        elif not any(
            w in q_text
            for w in ["contract", "contracts", "agreement", "agreements", "legal", "governing"]
        ):
            return False

    if file_id and file_id.lower() in q_text and not (q_persons or q_orgs_norm):
        return False

    return True

