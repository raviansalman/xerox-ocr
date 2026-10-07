#!/usr/bin/env python3
"""
Document Processing Web Interface

Professional web application for document processing and intelligent search
using advanced OCR and text extraction capabilities.
"""
import warnings
# Suppress HuggingFace deprecation so logs are readable (resume_download removal in 1.0)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

import os
import sys
import time
import json
import asyncio
import logging
import requests
import re
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, unquote as _url_unquote
from dotenv import load_dotenv
from src.semantic.semantic_pipeline import get_global_semantic_pipeline, embed_text as embed_text_doc

# Use process-wide singleton so server binds fast and we don't load the model multiple times
def get_semantic_engine():
    return get_global_semantic_pipeline()


def _apply_constraint_boost(items: list, meta: dict) -> None:
    """Ranking-only constraint boosts; delegates to src.semantic.constraint_ranking."""
    from src.semantic.constraint_ranking import apply_constraint_boost

    apply_constraint_boost(items, meta, get_semantic_engine=get_semantic_engine)


def _prune_tx_peer_from_search_results(items: list, query_meta: dict) -> list:
    """
    Drop hits whose visible text/filename/file_id mentions a peer city
    mutually exclusive with the queried location anchor.
    """
    anchors = [
        str(c).lower().strip()
        for c in (query_meta.get("location_anchor_cities") or [])
        if c
    ]
    if len(anchors) != 1:
        return items
        
    from src.semantic.semantic_utils import LOCATION_PEERS
    ac = anchors[0]
    peers = LOCATION_PEERS.get(ac)
    if not peers:
        return items
        
    out: list = []
    for it in items:
        md = it.get("metadata") or {}
        raw_fn = str(md.get("original_filename") or md.get("filename") or "")
        try:
            dec_fn = _url_unquote(raw_fn.split("?")[0]).lower()
        except Exception:
            dec_fn = raw_fn.lower()
        fid = str(it.get("file_id") or it.get("source_file") or "")
        fid_tail = os.path.basename(fid.split("?")[0]).lower()
        surface = f"{it.get('text') or ''} {dec_fn} {fid_tail}".lower()
        
        has_peer_match = any(re.search(rf"\b{re.escape(peer)}\b", surface) for peer in peers)
        if has_peer_match:
            continue
        out.append(it)
    return out


# Load environment variables
load_dotenv()

# Workflow configuration from environment variables
WORKFLOW_API_URL = os.getenv('WORKFLOW_API_URL', '')
WORKFLOW_ACCESS_KEY = os.getenv('WORKFLOW_ACCESS_KEY', '')
WORKFLOW_STOR_API_KEY = os.getenv('WORKFLOW_STOR_API_KEY', '')
WORKFLOW_ENABLED = os.getenv('WORKFLOW_ENABLED', 'false').lower() == 'true'
WORKFLOW_TIMEOUT = int(os.getenv('WORKFLOW_TIMEOUT', '10'))

# Large file routing (prevents big files from blocking the main queue)
LARGE_FILE_BYTES = int(os.getenv("LARGE_FILE_BYTES", str(5 * 1024 * 1024)))  # 5MB
LARGE_TEXT_CHARS = int(os.getenv("LARGE_TEXT_CHARS", "2000000"))  # 2M chars
LARGE_CSV_ROWS = int(os.getenv("LARGE_CSV_ROWS", "100000"))  # 100k rows

# User-sharded queues: each user maps to a shard so one user's 500 files don't block others
QUEUE_SHARD_COUNT = int(os.getenv("QUEUE_SHARD_COUNT", "8"))
SPREADSHEET_URL_FORCE_LARGE = os.getenv("SPREADSHEET_URL_FORCE_LARGE", "true").lower() == "true"
CELERY_DEFAULT_QUEUE = os.getenv("CELERY_DEFAULT_QUEUE", "ultimate_processing")
CELERY_LARGE_QUEUE = os.getenv("CELERY_LARGE_QUEUE", "ultimate_processing_large")

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level chunk cache for content-scan (keyed by user_id).
# Avoids re-fetching 16K chunks on every content-scan query within the TTL.
_CONTENT_SCAN_CACHE: dict = {}
_CONTENT_SCAN_TTL: int = 180  # 3 minutes

# Import vector integration and Celery tasks
try:
    from src.ultimate_vector_integration import UltimateVectorIntegration
    from src.ultimate_tasks import process_ultimate_document_task
    VECTOR_INTEGRATION_AVAILABLE = True
except ImportError as e:
    VECTOR_INTEGRATION_AVAILABLE = False
    logging.warning(f"Vector integration not available: {e}")

# Global storage for processed files
processed_files: Dict[str, Dict[str, Any]] = {}

@dataclass
class ProcessedFile:
    """Represents a processed file with metadata."""
    file_id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    text_content: str
    processing_time: float
    confidence: float
    extraction_method: str
    keywords: List[str]
    created_at: str
    # Vision analysis fields
    objects_detected: List[Dict[str, Any]] = None
    image_caption: str = ""
    visual_elements: List[str] = None
    # Metadata for API integration
    metadata: Dict[str, Any] = None

class DocumentProcessingUI:
    """Professional document processing interface with advanced search capabilities and vector database integration."""
    
    def __init__(self):
        """
        Initialize the UI with document processor.

        Vector integration is now initialized lazily on first use to keep
        the API process lightweight at startup; heavy model and Milvus
        initialization happens only when search/admin endpoints are used.
        """
        self.processor = None
        self.vector_integration = None
        self.available = False
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize the document processor."""
        try:
            from src.ultimate_search_processor import DocumentProcessor
            self.processor = DocumentProcessor()
            self.available = True
            logger.info("Document processor initialized successfully")
        except Exception as e:
            self.processor = None
            self.available = False
            logger.error(f"Document processor not available: {e}")
    
    def _initialize_vector_integration(self):
        """Initialize vector database integration."""
        if VECTOR_INTEGRATION_AVAILABLE:
            try:
                self.vector_integration = UltimateVectorIntegration()
                logger.info("Vector database integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector integration: {e}")
                self.vector_integration = None
        else:
            logger.warning("Vector integration not available - running in standalone mode")

    def get_vector_integration(self) -> Optional["UltimateVectorIntegration"]:
        """
        Lazily initialize and return vector integration.

        This avoids loading models and connecting to Milvus in the API
        process until a search/admin endpoint actually needs it.
        """
        if self.vector_integration is None and VECTOR_INTEGRATION_AVAILABLE:
            try:
                self._initialize_vector_integration()
            except Exception as exc:
                logger.error(f"Lazy vector integration init failed: {exc}")
                self.vector_integration = None
        return self.vector_integration

def create_html_ui():
    """Create the HTML UI for the Ultimate Document Processor."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ultimate Document Processor</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <meta name="version" content="v2.0-no-polling">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }

            .content {
                padding: 30px;
            }

            .section {
                margin-bottom: 40px;
                padding: 25px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                background: #fafafa;
            }

            .section h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.5em;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }


            .url-container {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }

            .url-input {
                flex: 1;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }

            .url-input:focus {
                outline: none;
                border-color: #3498db;
            }

            .url-btn {
                padding: 12px 24px;
                background: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background 0.3s;
            }

            .url-btn:hover {
                background: #219a52;
            }

            .search-container {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }

            .search-input {
                flex: 1;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
            }

            .search-btn {
                background: #e74c3c;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.3s ease;
            }

            .search-btn:hover {
                background: #c0392b;
            }

            .files-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }

            .file-card {
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }

            .file-card:hover {
                transform: translateY(-5px);
            }

            .file-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }

            .file-name {
                font-weight: bold;
                color: #2c3e50;
                font-size: 1.1em;
            }

            .file-type {
                background: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
            }

            .file-info {
                margin-bottom: 10px;
                font-size: 0.9em;
                color: #666;
            }
            
            .object-info {
                background: #e8f5e8;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 5px 0;
                font-size: 0.85em;
                color: #2d5a2d;
                border-left: 3px solid #4caf50;
            }
            
            .caption-info {
                background: #e3f2fd;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 5px 0;
                font-size: 0.85em;
                color: #1565c0;
                border-left: 3px solid #2196f3;
            }
            
            .visual-info {
                background: #fff3e0;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 5px 0;
                font-size: 0.85em;
                color: #e65100;
                border-left: 3px solid #ff9800;
            }

            .search-results {
                margin-top: 20px;
            }

            .result-item {
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            
            .result-badges {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .match-badge {
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            .match-badge.object {
                background: #e8f5e8;
                color: #2d5a2d;
            }
            
            .match-badge.description {
                background: #e3f2fd;
                color: #1565c0;
            }
            
            .match-badge.visual {
                background: #fff3e0;
                color: #e65100;
            }
            
            .match-badge.text {
                background: #f3e5f5;
                color: #7b1fa2;
            }
            
            .search-object-info {
                background: #e8f5e8;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 5px 0;
                font-size: 0.85em;
                color: #2d5a2d;
                border-left: 3px solid #4caf50;
            }
            
            .search-caption-info {
                background: #e3f2fd;
                padding: 5px 8px;
                border-radius: 4px;
                margin: 5px 0;
                font-size: 0.85em;
                color: #1565c0;
                border-left: 3px solid #2196f3;
            }

            .result-filename {
                font-weight: bold;
                color: #2c3e50;
            }

            .result-matches {
                background: #27ae60;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
            }

            .result-context {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
                font-size: 0.9em;
                color: #333;
            }

            .status {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-weight: bold;
            }

            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }

            .delete-btn {
                background: #e74c3c;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8em;
            }

            .delete-btn:hover {
                background: #c0392b;
            }

            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }

            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }

            .stat-label {
                color: #666;
                margin-top: 5px;
            }

            .api-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }

            .api-endpoint {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }

            .api-endpoint h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.1em;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }

            .api-form {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            .api-form .url-input {
                margin-bottom: 0;
            }

            .api-results {
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }

            .api-result-item {
                background: white;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border: 1px solid #ddd;
            }

            .api-result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }

            .api-result-id {
                font-weight: bold;
                color: #2c3e50;
            }

            .api-result-confidence {
                background: #27ae60;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8em;
            }

            .api-result-context {
                background: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.9em;
                color: #333;
            }

            @media (max-width: 768px) {
                .api-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Document Processing System</h1>
                <p>Advanced OCR and Text Extraction Platform</p>
                <div class="api-key-bar" style="margin-top:12px;">
                    <input type="password" id="apiKeyInput" placeholder="API key" autocomplete="off" style="padding:6px;width:280px;">
                    <button type="button" onclick="saveApiKey()">Use key</button>
                    <span id="apiKeyState" style="margin-left:8px;font-size:13px;"></span>
                </div>
            </div>

            <div class="content">
                <!-- Statistics -->
                <div class="stats" id="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="total-files">0</div>
                        <div class="stat-label">Files Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="total-text">0</div>
                        <div class="stat-label">Characters Extracted</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="avg-confidence">0%</div>
                        <div class="stat-label">Avg Confidence</div>
                    </div>
                </div>

                <!-- Session file cards (browser-side); used by delete refresh + stats -->
                <div class="section" style="margin-top: 8px;">
                    <h2 style="font-size: 1.1em;">Files in this browser session</h2>
                    <div id="filesGrid" class="files-grid"></div>
                </div>

                <!-- API Integration Section -->
                <div class="section">
                    <h2>API Integration</h2>
                    <div class="api-container">
                        <div class="api-endpoint">
                            <h3>POST /process (URL)</h3>
                            <div class="api-form">
                                <input type="text" id="apifileUrl" class="url-input" placeholder="File URL (e.g., https://file-view-stage.storagechain.io/api/file/view/access/...)">
                                <input type="text" id="apiUserId" class="url-input" placeholder="User ID" value="user000">
                                <input type="text" id="apiBucketId" class="url-input" placeholder="Bucket ID (optional, e.g. bucket-user777-main)" value="">
                                <input type="text" id="apiConnectionId" class="url-input" placeholder="Connection ID (optional, stable per S3 link)" value="">
                                <input type="text" id="apiPath" class="url-input" placeholder="Path (optional, e.g. Archive/Legal)" value="">
                                <input type="text" id="apiFileId" class="url-input" placeholder="File ID (auto-generated if empty)" value="">
                                <select id="apiFileType" class="url-input">
                                    <option value="">Auto-detect from URL</option>
                                    <option value="application/pdf">PDF Document</option>
                                    <option value="application/msword">DOC Document</option>
                                    <option value="application/vnd.openxmlformats-officedocument.wordprocessingml.document">DOCX Document</option>
                                    <option value="application/vnd.ms-powerpoint">PPT Document</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.presentation">PPTX Presentation</option>
                                    <option value="application/vnd.ms-powerpoint.presentation.macroEnabled.12">PPTM (with macros)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.slideshow">PPSX Slideshow</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.template">POTX Template</option>
                                    <option value="application/vnd.ms-powerpoint.template.macroEnabled.12">POTM Template (with macros)</option>
                                    <option value="application/vnd.ms-powerpoint.slideshow.macroEnabled.12">PPSM Slideshow (with macros)</option>
                                    <option value="application/vnd.oasis.opendocument.presentation">ODP (OpenDocument)</option>
                                    <option value="application/vnd.ms-excel">XLS (Legacy Excel)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">XLSX Spreadsheet</option>
                                    <option value="application/vnd.ms-excel.sheet.macroEnabled.12">XLSM (with macros)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.template">XLTX Template</option>
                                    <option value="application/vnd.ms-excel.template.macroEnabled.12">XLTM Template (with macros)</option>
                                    <option value="application/vnd.ms-excel.sheet.binary.macroEnabled.12">XLSB (Binary with macros)</option>
                                    <option value="application/vnd.oasis.opendocument.spreadsheet">ODS (OpenDocument Spreadsheet)</option>
                                    <option value="text/csv">CSV (Comma Separated Values)</option>
                                    <option value="application/csv">CSV (Alternative MIME)</option>
                                    <option value="image/jpeg">JPEG Image</option>
                                    <option value="image/png">PNG Image</option>
                                    <option value="image/gif">GIF Image</option>
                                    <option value="image/bmp">BMP Image</option>
                                    <option value="image/webp">WebP Image</option>
                                    <option value="image/tiff">TIFF Image</option>
                                    <option value="image/svg+xml">SVG Image</option>
                                </select>
                                <button class="url-btn" onclick="processApiUrl()">Process via URL</button>
                            </div>
                        </div>
                        <div class="api-endpoint">
                            <h3>POST /process-file (Upload & Process)</h3>
                            <div class="api-form">
                                <input type="file" id="fileUpload" class="url-input" accept=".pdf,.doc,.docx,.ppt,.pptx,.pptm,.ppsx,.potx,.potm,.ppsm,.odp,.xls,.xlsx,.xlsm,.xltx,.xltm,.xlsb,.ods,.csv,.jpg,.jpeg,.png,.gif,.bmp,.webp,.tiff" style="padding: 8px;">
                                <input type="text" id="uploadUserId" class="url-input" placeholder="User ID" value="user000">
                                <input type="text" id="uploadBucketId" class="url-input" placeholder="Bucket ID (optional)" value="">
                                <input type="text" id="uploadConnectionId" class="url-input" placeholder="Connection ID (optional)" value="">
                                <input type="text" id="uploadPath" class="url-input" placeholder="Path (optional, e.g. Archive)" value="">
                                <input type="text" id="uploadFileId" class="url-input" placeholder="File ID (auto-generated if empty)" value="">
                                <select id="uploadFileType" class="url-input">
                                    <option value="">Auto-detect from file</option>
                                    <option value="application/pdf">PDF Document</option>
                                    <option value="application/msword">DOC Document</option>
                                    <option value="application/vnd.openxmlformats-officedocument.wordprocessingml.document">DOCX Document</option>
                                    <option value="application/vnd.ms-powerpoint">PPT Document</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.presentation">PPTX Presentation</option>
                                    <option value="application/vnd.ms-powerpoint.presentation.macroEnabled.12">PPTM (with macros)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.slideshow">PPSX Slideshow</option>
                                    <option value="application/vnd.openxmlformats-officedocument.presentationml.template">POTX Template</option>
                                    <option value="application/vnd.ms-powerpoint.template.macroEnabled.12">POTM Template (with macros)</option>
                                    <option value="application/vnd.ms-powerpoint.slideshow.macroEnabled.12">PPSM Slideshow (with macros)</option>
                                    <option value="application/vnd.oasis.opendocument.presentation">ODP (OpenDocument)</option>
                                    <option value="application/vnd.ms-excel">XLS (Legacy Excel)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet">XLSX Spreadsheet</option>
                                    <option value="application/vnd.ms-excel.sheet.macroEnabled.12">XLSM (with macros)</option>
                                    <option value="application/vnd.openxmlformats-officedocument.spreadsheetml.template">XLTX Template</option>
                                    <option value="application/vnd.ms-excel.template.macroEnabled.12">XLTM Template (with macros)</option>
                                    <option value="application/vnd.ms-excel.sheet.binary.macroEnabled.12">XLSB (Binary with macros)</option>
                                    <option value="application/vnd.oasis.opendocument.spreadsheet">ODS (OpenDocument Spreadsheet)</option>
                                    <option value="text/csv">CSV (Comma Separated Values)</option>
                                    <option value="application/csv">CSV (Alternative MIME)</option>
                                    <option value="image/jpeg">JPEG Image</option>
                                    <option value="image/png">PNG Image</option>
                                    <option value="image/gif">GIF Image</option>
                                    <option value="image/bmp">BMP Image</option>
                                    <option value="image/webp">WebP Image</option>
                                    <option value="image/tiff">TIFF Image</option>
                                </select>
                                <button class="url-btn" onclick="processFileUpload()">Upload & Process</button>
                            </div>
                            <div id="fileUploadStatus" style="margin-top: 10px;"></div>
                        </div>
                        <div class="api-endpoint">
                            <h3>POST /search</h3>
                            <div class="api-form">
                                <input type="text" id="apiSearchText" class="url-input" placeholder="Search text">
                                <input type="text" id="apiSearchUserId" class="url-input" placeholder="User ID" value="user000">
                                <input type="text" id="apiSearchBucketId" class="url-input" placeholder="Bucket ID (optional)" value="">
                                <input type="text" id="apiSearchConnectionId" class="url-input" placeholder="Connection ID (optional)" value="">
                                <input type="text" id="apiSearchPath" class="url-input" placeholder="Path (optional, e.g. Archive)" value="">
                                <input type="number" id="apiSearchMinScore" class="url-input" placeholder="Min score (e.g. 0.3)" step="0.01" min="0" max="1" value="">
                                <select id="apiSearchMethod" class="url-input" style="margin-top: 10px;">
                                    <option value="both">Both (Semantic + Vector)</option>
                                    <option value="semantic_pure"> Semantic Mode </option>
                                    <option value="vector">Vector Search Only</option>
                                </select>
                                <button class="url-btn" onclick="searchApi()">Search via API</button>
                            </div>
                        </div>
                    </div>
                    <div id="apiStatus"></div>
                </div>

                <!-- Delete one file: all vectors for that file_id + user_id -->
                <div class="section">
                    <h2>Delete all vectors for one file</h2>
                    <p style="color:#555;font-size:0.95em;margin:0 0 12px 0;">
                        Removes <strong>every chunk</strong> in Milvus for the given <strong>file ID</strong> and <strong>user ID</strong> (one file per request).                         If the file was ingested with scoping, set <strong>Bucket ID</strong> and/or <strong>Connection ID</strong> (same values as at ingest).
                        To drop an entire S3 link for a user, use Admin → purge with optional bucket and/or connection. To wipe <em>all</em> vectors for the user, leave those fields empty there.
                    </p>
                    <div class="api-container">
                        <div class="api-endpoint" style="grid-column: 1 / -1;">
                            <h3>Delete vectors for file + user</h3>
                            <p style="font-size:0.85em;color:#666;margin:0 0 8px 0;">API: <code>DELETE</code> or <code>POST</code> <code>/delete-document</code></p>
                            <div class="api-form">
                                <input type="text" id="deleteFileId" class="url-input" placeholder="File ID (Milvus id from search — often upload_… or UUID)" required>
                                <input type="text" id="deleteUserId" class="url-input" placeholder="User ID (required — same tenant as ingest / search)">
                                <input type="text" id="deleteBucketId" class="url-input" placeholder="Bucket ID (optional — only if used at ingest)">
                                <input type="text" id="deleteConnectionId" class="url-input" placeholder="Connection ID (optional — only if used at ingest)">
                                <input type="text" id="deletePath" class="url-input" placeholder="Path (optional — only if used at ingest)">
                                <button class="url-btn" onclick="deleteDocumentFromForm()" style="background: #e74c3c;">Delete all vectors for this file</button>
                            </div>
                            <div id="deleteResult" style="margin-top: 15px;"></div>
                            <div style="margin-top: 15px; padding: 10px; background: #e8f4fd; border-radius: 5px; border-left: 4px solid #2196f3; font-size: 0.9em;">
                                <strong>File ID vs filename:</strong> Use the <strong>stored file id</strong> (bold in API search results), not only the <code>.pdf</code> name. Use <strong>Fill delete form from search</strong> on a result row, or paste the id here.
                            </div>
                            <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;">
                                <strong>Warning:</strong> Permanently deletes <strong>all vector chunks</strong> for this file in Milvus. Cannot be undone.
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Additional API Endpoints -->
                <div class="section">
                    <h2>Additional API Endpoints</h2>
                    <div class="api-container">
                        <div class="api-endpoint">
                            <h3>GET /health</h3>
                            <div class="api-form">
                                <button class="url-btn" onclick="checkHealth()">Check System Health</button>
                            </div>
                        </div>
                        <div class="api-endpoint">
                            <h3>GET /collections</h3>
                            <div class="api-form">
                                <button class="url-btn" onclick="getCollections()">View Collections</button>
                            </div>
                        </div>
                    </div>
                    <div id="additionalApiStatus"></div>
                </div>

                <!-- Admin / Monitoring Endpoints -->
                <div class="section">
                    <h2>Admin &amp; Scalability Debug</h2>
                    <div class="api-container">
                        <div class="api-endpoint">
                            <h3>GET /admin/stuck-jobs</h3>
                            <div class="api-form">
                                <button class="url-btn" onclick="getStuckJobs()">View Stuck Jobs</button>
                            </div>
                        </div>
                        <div class="api-endpoint">
                            <h3>GET /admin/queues</h3>
                            <div class="api-form">
                                <button class="url-btn" onclick="getQueueConfig()">View Queue Configuration</button>
                            </div>
                        </div>
                        <div class="api-endpoint" style="grid-column: 1 / -1;">
                            <h3>Load all vectors (summary)</h3>
                            <p style="font-size: 0.85em; color: #666; margin: 0 0 4px 0;">API: <code>GET /admin/vector-storage-by-user</code></p>
                            <p style="font-size: 0.9em; color: #555; margin: 0 0 8px 0;">
                                Scans Milvus (up to <strong>max rows</strong>) and shows <strong>chunk</strong> and <strong>file</strong> counts per <strong>user ID</strong>. Requires an admin API key.
                            </p>
                            <div class="api-form">
                                <input type="number" id="vectorStatsMaxRows" class="url-input" placeholder="Max rows to scan (default 50000)" min="1000" max="2000000" value="50000" style="max-width: 220px;">
                                <input type="text" id="vectorStatsFilterUserId" class="url-input" placeholder="Optional: one User ID to scan only that tenant" style="max-width: 280px;">
                                <input type="password" id="vectorStatsAdminKey" class="url-input" placeholder="Admin key (if server requires it)" autocomplete="off" style="max-width: 280px;">
                                <button class="url-btn" onclick="getVectorStorageByUser()">Load all vectors</button>
                            </div>
                        </div>
                        <div class="api-endpoint" style="grid-column: 1 / -1;">
                            <h3>Delete all data for a user ID</h3>
                            <p style="font-size: 0.85em; color: #666; margin: 0 0 4px 0;">API: <code>POST /admin/purge-user-vectors</code></p>
                            <p style="font-size: 0.9em; color: #555; margin: 0 0 8px 0;">
                                Wipes vector rows for a <strong>user ID</strong>. Leave bucket and connection empty to remove <em>everything</em> for that user.
                                Set <code>bucket_id</code> and/or <code>connectionId</code> to match ingest (use both together for a narrow AND). For one file, use <em>Delete all vectors for this file</em> above. Same admin key as <em>Load all vectors</em>.
                            </p>
                            <div class="api-form">
                                <input type="text" id="adminDeleteAllDataUserId" class="url-input" placeholder="User ID (required)" style="max-width: 280px;">
                                <input type="text" id="adminDeleteAllDataBucketId" class="url-input" placeholder="Optional: bucket_id (same as ingest)" style="max-width: 280px;">
                                <input type="text" id="adminDeleteAllDataConnectionId" class="url-input" placeholder="Optional: connectionId (same as POST /process)" style="max-width: 280px;">
                                <button type="button" class="url-btn" style="background:#c0392b;" onclick="deleteAllDataForUserId()">Delete stored data (scoped or full user)</button>
                            </div>
                        </div>
                        <div class="api-endpoint">
                            <h3>POST /admin/route-test</h3>
                            <div class="api-form">
                                <input type="text" id="routeTestFileType" class="url-input" placeholder="MIME Type (e.g. application/pdf)">
                                <input type="text" id="routeTestFilename" class="url-input" placeholder="Filename (e.g. report.xlsx)">
                                <input type="number" id="routeTestSizeBytes" class="url-input" placeholder="Size in bytes (optional)">
                                <button class="url-btn" onclick="testRouting()">Test Routing</button>
                            </div>
                        </div>
                    </div>
                    <div id="adminApiStatus" style="margin-top: 15px;"></div>
                </div>
            </div>
        </div>

        <script>
            // Force cache refresh - multiple methods
            const CACHE_VERSION = Date.now();
            const RANDOM_ID = Math.random().toString(36).substr(2, 9);
            console.log('UI loaded with cache version:', CACHE_VERSION, 'Random ID:', RANDOM_ID);
            
            // Clear any existing intervals/timeouts
            if (window.pollingInterval) {
                clearInterval(window.pollingInterval);
                window.pollingInterval = null;
            }
            if (window.pollingTimeout) {
                clearTimeout(window.pollingTimeout);
                window.pollingTimeout = null;
            }
            
            let processedFiles = {};

            /** Escape any server or document text before it goes into innerHTML. */
            function escHtml(v) {
                return String(v === undefined || v === null ? '' : v).replace(/[&<>"']/g, function (c) {
                    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
                });
            }

            /** Every API call is authenticated: send the key held in this tab's sessionStorage. */
            const API_KEY_STORE = 'xocr_api_key';
            function getApiKey() {
                try { return sessionStorage.getItem(API_KEY_STORE) || ''; } catch (e) { return ''; }
            }
            function showKeyState() {
                const el = document.getElementById('apiKeyState');
                if (el) el.textContent = getApiKey() ? 'Key set for this tab' : 'No API key set';
            }
            function saveApiKey() {
                const input = document.getElementById('apiKeyInput');
                try { sessionStorage.setItem(API_KEY_STORE, input.value.trim()); } catch (e) {}
                input.value = '';
                showKeyState();
            }
            const _origFetch = window.fetch.bind(window);
            window.fetch = function (resource, init) {
                init = init || {};
                const key = getApiKey();
                if (key) {
                    const headers = new Headers(init.headers || {});
                    headers.set('X-API-Key', key);
                    init.headers = headers;
                }
                return _origFetch(resource, init);
            };
            showKeyState();

            /** FastAPI errors: detail may be string or { message, hint, ... } */
            function formatApiDetail(payload) {
                if (!payload || typeof payload !== 'object') return String(payload || 'Unknown error');
                if (typeof payload.detail === 'string') return payload.detail;
                if (payload.detail && typeof payload.detail === 'object') {
                    const d = payload.detail;
                    const parts = [d.message, d.hint, d.file_id ? 'file_id: ' + d.file_id : ''].filter(Boolean);
                    if (parts.length) return parts.join(' — ');
                    try { return JSON.stringify(d); } catch (e) { return 'Error (see console)'; }
                }
                if (payload.message) return String(payload.message);
                if (payload.error) return String(payload.error);
                try { return JSON.stringify(payload); } catch (e2) { return 'Request failed'; }
            }

            function fillDeleteFromSearchResult(index) {
                const list = window.__lastSearchResults;
                if (!list || !list[index]) {
                    alert('No search result at this index; run search again.');
                    return;
                }
                const r = list[index];
                const fid = r.file_id || '';
                if (!fid) {
                    alert('This hit has no file_id; cannot pre-fill delete.');
                    return;
                }
                document.getElementById('deleteFileId').value = fid;
                const su = document.getElementById('apiSearchUserId');
                const du = document.getElementById('deleteUserId');
                if (su && du) {
                    du.value = (su.value || '').trim();
                }
                const delBox = document.getElementById('deleteResult');
                if (delBox) {
                    delBox.innerHTML = '<div class="status info">File ID and User ID filled from this search hit. Click <strong>Delete all vectors for this file</strong> when ready.</div>';
                }
            }

            // Removed unused searchDocuments and displaySearchResults functions
            // Search functionality is handled through the API search form

            function updateFilesGrid() {
                const filesGrid = document.getElementById('filesGrid');
                if (!filesGrid) {
                    return;
                }
                const files = Object.values(processedFiles);
                
                if (files.length === 0) {
                    filesGrid.innerHTML = '<p style="color: #666; text-align: center; grid-column: 1/-1;">No files processed yet</p>';
                    return;
                }

                let html = '';
                files.forEach(file => {
                    // Build object detection info
                    let objectInfo = '';
                    if (file.objects_detected && file.objects_detected.length > 0) {
                        const objects = file.objects_detected.map(obj => 
                            `${escHtml(obj.class)} (${(obj.confidence * 100).toFixed(1)}%)`
                        ).join(', ');
                        objectInfo = `<div class="object-info">Objects: ${objects}</div>`;
                    }
                    
                    // Build image caption info
                    let captionInfo = '';
                    if (file.image_caption) {
                        captionInfo = `<div class="caption-info">Description: ${escHtml(file.image_caption)}</div>`;
                    }
                    
                    // Build visual elements info
                    let visualInfo = '';
                    if (file.visual_elements && file.visual_elements.length > 0) {
                        visualInfo = `<div class="visual-info">Elements: ${escHtml(file.visual_elements.join(', '))}</div>`;
                    }
                    
                    html += `
                        <div class="file-card">
                            <div class="file-header">
                                <span class="file-name">${escHtml(file.filename)}</span>
                                <span class="file-type">${escHtml(file.file_type)}</span>
                            </div>
                            <div class="file-info">
                                <div>Size: ${(file.file_size / 1024).toFixed(1)} KB</div>
                                <div>Processed in: ${file.processing_time.toFixed(2)}s</div>
                                <div>Confidence: ${file.confidence.toFixed(1)}%</div>
                                <div>Text: ${file.text_content.length} characters</div>
                                ${objectInfo}
                                ${captionInfo}
                                ${visualInfo}
                            </div>
                            <button class="delete-btn" onclick="deleteFile(${escHtml(JSON.stringify(String(file.file_id)))})">Delete vectors for this file</button>
                        </div>
                    `;
                });

                filesGrid.innerHTML = html;
            }

            function deleteFile(fileId, userId = null, bucketId = null, path = null, connectionId = null) {
                if (!confirm('Delete all vector chunks for this file in Milvus? This cannot be undone. (Add User ID in the form below if your rows are tenant-scoped.)')) {
                    return;
                }
                const payload = { file_id: fileId };
                if (userId) payload.user_id = userId;
                if (bucketId) payload.bucket_id = bucketId;
                if (path) payload.path = path;
                if (connectionId) payload.connectionId = connectionId;

                const delResultEl = document.getElementById('deleteResult');
                if (delResultEl) {
                    delResultEl.innerHTML = '<div class="status info">Deleting…</div>';
                }

                fetch('/delete-document', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify(payload)
                })
                    .then(async (response) => {
                        const text = await response.text();
                        let data = {};
                        if (text) {
                            try {
                                data = JSON.parse(text);
                            } catch (parseErr) {
                                throw new Error('Server did not return JSON: ' + text.slice(0, 240));
                            }
                        }
                        if (!response.ok) {
                            const msg = formatApiDetail(data);
                            if (delResultEl) {
                                delResultEl.innerHTML = '<div class="status error"><strong>Delete failed</strong> (' + response.status + '): ' +
                                    String(msg).replace(/</g, '&lt;') + '</div>';
                            }
                            alert('Delete failed (' + response.status + '): ' + msg);
                            return null;
                        }
                        return data;
                    })
                    .then((data) => {
                        if (!data) return;
                        if (data.success) {
                            const fd = data.files_deleted ?? 0;
                            const cd = data.chunks_deleted ?? 0;
                            alert('All vectors removed for this file.\\nLogical files: ' + fd +
                                '\\nTotal chunks removed: ' + cd +
                                '\\nDocument chunks: ' + (data.document_chunks_deleted ?? 0) +
                                '\\nImage chunks: ' + (data.image_chunks_deleted ?? 0));
                            delete processedFiles[fileId];
                            updateFilesGrid();
                            updateStats();
                            if (delResultEl) {
                                const ids = (data.distinct_file_ids_removed || []).slice(0, 5).join(', ');
                                delResultEl.innerHTML =
                                    '<div class="status success"><strong>OK</strong> — removed <strong>' + fd + '</strong> file(s), ' +
                                    '<strong>' + cd + '</strong> chunk(s).' +
                                    (ids ? '<br><small>Ids: ' + ids.replace(/</g, '&lt;') + '</small>' : '') + '</div>';
                            }
                        } else {
                            const msg = formatApiDetail(data);
                            if (delResultEl) {
                                delResultEl.innerHTML = '<div class="status error">' + String(msg).replace(/</g, '&lt;') + '</div>';
                            }
                            alert('Failed to delete vectors for file: ' + msg);
                        }
                    })
                    .catch((error) => {
                        console.error('Delete error:', error);
                        const m = error && error.message ? error.message : String(error);
                        if (delResultEl) {
                            delResultEl.innerHTML = '<div class="status error">' + m.replace(/</g, '&lt;') + '</div>';
                        }
                        alert('Error deleting vectors for file: ' + m);
                    });
            }
            
            function deleteDocumentFromForm() {
                const fileId = document.getElementById('deleteFileId').value.trim();
                const userId = document.getElementById('deleteUserId').value.trim() || null;
                const bucketId = document.getElementById('deleteBucketId').value.trim() || null;
                const connectionId = (document.getElementById('deleteConnectionId') || {}).value.trim() || null;
                const path = document.getElementById('deletePath').value.trim() || null;
                
                if (!fileId) {
                    alert('File ID is required (the stored id for this file in Milvus).');
                    return;
                }
                
                deleteFile(fileId, userId, bucketId, path, connectionId);
            }

            function updateStats() {
                const files = Object.values(processedFiles);
                const totalFiles = files.length;
                const totalText = files.reduce((sum, file) => sum + file.text_content.length, 0);
                const avgConfidence = files.length > 0 ? 
                    (files.reduce((sum, file) => sum + file.confidence, 0) / files.length).toFixed(1) : 0;

                const elFiles = document.getElementById('total-files');
                const elText = document.getElementById('total-text');
                const elConf = document.getElementById('avg-confidence');
                if (elFiles) elFiles.textContent = totalFiles;
                if (elText) elText.textContent = totalText.toLocaleString();
                if (elConf) elConf.textContent = avgConfidence + '%';
            }

            // API Integration Functions
            function processApiUrl() {
                console.log('processApiUrl called - NEW VERSION v2.0-no-polling');
                
                const fileUrl = document.getElementById('apifileUrl').value.trim();
                const userId = document.getElementById('apiUserId').value.trim() || 'user000';
                const bucketId = document.getElementById('apiBucketId').value.trim();
                const connectionId = (document.getElementById('apiConnectionId') || {}).value.trim();
                const path = document.getElementById('apiPath').value.trim();
                const fileId = document.getElementById('apiFileId').value.trim();
                const fileType = document.getElementById('apiFileType').value;

                if (!fileUrl) {
                    alert('Please enter a file URL');
                    return;
                }

                const apiStatus = document.getElementById('apiStatus');
                apiStatus.innerHTML = '<div class="status info">🔄 Processing via API...</div>';

                // Build request body - only include fileId and fileType if they have values
                const requestBody = {
                    fileUrl: fileUrl,
                    userId: userId
                };
                if (bucketId) {
                    requestBody.bucketId = bucketId;
                }
                if (connectionId) {
                    requestBody.connectionId = connectionId;
                }
                if (path) {
                    requestBody.path = path;
                }
                
                // Only include fileId if provided (empty string triggers auto-generation)
                if (fileId && fileId.trim()) {
                    requestBody.fileId = fileId;
                }
                
                // Only include fileType if provided (empty string triggers auto-detection)
                if (fileType && fileType.trim()) {
                    requestBody.fileType = fileType;
                }

                fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.status === 'processing' && data.task_id) {
                        // Wait for task to complete directly
                        apiStatus.innerHTML = '<div class="status info">🔄 Processing in background... Please wait...</div>';
                        waitForTaskCompletion(data.task_id, apiStatus);
                    } else if (data.success) {
                        // Direct results (shouldn't happen with current API)
                        displayProcessingResults(data, apiStatus);
                    } else {
                        apiStatus.innerHTML = '<div class="status error">API Processing failed: ' + escHtml(formatApiDetail(data)) + '</div>';
                    }
                })
                .catch(error => {
                    apiStatus.innerHTML = '<div class="status error">API Processing failed: ' + escHtml(error.message) + '</div>';
                });
            }

            function processFileUpload() {
                const fileInput = document.getElementById('fileUpload');
                const userId = document.getElementById('uploadUserId').value.trim();
                const bucketId = document.getElementById('uploadBucketId').value.trim();
                const connectionId = (document.getElementById('uploadConnectionId') || {}).value.trim();
                const path = document.getElementById('uploadPath').value.trim();
                let fileId = document.getElementById('uploadFileId').value.trim();
                const fileType = document.getElementById('uploadFileType').value;
                const statusElement = document.getElementById('fileUploadStatus');

                if (!fileInput.files || fileInput.files.length === 0) {
                    alert('Please select a file to upload');
                    return;
                }

                if (!userId) {
                    alert('Please enter user ID');
                    return;
                }

                // Auto-generate file ID if not provided
                if (!fileId) {
                    const timestamp = Date.now();
                    const random = Math.random().toString(36).substr(2, 5);
                    fileId = `upload_${timestamp}_${random}`;
                    document.getElementById('uploadFileId').value = fileId;
                }

                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);
                formData.append('fileId', fileId);
                formData.append('userId', userId);
                if (fileType) {
                    formData.append('fileType', fileType);
                }
                if (bucketId) {
                    formData.append('bucketId', bucketId);
                }
                if (connectionId) {
                    formData.append('connectionId', connectionId);
                }
                if (path) {
                    formData.append('path', path);
                }

                statusElement.innerHTML = '<div class="status info">🔄 Uploading and processing file...</div>';

                fetch('/process-file', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.status === 'processing' && data.task_id) {
                        statusElement.innerHTML = '<div class="status info">🔄 File uploaded! Processing in background... Please wait...</div>';
                        waitForTaskCompletion(data.task_id, statusElement);
                    } else if (data.success) {
                        displayProcessingResults(data, statusElement);
                    } else {
                        statusElement.innerHTML = '<div class="status error">❌ Upload failed: ' + escHtml(formatApiDetail(data)) + '</div>';
                    }
                })
                .catch(error => {
                    statusElement.innerHTML = '<div class="status error">❌ Upload failed: ' + escHtml(error.message) + '</div>';
                });
            }

            // Auto-update file type when file is selected
            document.addEventListener('DOMContentLoaded', function() {
                const fileInput = document.getElementById('fileUpload');
                const fileTypeSelect = document.getElementById('uploadFileType');
                
                if (fileInput) {
                    fileInput.addEventListener('change', function() {
                        const file = this.files[0];
                        if (file) {
                            const fileName = file.name.toLowerCase();
                            let detectedType = '';
                            
                            if (fileName.endsWith('.pdf')) {
                                detectedType = 'application/pdf';
                            } else if (fileName.endsWith('.docx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
                            } else if (fileName.endsWith('.doc')) {
                                detectedType = 'application/msword';
                            } else if (fileName.endsWith('.pptx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.presentationml.presentation';
                            } else if (fileName.endsWith('.ppt')) {
                                detectedType = 'application/vnd.ms-powerpoint';
                            } else if (fileName.endsWith('.pptm')) {
                                detectedType = 'application/vnd.ms-powerpoint.presentation.macroEnabled.12';
                            } else if (fileName.endsWith('.ppsx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.presentationml.slideshow';
                            } else if (fileName.endsWith('.potx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.presentationml.template';
                            } else if (fileName.endsWith('.potm')) {
                                detectedType = 'application/vnd.ms-powerpoint.template.macroEnabled.12';
                            } else if (fileName.endsWith('.ppsm')) {
                                detectedType = 'application/vnd.ms-powerpoint.slideshow.macroEnabled.12';
                            } else if (fileName.endsWith('.xls')) {
                                detectedType = 'application/vnd.ms-excel';
                            } else if (fileName.endsWith('.xlsx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
                            } else if (fileName.endsWith('.xlsm')) {
                                detectedType = 'application/vnd.ms-excel.sheet.macroEnabled.12';
                            } else if (fileName.endsWith('.xltx')) {
                                detectedType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.template';
                            } else if (fileName.endsWith('.xltm')) {
                                detectedType = 'application/vnd.ms-excel.template.macroEnabled.12';
                            } else if (fileName.endsWith('.xlsb')) {
                                detectedType = 'application/vnd.ms-excel.sheet.binary.macroEnabled.12';
                            } else if (fileName.endsWith('.ods')) {
                                detectedType = 'application/vnd.oasis.opendocument.spreadsheet';
                            } else if (fileName.endsWith('.csv')) {
                                detectedType = 'text/csv';
                            } else if (fileName.endsWith('.odp')) {
                                detectedType = 'application/vnd.oasis.opendocument.presentation';
                            } else if (fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) {
                                detectedType = 'image/jpeg';
                            } else if (fileName.endsWith('.png')) {
                                detectedType = 'image/png';
                            } else if (fileName.endsWith('.gif')) {
                                detectedType = 'image/gif';
                            } else if (fileName.endsWith('.bmp')) {
                                detectedType = 'image/bmp';
                            } else if (fileName.endsWith('.webp')) {
                                detectedType = 'image/webp';
                            } else if (fileName.endsWith('.tiff') || fileName.endsWith('.tif')) {
                                detectedType = 'image/tiff';
                            }
                            
                            if (detectedType && fileTypeSelect) {
                                // Set the detected type if it matches an option
                                for (let option of fileTypeSelect.options) {
                                    if (option.value === detectedType) {
                                        fileTypeSelect.value = detectedType;
                                        break;
                                    }
                                }
                            }
                            
                            // Auto-generate file ID from filename if empty
                            const uploadFileId = document.getElementById('uploadFileId');
                            if (uploadFileId && !uploadFileId.value.trim()) {
                                const baseName = file.name.replace(/\\.[^/.]+$/, '');
                                const timestamp = Date.now();
                                uploadFileId.value = `upload_${baseName}_${timestamp}`.substring(0, 50);
                            }
                        }
                    });
                }
            });

            function waitForTaskCompletion(taskId, statusElement) {
                console.log('waitForTaskCompletion called - Optimized polling, taskId:', taskId);
                
                let pollCount = 0;
                const maxPolls = 60; // Increased to 60 polls (5 minutes total)
                const pollInterval = 5000; // Poll every 5 seconds for better responsiveness
                
                function pollTaskStatus() {
                    pollCount++;
                    
                    fetch(`/task-status/${taskId}`)
                    .then(response => response.json())
                    .then(data => {
                        const taskState = data.state || data.status || 'unknown';
                        const taskStatus = data.status || 'processing';
                        const progress = data.progress || 0;
                        const statusMsg = data.status_message || data.message || data.status || 'Processing...';
                        
                        // Update status display immediately to show processing stage
                        statusElement.innerHTML = '<div class="status info">🔄 ' + escHtml(statusMsg) + ' (' + escHtml(progress) + '%)</div>';
                        
                        if (taskState === 'SUCCESS' || taskStatus === 'completed' || taskStatus === 'COMPLETED' || progress >= 100) {
                            // Task completed successfully, display results
                            if (data.result) {
                                displayProcessingResults(data.result, statusElement);
                            } else {
                                statusElement.innerHTML = '<div class="status success">✅ Processing completed successfully!</div>';
                            }
                        } else if (taskState === 'FAILURE' || taskStatus === 'failed' || taskStatus === 'FAILED') {
                            statusElement.innerHTML = '<div class="status error">❌ Processing failed: ' + escHtml(data.message || data.error || data.status_message || 'Unknown error') + '</div>';
                        } else if (pollCount >= maxPolls) {
                            // Max polls reached - task still processing
                            const statusMsg = data.status_message || data.message || data.status || 'Processing...';
                            statusElement.innerHTML = '<div class="status info">🔄 ' + escHtml(statusMsg) + ' (' + escHtml(progress) + '%) - Task still processing in background. Check back later or refresh page.</div>';
                        } else {
                            // Still processing - poll again after 5 seconds
                            setTimeout(pollTaskStatus, pollInterval);
                        }
                    })
                    .catch(error => {
                        console.error('Error checking task status:', error);
                        if (pollCount < maxPolls) {
                            // Retry on error after 5 seconds
                            setTimeout(pollTaskStatus, pollInterval);
                        } else {
                        statusElement.innerHTML = '<div class="status error">❌ Error checking task status: ' + escHtml(error.message) + '</div>';
                        }
                    });
                }
                
                // Start polling immediately
                statusElement.innerHTML = '<div class="status info">🔄 Processing in background... Please wait...</div>';
                pollTaskStatus();
            }

            function displayProcessingResults(data, statusElement) {
                // Handle both direct API response and task result data
                const result = data.result || data;
                
                // Safely access properties with fallbacks
                const fileId = result.file_id || 'Unknown';
                const confidence = result.confidence || 0;
                const processingTime = result.processing_time || 0;
                const textContent = result.text_content || '';
                const textLength = result.text_length || textContent.length || 0;
                const keywords = result.keywords || [];
                const objectsDetected = result.objects_detected || [];
                const extractionMethod = result.extraction_method || 'unknown';
                
                statusElement.innerHTML = `
                    <div class="status success">✅ Processing Completed Successfully!</div>
                            <div class="api-results">
                                <h4>Processing Results:</h4>
                                <div class="api-result-item">
                                    <div class="api-result-header">
                                <span class="api-result-id">File ID: ${escHtml(fileId)}</span>
                                <span class="api-result-confidence">${confidence.toFixed(1)}%</span>
                                    </div>
                            <div><strong>Processing Time:</strong> ${processingTime.toFixed(2)}s</div>
                            <div><strong>Text Length:</strong> ${textLength} characters</div>
                            <div><strong>Keywords:</strong> ${keywords.length} extracted</div>
                            <div><strong>Objects:</strong> ${objectsDetected.length} detected</div>
                            <div><strong>Extraction Method:</strong> ${escHtml(extractionMethod)}</div>
                            ${textContent ? `<div class="api-result-context">${escHtml(textContent.substring(0, 200))}${textContent.length > 200 ? '...' : ''}</div>` : '<div class="api-result-context">Text extracted successfully (content not shown in summary)</div>'}
                                </div>
                            </div>
                        `;
                        
                        // Clear form
                const fileUrlInput = document.getElementById('apifileUrl');
                const fileIdInput = document.getElementById('apiFileId');
                if (fileUrlInput) fileUrlInput.value = '';
                if (fileIdInput) fileIdInput.value = 'api_file_' + Date.now();
                
                // Update stats if function exists
                if (typeof updateStats === 'function') {
                        updateStats();
                    }
            }

            function searchApi() {
                const searchText = document.getElementById('apiSearchText').value.trim();
                const userId = document.getElementById('apiSearchUserId').value.trim();
                const bucketId = document.getElementById('apiSearchBucketId').value.trim();
                const connectionId = (document.getElementById('apiSearchConnectionId') || {}).value.trim();
                const path = document.getElementById('apiSearchPath').value.trim();
                const minScoreRaw = document.getElementById('apiSearchMinScore').value.trim();

                if (!searchText) {
                    alert('Please enter search text');
                    return;
                }

                if (!userId) {
                    alert('Please enter user ID');
                    return;
                }

                const apiStatus = document.getElementById('apiStatus');
                apiStatus.innerHTML = '<div class="status info">Searching via API...</div>';

                const requestBody = { searchedText: searchText };
                if (userId) {
                    requestBody.userId = userId;
                }

                const searchMethod = document.getElementById('apiSearchMethod').value;
                
                // Map UI options to API parameters
                let actualSearchMethod = searchMethod;
                let semanticMode = false;
                
                if (searchMethod === 'both') {
                    // Original BOTH mode: Hybrid semantic (with hardcoded filters) + Vector
                    actualSearchMethod = 'both';
                    semanticMode = false;  // Use original hybrid semantic mode
                } else if (searchMethod === 'semantic_pure') {
                    // Pure Semantic Mode: No hardcoded filters, trusts embedding model
                    actualSearchMethod = 'semantic';
                    semanticMode = true;
                } else if (searchMethod === 'vector') {
                    // Vector Search Only
                    actualSearchMethod = 'vector';
                    semanticMode = false;
                }

                const body = {
                    query: searchText,
                    userId: userId || 'user000',
                    search_method: actualSearchMethod,
                    semantic_mode: semanticMode
                };
                if (bucketId) {
                    body.bucket_id = bucketId;
                }
                if (connectionId) {
                    body.connectionId = connectionId;
                }
                if (path) {
                    body.path = path;
                }
                if (minScoreRaw) {
                    const v = parseFloat(minScoreRaw);
                    if (!isNaN(v)) {
                        body.min_score = v;
                    }
                }

                fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body)
                })
                .then(response => response.json().then(data => {
                    if (!response.ok) throw new Error(formatApiDetail(data) + ' (HTTP ' + response.status + ')');
                    return data;
                }))
                .then(data => {
                    // Handle search-vector response format
                    let results = [];
                    if (data.success && data.results) {
                        results = data.results;
                    } else if (Array.isArray(data)) {
                        results = data;
                    } else if (data.results) {
                        results = data.results;
                    }

                    window.__lastSearchResults = results;
                    
                        let resultsHtml = `
                            <div class="status success">API Search Complete!</div>
                            <div class="api-results">
                            <h4>Search Results (${results.length} found):</h4>
                        `;
                        
                    if (results.length === 0) {
                            resultsHtml += '<div class="api-result-item">No results found</div>';
                        } else {
                        results.forEach((result, index) => {
                            const filename = result.metadata?.filename || result.file_id || 'Unknown File';
                            const confidence = result.similarity_score || result.confidence || 0;
                            const escFid = escHtml(result.file_id || '');
                            const tx = (result.text || '');
                            
                                resultsHtml += `
                                    <div class="api-result-item">
                                        <div class="api-result-header">
                                        <span class="api-result-id" title="Milvus file_id — use for delete">${escFid}</span>
                                        <span class="api-result-confidence">${(confidence * 100).toFixed(1)}%</span>
                                        </div>
                                    <div><strong>Filename:</strong> ${escHtml(filename)}</div>
                                    <div style="font-size:12px;color:#555;margin:4px 0 0 0;">Bold line above is the id stored in Milvus; the .pdf name alone may not delete.</div>
                                    <div><strong>Text:</strong> ${escHtml(tx.substring(0, 200))}${tx.length > 200 ? '...' : ''}</div>
                                    <div><strong>Method:</strong> ${escHtml(result.extraction_method || result.search_method || 'vector_search')}</div>
                                    <button type="button" class="url-btn" style="margin-top:8px;font-size:12px;padding:6px 12px;background:#c0392b;" onclick="fillDeleteFromSearchResult(${index})">Fill delete form — this file + user</button>
                                    </div>
                                `;
                            });
                        }
                        
                        resultsHtml += '</div>';
                        apiStatus.innerHTML = resultsHtml;
                })
                .catch(error => {
                    apiStatus.innerHTML = '<div class="status error">API Search failed: ' + escHtml(error.message) + '</div>';
                });
            }

            // Additional API Functions
            function checkHealth() {
                const statusElement = document.getElementById('additionalApiStatus');
                statusElement.innerHTML = '<div class="status info">Checking system health...</div>';

                fetch('/health')
                .then(response => response.json())
                .then(data => {
                    const cls = data.status === 'healthy' ? 'success' : (data.status === 'degraded' ? 'info' : 'error');
                    let healthHtml = '<div class="status ' + cls + '">System status: ' + escHtml(data.status) + '</div>';
                    healthHtml += '<div class="api-results">';
                    healthHtml += '<h4>Dependencies:</h4>';
                    Object.values(data.checks || {}).forEach((c) => {
                        healthHtml += '<div class="api-result-item"><strong>' + escHtml(c.name) + ':</strong> ' +
                            escHtml(c.ok ? 'ok' : c.status) + (c.critical ? '' : ' (non-critical)') +
                            (c.detail ? ' - ' + escHtml(c.detail) : '') + ' <small>(' + escHtml(c.latency_ms) + ' ms)</small></div>';
                    });
                    healthHtml += `<div class="api-result-item"><strong>Timestamp:</strong> ${escHtml(data.timestamp)}</div>`;
                    
                    if (data.vector_stats) {
                        healthHtml += '<h4>Vector Database:</h4>';
                        healthHtml += `<div class="api-result-item"><strong>Total Chunks:</strong> ${data.vector_stats.total_chunks}</div>`;
                        healthHtml += `<div class="api-result-item"><strong>Total Documents:</strong> ${data.vector_stats.total_documents}</div>`;
                        healthHtml += `<div class="api-result-item"><strong>Collection:</strong> ${data.vector_stats.collection_name}</div>`;
                    }
                    
                    healthHtml += '</div>';
                    statusElement.innerHTML = healthHtml;
                })
                .catch(error => {
                    statusElement.innerHTML = '<div class="status error">Health check failed: ' + escHtml(error.message) + '</div>';
                });
            }

            function getCollections() {
                const statusElement = document.getElementById('additionalApiStatus');
                statusElement.innerHTML = '<div class="status info">Fetching collections...</div>';

                fetch('/collections')
                .then(response => response.json())
                .then(data => {
                    let collectionsHtml = '<div class="status success">Collections Retrieved!</div>';
                    collectionsHtml += '<div class="api-results">';
                    collectionsHtml += '<h4>Available Collections:</h4>';
                    
                    if (data.collections && data.collections.length > 0) {
                        data.collections.forEach(collection => {
                            collectionsHtml += `<div class="api-result-item"><strong>${collection.name}</strong> - ${collection.description}</div>`;
                        });
                    } else {
                        collectionsHtml += '<div class="api-result-item">No collections found</div>';
                    }
                    
                    collectionsHtml += '</div>';
                    statusElement.innerHTML = collectionsHtml;
                })
                .catch(error => {
                    statusElement.innerHTML = '<div class="status error">Failed to fetch collections: ' + escHtml(error.message) + '</div>';
                });
            }

            function getStuckJobs() {
                const statusElement = document.getElementById('adminApiStatus');
                statusElement.innerHTML = '<div class="status info">Fetching stuck jobs...</div>';

                fetch('/admin/stuck-jobs')
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="status success">Stuck jobs retrieved.</div>';
                        html += '<pre style="margin-top:10px; max-height:300px; overflow:auto; background:#f5f5f5; padding:10px; border-radius:6px;">';
                        html += escHtml(JSON.stringify(data, null, 2));
                        html += '</pre>';
                        statusElement.innerHTML = html;
                    })
                    .catch(error => {
                        statusElement.innerHTML = '<div class="status error">Failed to fetch stuck jobs: ' + escHtml(error.message) + '</div>';
                    });
            }

            function getQueueConfig() {
                const statusElement = document.getElementById('adminApiStatus');
                statusElement.innerHTML = '<div class="status info">Fetching queue configuration...</div>';

                fetch('/admin/queues')
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="status success">Queue configuration retrieved.</div>';
                        html += '<pre style="margin-top:10px; max-height:300px; overflow:auto; background:#f5f5f5; padding:10px; border-radius:6px;">';
                        html += escHtml(JSON.stringify(data, null, 2));
                        html += '</pre>';
                        statusElement.innerHTML = html;
                    })
                    .catch(error => {
                        statusElement.innerHTML = '<div class="status error">Failed to fetch queue configuration: ' + escHtml(error.message) + '</div>';
                    });
            }

            function getVectorStorageByUser() {
                const statusElement = document.getElementById('adminApiStatus');
                const maxRowsRaw = (document.getElementById('vectorStatsMaxRows') || {}).value;
                const maxRows = Math.max(1000, Math.min(2000000, parseInt(maxRowsRaw, 10) || 50000));
                const adminKeyEl = document.getElementById('vectorStatsAdminKey');
                const adminKey = adminKeyEl ? adminKeyEl.value.trim() : '';
                const filterUidEl = document.getElementById('vectorStatsFilterUserId');
                const filterUid = filterUidEl ? filterUidEl.value.trim() : '';

                statusElement.innerHTML = '<div class="status info">Loading all vectors (scanning Milvus; may take a while)…</div>';

                const headers = { 'Accept': 'application/json' };
                if (adminKey) headers['X-Admin-Key'] = adminKey;

                let url = '/admin/vector-storage-by-user?max_rows=' + encodeURIComponent(String(maxRows));
                if (filterUid) {
                    url += '&user_id=' + encodeURIComponent(filterUid);
                }

                fetch(url, { method: 'GET', headers: headers })
                    .then(async (response) => {
                        const text = await response.text();
                        let data = {};
                        if (text) {
                            try {
                                data = JSON.parse(text);
                            } catch (e) {
                                throw new Error('Invalid JSON: ' + text.slice(0, 200));
                            }
                        }
                        if (!response.ok) {
                            throw new Error(formatApiDetail(data) || ('HTTP ' + response.status));
                        }
                        return data;
                    })
                    .then((data) => {
                        const merged = data.merged_by_user_id || {};
                        const userIds = Object.keys(merged).sort();
                        let table = '<table style="width:100%; border-collapse:collapse; font-size:13px; margin-top:10px;">';
                        table += '<thead><tr style="background:#eee;">' +
                            '<th style="text-align:left;padding:8px;border:1px solid #ccc;">user_id</th>' +
                            '<th style="text-align:right;padding:8px;border:1px solid #ccc;">chunks (total)</th>' +
                            '<th style="text-align:right;padding:8px;border:1px solid #ccc;">files (doc coll.)</th>' +
                            '<th style="text-align:right;padding:8px;border:1px solid #ccc;">chunks (doc)</th>' +
                            '<th style="text-align:right;padding:8px;border:1px solid #ccc;">files (image)</th>' +
                            '<th style="text-align:right;padding:8px;border:1px solid #ccc;">chunks (image)</th>' +
                            '</tr></thead><tbody>';
                        userIds.forEach((uid) => {
                            const m = merged[uid] || {};
                            table += '<tr>' +
                                '<td style="padding:6px;border:1px solid #ccc;word-break:break-all;">' + uid.replace(/</g, '&lt;') + '</td>' +
                                '<td style="text-align:right;padding:6px;border:1px solid #ccc;">' + (m.chunk_count_total ?? 0) + '</td>' +
                                '<td style="text-align:right;padding:6px;border:1px solid #ccc;">' + (m.file_count_document_collection ?? 0) + '</td>' +
                                '<td style="text-align:right;padding:6px;border:1px solid #ccc;">' + (m.chunk_count_document_collection ?? 0) + '</td>' +
                                '<td style="text-align:right;padding:6px;border:1px solid #ccc;">' + (m.file_count_image_collection ?? 0) + '</td>' +
                                '<td style="text-align:right;padding:6px;border:1px solid #ccc;">' + (m.chunk_count_image_collection ?? 0) + '</td>' +
                                '</tr>';
                        });
                        table += '</tbody></table>';

                        const dc = data.document_collection || {};
                        const ic = data.image_collection || {};
                        const note = (data.note || '') + ' Document scan: ' + (dc.scanned_chunks ?? '?') + ' / cap ' + (dc.max_rows_cap ?? '?') +
                            ' chunks. ' + (ic.scanned_chunks != null ? ('Image scan: ' + ic.scanned_chunks + ' / cap ' + (ic.max_rows_cap || '') + '. ') : '');

                        let html = '<div class="status success"><strong>All vectors</strong> — ' + userIds.length + ' user id(s) in summary</div>';
                        html += '<p style="font-size:12px;color:#444;margin:8px 0;">' + note.replace(/</g, '&lt;') + '</p>';
                        html += table;
                        html += '<details style="margin-top:12px;"><summary style="cursor:pointer;">Raw JSON</summary>';
                        html += '<pre style="margin-top:8px; max-height:320px; overflow:auto; background:#f5f5f5; padding:10px; border-radius:6px; font-size:11px;">';
                        html += JSON.stringify(data, null, 2).replace(/</g, '&lt;');
                        html += '</pre></details>';
                        statusElement.innerHTML = html;
                    })
                    .catch((error) => {
                        statusElement.innerHTML = '<div class="status error">Load all vectors failed: ' +
                            String(error.message || error).replace(/</g, '&lt;') + '</div>';
                    });
            }

            function deleteAllDataForUserId() {
                const statusElement = document.getElementById('adminApiStatus');
                const uid = (document.getElementById('adminDeleteAllDataUserId') || {}).value.trim();
                const bkt = (document.getElementById('adminDeleteAllDataBucketId') || {}).value.trim();
                const conn = (document.getElementById('adminDeleteAllDataConnectionId') || {}).value.trim();
                const adminKeyEl = document.getElementById('vectorStatsAdminKey');
                const adminKey = adminKeyEl ? adminKeyEl.value.trim() : '';
                if (!uid) {
                    alert('Enter the User ID');
                    return;
                }
                let scopeMsg = 'ALL data for user "' + uid + '"';
                if (bkt && conn) {
                    scopeMsg = 'user "' + uid + '" with bucket "' + bkt + '" AND connection "' + conn + '"';
                } else if (bkt) {
                    scopeMsg = 'user "' + uid + '" bucket "' + bkt + '" only';
                } else if (conn) {
                    scopeMsg = 'user "' + uid + '" connection "' + conn + '" only';
                }
                if (!confirm('Delete stored vector data: ' + scopeMsg + '? This cannot be undone.')) {
                    return;
                }
                statusElement.innerHTML = '<div class="status info">Deleting vector data…</div>';
                const headers = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
                if (adminKey) headers['X-Admin-Key'] = adminKey;
                const body = {
                    user_id: uid,
                    confirm: 'purge-all-vectors-for-user',
                };
                if (bkt) body.bucket_id = bkt;
                if (conn) body.connectionId = conn;
                fetch('/admin/purge-user-vectors', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(body),
                })
                    .then(async (r) => {
                        const text = await r.text();
                        let data = {};
                        try { data = text ? JSON.parse(text) : {}; } catch (e) { throw new Error(text.slice(0, 300)); }
                        if (!r.ok) throw new Error(JSON.stringify(data.detail || data));
                        return data;
                    })
                    .then((data) => {
                        statusElement.innerHTML = '<div class="status success"><strong>Deleted all data for user ID</strong></div>' +
                            '<pre style="margin-top:10px;max-height:280px;overflow:auto;background:#f5f5f5;padding:10px;font-size:11px;">' +
                            JSON.stringify(data, null, 2).replace(/</g, '&lt;') + '</pre>';
                    })
                    .catch((err) => {
                        statusElement.innerHTML = '<div class="status error">Delete all data for user ID failed: ' +
                            String(err.message || err).replace(/</g, '&lt;') + '</div>';
                    });
            }

            function testRouting() {
                const statusElement = document.getElementById('adminApiStatus');
                const fileType = document.getElementById('routeTestFileType').value.trim();
                const filename = document.getElementById('routeTestFilename').value.trim();
                const sizeBytesRaw = document.getElementById('routeTestSizeBytes').value.trim();

                const sizeBytes = sizeBytesRaw ? parseInt(sizeBytesRaw, 10) : 0;

                statusElement.innerHTML = '<div class="status info">Testing routing...</div>';

                const body = {
                    file_type: fileType || null,
                    filename: filename || null,
                    size_bytes: isNaN(sizeBytes) ? 0 : sizeBytes,
                };

                fetch('/admin/route-test', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body),
                })
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="status success">Routing decision:</div>';
                        html += '<pre style="margin-top:10px; max-height:300px; overflow:auto; background:#f5f5f5; padding:10px; border-radius:6px;">';
                        html += escHtml(JSON.stringify(data, null, 2));
                        html += '</pre>';
                        statusElement.innerHTML = html;
                    })
                    .catch(error => {
                        statusElement.innerHTML = '<div class="status error">Routing test failed: ' + escHtml(error.message) + '</div>';
                    });
            }

            // Initialize
            updateFilesGrid();
            updateStats();
        </script>
    </body>
    </html>
    """
    return html_content

def create_fastapi_app():
    """Create FastAPI app with document processing integration."""
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Query, Header, Depends
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import tempfile
        import shutil
        from src.security import AuthError, Principal, authenticate, require_role, resolve_tenant
        from src.net_safety import UnsafeURLError, check_url, redact_url, safe_head

        app = FastAPI(title="Ultimate Document Processor", version="1.0.0")

        # CORS: the bundled UI is same-origin, so cross-origin access is off unless allowlisted.
        _cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
        if _cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=_cors_origins,
                allow_credentials="*" not in _cors_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        def _auth_http(e: AuthError) -> HTTPException:
            headers = {"WWW-Authenticate": "Bearer"} if e.status_code == 401 else None
            return HTTPException(status_code=e.status_code, detail=e.detail, headers=headers)

        def principal_dep(request: Request) -> Principal:
            try:
                return authenticate(request.headers)
            except AuthError as e:
                raise _auth_http(e)

        def _require(principal: Principal, role: str) -> None:
            try:
                require_role(principal, role)
            except AuthError as e:
                raise _auth_http(e)

        def _tenant(principal: Principal, requested, missing_detail: str = "userId is required") -> str:
            try:
                return resolve_tenant(principal, requested, missing_detail)
            except AuthError as e:
                raise _auth_http(e)
        
        # Initialize processor (lightweight); vector integration and semantic
        # pipeline are now initialized lazily on first use to keep the API
        # process thin and responsive under load.
        ui_processor = DocumentProcessingUI()

        logger.info("Ultimate UI initialized - server ready; models will load lazily on demand")
        
        @app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main UI."""
            return HTMLResponse(content=create_html_ui())
        
        # Removed duplicate task-status endpoint - using the one below
        
        # API Endpoints for Backend Integration
        def _estimate_csv_metrics(content: bytes) -> tuple[int, int]:
            if not content:
                return 0, 0
            rows = content.count(b"\n")
            est_chars = len(content)
            return rows, est_chars

        def _estimate_xlsx_chars(file_path: str) -> int:
            try:
                import zipfile
                total = 0
                with zipfile.ZipFile(file_path, "r") as zf:
                    for info in zf.infolist():
                        name = info.filename
                        if name.startswith("xl/worksheets/") or name.endswith("sharedStrings.xml"):
                            total += info.file_size
                return total
            except Exception:
                return 0

        def _user_queue_shard(user_id: str) -> int:
            """Stable shard 0..N-1 for user_id. Ensures same user always maps to same shard."""
            import hashlib
            h = hashlib.sha256((user_id or "").encode()).hexdigest()
            return int(h[:8], 16) % QUEUE_SHARD_COUNT

        def _choose_processing_queue(
            file_size_bytes: int,
            file_type: str,
            file_path: str = "",
            content: bytes = b"",
            prefer_large_if_unknown_size: bool = False,
            user_id: str = "",
        ) -> tuple[str, int]:
            """
            Central routing engine for file processing.

            Decides which Celery queue to use based on:
              - file type (spreadsheet, image, pdf, word, powerpoint, ocr)
              - approximate size (bytes / rows / chars)
              - prefer_large_if_unknown_size: when True and size cannot be determined
                (e.g. URL with no Content-Length), use the type-specific *_large queue
                for every category (spreadsheet, image, pdf, word, powerpoint, ocr).

            This is intentionally isolated routing logic; it does NOT touch any
            vector or semantic processing code.
            """
            est_chars = 0

            ft = (file_type or "").lower()
            path_lower = (file_path or "").lower()

            # ---- Helper: classify logical file category ----
            def _classify_category() -> str:
                # Spreadsheets
                if ft in {
                    "text/csv",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.oasis.opendocument.spreadsheet",
                } or path_lower.endswith((".csv", ".xls", ".xlsx", ".ods", ".xlsb")):
                    return "spreadsheet"

                # PowerPoint
                if ft in {
                    "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
                } or path_lower.endswith((".ppt", ".pptx", ".ppsx", ".pptm")):
                    return "powerpoint"

                # Word
                if ft in {
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                } or path_lower.endswith((".doc", ".docx")):
                    return "word"

                # PDF (documents)
                if ft == "application/pdf" or path_lower.endswith(".pdf"):
                    return "pdf"

                # Images
                if ft.startswith("image/") or path_lower.endswith(
                    (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg")
                ):
                    return "image"

                # OCR-only (scanned docs with unknown type) – currently treated as pdf
                return "other"

            category = _classify_category()

            # ---- Helper: compute size-based "large" decision ----
            is_csv = ft == "text/csv" or path_lower.endswith(".csv")
            is_xlsx = ft in {
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.oasis.opendocument.spreadsheet",
            } or path_lower.endswith((".xls", ".xlsx", ".ods", ".xlsb"))

            is_large = False

            if file_size_bytes and file_size_bytes >= LARGE_FILE_BYTES:
                is_large = True
            else:
                if is_csv and content:
                    rows, est_chars = _estimate_csv_metrics(content)
                    if rows >= LARGE_CSV_ROWS or est_chars >= LARGE_TEXT_CHARS:
                        is_large = True
                elif is_xlsx and file_path:
                    est_chars = _estimate_xlsx_chars(file_path)
                    if est_chars >= LARGE_TEXT_CHARS:
                        is_large = True
                elif prefer_large_if_unknown_size and not file_size_bytes:
                    # Size unknown (e.g. URL with no Content-Length): use type-specific large queue
                    is_large = True

            # ---- Map category + size to specific queues ----
            # Defaults (backward compatible)
            base_default = CELERY_DEFAULT_QUEUE
            base_large = CELERY_LARGE_QUEUE

            # Allow overriding via environment for flexibility
            spreadsheet_q = os.getenv("CELERY_SPREADSHEET_QUEUE", "ultimate_spreadsheet")
            spreadsheet_large_q = os.getenv("CELERY_SPREADSHEET_LARGE_QUEUE", "ultimate_spreadsheet_large")

            image_q = os.getenv("CELERY_IMAGE_QUEUE", "ultimate_image")
            image_large_q = os.getenv("CELERY_IMAGE_LARGE_QUEUE", "ultimate_image_large")

            pdf_q = os.getenv("CELERY_PDF_QUEUE", "ultimate_pdf")
            pdf_large_q = os.getenv("CELERY_PDF_LARGE_QUEUE", "ultimate_pdf_large")

            word_q = os.getenv("CELERY_WORD_QUEUE", "ultimate_word")
            word_large_q = os.getenv("CELERY_WORD_LARGE_QUEUE", "ultimate_word_large")

            ppt_q = os.getenv("CELERY_POWERPOINT_QUEUE", "ultimate_powerpoint")
            ppt_large_q = os.getenv("CELERY_POWERPOINT_LARGE_QUEUE", "ultimate_powerpoint_large")

            ocr_q = os.getenv("CELERY_OCR_QUEUE", "ultimate_ocr")
            ocr_large_q = os.getenv("CELERY_OCR_LARGE_QUEUE", "ultimate_ocr_large")

            if category == "spreadsheet":
                queue = spreadsheet_large_q if is_large else spreadsheet_q
            elif category == "image":
                queue = image_large_q if is_large else image_q
            elif category == "pdf":
                queue = pdf_large_q if is_large else pdf_q
            elif category == "word":
                queue = word_large_q if is_large else word_q
            elif category == "powerpoint":
                queue = ppt_large_q if is_large else ppt_q
            elif category == "ocr":
                queue = ocr_large_q if is_large else ocr_q
            else:
                # Unknown / plain text / other → OCR queue for fully distributed processing
                queue = ocr_large_q if is_large else ocr_q

            # User isolation: append shard only when QUEUE_SHARD_COUNT > 1
            # 0 or 1 = use base queues (ultimate_pdf, etc.) for workers on non-sharded queues
            if QUEUE_SHARD_COUNT > 1:
                shard = _user_queue_shard(user_id)
                queue = f"{queue}_u{shard}"

            return queue, est_chars

        def _get_url_content_length(url: str) -> int:
            try:
                resp = safe_head(url, timeout=10)
                if resp.status_code >= 400:
                    return 0
                length = resp.headers.get("Content-Length")
                return int(length) if length and length.isdigit() else 0
            except Exception:
                return 0

        @app.get("/admin/jobs/{user_id}")
        async def list_user_jobs(user_id: str, principal: Principal = Depends(principal_dep)):
            """Returns the list of jobs and their statuses for a specific user."""
            if not (principal.has("admin") or principal.tenant == user_id):
                raise HTTPException(status_code=403, detail="Not allowed to list jobs for this user")
            try:
                from src.job_registry import get_job_registry
                reg = get_job_registry()
                if not reg:
                    raise HTTPException(status_code=503, detail="Job registry not configured")
                    
                jobs = reg.list_jobs(user_id)
                return {"user_id": user_id, "jobs": jobs}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to list jobs for user {user_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/process")
        async def process_document_api(request: dict, principal: Principal = Depends(principal_dep)):
            """
            Process document from URL: route by type/size, enqueue to the same Celery queues as /process-file.
            All routing is done here—spreadsheet→ultimate_spreadsheet, pdf→ultimate_pdf,
            image→ultimate_image, word→ultimate_word, powerpoint→ultimate_powerpoint,
            other/OCR→ultimate_ocr. Large files go to *_large queues. Same implementation as /process-file.
            """
            _require(principal, "uploader")
            user_id = _tenant(principal, request.get("userId"))
            try:
                if not VECTOR_INTEGRATION_AVAILABLE:
                    raise HTTPException(status_code=500, detail="Background processing not available")

                image_url = request.get("fileUrl")
                file_id = request.get("fileId") or None  # Convert empty string to None
                file_type = request.get("fileType") or None  # Convert empty string to None

                # Optional bucket + path + connection scoping for ingestion
                bucket_id = request.get("bucketId") or request.get("bucket_id")
                path = request.get("path")
                # Prefer camelCase connectionId (matches userId/fileId/bucketId); connection_id still accepted
                connection_id = request.get("connectionId") or request.get("connection_id")

                if not image_url:
                    raise HTTPException(status_code=400, detail="fileUrl is required")
                try:
                    check_url(image_url)
                except UnsafeURLError as e:
                    raise HTTPException(status_code=400, detail=f"fileUrl not allowed: {e}")

                # Strict rule: fileId must be provided and non-empty; never auto-generate.
                if not file_id or (isinstance(file_id, str) and not file_id.strip()):
                    raise HTTPException(status_code=400, detail="fileId is required and cannot be empty")
                
                # user_id comes from the authenticated principal (see _tenant above)
                if hasattr(file_id, 'str'):
                    file_id = str(file_id)
                elif hasattr(file_id, '__str__'):
                    file_id = str(file_id)
                
                # Auto-detect file type if not provided or empty
                if not file_type or (isinstance(file_type, str) and not file_type.strip()):
                    logger.info(f"File type not provided, attempting auto-detection from URL: {redact_url(image_url)}")
                    
                    # FIRST: Try to detect from URL extension (most reliable)
                    filename = image_url.split('/')[-1].split('?')[0]  # Remove query params
                    extension = os.path.splitext(filename)[1].lower()
                    
                    # Map extensions to MIME types
                    extension_to_mime = {
                        '.pdf': 'application/pdf',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.ppt': 'application/vnd.ms-powerpoint',
                        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                        '.pptm': 'application/vnd.ms-powerpoint.presentation.macroEnabled.12',
                        '.ppsx': 'application/vnd.openxmlformats-officedocument.presentationml.slideshow',
                        '.potx': 'application/vnd.openxmlformats-officedocument.presentationml.template',
                        '.potm': 'application/vnd.ms-powerpoint.template.macroEnabled.12',
                        '.ppsm': 'application/vnd.ms-powerpoint.slideshow.macroEnabled.12',
                        '.odp': 'application/vnd.oasis.opendocument.presentation',
                        '.xls': 'application/vnd.ms-excel',
                        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        '.xlsm': 'application/vnd.ms-excel.sheet.macroEnabled.12',
                        '.xltx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.template',
                        '.xltm': 'application/vnd.ms-excel.template.macroEnabled.12',
                        '.xlsb': 'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
                        '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
                        '.csv': 'text/csv',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.bmp': 'image/bmp',
                        '.webp': 'image/webp',
                        '.tiff': 'image/tiff',
                        '.tif': 'image/tiff',
                        '.svg': 'image/svg+xml',
                        '.txt': 'text/plain',
                        '.html': 'text/html',
                        '.htm': 'text/html',
                        '.md': 'text/markdown',
                        '.rtf': 'application/rtf'
                    }
                    
                    if extension in extension_to_mime:
                        file_type = extension_to_mime[extension]
                        logger.info(f"Detected file type from URL extension '{extension}': {file_type}")
                    else:
                        # FALLBACK: Try to get Content-Type from URL (HEAD request)
                        try:
                            head_response = safe_head(image_url, timeout=5)
                            content_type = head_response.headers.get('Content-Type', '').split(';')[0].strip()
                            
                            if content_type:
                                file_type = content_type
                                logger.info(f"Detected file type from Content-Type header: {file_type}")
                        except Exception as e:
                            logger.warning(f"Failed to get Content-Type from URL: {e}")
                        
                        # Final fallback: Default to PDF if unknown
                        if not file_type:
                            file_type = 'application/pdf'
                            logger.warning(f"Unknown extension '{extension}', defaulting to PDF")
                
                # Ensure file_type is set (should be set by auto-detection or provided)
                if not file_type:
                    # Final fallback - try to detect from URL one more time
                    filename = image_url.split('/')[-1].split('?')[0]
                    extension = os.path.splitext(filename)[1].lower()
                    extension_to_mime = {
                        '.pdf': 'application/pdf',
                        '.doc': 'application/msword',
                        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        '.ppt': 'application/vnd.ms-powerpoint',
                        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                        '.pptm': 'application/vnd.ms-powerpoint.presentation.macroEnabled.12',
                        '.ppsx': 'application/vnd.openxmlformats-officedocument.presentationml.slideshow',
                        '.potx': 'application/vnd.openxmlformats-officedocument.presentationml.template',
                        '.potm': 'application/vnd.ms-powerpoint.template.macroEnabled.12',
                        '.ppsm': 'application/vnd.ms-powerpoint.slideshow.macroEnabled.12',
                        '.odp': 'application/vnd.oasis.opendocument.presentation',
                        '.xls': 'application/vnd.ms-excel',
                        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        '.xlsm': 'application/vnd.ms-excel.sheet.macroEnabled.12',
                        '.xltx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.template',
                        '.xltm': 'application/vnd.ms-excel.template.macroEnabled.12',
                        '.xlsb': 'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
                        '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
                        '.csv': 'text/csv',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.bmp': 'image/bmp',
                        '.webp': 'image/webp',
                        '.tiff': 'image/tiff',
                        '.tif': 'image/tiff',
                        '.svg': 'image/svg+xml',
                        '.txt': 'text/plain',
                        '.html': 'text/html',
                        '.md': 'text/markdown',
                        '.rtf': 'application/rtf'
                    }
                    file_type = extension_to_mime.get(extension, 'application/pdf')
                    logger.warning(f"File type not detected, using fallback: {file_type}")
                
                logger.info(
                    f"Processing file: file_id={file_id}, file_type={file_type}, "
                    f"user_id={user_id}, bucket_id={bucket_id}, path={path}, connection_id={connection_id}"
                )
                
                # Update workflow status to PROCESSING when task is submitted
                # This ensures backend sees "processing" status right away, even if task is queued
                from src.workflow_manager import get_workflow_manager
                workflow_manager = get_workflow_manager()
                workflow_manager.update_processing(file_id)
                logger.info(f"[WORKFLOW] Updated {file_id} to PROCESSING status (immediate, before task execution)")
                
                # Submit task to Celery for background processing (same routing as /process-file)
                url_size_bytes = _get_url_content_length(image_url) if image_url else 0
                url_filename = (image_url or "").split("/")[-1].split("?")[0].strip()
                # When size is unknown (URL with no Content-Length), use type-specific large queue for all types
                prefer_large = not url_size_bytes
                queue_name, est_chars = _choose_processing_queue(
                    url_size_bytes,
                    file_type or "",
                    file_path=url_filename,
                    content=b"",
                    prefer_large_if_unknown_size=prefer_large,
                    user_id=user_id or "",
                )
                
                # Handle empty strings for bucket_id and path (convert to None)
                bucket_id_value = bucket_id.strip() if (bucket_id and isinstance(bucket_id, str) and bucket_id.strip()) else None
                path_value = path.strip() if (path and isinstance(path, str) and path.strip()) else None
                conn_value = (
                    connection_id.strip()
                    if (connection_id and isinstance(connection_id, str) and connection_id.strip())
                    else None
                )

                task = process_ultimate_document_task.apply_async(
                    kwargs={
                        "url": image_url,
                        "file_id": file_id,
                        "user_id": user_id,
                        "original_filename": image_url.split('/')[-1],
                        "file_type": file_type,
                        "bucket_id": bucket_id_value,
                        "path": path_value,
                        "connection_id": conn_value,
                    },
                    queue=queue_name,
                )
                logger.info(
                    f"[QUEUE] Routed file_id={file_id} size={url_size_bytes} "
                    f"est_chars={est_chars} -> {queue_name}"
                )
                
                logger.info(f"Background processing task submitted: {task.id} for file {file_id}")
                
                # Register job in the lifecycle registry so /admin/jobs can track state.
                try:
                    from src.job_registry import get_job_registry as _get_registry
                    _reg = _get_registry()
                    if _reg:
                        _reg.register(user_id=user_id, file_id=file_id, task_id=task.id)
                except Exception as _reg_err:
                    logger.warning(f"[JOB REGISTRY] register() failed (non-fatal): {_reg_err}")
                
                return {
                    "success": True,
                    "file_id": file_id,
                    "task_id": task.id,
                    "status": "processing",
                    "message": "File picked by service and processing has started.",
                    "file_type": file_type,  # Include detected/provided file type in response
                    "queue": queue_name,
                    "file_id_auto_generated": not request.get("fileId") or (isinstance(request.get("fileId"), str) and not request.get("fileId").strip()),
                    "file_type_auto_detected": not request.get("fileType") or (isinstance(request.get("fileType"), str) and not request.get("fileType").strip()),
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "text_content": "",
                    "extraction_method": "background_processing",
                    "keywords": [],
                    "objects_detected": [],
                    "image_caption": "",
                    "visual_elements": []
                }
                
            except HTTPException:
                raise
            except Exception:
                logger.exception("API processing failed")
                raise HTTPException(status_code=500, detail="Processing failed; see server logs")
        
        @app.post("/process-file")
        async def process_file_upload(
            file: UploadFile = File(...),
            fileId: str = Form(None),
            userId: Optional[str] = Form(None),
            fileType: str = Form(None),
            bucketId: Optional[str] = Form(None),  # Allow None but preserve actual values
            path: Optional[str] = Form(None),  # Allow None but preserve actual values
            connectionId: Optional[str] = Form(None),
            principal: Principal = Depends(principal_dep),
        ):
            """
            Process uploaded file: save, route by type/size, enqueue to the right Celery queue.
            All routing is done here—spreadsheet→ultimate_spreadsheet, pdf→ultimate_pdf,
            image→ultimate_image, word→ultimate_word, powerpoint→ultimate_powerpoint,
            other/OCR→ultimate_ocr. Large files go to *_large queues. Clients just POST;
            the API distributes work so the system scales without hanging.
            """
            _require(principal, "uploader")
            userId = _tenant(principal, userId)
            try:
                if not VECTOR_INTEGRATION_AVAILABLE:
                    raise HTTPException(status_code=500, detail="Background processing not available")
                
                # Strict rule: fileId must be provided and non-empty; never auto-generate.
                if fileId and isinstance(fileId, str):
                    fileId = fileId.strip()
                if not fileId:
                    raise HTTPException(status_code=400, detail="fileId is required and cannot be empty")
                
                if hasattr(fileId, 'str'):
                    fileId = str(fileId)
                elif hasattr(fileId, '__str__'):
                    fileId = str(fileId)
                
                # Handle empty string for fileType
                if fileType and isinstance(fileType, str):
                    fileType = fileType.strip() or None
                
                # Auto-detect file type if not provided or empty
                if not fileType:
                    logger.info(f"File type not provided, attempting auto-detection from uploaded file")
                    
                    # First, try to use content type from upload
                    if file.content_type:
                        fileType = file.content_type.split(';')[0].strip()
                        logger.info(f"Detected file type from Content-Type header: {fileType}")
                    
                    # If still not detected, try to infer from filename
                    if not fileType:
                        filename = file.filename or ""
                        extension = os.path.splitext(filename)[1].lower()
                        
                        # Map extensions to MIME types (comprehensive list)
                        extension_to_mime = {
                            '.pdf': 'application/pdf',
                            '.doc': 'application/msword',
                            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            '.ppt': 'application/vnd.ms-powerpoint',
                            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                            '.pptm': 'application/vnd.ms-powerpoint.presentation.macroEnabled.12',
                            '.ppsx': 'application/vnd.openxmlformats-officedocument.presentationml.slideshow',
                            '.potx': 'application/vnd.openxmlformats-officedocument.presentationml.template',
                            '.potm': 'application/vnd.ms-powerpoint.template.macroEnabled.12',
                            '.ppsm': 'application/vnd.ms-powerpoint.slideshow.macroEnabled.12',
                            '.odp': 'application/vnd.oasis.opendocument.presentation',
                            '.xls': 'application/vnd.ms-excel',
                            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            '.xlsm': 'application/vnd.ms-excel.sheet.macroEnabled.12',
                            '.xltx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.template',
                            '.xltm': 'application/vnd.ms-excel.template.macroEnabled.12',
                            '.xlsb': 'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
                            '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.gif': 'image/gif',
                            '.bmp': 'image/bmp',
                            '.webp': 'image/webp',
                            '.tiff': 'image/tiff',
                            '.tif': 'image/tiff',
                            '.svg': 'image/svg+xml',
                            '.txt': 'text/plain',
                            '.html': 'text/html',
                            '.htm': 'text/html',
                            '.md': 'text/markdown',
                            '.rtf': 'application/rtf',
                        '.csv': 'text/csv'
                        }
                        
                        if extension in extension_to_mime:
                            fileType = extension_to_mime[extension]
                            logger.info(f"Detected file type from extension '{extension}': {fileType}")
                        else:
                            # Default fallback
                            fileType = file.content_type or 'application/octet-stream'
                            logger.warning(f"Unknown extension '{extension}', using: {fileType}")
                
                # Save uploaded file to shared volume location (accessible by both containers)
                # Use temp_uploads directory which is shared between containers
                temp_uploads_dir = os.getenv("UPLOAD_DIR", "/app/temp_uploads")
                os.makedirs(temp_uploads_dir, exist_ok=True)
                
                # Sanitize filename to avoid path issues
                safe_filename = file.filename or f"upload_{fileId}"
                # Remove any path separators and special characters
                safe_filename = os.path.basename(safe_filename).replace(" ", "_")
                # Ensure unique filename
                timestamp = int(time.time() * 1000)
                safe_filename = f"{timestamp}_{safe_filename}"
                
                temp_file_path = os.path.join(temp_uploads_dir, safe_filename)
                
                max_upload = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
                buf = bytearray()
                with open(temp_file_path, 'wb') as f:
                    while True:
                        piece = await file.read(1024 * 1024)
                        if not piece:
                            break
                        buf.extend(piece)
                        if len(buf) > max_upload:
                            break
                        f.write(piece)
                if len(buf) > max_upload:
                    os.remove(temp_file_path)
                    raise HTTPException(status_code=413, detail=f"File exceeds MAX_UPLOAD_BYTES ({max_upload} bytes)")
                content = bytes(buf)
                
                logger.info(f"File uploaded: {file.filename} ({len(content)} bytes) to {temp_file_path}")
                
                # Update workflow status to PROCESSING immediately when task is submitted
                # This ensures backend sees "processing" status right away, even if task is queued
                from src.workflow_manager import get_workflow_manager
                workflow_manager = get_workflow_manager()
                workflow_manager.update_processing(fileId)
                logger.info(f"[WORKFLOW] Updated {fileId} to PROCESSING status (immediate, before task execution)")
                
                # Submit task to Celery for background processing
                queue_name, est_chars = _choose_processing_queue(
                    len(content),
                    fileType or "",
                    file_path=temp_file_path,
                    content=content,
                    user_id=userId or "",
                )
                # Handle empty strings for bucketId and path (convert to None)
                bucket_id_value = bucketId.strip() if (bucketId and isinstance(bucketId, str) and bucketId.strip()) else None
                path_value = path.strip() if (path and isinstance(path, str) and path.strip()) else None
                conn_value = (
                    connectionId.strip()
                    if (connectionId and isinstance(connectionId, str) and connectionId.strip())
                    else None
                )

                task = process_ultimate_document_task.apply_async(
                    kwargs={
                        "url": None,
                        "file_path": temp_file_path,
                        "file_id": fileId,
                        "user_id": userId,
                        "original_filename": file.filename,
                        "file_type": fileType,
                        "bucket_id": bucket_id_value,
                        "path": path_value,
                        "connection_id": conn_value,
                    },
                    queue=queue_name,
                )
                logger.info(
                    f"[QUEUE] Routed file_id={fileId} size={len(content)} "
                    f"est_chars={est_chars} -> {queue_name}"
                )
                
                logger.info(f"Background processing task submitted: {task.id} for file {fileId}")

                # Register job in the lifecycle registry so /admin/jobs can track state.
                try:
                    from src.job_registry import get_job_registry as _get_registry
                    _reg = _get_registry()
                    if _reg:
                        _reg.register(user_id=userId, file_id=fileId, task_id=task.id)
                except Exception as _reg_err:
                    logger.warning(f"[JOB REGISTRY] register() failed (non-fatal): {_reg_err}")
                
                # Determine if file_id was auto-generated (starts with "auto_file_")
                file_id_was_auto_generated = fileId.startswith("auto_file_") if fileId else False
                
                # Determine if file_type was auto-detected (check if it matches content_type or was inferred)
                file_type_was_auto_detected = (
                    file.content_type and fileType == file.content_type
                ) or (
                    file.filename and fileType in [
                        'application/pdf', 'application/msword',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp',
                        'image/tiff', 'image/svg+xml', 'text/plain', 'text/html', 'text/markdown',
                        'application/rtf'
                    ] and os.path.splitext(file.filename)[1].lower() in [
                        '.pdf', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.bmp',
                        '.webp', '.tiff', '.tif', '.svg', '.txt', '.html', '.htm', '.md', '.rtf', '.csv'
                    ]
                )
                
                return {
                    "success": True,
                    "file_id": fileId,
                    "task_id": task.id,
                    "status": "processing",
                    "message": "File uploaded and processing has started.",
                    "filename": file.filename,
                    "file_type": fileType,
                    "queue": queue_name,
                    "file_id_auto_generated": file_id_was_auto_generated,
                    "file_type_auto_detected": file_type_was_auto_detected,
                    "file_size": len(content),
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "text_content": "",
                    "extraction_method": "background_processing",
                    "keywords": [],
                    "objects_detected": [],
                    "image_caption": "",
                    "visual_elements": []
                }
                
            except HTTPException:
                raise
            except Exception:
                logger.exception("Error in process_file_upload")
                raise HTTPException(status_code=500, detail="Upload processing failed; see server logs")
        
        @app.get("/task-status/{task_id}")
        async def get_task_status(task_id: str, user_id: str = None, principal: Principal = Depends(principal_dep)):
            """Get status of a background processing task. Supports Celery task_id (UUID) or file_id."""
            _require(principal, "reader")
            if principal.tenant:
                scope_uid = _tenant(principal, user_id)
            elif user_id:
                scope_uid = _tenant(principal, user_id)
            else:
                scope_uid = None  # service/admin key without a userId: Celery task ids only
            try:
                if not VECTOR_INTEGRATION_AVAILABLE:
                    raise HTTPException(status_code=500, detail="Task status not available")

                # If not a UUID, treat as file_id and resolve task_id from job registry
                celery_task_id = task_id
                import re
                uuid_pattern = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
                if not re.match(uuid_pattern, task_id):
                    from src.job_registry import get_job_registry
                    reg = get_job_registry()
                    uids_to_try = [scope_uid] if scope_uid else []
                    if reg:
                        for uid in uids_to_try:
                            job = reg.get_job(uid, task_id)
                            if job:
                                celery_task_id = job.get("task_id") or celery_task_id
                                st = job.get("status", "PENDING")
                                if st in ("SUCCESS", "FAILED", "FAILED_STALE", "CANCELLED"):
                                    tid = job.get("task_id", "")
                                    return {
                                        "task_id": tid or task_id, "file_id": task_id, "state": st,
                                        "status": st.lower().replace("_", " "),
                                        "status_message": f"Job status: {st}",
                                        "progress": 100 if st == "SUCCESS" else 0,
                                        "result": job if st == "SUCCESS" else None,
                                    }
                                elif celery_task_id != task_id:
                                    break

                if not principal.has("service"):
                    from src.job_registry import get_job_registry
                    _owner_reg = get_job_registry()
                    _owner = _owner_reg.task_owner(celery_task_id) if _owner_reg else None
                    if _owner != principal.tenant:
                        raise HTTPException(status_code=404, detail="Task not found")

                from src.ultimate_celery_app import celery_app
                task = celery_app.AsyncResult(celery_task_id)

                # When Celery returns PENDING, check job registry (Celery result may have expired).
                if task.state == "PENDING":
                    from src.job_registry import get_job_registry
                    reg = get_job_registry()
                    if reg:
                        job = reg.get_job_by_task_id(celery_task_id)
                        if job:
                            st = job.get("status", "PENDING")
                            if st in ("SUCCESS", "FAILED", "FAILED_STALE", "CANCELLED"):
                                return {
                                    "task_id": celery_task_id,
                                    "file_id": job.get("file_id", task_id),
                                    "state": st,
                                    "status": st.lower().replace("_", " "),
                                    "status_message": f"Job status: {st}",
                                    "progress": 100 if st == "SUCCESS" else 0,
                                    "result": job if st == "SUCCESS" else None,
                                }

                # Handle task state properly
                if task.state == 'PENDING':
                    response = {
                        'task_id': task_id,
                        'state': task.state,
                        'status': 'pending',
                        'status_message': 'Task is waiting to be processed...',
                        'progress': 0
                    }
                elif task.state == 'PROCESSING' or task.state == 'STARTED':
                    # Get task info which contains status and progress
                    task_info = task.info or {}
                    if isinstance(task_info, dict):
                        response = {
                            'task_id': task_id,
                            'state': task.state,
                            'status': 'processing',
                            'status_message': task_info.get('status', 'Processing...'),
                            'progress': task_info.get('progress', 0),
                            'message': task_info.get('status', 'Processing...')
                        }
                    else:
                        response = {
                            'task_id': task_id,
                            'state': task.state,
                            'status': 'processing',
                            'status_message': 'Processing...',
                            'progress': 0
                    }
                elif task.state == 'SUCCESS':
                    # Task completed successfully
                    task_result = task.result
                    response = {
                        'task_id': task_id,
                        'state': task.state,
                        'status': 'completed',
                        'status_message': 'Task completed successfully',
                        'progress': 100,
                        'result': task_result
                    }
                elif task.state == 'FAILURE':
                    # Task failed
                    error_info = task.info
                    error_msg = str(error_info) if error_info else 'Unknown error'
                    response = {
                        'task_id': task_id,
                        'state': task.state,
                        'status': 'failed',
                        'status_message': 'Task failed',
                        'progress': 0,
                        'error': error_msg,
                        'message': error_msg
                    }
                else:
                    # Unknown state
                    response = {
                        'task_id': task_id,
                        'state': task.state,
                        'status': 'unknown',
                        'status_message': f'Task state: {task.state}',
                        'progress': 0
                    }
                
                return response
                
            except HTTPException:
                raise
            except Exception:
                logger.exception("Task status API failed")
                raise HTTPException(status_code=500, detail="Task status failed; see server logs")

        @app.get("/admin/stuck-jobs")
        async def get_stuck_jobs(principal: Principal = Depends(principal_dep)):
            """
            Return list of tasks that appear stuck based on heartbeat data.

            This endpoint is read-only and does not modify any task state or
            vector/semantic behavior. It simply surfaces the output of
            src.scalability_utils.detect_stuck_jobs() for monitoring.
            """
            _require(principal, "admin")
            try:
                from src.scalability_utils import detect_stuck_jobs
                jobs = detect_stuck_jobs()
                return {
                    "success": True,
                    "count": len(jobs),
                    "jobs": jobs,
                }
            except ImportError as e:
                logger.warning(f"[ADMIN] scalability_utils not available: {e}")
                raise HTTPException(status_code=503, detail="Stuck-jobs monitoring not available (scalability_utils missing). Rebuild image or copy src/ into container.")
            except Exception as e:
                logger.error(f"[ADMIN] Failed to retrieve stuck jobs: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve stuck jobs: {e}")

        @app.get("/admin/queues")
        async def get_queue_config(principal: Principal = Depends(principal_dep)):
            """
            Return the current queue configuration for all file categories.

            This is a read-only view of how the routing engine is configured.
            """
            _require(principal, "admin")
            try:
                # Base queues
                base_default = CELERY_DEFAULT_QUEUE
                base_large = CELERY_LARGE_QUEUE

                cfg = {
                    "base": {
                        "default_queue": base_default,
                        "large_queue": base_large,
                        "large_file_bytes": LARGE_FILE_BYTES,
                        "large_csv_rows": LARGE_CSV_ROWS,
                        "large_text_chars": LARGE_TEXT_CHARS,
                    },
                    "spreadsheet": {
                        "queue": os.getenv("CELERY_SPREADSHEET_QUEUE", "ultimate_spreadsheet"),
                        "large_queue": os.getenv("CELERY_SPREADSHEET_LARGE_QUEUE", "ultimate_spreadsheet_large"),
                    },
                    "image": {
                        "queue": os.getenv("CELERY_IMAGE_QUEUE", "ultimate_image"),
                        "large_queue": os.getenv("CELERY_IMAGE_LARGE_QUEUE", "ultimate_image_large"),
                    },
                    "pdf": {
                        "queue": os.getenv("CELERY_PDF_QUEUE", "ultimate_pdf"),
                        "large_queue": os.getenv("CELERY_PDF_LARGE_QUEUE", "ultimate_pdf_large"),
                    },
                    "word": {
                        "queue": os.getenv("CELERY_WORD_QUEUE", "ultimate_word"),
                        "large_queue": os.getenv("CELERY_WORD_LARGE_QUEUE", "ultimate_word_large"),
                    },
                    "powerpoint": {
                        "queue": os.getenv("CELERY_POWERPOINT_QUEUE", "ultimate_powerpoint"),
                        "large_queue": os.getenv("CELERY_POWERPOINT_LARGE_QUEUE", "ultimate_powerpoint_large"),
                    },
                    "ocr": {
                        "queue": os.getenv("CELERY_OCR_QUEUE", "ultimate_ocr"),
                        "large_queue": os.getenv("CELERY_OCR_LARGE_QUEUE", "ultimate_ocr_large"),
                    },
                }
                return {"success": True, "queues": cfg}
            except Exception as e:
                logger.error(f"[ADMIN] Failed to build queue config: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to build queue config: {e}")

        @app.get("/admin/vector-storage-by-user")
        async def vector_storage_by_user(
            max_rows: Optional[int] = Query(
                default=None,
                description="Max Milvus entities to scan per collection (default: VECTOR_STATS_MAX_ROWS)",
                ge=1,
                le=2_000_000,
            ),
            user_id: Optional[str] = Query(
                default=None,
                description="If set, only scan rows for this user_id (accurate stats for that tenant)",
            ),
            principal: Principal = Depends(principal_dep),
        ):
            """
            UI label: **Load all vectors (summary)** — per-user file and chunk counts in document
            and image Milvus collections. Scan is bounded by max_rows.

            Requires an admin API key.
            """
            _require(principal, "admin")
            try:
                vi = ui_processor.get_vector_integration()
                if not vi:
                    raise HTTPException(status_code=503, detail="Vector integration not available")
                fu = (user_id or "").strip() or None
                data = vi.get_vector_storage_by_user(max_rows=max_rows, filter_user_id=fu)
                data["success"] = True
                data["note"] = (
                    "file_count = distinct file_id/source_file/object_id per collection; "
                    "chunk_count = Milvus entities (chunks). "
                    "merged_by_user_id aggregates both collections. "
                    + (
                        f"Scan filtered to user_id={fu!r} (counts are complete for that user up to max_rows). "
                        if fu
                        else "Without ?user_id=, the scan walks the whole collection only up to max_rows — "
                        "some tenants may be missing if the cap is exceeded. Use ?user_id=user888 for one tenant."
                    )
                )
                return data
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[ADMIN] vector-storage-by-user failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/admin/purge-user-vectors")
        async def admin_purge_user_vectors(
            payload: dict,
            principal: Principal = Depends(principal_dep),
        ):
            """
            Admin purge: removes Milvus rows (document + image collections) for user_id.
            Omit bucket_id and connectionId to wipe all S3 connections for that user. Set either or both
            to match ingest and narrow the purge (both filter as AND when both are set).
            For one file, use /delete-document.
            Requires an admin API key.

            Body JSON (connectionId preferred; connection_id accepted too):
              { "user_id": "...", "confirm": "purge-all-vectors-for-user",
                "bucket_id": "optional — same as ingest",
                "connectionId": "optional — same as POST /process" }
            """
            _require(principal, "admin")
            uid = str(payload.get("user_id") or payload.get("userId") or "").strip()
            if not uid:
                raise HTTPException(status_code=400, detail="user_id required")
            if payload.get("confirm") != "purge-all-vectors-for-user":
                raise HTTPException(
                    status_code=400,
                    detail='Set confirm to the literal string "purge-all-vectors-for-user".',
                )
            bkt_raw = payload.get("bucket_id") or payload.get("bucketId")
            bkt = str(bkt_raw).strip() if bkt_raw is not None and str(bkt_raw).strip() else None
            conn_raw = payload.get("connectionId") or payload.get("connection_id")
            conn = str(conn_raw).strip() if conn_raw is not None and str(conn_raw).strip() else None
            try:
                vi = ui_processor.get_vector_integration()
                if not vi:
                    raise HTTPException(status_code=503, detail="Vector integration not available")
                result = vi.purge_user_vectors(uid, bucket_id=bkt, connection_id=conn)
                if not result.get("success"):
                    raise HTTPException(status_code=500, detail=result)
                _CONTENT_SCAN_CACHE.pop(uid, None)
                try:
                    from src.ultimate_tasks import rebuild_user_metadata_index_task as _rebuild_meta

                    _rebuild_meta.delay(uid)
                    result["metadata_rebuild_queued"] = True
                except Exception as ex:
                    logger.warning("[ADMIN] metadata rebuild queue failed: %s", ex)
                    result["metadata_rebuild_queued"] = False
                    result["metadata_rebuild_error"] = str(ex)
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[ADMIN] purge-user-vectors failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/admin/route-test")
        async def route_test(payload: dict, principal: Principal = Depends(principal_dep)):
            """
            Test the routing engine without actually enqueuing a task.

            Accepts:
              - file_type (or fileType)
              - filename
              - size_bytes (or file_size_bytes)
            """
            _require(principal, "admin")
            try:
                file_type = payload.get("file_type") or payload.get("fileType") or ""
                filename = payload.get("filename") or ""
                size_bytes_raw = payload.get("size_bytes") or payload.get("file_size_bytes") or 0
                try:
                    size_bytes = int(size_bytes_raw)
                except Exception:
                    size_bytes = 0

                queue, est_chars = _choose_processing_queue(
                    file_size_bytes=size_bytes,
                    file_type=file_type or "",
                    file_path=filename or "",
                    content=b"",
                    user_id=payload.get("user_id", "route-test"),
                )

                return {
                    "success": True,
                    "queue": queue,
                    "estimated_chars": est_chars,
                    "inputs": {
                        "file_type": file_type,
                        "filename": filename,
                        "size_bytes": size_bytes,
                    },
                }
            except Exception as e:
                logger.error(f"[ADMIN] Route test failed: {e}")
                raise HTTPException(status_code=500, detail=f"Route test failed: {e}")


        
        def _normalize_temporal_phrasing(query: str) -> str:
            """
            Normalize common temporal phrasings so that equivalent queries
            follow the same path through the semantic pipeline.

            Examples:
              - 'documents from 2025'        -> 'what are the documents from 2025?'
              - 'documents from 2020-2025'   -> 'what are the documents from 2020-2025?'
            """
            if not query:
                return query
            q = query.strip()
            q_lower = q.lower()

            # documents from YYYY
            m = re.match(r"^documents\s+from\s+(\d{4})\s*\??$", q_lower)
            if m:
                year = m.group(1)
                return f"What are the documents from {year}?"

            # documents from YYYY-YYYY or YYYY–YYYY
            m = re.match(r"^documents\s+from\s+(\d{4})\s*[-–]\s*(\d{4})\s*\??$", q_lower)
            if m:
                y1, y2 = m.group(1), m.group(2)
                return f"What are the documents from {y1}-{y2}?"

            return query

        
        # Searches run on one dedicated thread, off the event loop: a slow search no longer freezes
        # health checks and other requests, and searches stay serialized exactly as before (the body
        # writes per-request state onto process-wide singletons, KD-SRCH-03, so do not raise this
        # above 1 until that is fixed).
        from concurrent.futures import ThreadPoolExecutor as _SearchPool
        _SEARCH_EXECUTOR = _SearchPool(max_workers=1, thread_name_prefix="search")

        @app.post("/search")
        async def search_vector_api(request: dict, principal: Principal = Depends(principal_dep)):
            """
            Unified semantic + vector search API.
            Uses MPNet embeddings for both documents & image captions.
            Cross-encoder re-ranking ensures zero false positives.

            Tenant scoping: use the same keys as ingest where possible — bucketId or bucket_id, path,
            connectionId (preferred) or connection_id (alias). Milvus filter field remains connection_id.
            """
            _require(principal, "reader")
            user_id = _tenant(principal, request.get("userId") or request.get("user_id"),
                              "userId or user_id is required")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_SEARCH_EXECUTOR, _search_blocking, request, user_id)

        def _search_blocking(request: dict, user_id: str):
            """Body of POST /search, unchanged; runs on _SEARCH_EXECUTOR."""
            try:
                raw_query = request.get("query")
                query = _normalize_temporal_phrasing(raw_query)
                limit = int(request.get("limit", 1000))  # Default to 1000, effectively no limit for most use cases
                # Production: default to "both" for time-bounded search; set SEARCH_DEFAULT_METHOD=semantic for legacy
                _default_method = os.getenv("SEARCH_DEFAULT_METHOD", "both").strip().lower() or "both"
                search_method = request.get("searchMethod") or request.get("search_method") or _default_method

                # Auto-upgrade "vector"-only mode to "both" when the query has structured
                # intent (date, person, location). Pure vector search has no awareness of
                # the metadata index, so date/entity queries return semantically irrelevant
                # results.  This does NOT affect queries that are genuinely vector-only
                # (no structured signals), where "vector" mode is kept as-is.
                if search_method == "vector":
                    try:
                        from src.semantic.query_enhancement import enhance_query as _eq_check
                        _, _qm_check = _eq_check(query)
                        _has_structure = bool(
                            _qm_check.get("full_date_query")
                            or _qm_check.get("date")
                            or _qm_check.get("date_range")
                            or _qm_check.get("month_year")
                            or _qm_check.get("persons")
                            or _qm_check.get("locations")
                        )
                        if _has_structure:
                            search_method = "both"
                            logger.info(
                                f"[AUTO-UPGRADE] search_method 'vector' → 'both' "
                                f"(temporal/entity intent detected for query: {query[:60]!r})"
                            )
                    except Exception:
                        pass

                # bucket + path + connection + min_score filters
                bucket_id = request.get("bucketId") or request.get("bucket_id")
                path = request.get("path")
                connection_id = request.get("connectionId") or request.get("connection_id")
                try:
                    min_score = float(request.get("min_score", 0.35))
                except Exception:
                    min_score = 0.35
                logger.info(f"[SEARCH API] query='{query}', search_method='{search_method}', user_id='{user_id}'")

                if not query:
                    raise HTTPException(status_code=400, detail="query is required")
                
                # Precompute enhanced query metadata once so both semantic and vector
                # paths can see consistent temporal/org/person intent.
                try:
                    from src.semantic.query_enhancement import enhance_query
                    _, query_meta = enhance_query(query)
                except Exception:
                    query_meta = {"original_query": query, "normalized_query": query}

                # Single string for Milvus + semantic embedding: must match enhance_query
                # canonical forms (e.g. bank statement vs bank statements, austin texas → austin).
                _effective_search_q = (
                    (query_meta or {}).get("vector_query")
                    or (query_meta or {}).get("normalized_query")
                    or query
                )
                _effective_search_q = (
                    str(_effective_search_q).strip()
                    if _effective_search_q
                    else query
                )
                _ranking_q = str(
                    (query_meta or {}).get("normalized_query") or query
                ).strip()

                # Store query metadata inside vector integration for temporal-first narrowing (Option C)
                vi = ui_processor.get_vector_integration()
                if vi:
                    try:
                        vi._last_query_meta = query_meta
                    except Exception:
                        pass
                
                # ✅ OPTIMIZED: Helper functions for semantic and vector search
                def run_semantic_search():
                    """Run semantic search in thread pool."""
                    if search_method not in ["semantic", "both"]:
                        return []
                    logger.info(
                        f"Applying semantic reranking for query: {_effective_search_q[:50]}..."
                    )
                    search_type_param = request.get("search_type", "text")
                    
                    if search_type_param == "text":
                        # Check if semantic_mode is requested (pure semantic, no hardcoded filters)
                        semantic_mode = request.get("semantic_mode", False) or request.get("pure_semantic", False)
                        # In BOTH mode, pass a soft per-call time budget down to the
                        # semantic engine so it can bail out early instead of blocking
                        # the API. Pure semantic mode (search_method == "semantic")
                        # keeps full-strength behavior (max_seconds=None).
                        from os import getenv
                        if search_method == "both":
                            # BOTH mode: fast, vector-first behavior.
                            # - Use a small semantic budget (80% of outer timeout).
                            # - Enable metadata-first router to catch authoritative hits
                            #   (e.g. specific dates/locations) while staying fast.
                            try:
                                budget = float(getenv("SEMANTIC_MAX_SECONDS_BOTH", "8.0")) * 0.8
                            except Exception:
                                budget = 6.0
                            semantic_results = get_semantic_engine().search_documents(
                                _effective_search_q,
                                top_k=min(limit, 50),
                                user_id=user_id,
                                semantic_mode=semantic_mode,
                                max_seconds=budget,
                                metadata_first=True,
                            )
                        else:
                            # Pure semantic mode: full-strength semantic router.
                            semantic_results = get_semantic_engine().search_documents(
                                _effective_search_q,
                                top_k=limit,
                                user_id=user_id,
                                semantic_mode=semantic_mode,
                            )
                    elif search_type_param == "image":
                        semantic_results = get_semantic_engine().search_images(
                            _effective_search_q, top_k=limit, user_id=user_id
                        )
                    else:
                        semantic_results = get_semantic_engine().search_documents(
                            _effective_search_q,
                            top_k=limit,
                            user_id=user_id,
                        )

                    # Apply bucket/path-level restrictions at semantic layer
                    if bucket_id:
                        semantic_results = [
                            r for r in semantic_results
                            if (r.get("metadata") or {}).get("bucket_id") == bucket_id
                        ]
                    if connection_id:
                        semantic_results = [
                            r for r in semantic_results
                            if (r.get("metadata") or {}).get("connection_id") == connection_id
                        ]
                    if path:
                        semantic_results = [
                            r for r in semantic_results
                            if (r.get("metadata") or {}).get("path", "").startswith(path)
                        ]

                    formatted_semantic = [
                        {
                            "file_id": r.get("file_id", r.get("source_file", "")),
                            "text": r.get("text", ""),
                            # Cap scores at 1.0 (100%)
                            "similarity_score": min(1.0, float(r.get("similarity_score", r.get("score", 0.0)))),
                            "confidence": min(1.0, float(r.get("confidence", r.get("similarity_score", r.get("score", 0.0))))),
                            "extraction_method": r.get("extraction_method", "semantic_search"),
                            "search_method": r.get("search_method", "semantic_text_search"),
                            "metadata": r.get("metadata", {}),  # Already includes years, full_dates, month_year, month_only
                            "_temporal_features": r.get("_temporal_features", {}),  # Temporal soft features
                            "source_file": r.get("source_file", r.get("file_id", ""))
                        }
                        for r in semantic_results
                    ]
                    logger.info(f"Semantic search returned {len(formatted_semantic)} document-level results")
                    return formatted_semantic
                
                def run_vector_search():
                    """Run vector search in thread pool."""
                    if search_method not in ["vector", "both"]:
                        return []
                    vi = ui_processor.get_vector_integration()
                    if not vi:
                        logger.warning("Vector integration not available for exact text search")
                        return []
                    
                    # Same canonical string as semantic path (vector_query / normalized_query / query)
                    vector_query_text = _effective_search_q
                    
                    # Ensure user_id is provided (required for search)
                    if not user_id:
                        raise HTTPException(status_code=400, detail="userId is required for vector search")
                    # Use limit as-is (removed cap to allow unlimited results)
                    vector_limit = limit
                    # Build filter conditions for bucket/path scoping
                    filter_conditions = {}
                    if bucket_id:
                        filter_conditions["bucket_id"] = bucket_id
                    if path:
                        filter_conditions["path"] = path
                    if connection_id:
                        filter_conditions["connection_id"] = connection_id
                    # IMPORTANT: Use recall-friendly threshold (0.15) for 100% match - partial/exact.
                    # Higher values filter out valid partial matches.
                    # 0.0 for 100% recall: surface all vector matches
                    vector_min_score = float(query_meta.get("vector_min_score", 0.0))
                    vector_results = vi.search_documents(
                        query=vector_query_text,
                        user_id=user_id,
                        limit=vector_limit,
                        similarity_threshold=vector_min_score,
                        filter_conditions=filter_conditions or None,
                    )
                    
                    # Attach temporal metadata to vector results
                    from src.semantic.temporal_engine import TemporalReasoningEngine
                    temporal_engine = TemporalReasoningEngine()
                    
                    formatted_vector = []
                    for result in vector_results:
                        file_id = result.file_id or result.metadata.get("source_file", "")
                        result_meta = result.metadata or {}
                        result_text = result.text or ""
                        
                        # Extract temporal metadata for vector results
                        doc_years = temporal_engine.extract_doc_years(file_id, result_meta, result_text)
                        profile = temporal_engine.extract_doc_temporal_profile(file_id, result_meta, result_text)
                        
                        # Ensure metadata has all temporal fields
                        if doc_years:
                            result_meta["years"] = doc_years
                        if not result_meta.get("full_dates"):
                            result_meta["full_dates"] = profile.get("explicit_dates", [])
                        if not result_meta.get("month_year") and profile.get("month_years"):
                            result_meta["month_year"] = profile.get("month_years", [])[0]
                        if not result_meta.get("month_only"):
                            # Extract month-only if available
                            month_only = result_meta.get("month")
                            if month_only:
                                result_meta["month_only"] = month_only
                        
                        # Compute temporal soft features
                        temporal_soft = temporal_engine.compute_temporal_soft_features(
                            text=result_text,
                            metadata=result_meta
                        )
                        
                        formatted_vector.append({
                            "file_id": file_id,
                            "text": result_text,
                            # Cap scores at 1.0 (100%)
                            "similarity_score": min(1.0, float(result.similarity_score or 0.0)),
                            "confidence": min(1.0, float(result.confidence or 0.0)),
                            "extraction_method": result.extraction_method or "vector_search",  # Use same method name as vector mode
                            "search_method": "vector",  # Use same method name as vector mode
                            "metadata": result_meta,  # Includes years, full_dates, month_year, month_only
                            "_temporal_features": temporal_soft,  # Temporal soft features
                            "source_file": file_id
                        })
                    logger.info(f"Vector search (exact text) returned {len(formatted_vector)} results")
                    return formatted_vector
                
                # ✅ INTELLIGENT ROUTING FOR BOTH MODE
                if search_method == "both":
                    # Query enhancement engine has already determined if semantic search is needed
                    needs_semantic = query_meta.get("needs_semantic", False)

                    # Run vector and semantic CONCURRENTLY so total latency =
                    # max(vector_time, semantic_time) instead of their sum.
                    # Previously vector ran first (blocking), then semantic ran
                    # sequentially — this caused 10-25s total for queries where
                    # one path was fast and the other slow.
                    import concurrent.futures as _cf_both
                    SEMANTIC_TIMEOUT = float(os.getenv("SEMANTIC_MAX_SECONDS_BOTH", "8.0"))
                    VECTOR_TIMEOUT   = float(os.getenv("VECTOR_MAX_SECONDS_BOTH",   "20.0"))

                    formatted_vector   = []
                    formatted_semantic = []

                    logger.info(f"[BOTH MODE] Running vector+semantic concurrently (needs_semantic={needs_semantic})")
                    with _cf_both.ThreadPoolExecutor(max_workers=2) as _exec:
                        _fut_vec = _exec.submit(run_vector_search)
                        _fut_sem = _exec.submit(run_semantic_search) if needs_semantic else None

                        try:
                            formatted_vector = _fut_vec.result(timeout=VECTOR_TIMEOUT)
                        except _cf_both.TimeoutError:
                            logger.warning(f"[BOTH MODE] Vector search exceeded {VECTOR_TIMEOUT}s; using semantic only")
                        except Exception as _ve:
                            logger.warning(f"[BOTH MODE] Vector search failed: {_ve}")

                        if _fut_sem is not None:
                            try:
                                formatted_semantic = _fut_sem.result(timeout=SEMANTIC_TIMEOUT)
                            except _cf_both.TimeoutError:
                                logger.warning(f"[BOTH MODE] Semantic search exceeded {SEMANTIC_TIMEOUT}s; using vector only")
                            except Exception as _se:
                                logger.warning(f"[BOTH MODE] Semantic search failed: {_se}")
                        else:
                            logger.info("[BOTH MODE] General keyword query — vector only")
                else:
                    # For "semantic" or "vector" modes: apply request timeout (production-ready)
                    try:
                        search_request_timeout = float(os.getenv("SEARCH_REQUEST_TIMEOUT", "60.0"))
                    except Exception:
                        search_request_timeout = 60.0
                    formatted_semantic = []
                    formatted_vector = []
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        if search_method in ["semantic", "both"]:
                            fut_sem = executor.submit(run_semantic_search)
                            try:
                                formatted_semantic = fut_sem.result(timeout=search_request_timeout)
                            except concurrent.futures.TimeoutError:
                                logger.warning(
                                    f"[SEARCH] Semantic search exceeded {search_request_timeout}s; "
                                    "returning partial/vector-only results"
                                )
                            except Exception as e:
                                logger.warning(f"[SEARCH] Semantic search failed: {e}")
                        if search_method in ["vector", "both"]:
                            fut_vec = executor.submit(run_vector_search)
                            try:
                                formatted_vector = fut_vec.result(timeout=search_request_timeout)
                            except concurrent.futures.TimeoutError:
                                logger.warning(
                                    f"[SEARCH] Vector search exceeded {search_request_timeout}s; "
                                    "returning partial results"
                                )
                            except Exception as e:
                                logger.warning(f"[SEARCH] Vector search failed: {e}")

                # ✅ Merge + Deduplicate + Final Sort (v5.0: Preserve temporal metadata)
                # Keep highest scoring result for each file_id (normalized for deduplication)
                merged = formatted_semantic + formatted_vector
                logger.info(f"[MERGE] Semantic: {len(formatted_semantic)}, Vector: {len(formatted_vector)}, Merged: {len(merged)}")
                
                # Normalize file_id for deduplication (handle underscores, hyphens, case differences)
                def normalize_file_id_for_dedup(file_id: str) -> str:
                    """Normalize file_id by removing special chars, lowercasing, for deduplication."""
                    if not file_id:
                        return ""
                    # Remove common suffixes and extensions
                    normalized = file_id.lower()
                    # Remove .pdf, .docx, etc. (case-insensitive)
                    normalized = re.sub(r'\.(pdf|docx?|txt)$', '', normalized, flags=re.IGNORECASE)
                    # Normalize separators: replace underscores, hyphens, and spaces with nothing
                    # This ensures "Misfits_Gaming_Agreement_4-11_" and "misfits_gaming_agreement_4_11__pdf" become the same
                    normalized = re.sub(r'[_\-\s]+', '', normalized)
                    return normalized
                
                unique_dict = {}
                for item in merged:
                    file_id = item.get("file_id", "") or item.get("source_file", "")
                    if not file_id:
                        logger.warning(f"[MERGE] Skipping item with no file_id: {item.keys()}")
                        continue
                    # Use normalized file_id for deduplication
                    normalized_file_id = normalize_file_id_for_dedup(file_id)
                    score = float(item.get("similarity_score") or item.get("score") or 0.0)
                    # Boost semantic/metadata scores because they represent higher precision intent matches
                    if item.get("search_method") in ["metadata_first", "semantic"]:
                        score += 0.15  # Semantic boost
                    
                    # Keep result with higher score, preserving all temporal metadata
                    existing_score = float(unique_dict.get(normalized_file_id, {}).get("similarity_score") or unique_dict.get(normalized_file_id, {}).get("score") or 0.0) if normalized_file_id in unique_dict else -1
                    existing_method = unique_dict.get(normalized_file_id, {}).get("search_method", "")
                    if existing_method in ["metadata_first", "semantic"]:
                        existing_score += 0.15

                    if normalized_file_id not in unique_dict or score > existing_score:
                        # Update the actual item score with the boosted value so it affects final sorting
                        if item.get("search_method") in ["metadata_first", "semantic"]:
                            # Store boosted score in the item
                            boosted_val = min(0.99, score)
                            item["similarity_score"] = boosted_val
                            item["confidence"] = boosted_val
                            if "score" in item:
                                item["score"] = boosted_val
                        
                        # Ensure temporal metadata is preserved in merged results
                        item_meta = item.get("metadata", {})
                        if not item_meta.get("years") and item_meta.get("years") is not None:
                            # Preserve years if present
                            pass
                        if not item_meta.get("full_dates") and item_meta.get("full_dates") is not None:
                            # Preserve full_dates if present
                            pass
                        if not item_meta.get("month_year") and item_meta.get("month_year") is not None:
                            # Preserve month_year if present
                            pass
                        if not item_meta.get("month_only") and item_meta.get("month_only") is not None:
                            # Preserve month_only if present
                            pass
                        # Ensure _temporal_features is preserved
                        if not item.get("_temporal_features"):
                            item["_temporal_features"] = {}
                        unique_dict[normalized_file_id] = item
                unique = list(unique_dict.values())
                logger.info(f"[MERGE] After dedup: {len(unique)} unique results")

                # BOTH mode merges vector + semantic; vector path does not apply semantic_pipeline
                # post-filters (required_keywords, regex, city anchors). Prune merged list here.
                if query_meta and search_method == "both":
                    _req_kws = [
                        str(k).strip().lower()
                        for k in (query_meta.get("required_keywords") or [])
                        if k and str(k).strip()
                    ]
                    _req_rx = query_meta.get("required_text_regex")
                    _anchors = [str(c).lower() for c in (query_meta.get("location_anchor_cities") or []) if c]
                    _meta_idx_merge = None
                    try:
                        _eng_merge = get_semantic_engine()
                        if _eng_merge and getattr(_eng_merge, "metadata_index", None):
                            _meta_idx_merge = _eng_merge.metadata_index
                            if user_id and hasattr(_eng_merge, "_ensure_metadata_index_for_user"):
                                _eng_merge._ensure_metadata_index_for_user(user_id)
                    except Exception:
                        pass
                    if _req_kws or _req_rx or _anchors:
                        from src.semantic.semantic_utils import tx_metro_snippet_has_wrong_peer_only

                        def _merged_passes_constraints(it: dict) -> bool:
                            md = it.get("metadata") or {}
                            raw_fn = str(md.get("original_filename") or md.get("filename") or "")
                            try:
                                dec_fn = _url_unquote(raw_fn.split("?")[0]).lower()
                            except Exception:
                                dec_fn = raw_fn.lower()
                            blob = (
                                (it.get("text") or "")
                                + " "
                                + raw_fn.lower()
                                + " "
                                + dec_fn
                                + " "
                                + str(it.get("file_id") or it.get("source_file") or "").lower()
                            ).lower()
                            if _meta_idx_merge:
                                _fid_m = it.get("file_id") or it.get("source_file") or ""
                                if _fid_m:
                                    try:
                                        _dm = _meta_idx_merge.get_metadata(_fid_m) or {}
                                        _ft_m = (_dm.get("full_text") or "")[:8000]
                                        if _ft_m:
                                            blob = blob + " " + _ft_m.lower()
                                    except Exception:
                                        pass
                            if _req_kws and not all(k in blob for k in _req_kws):
                                return False
                            if _req_rx and not re.search(_req_rx, blob, re.IGNORECASE):
                                return False
                            for city in _anchors:
                                if not re.search(rf"\b{re.escape(city)}\b", blob):
                                    return False
                            # Full-doc blob can include the anchor while the returned chunk is the wrong metro.
                            if tx_metro_snippet_has_wrong_peer_only(
                                _anchors,
                                it.get("text") or "",
                                dec_fn,
                                str(it.get("file_id") or it.get("source_file") or ""),
                            ):
                                return False
                            return True

                        _before_c = len(unique)
                        unique = [x for x in unique if _merged_passes_constraints(x)]
                        logger.info(
                            f"[MERGE] Constraint filter (keywords/regex/anchors): {_before_c} -> {len(unique)}"
                        )

                    # Topic + calendar year: raw vector hits often chase "2019" in spreadsheets
                    # and bury itinerary PDFs. When semantic path already returned hits, drop vector-only rows.
                    _ql = (query or "").strip()
                    _words = _ql.split()
                    if (
                        query_meta.get("date")
                        and len(_words) >= 5
                        and (query_meta.get("intent_type") or "").lower() == "temporal"
                    ):
                        _non_vec = [x for x in unique if (x.get("search_method") or "") != "vector"]
                        if len(_non_vec) >= 1:
                            _before_t = len(unique)
                            unique = _non_vec
                            logger.info(
                                f"[MERGE] Temporal topic query: dropped vector-only rows {_before_t} -> {len(unique)}"
                            )

                # 0.0 for 100% recall: never filter by score; lexical boost ranks strongest first
                MIN_API_SCORE = float(os.getenv("SEARCH_MIN_SCORE", "0.0"))
                MIN_IMAGE_SCORE = 0.50  # Higher threshold for images to reduce false positives
                
                filtered = []
                for item in unique:
                    # Get score, handling None values
                    score = item.get("similarity_score") or item.get("score") or 0.0
                    score = float(score) if score is not None else 0.0

                    is_lexical_match = (
                        item.get("search_method") == "metadata_first" or
                        item.get("extraction_method") == "metadata_hybrid"
                    )
                    
                    # Check if this is an image result
                    is_image = item.get("extraction_method", "").lower() in ["image_captioning", "image_ocr", "image_search"] or \
                              item.get("search_method", "").lower() in ["image_search", "image_semantic_search"]
                    
                    min_score = MIN_IMAGE_SCORE if is_image else MIN_API_SCORE
                    file_id_debug = item.get("file_id", "")[:50]
                    
                    # Always keep lexical matches, regardless of score
                    if is_lexical_match:
                        filtered.append(item)
                        logger.debug(f"[FILTER] Keeping lexical match: {file_id_debug} | score: {score:.3f} (bypassed threshold)")
                    elif score >= min_score:
                        filtered.append(item)
                        logger.debug(f"[FILTER] Keeping: {file_id_debug} | score: {score:.3f} >= {min_score:.3f}")
                    else:
                        logger.debug(f"[FILTER] Filtered out: {file_id_debug} | score: {score:.3f} < {min_score:.3f}")
                logger.info(f"[FILTER] After filtering: {len(filtered)} results (from {len(unique)} unique)")
                
                STOPWORDS = frozenset({"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or"})

                def _ranking_key(item: dict, q: str) -> tuple:
                    """Sort key: (phrase_match, filename_token_hits, _ranking_score).
                    Primary signal remains the boosted score; phrase_match and decoded-basename
                    token overlap break ties without reintroducing haystack term-count dominance."""
                    import urllib.parse as _rk_u

                    base = float(item.get("_ranking_score", item.get("similarity_score", item.get("score", 0.0))) or 0.0)
                    text = (item.get("text") or "") + " " + str((item.get("metadata") or {}).get("filename", ""))
                    text_lower = text.lower()
                    q_clean = " ".join(q.lower().split())
                    phrase_match = 1 if len(q_clean) >= 4 and q_clean in text_lower else 0
                    _m = item.get("metadata") or {}
                    _raw_fn = str(_m.get("original_filename") or _m.get("filename") or "")
                    _fn_base = _rk_u.unquote(_raw_fn.split("?")[0]).lower().split("/")[-1]
                    sig_tokens = [t for t in q_clean.split() if len(t) >= 4 and t not in STOPWORDS]
                    fn_hits = sum(1 for t in sig_tokens if t in _fn_base) if _fn_base else 0
                    return (phrase_match, fn_hits, base)


                if search_method == "vector":
                    # Pure vector mode: apply constraint boost then sort
                    _vm = {}
                    try:
                        from src.semantic.query_enhancement import enhance_query as _eq
                        _, _vm = _eq(query)
                        _apply_constraint_boost(filtered, _vm)
                    except Exception:
                        pass
                    final = sorted(
                            filtered,
                            key=lambda x: _ranking_key(x, _ranking_q),
                            reverse=True,
                        )[:limit]
                else:
                    try:
                        from src.semantic.semantic_components import validate_constraints
                        from src.semantic.query_enhancement import enhance_query
                        
                        # Get enhanced query metadata
                        _, query_meta = enhance_query(query)
                        
                        validated = []
                        for item in filtered:
                            file_meta = item.get("metadata", {})
                            doc_text = item.get("text", "")
                            file_id = item.get("file_id", "")

                            is_vector_hit = (item.get("search_method") or "").lower() == "vector"
                            is_lexical_match = (
                                item.get("search_method") == "metadata_first" or
                                item.get("extraction_method") == "metadata_hybrid"
                            )
                            
                            if is_vector_hit:
                                import urllib.parse as _vm_urlparse
                                _q_lower_v = (query_meta.get("original_query", "") or "").lower()
                                _vm_meta = item.get("metadata") or {}
                                _vm_raw_fn = str(_vm_meta.get("original_filename") or _vm_meta.get("filename") or "")
                                _vm_fn = _vm_urlparse.unquote(_vm_raw_fn.split("?")[0]).lower()
                                _vm_score = float(item.get("similarity_score") or 0)

                                # ── Temporal quality filter ─────────────────────────────────────
                                # For full-date queries (month+day+year like "June 1, 2020"),
                                # vector hits with low similarity are nearly always false positives —
                                # the model finds "2020" in the embedding space but the document has
                                # nothing to do with that specific date.  Apply a strict floor.
                                # For year-only queries, use a softer floor.
                                _has_full_date_v = bool(query_meta.get("full_date_query") and query_meta.get("date"))
                                _has_year_only_v = bool(query_meta.get("date") and not _has_full_date_v)
                                if _has_full_date_v and _vm_score < 0.50:
                                    logger.debug(f"[TEMPORAL-FILTER-V] Low-score vector for full-date query (sim={_vm_score:.3f}): {_vm_fn[:50]}")
                                    continue
                                if _has_year_only_v and _vm_score < 0.35:
                                    logger.debug(f"[TEMPORAL-FILTER-V] Low-score vector for year query (sim={_vm_score:.3f}): {_vm_fn[:50]}")
                                    continue

                                # ── Person-query quality filter ──────────────────────────────────
                                # Drop images (jpg/png/gif) that have no person name in their
                                # filename, and drop very low-score (<0.30) vector hits unless
                                # the person name is in the filename.
                                _is_person_q_v = bool(
                                    query_meta.get("intent_type") == "entity"
                                    and query_meta.get("persons")
                                )
                                if _is_person_q_v:
                                    _persons_v = [p.lower() for p in (query_meta.get("persons") or []) if len(p) >= 4]
                                    _fn_has_person_v = any(p in _vm_fn for p in _persons_v)
                                    _IMAGE_EXTS_V = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.tiff'}
                                    _vm_ext_v = ("." + _vm_fn.rsplit(".", 1)[-1]) if "." in _vm_fn else ""
                                    if _vm_ext_v in _IMAGE_EXTS_V and not _fn_has_person_v:
                                        logger.debug(f"[PERSON-FILTER-V] Dropped image without person in filename: {_vm_fn[:50]}")
                                        continue
                                    if _vm_score < 0.30 and not _fn_has_person_v:
                                        logger.debug(f"[PERSON-FILTER-V] Dropped low-score (sim={_vm_score:.3f}): {_vm_fn[:50]}")
                                        continue

                                # ── NDA filter ───────────────────────────────────────────────────
                                # Prevent board decks, retainer agreements, HR policies from
                                # appearing in NDA queries.
                                _NDA_TERMS_V = ['nda', 'ndas', 'mnda', 'non-disclosure', 'non disclosure', 'nondisclosure']
                                if any(term in _q_lower_v for term in _NDA_TERMS_V):
                                    _vm_text = (doc_text or "").lower()
                                    _FN_NDA_SIGNALS = ['nda', 'non-disclosure', 'nondisclosure', 'non disclosure']
                                    _fn_nda = any(t in _vm_fn for t in _FN_NDA_SIGNALS)
                                    _TEXT_NDA_PHRASES = ['non-disclosure agreement', 'nondisclosure agreement',
                                                         'non disclosure agreement', 'this nda', 'under this nda',
                                                         'pursuant to this nda', 'the nda']
                                    _text_nda = any(p in _vm_text for p in _TEXT_NDA_PHRASES)
                                    if not (_fn_nda or _text_nda):
                                        logger.debug(f"[NDA-FILTER-V] Removed non-NDA vector hit: {file_id[:50]}")
                                        continue
                                validated.append(item)
                                continue

                            if is_lexical_match:
                                from src.semantic.semantic_components import validate_temporal_constraints, validate_constraints
                                
                                # Check temporal constraints (strict for temporal queries)
                                temporal_ok = validate_temporal_constraints(
                                    query_meta, file_meta, doc_text, from_lexical_index=True
                                )

                                person_ok = True
                                if query_meta.get("persons") and any(
                                    phrase in (query_meta.get("original_query", "") or "").lower() 
                                    for phrase in ["signed by", "documents by", "files by", "docs by", "agreements by", "contracts by", "signed between", "executed by"]
                                ):
                                    # For person queries, use full validation to check person presence
                                    person_ok = validate_constraints(query_meta, file_meta, doc_text=doc_text, file_id=file_id)
                                
                                nda_ok = True
                                q_text_lower = (query_meta.get("original_query", "") or "").lower()
                                # NDA queries: ANY query containing NDA terms should enforce
                                # NDA validation on lexical (metadata_first) matches.
                                # This prevents METAFESTO agreements from appearing when user
                                # queries "NDA signed in austin texas" (they satisfy location
                                # but are NOT NDAs).
                                _NDA_TERMS = [
                                    'nda', 'ndas', 'mnda', 'mndas',
                                    'non-disclosure', 'non disclosure', 'nondisclosure',
                                    'mutual non-disclosure', 'mutual non disclosure',
                                ]
                                is_nda_query = any(term in q_text_lower for term in _NDA_TERMS)
                                if is_nda_query:
                                    # Check the decoded filename for NDA signal (fast path)
                                    _item_meta_nda = item.get("metadata") or {}
                                    _raw_fn_nda = str(
                                        _item_meta_nda.get("original_filename")
                                        or _item_meta_nda.get("filename") or ""
                                    )
                                    import urllib.parse as _nda_urlparse
                                    _decoded_fn_nda = _nda_urlparse.unquote(_raw_fn_nda.split("?")[0]).lower()
                                    _FN_NDA = ['nda', 'non-disclosure', 'nondisclosure', 'non disclosure']
                                    _fn_has_nda = any(t in _decoded_fn_nda for t in _FN_NDA)
                                    _vm_text2 = (doc_text or "").lower()
                                    _TEXT_NDA_PH = ['non-disclosure agreement', 'nondisclosure agreement',
                                                    'non disclosure agreement', 'this nda', 'under this nda',
                                                    'the nda']
                                    _text_has_nda = any(p in _vm_text2 for p in _TEXT_NDA_PH)
                                    if not (_fn_has_nda or _text_has_nda):
                                        nda_ok = False
                                        logger.debug(f"[NDA-FILTER] Removed non-NDA from NDA query: {file_id[:50]}")
                                
                                if temporal_ok and person_ok and nda_ok:
                                    validated.append(item)
                                else:
                                    logger.info(f"[VALIDATOR] Removed lexical match: temporal_ok={temporal_ok}, person_ok={person_ok}, nda_ok={nda_ok}, file_id={file_id}")
                            else:
                                # For semantic matches and vector hits under strong constraints, use full validation
                                if validate_constraints(query_meta, file_meta, doc_text=doc_text, file_id=file_id):
                                    validated.append(item)
                                else:
                                    logger.debug(f"[VALIDATOR] Removed: {file_id}")
                        
                        logger.info(
                            f"[VALIDATOR] After constraint validation: {len(validated)} results (from {len(filtered)} filtered)"
                        )

                        # ── Filename-NDA Supplement ──────────────────────────────────────────
                        # If query has NDA terms, do a Milvus scalar scan for files with
                        # "nda" or "non-disclosure" in their filename and inject any that
                        # are missing from the validated pool.  This recovers files like
                        # "eightM Corp Storage Chain LLC_mutual_NDA.pdf" that vector search
                        # missed because the filename is URL-encoded in the embeddings.
                        # Do NOT inject generic NDAs when the query requires a specific brand keyword
                        # or regex enforcement — that would undo precision.
                        _q_nda_supp = (query_meta.get("original_query", "") or "").lower()
                        _NDA_TERMS_SUPP = ['nda', 'ndas', 'mnda', 'non-disclosure', 'non disclosure', 'nondisclosure']
                        _skip_nda_supp = bool(
                            query_meta.get("required_keywords")
                            or query_meta.get("required_text_regex")
                            or query_meta.get("location_anchor_cities")
                        )
                        if (
                            any(t in _q_nda_supp for t in _NDA_TERMS_SUPP)
                            and vi
                            and user_id
                            and not _skip_nda_supp
                        ):
                            try:
                                import urllib.parse as _supp_urlparse
                                _existing_fids = {r.get("file_id") for r in validated}
                                # Use search_documents (correct method name) with "mutual nda"
                                # as a focused query to surface all mutual NDA files
                                _supp_results = vi.search_documents(
                                    query="non-disclosure agreement NDA mutual",
                                    user_id=user_id,
                                    limit=40,
                                    similarity_threshold=0.30,
                                )
                                for _sr in (_supp_results or []):
                                    _sr_dict = _sr if isinstance(_sr, dict) else (
                                        _sr.__dict__ if hasattr(_sr, '__dict__') else {}
                                    )
                                    # VectorSearchResult might use different field names
                                    _sr_fid = (_sr_dict.get("file_id") or
                                               getattr(_sr, "file_id", None) or
                                               _sr_dict.get("id", ""))
                                    if _sr_fid in _existing_fids:
                                        continue
                                    _sr_meta = (_sr_dict.get("metadata") or
                                                getattr(_sr, "metadata", {}) or {})
                                    _sr_raw_fn = str(_sr_meta.get("original_filename") or _sr_meta.get("filename") or "")
                                    _sr_fn = _supp_urlparse.unquote(_sr_raw_fn.split("?")[0]).lower()
                                    if any(t in _sr_fn for t in ['nda', 'non-disclosure', 'nondisclosure']):
                                        # Convert to dict format for consistency
                                        if not isinstance(_sr_dict, dict) or "similarity_score" not in _sr_dict:
                                            _sr_dict = {
                                                "file_id": _sr_fid,
                                                "similarity_score": getattr(_sr, "score", 0.5),
                                                "text": getattr(_sr, "text", ""),
                                                "metadata": _sr_meta,
                                                "search_method": "filename_supplement",
                                            }
                                        else:
                                            _sr_dict["search_method"] = "filename_supplement"
                                        validated.append(_sr_dict)
                                        _existing_fids.add(_sr_fid)
                                        logger.info(f"[NDA-SUPP] Added NDA file: {_sr_fn[:60]}")
                            except Exception as _supp_err:
                                logger.debug(f"[NDA-SUPP] Supplement scan failed: {_supp_err}", exc_info=True)
                        # ── End Filename-NDA Supplement ──────────────────────────────────────

                        # ── General Filename Keyword Supplement ───────────────────────────────
                        # For short keyword queries (1-3 words, no NLP signals like location/date),
                        # run search_lexical to find ALL files whose name contains the keyword.
                        # This ensures all matching files are returned even if they have sparse content.
                        _q_supp2 = (query_meta.get("original_query") or query or "").strip()
                        _q_supp2_words = [w for w in _q_supp2.lower().split()
                                          if len(w) >= 3 and w not in {
                                              'the', 'and', 'for', 'all', 'any', 'get', 'find',
                                              'show', 'list', 'me', 'my', 'can', 'you', 'please',
                                              'files', 'file', 'documents', 'document', 'docs',
                                              'related', 'about', 'with', 'from', 'search',
                                          }]
                        # Strip file-format tokens so file-type queries do not contaminate
                        # the filename scan with format terms.
                        _FILENAME_QUERY_TYPE_WORDS = frozenset({
                            'excel', 'excels', 'xlsx', 'xls', 'spreadsheet', 'spreadsheets', 'csv',
                            'powerpoint', 'pptx', 'ppt', 'presentation', 'presentations',
                            'slides', 'slideshow', 'deck', 'decks', 'word', 'docx', 'doc',
                            'pdf', 'image', 'images', 'photo', 'photos', 'jpeg', 'jpg', 'png',
                            'gif', 'text', 'txt', 'document', 'documents',
                        })
                        _entity_lex_tokens = [
                            w for w in _q_supp2_words
                            if w not in _FILENAME_QUERY_TYPE_WORDS
                        ]
                        _lex_filename_query = (
                            " ".join(_entity_lex_tokens).strip()
                            if _entity_lex_tokens
                            else _q_supp2.strip()
                        )
                        # Block general lexical only for location/date/NDA, or "type-only"
                        # queries with no entity token (e.g. "excel files" with no brand).
                        # If file_extensions is set BUT we still have entity tokens,
                        # run filename scan on the entity.
                        _has_blocking_signals = bool(
                            query_meta.get("locations") or
                            query_meta.get("dates") or
                            any(t in _q_supp2.lower() for t in ['nda', 'non-disclosure'])
                        )
                        if (
                            query_meta.get("file_extensions")
                            and not _entity_lex_tokens
                        ):
                            _has_blocking_signals = True
                        # Allow up to 8 content words; multi-token scan uses type words for
                        # extension-aware lexical matching (see search_lexical_on_chunks).
                        _do_lex_supp = (
                            vi and user_id and
                            1 <= len(_q_supp2_words) <= 8 and
                            not _has_blocking_signals and
                            len(_lex_filename_query) >= 2
                        )
                        # _added_lex must be defined before the try so _do_content_scan
                        # can reference it even if search_lexical raises an exception.
                        _added_lex = 0
                        if _do_lex_supp:
                            try:
                                import urllib.parse as _lex_urlparse
                                _existing_fids2 = {r.get("file_id") for r in validated}
                                # Also track by decoded filename to suppress duplicate ingestions
                                def _lex_fn(item):
                                    meta = (item.get("metadata") or {} if isinstance(item, dict)
                                            else getattr(item, "metadata", {}) or {})
                                    raw = meta.get("original_filename") or meta.get("filename") or ""
                                    return _lex_urlparse.unquote(str(raw).split("?")[0]).split("/")[-1].strip().lower()
                                _existing_fns2 = {_lex_fn(r) for r in validated if _lex_fn(r)}
                                # Always use search_lexical (Milvus metadata + CamelCase/URL
                                # normalization). A prior cache-based scan diverged from this
                                # logic and dropped valid filename hits after re-ingest.
                                _lex_limit = min(4000, max(500, int(limit) * 6))
                                _lex_scan_multi = " ".join(_q_supp2_words)
                                _lex_queries = []
                                if _lex_filename_query:
                                    _lex_queries.append(_lex_filename_query.strip())
                                if (
                                    _lex_scan_multi.strip()
                                    and _lex_scan_multi.strip() != _lex_filename_query.strip()
                                    and len(_lex_scan_multi.strip()) >= 4
                                ):
                                    _lex_queries.append(_lex_scan_multi.strip())
                                _lex_seen_fids: set = set()
                                _lex_results = []
                                for _lxq in dict.fromkeys(_lex_queries):
                                    _lr_part = vi.search_lexical(
                                        _lxq,
                                        user_id=user_id,
                                        limit=_lex_limit,
                                    )
                                    for _lr in _lr_part or []:
                                        _lrf = (
                                            _lr.get("file_id")
                                            if isinstance(_lr, dict)
                                            else getattr(_lr, "file_id", "")
                                        )
                                        if _lrf and _lrf not in _lex_seen_fids:
                                            _lex_seen_fids.add(_lrf)
                                            _lex_results.append(_lr)
                                for _lr in _lex_results:
                                    _lr_dict = _lr if isinstance(_lr, dict) else {
                                        "file_id": getattr(_lr, "file_id", ""),
                                        "similarity_score": float(getattr(_lr, "similarity_score", 0.5) or 0.5),
                                        "text": getattr(_lr, "text", ""),
                                        "metadata": getattr(_lr, "metadata", {}),
                                        "search_method": "lexical_supplement",
                                    }
                                    _lr_fid = _lr_dict.get("file_id") or getattr(_lr, "file_id", "")
                                    _lr_fn = _lex_fn(_lr)
                                    # Skip if same file_id OR same filename (handles duplicate ingestions)
                                    if _lr_fid and _lr_fid not in _existing_fids2 and _lr_fn not in _existing_fns2:
                                        _lr_dict["search_method"] = "lexical_supplement"
                                        if not isinstance(_lr, dict):
                                            _lr_dict["file_id"] = _lr_fid
                                            _lr_dict["metadata"] = getattr(_lr, "metadata", {})
                                        validated.append(_lr_dict)
                                        _existing_fids2.add(_lr_fid)
                                        _existing_fns2.add(_lr_fn)
                                        _added_lex += 1
                                if _added_lex:
                                    logger.info(
                                        f"[LEX-SUPP] Added {_added_lex} files filename_q="
                                        f"{_lex_filename_query[:50]!r}"
                                    )
                            except Exception as _lex_err:
                                logger.debug(f"[LEX-SUPP] Lexical supplement failed: {_lex_err}", exc_info=True)
                        # ── End General Filename Keyword Supplement ───────────────────────────

                        # ── Content Keyword Scan Supplement ──────────────────────────────────
                        # Triggered when the lexical (filename) supplement found 0 results AND
                        # the query is a meaningful keyword/phrase (1-4 words, all ≥3 chars).
                        #
                        # Single-keyword use case (e.g. "Coinstore"):
                        #   Finds the keyword verbatim in chunk text, adds with high score.
                        #
                        # Multi-word use case (e.g. "1100 W Town and Country Rd", "March 5, 2025"):
                        #   All keywords must appear in the SAME chunk (AND-match).
                        #
                        # Performance: uses module-level cache (_CONTENT_SCAN_CACHE) so the
                        # 16K-chunk Milvus fetch only happens once per user per 3 minutes.
                        _cs_is_temporal = bool(
                            query_meta.get("full_date_query") or query_meta.get("date")
                        )
                        _cs_starts_with_num = bool(
                            _q_supp2_words and _q_supp2_words[0].isdigit()
                        )
                        # Person-style scan: NER multi-word person, OR two alpha tokens
                        # with no location/date (covers "chris dominguez" when NER is thin).
                        _cs_ner_person = bool(
                            query_meta.get("persons")
                            and any(
                                " " in (p or "").strip()
                                for p in (query_meta.get("persons") or [])
                            )
                        )
                        _cs_two_name_heuristic = bool(
                            len(_q_supp2_words) == 2
                            and all(w.isalpha() and len(w) >= 3 for w in _q_supp2_words)
                            and not query_meta.get("locations")
                            and not _cs_is_temporal
                            and not _cs_starts_with_num
                        )
                        _cs_is_person = _cs_ner_person or _cs_two_name_heuristic
                        _cs_multi_ok = (
                            len(_q_supp2_words) == 1 or  # single keyword
                            _cs_starts_with_num or        # street address
                            _cs_is_temporal or            # date query
                            _cs_is_person                 # person name query
                        )
                        _do_content_scan = (
                            _do_lex_supp and
                            _added_lex == 0 and
                            _cs_multi_ok and
                            1 <= len(_q_supp2_words) <= 4 and
                            all(len(w) >= 3 for w in _q_supp2_words) and
                            (len(_q_supp2_words) > 1 or len(_q_supp2_words[0]) >= 5) and
                            vi and user_id and
                            hasattr(vi, 'vector_db')
                        )
                        if _do_content_scan:
                            try:
                                import urllib.parse as _cs_urlparse
                                import time as _cs_time
                                import re as _cs_re
                                _cs_keywords = [w.lower() for w in _q_supp2_words]
                                # Use module-level cache to avoid re-fetching 16K chunks.
                                _cs_cache_entry = _CONTENT_SCAN_CACHE.get(user_id)
                                if (_cs_cache_entry and
                                        (_cs_time.time() - _cs_cache_entry['ts']) < _CONTENT_SCAN_TTL):
                                    _cs_chunks = _cs_cache_entry['chunks']
                                    logger.debug(f"[CONTENT-SCAN] cache hit ({len(_cs_chunks)} chunks)")
                                else:
                                    _cs_chunks = vi.vector_db.query_all_chunks(
                                        user_id=user_id,
                                        limit=16384,
                                        include_text=True,
                                    )
                                    _CONTENT_SCAN_CACHE[user_id] = {
                                        'ts': _cs_time.time(), 'chunks': _cs_chunks
                                    }
                                    logger.info(
                                        f"[CONTENT-SCAN] fetched+cached {len(_cs_chunks or [])} chunks"
                                    )
                                # file_id → (best_score, meta, text_with_keyword)
                                _cs_best = {}

                                def _csc_keywords_in_text(txt_low: str, kws: list, as_person: bool) -> bool:
                                    """Match keywords in chunk text without person-query false positives.
                                    Non-person: substring AND (addresses, dates, brands).
                                    Person-style: whole-word / abbreviated-stem (chris ↔ christopher)."""
                                    if not kws:
                                        return False
                                    if not as_person:
                                        return all(kw in txt_low for kw in kws)
                                    for _kw in kws:
                                        if len(_kw) < 2:
                                            continue
                                        if len(_kw) >= 4:
                                            _pat = rf'\b{_cs_re.escape(_kw)}(?:[a-z]{{1,22}})?\b'
                                        else:
                                            _pat = rf'\b{_cs_re.escape(_kw)}\b'
                                        if not _cs_re.search(_pat, txt_low):
                                            return False
                                    return True

                                for _csc in (_cs_chunks or []):
                                    _csc_text = (getattr(_csc, 'text', '') or '').lower()
                                    if not _csc_keywords_in_text(
                                            _csc_text, _cs_keywords, _cs_is_person):
                                        continue
                                    _csc_meta = dict(getattr(_csc, 'metadata', {}) or {})
                                    _csc_fid = (_csc_meta.get('file_id') or
                                                _csc_meta.get('source_file') or
                                                getattr(_csc, 'chunk_id', ''))
                                    if not _csc_fid:
                                        continue
                                    # Extract clean filename from [FILE: ...] header in chunk text
                                    # so deduplication by filename works correctly.
                                    if not _csc_meta.get('original_filename'):
                                        _csc_raw_text = getattr(_csc, 'text', '') or ''
                                        _csc_file_tags = _cs_re.findall(
                                            r'\[FILE:\s*([^\]]+)\]', _csc_raw_text)
                                        for _tag in _csc_file_tags:
                                            _tag = _tag.strip()
                                            # Skip ObjectID-style tags (24 hex chars)
                                            if len(_tag) == 24 and all(
                                                    c in '0123456789abcdef' for c in _tag.lower()):
                                                continue
                                            _csc_meta['original_filename'] = _tag
                                            break
                                    # Score: single kw uses frequency, multi-kw uses match count
                                    if len(_cs_keywords) == 1:
                                        _kw_count = _csc_text.count(_cs_keywords[0])
                                        _csc_score = min(0.95, 0.85 + 0.03 * _kw_count)
                                    else:
                                        _kw_density = sum(
                                            _csc_text.count(kw) for kw in _cs_keywords
                                        )
                                        _csc_score = min(0.92, 0.78 + 0.04 * len(_cs_keywords)
                                                         + 0.01 * min(_kw_density, 3))
                                    if (_csc_fid not in _cs_best or
                                            _csc_score > _cs_best[_csc_fid][0]):
                                        _cs_best[_csc_fid] = (
                                            _csc_score, _csc_meta,
                                            getattr(_csc, 'text', '') or ''
                                        )
                                _added_cs = 0
                                _cs_ranked = sorted(
                                    _cs_best.items(),
                                    key=lambda kv: -kv[1][0],
                                )
                                # Person queries: cap text-scan rows so weak tail files do not
                                # flood merge/ranking (Chris Dominguez had dozens of cap-table hits).
                                _cs_cap = (
                                    max(int(limit) + 4, 14)
                                    if _cs_is_person
                                    else min(len(_cs_ranked), max(int(limit) * 2, 45))
                                )
                                for _cs_fid, (_cs_score, _cs_meta, _cs_text_raw) in _cs_ranked[
                                    :_cs_cap
                                ]:
                                    validated.append({
                                        "file_id": _cs_fid,
                                        "similarity_score": _cs_score,
                                        "text": _cs_text_raw,
                                        "metadata": _cs_meta,
                                        "search_method": "content_scan",
                                    })
                                    _added_cs += 1
                                if _added_cs:
                                    logger.info(
                                        f"[CONTENT-SCAN] Added {_added_cs} files via text scan "
                                        f"for keywords {_cs_keywords}"
                                    )
                            except Exception as _cs_err:
                                logger.debug(
                                    f"[CONTENT-SCAN] Text scan failed: {_cs_err}", exc_info=True
                                )
                        # ── End Content Keyword Scan Supplement ───────────────────────────────

                        # ── Entity + Extension Supplement ─────────────────────────────────────
                        # When query has BOTH file_extensions AND entity keywords
                        # (e.g. "freightpal excel files"), run a lexical scan for the entity
                        # keyword and filter results to the required extension.
                        # This ensures "Copy of FreightPal Cap Table.xlsx" appears for
                        # "freightpal excel files" even when metadata_first returns only
                        # Curation Media cap tables (because they dominate by file-type).
                        _ent_ext_exts = [e.lower() for e in (query_meta.get("file_extensions") or [])]
                        _EXT_TYPE_WORDS = {
                            'powerpoint', 'excel', 'excels', 'spreadsheet', 'word', 'document', 'documents',
                            'file', 'files', 'pdf', 'csv', 'image', 'photo', 'pptx', 'xlsx',
                            'docx', 'ppt', 'xls', 'presentation', 'slides', 'deck', 'decks',
                            'slideshow', 'jpeg', 'png', 'jpg', 'the', 'and', 'for', 'all',
                            'any', 'get', 'find', 'show', 'list', 'me', 'my', 'can', 'you',
                            'please', 'related', 'about', 'with', 'from', 'search', 'give',
                        }
                        if _ent_ext_exts and vi and user_id:
                            _orig_q_ee = (query_meta.get("original_query") or query or "").strip()
                            _ee_words = [w for w in _orig_q_ee.lower().split()
                                         if len(w) >= 4 and w not in _EXT_TYPE_WORDS]
                            if _ee_words:
                                try:
                                    import urllib.parse as _ee_urlparse
                                    _existing_fids_ee = {r.get("file_id") for r in validated}
                                    _ee_keyword = " ".join(_ee_words[:2])
                                    _ee_lex = vi.search_lexical(
                                        _ee_keyword, user_id=user_id, limit=min(500, max(100, int(limit)))
                                    )
                                    _added_ee = 0
                                    for _ee_lr in (_ee_lex or []):
                                        _ee_dict = _ee_lr if isinstance(_ee_lr, dict) else {
                                            "file_id": getattr(_ee_lr, "file_id", ""),
                                            "similarity_score": float(getattr(_ee_lr, "similarity_score", 0.6) or 0.6),
                                            "text": getattr(_ee_lr, "text", ""),
                                            "metadata": getattr(_ee_lr, "metadata", {}),
                                            "search_method": "entity_ext_supplement",
                                        }
                                        _ee_fid = _ee_dict.get("file_id") or getattr(_ee_lr, "file_id", "")
                                        if _ee_fid in _existing_fids_ee:
                                            continue
                                        _ee_meta = (_ee_dict.get("metadata") or
                                                    getattr(_ee_lr, "metadata", {}) or {})
                                        _ee_raw = str(_ee_meta.get("original_filename") or _ee_meta.get("filename") or "")
                                        _ee_fn = _ee_urlparse.unquote(_ee_raw.split("?")[0]).lower()
                                        _ee_ext = ("." + _ee_fn.rsplit(".", 1)[-1]) if "." in _ee_fn else ""
                                        if _ee_ext in _ent_ext_exts:
                                            _ee_dict["search_method"] = "entity_ext_supplement"
                                            if not isinstance(_ee_lr, dict):
                                                _ee_dict["file_id"] = _ee_fid
                                                _ee_dict["metadata"] = _ee_meta
                                            validated.append(_ee_dict)
                                            _existing_fids_ee.add(_ee_fid)
                                            _added_ee += 1
                                    if _added_ee:
                                        logger.info(f"[ENT-EXT-SUPP] Added {_added_ee} files: kw='{_ee_keyword}' exts={_ent_ext_exts}")
                                except Exception as _ee_err:
                                    logger.debug(f"[ENT-EXT-SUPP] Failed: {_ee_err}", exc_info=True)
                        # ── End Entity + Extension Supplement ─────────────────────────────────

                        # ── Hard file-extension filter ─────────────────────────────────────────
                        # For explicit file-type queries ("powerpoint", "excel", etc.) hard-filter
                        # results to only return files with the requested extension.
                        # The penalty/boost in _apply_constraint_boost already reranks, but with
                        # 15-20 results, wrong-type files still appear in the user's list.
                        # Only apply if we have ≥2 correctly-typed results (to avoid empty results).
                        _hf_exts = [e.lower() for e in (query_meta.get("file_extensions") or [])]
                        if _hf_exts:
                            import urllib.parse as _hf_urlparse
                            _hf_keep, _hf_drop = [], []
                            for _hfi in validated:
                                # Ensure file-type queries with entities filter out unrelated files.
                                _hfi_meta = _hfi.get("metadata") or {}
                                _hfi_raw = str(
                                    _hfi_meta.get("original_filename")
                                    or _hfi_meta.get("filename")
                                    or ""
                                )
                                _hfi_fn = _hf_urlparse.unquote(
                                    _hfi_raw.split("?")[0]
                                ).lower()
                                _hfi_ext = (
                                    "." + _hfi_fn.rsplit(".", 1)[-1]
                                    if "." in _hfi_fn
                                    else ""
                                )
                                if _hfi_ext not in _hf_exts:
                                    _hf_drop.append(_hfi)
                                    continue
                                # Ensure file-type queries with entities filter out unrelated files.
                                if _entity_lex_tokens:
                                    if not any(
                                        et in _hfi_fn for et in _entity_lex_tokens
                                    ):
                                        _hf_drop.append(_hfi)
                                        continue
                                _hf_keep.append(_hfi)
                            _hf_min = 1 if _entity_lex_tokens else 2
                            # Entity + file-type: ALWAYS keep only extension + basename matches,
                            # even if that yields 0–1 rows — otherwise vector noise (every .xlsx)
                            # remains after re-ingest.
                            if _entity_lex_tokens and _hf_exts:
                                validated = _hf_keep
                                logger.info(
                                    f"[EXT-FILTER] Strict entity+ext: kept {len(_hf_keep)}/"
                                    f"{len(_hf_keep) + len(_hf_drop)} "
                                    f"ext={_hf_exts} entity={_entity_lex_tokens}"
                                )
                            elif len(_hf_keep) >= _hf_min:
                                validated = _hf_keep
                                logger.info(
                                    f"[EXT-FILTER] Kept {len(_hf_keep)}/"
                                    f"{len(_hf_keep) + len(_hf_drop)} "
                                    f"ext={_hf_exts} entity={_entity_lex_tokens or '—'}"
                                )
                        # ── End hard file-extension filter ─────────────────────────────────────

                        # Apply constraint-aware boost before sorting (does not change similarity_score)
                        _apply_constraint_boost(validated, query_meta)
                        _sorted_all = sorted(
                            validated,
                            key=lambda x: _ranking_key(x, _ranking_q),
                            reverse=True,
                        )
                        # Deduplicate by decoded filename: when the same file was ingested
                        # multiple times, keep only the highest-scored result for each unique filename.
                        import urllib.parse as _dedup_urlparse, re as _dedup_re
                        _seen_dedup_fns = set()
                        _deduped = []
                        for _di in _sorted_all:
                            _di_meta = _di.get("metadata") or {}
                            _di_raw = str(_di_meta.get("original_filename") or _di_meta.get("filename") or _di.get("file_id") or "")
                            _di_fn = _dedup_urlparse.unquote(_di_raw.split("?")[0]).split("/")[-1].strip().lower()
                            # Normalize name slightly to catch minor variants
                            _di_fn_norm = _dedup_re.sub(r'\s+', ' ', _di_fn).strip()
                            if _di_fn_norm and _di_fn_norm in _seen_dedup_fns:
                                continue
                            _seen_dedup_fns.add(_di_fn_norm)
                            _deduped.append(_di)

                        # Person name: trim weak vector hits with no name signal in filename
                        _p_full = [
                            p for p in (query_meta.get("persons") or [])
                            if p and " " in p.strip()
                        ]
                        _weak_embedding_hits = frozenset({
                            "vector",
                            "semantic_text_search",
                            "semantic_vector_fallback",
                        })
                        if (
                            _p_full
                            and query_meta.get("intent_type") == "entity"
                        ):
                            import urllib.parse as _pq_u
                            import re as _pq_re
                            _p_toks = []
                            for _p in _p_full:
                                for _t in _p.strip().lower().split():
                                    if len(_t) >= 3:
                                        _p_toks.append(_t)
                            _p_toks = list(dict.fromkeys(_p_toks))[:8]
                            _dedup2 = []
                            _keep_methods = frozenset({
                                "lexical_supplement",
                                "content_scan",
                                "metadata_first",
                                "entity_ext_supplement",
                                "filename_supplement",
                            })

                            def _person_tokens_in_body(_txt: str, _toks: list) -> bool:
                                """Same word-boundary/stem idea as content_scan (chris↔christopher)."""
                                _tl = (_txt or "").lower()
                                for _pt in _toks:
                                    if len(_pt) >= 4:
                                        _pat = rf"\b{_pq_re.escape(_pt)}(?:[a-z]{{1,22}})?\b"
                                    else:
                                        _pat = rf"\b{_pq_re.escape(_pt)}\b"
                                    if not _pq_re.search(_pat, _tl):
                                        return False
                                return True

                            for _it in _deduped:
                                _sm = str(_it.get("search_method") or "")
                                if _sm in _keep_methods:
                                    _dedup2.append(_it)
                                    continue
                                if _sm not in _weak_embedding_hits:
                                    _dedup2.append(_it)
                                    continue
                                _im = _it.get("metadata") or {}
                                _ir = str(
                                    _im.get("original_filename")
                                    or _im.get("filename")
                                    or ""
                                )
                                _ifn = _pq_u.unquote(
                                    _ir.split("?")[0]
                                ).lower()
                                _tx_full = (_it.get("text") or "")
                                _name_in_fn = any(_pt in _ifn for _pt in _p_toks)
                                # Drop weak vector (e.g. ADP @ 46%) unless name is provably
                                # in filename or in returned chunk text with word boundaries.
                                if _name_in_fn or _person_tokens_in_body(_tx_full, _p_toks):
                                    _dedup2.append(_it)
                            if _dedup2:
                                _deduped = _dedup2

                        # Brand / single-entity keyword: remove vector hits with no keyword in
                        # filename or chunk text to filter out unrelated vector hits.
                        # Skip when multi-word person query already handled above.
                        if (
                            _entity_lex_tokens
                            and _deduped
                            and not (
                                _p_full and query_meta.get("intent_type") == "entity"
                            )
                        ):
                            import urllib.parse as _ek_u
                            _EK_GENERIC = frozenset({
                                "corp", "inc", "llc", "ltd", "company", "companies",
                                "group", "holdings", "limited", "plc",
                            })
                            _ek_sig = [
                                t for t in _entity_lex_tokens
                                if len(t) >= 3 and t not in _EK_GENERIC
                            ]
                            if _ek_sig:
                                _dedup_ek = []
                                _km = frozenset({
                                    "lexical_supplement",
                                    "content_scan",
                                    "metadata_first",
                                    "entity_ext_supplement",
                                    "filename_supplement",
                                })
                                for _it in _deduped:
                                    _sm = str(_it.get("search_method") or "")
                                    if _sm in _km:
                                        _dedup_ek.append(_it)
                                        continue
                                    if _sm not in _weak_embedding_hits:
                                        _dedup_ek.append(_it)
                                        continue
                                    _im = _it.get("metadata") or {}
                                    _ir = str(
                                        _im.get("original_filename")
                                        or _im.get("filename")
                                        or ""
                                    )
                                    _ifn = _ek_u.unquote(
                                        _ir.split("?")[0]
                                    ).lower()
                                    _txl = (_it.get("text") or "").lower()
                                    _hay = _ifn + " " + _txl
                                    if len(_ek_sig) == 1:
                                        _ok_ek = _ek_sig[0] in _ifn or _ek_sig[0] in _txl
                                    else:
                                        _ok_ek = all(
                                            et in _hay for et in _ek_sig
                                        )
                                    if _ok_ek:
                                        _dedup_ek.append(_it)
                                if _dedup_ek:
                                    _deduped = _dedup_ek

                        final = _deduped[:limit]
                    except Exception as e:
                        # Fallback to original filtered results if validation fails
                        logger.warning(
                            f"[VALIDATOR] Constraint validation failed: {e}, using filtered results without validation"
                        )
                        final = sorted(
                            filtered,
                            key=lambda x: _ranking_key(x, _ranking_q),
                            reverse=True,
                        )[:limit]

                # 100% recall fallback: vector(0) -> year metadata -> lexical scan
                if len(final) == 0 and search_method in ("both", "vector") and vi and user_id:
                    fc = {}
                    if bucket_id:
                        fc["bucket_id"] = bucket_id
                    if path:
                        fc["path"] = path
                    if connection_id:
                        fc["connection_id"] = connection_id
                    fc_or_none = fc if fc else None

                    # 1) Vector with threshold 0
                    logger.info("[FALLBACK] Zero results - trying vector threshold 0")
                    vec_q = _effective_search_q
                    try:
                        fb_results = vi.search_documents(
                            query=vec_q,
                            user_id=user_id,
                            limit=limit,
                            similarity_threshold=0.0,
                            filter_conditions=fc_or_none,
                        )
                        for r in fb_results:
                            final.append({
                                "file_id": r.file_id,
                                "text": r.text or "",
                                "similarity_score": float(r.similarity_score or 0),
                                "confidence": float(r.confidence or 0),
                                "extraction_method": r.extraction_method,
                                "search_method": "vector",
                                "metadata": r.metadata or {},
                                "source_file": r.file_id,
                            })
                    except Exception as fb_err:
                        logger.warning(f"[FALLBACK] Vector retry failed: {fb_err}")

                    # 2) Year-based metadata query
                    if len(final) == 0:
                        import re as re_mod
                        years_in_query = [int(m.group(0)) for m in re_mod.finditer(r"\b(19|20)\d{2}\b", query)]
                        if years_in_query:
                            logger.info(f"[FALLBACK] Trying year metadata for {years_in_query}")
                            try:
                                yr_results = vi.search_by_years(years_in_query, user_id=user_id, limit=limit, filter_conditions=fc_or_none)
                                for r in yr_results:
                                    final.append({
                                        "file_id": r.file_id,
                                        "text": r.text or "",
                                        "similarity_score": float(r.similarity_score or 0),
                                        "confidence": float(r.confidence or 0),
                                        "extraction_method": r.extraction_method,
                                        "search_method": "metadata_years",
                                        "metadata": r.metadata or {},
                                        "source_file": r.file_id,
                                    })
                            except Exception as yr_err:
                                logger.warning(f"[FALLBACK] Year search failed: {yr_err}")

                    # 3) Lexical substring scan
                    if len(final) == 0 and len(query.strip()) >= 2:
                        logger.info("[FALLBACK] Trying lexical substring scan")
                        try:
                            lex_results = vi.search_lexical(query, user_id=user_id, limit=limit, filter_conditions=fc_or_none)
                            for r in lex_results:
                                final.append({
                                    "file_id": r.file_id,
                                    "text": r.text or "",
                                    "similarity_score": float(r.similarity_score or 0),
                                    "confidence": float(r.confidence or 0),
                                    "extraction_method": r.extraction_method,
                                    "search_method": "lexical",
                                    "metadata": r.metadata or {},
                                    "source_file": r.file_id,
                                })
                        except Exception as lex_err:
                            logger.warning(f"[FALLBACK] Lexical search failed: {lex_err}")

                    if final:
                        _fb_meta = query_meta if 'query_meta' in locals() and query_meta else {}
                        _apply_constraint_boost(final, _fb_meta)
                        final = sorted(
                            final,
                            key=lambda x: _ranking_key(x, _ranking_q),
                            reverse=True,
                        )[:limit]

                if "final" in locals() and final and "query_meta" in locals() and query_meta:
                    final = _prune_tx_peer_from_search_results(final, query_meta)
                
                # Original API response format (aligned with server integration)
                # Include query_meta in response for debugging and client-side routing
                response = {
                    "success": True,
                    "query": query,
                    "results": final,
                    "total_results": len(final)
                }
                # Add query_meta if available (from enhance_query call earlier)
                if 'query_meta' in locals() and query_meta:
                    try:
                        qm = query_meta
                        qm.setdefault("search_debug", {})
                        sd = qm["search_debug"]
                        if isinstance(sd, dict):
                            sd.setdefault("extracted_year", qm.get("date"))
                            sd.setdefault(
                                "extracted_entity",
                                (qm.get("query_decomposition") or {}).get("entity"),
                            )
                            sd.setdefault("intent_profile", (qm.get("query_decomposition") or {}).get("intent_profile"))
                            if final:
                                _top0 = final[0]
                                sd["top_result"] = {
                                    "file_id": _top0.get("file_id"),
                                    "search_method": _top0.get("search_method"),
                                    "_ranking_score": _top0.get("_ranking_score"),
                                    "similarity_score": _top0.get("similarity_score"),
                                }
                                sd["why_top"] = (
                                    "Highest _ranking_score after constraint boosts "
                                    "(entity/year/location/NDA signals when applicable)."
                                )
                            _sig = ["constraint_boost"]
                            if qm.get("entity_ranking_tokens"):
                                _sig.append("entity_year_ranking")
                            if qm.get("nda_year_query"):
                                _sig.append("nda_year_ranking")
                            sd["applied_filters"] = _sig
                    except Exception:
                        pass
                    response["query_meta"] = query_meta
                return response
                
            except HTTPException:
                # Re-raise HTTP exceptions (like 400 for missing user_id) as-is
                raise
            except ValueError as e:
                # Convert ValueError (like missing user_id) to HTTP 400
                if "user_id is required" in str(e).lower():
                    raise HTTPException(status_code=400, detail=str(e))
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
        @app.delete("/delete-document")
        @app.post("/delete-document")  # Also support POST for compatibility
        async def delete_document(request: Request, principal: Principal = Depends(principal_dep)):
            """
            UI label: **Delete all vectors for one file** — removes every chunk for the given
            file_id (and optional user_id / bucket / path filters) from document + image collections.

            JSON body (POST or DELETE with Content-Type: application/json):
            {
                "file_id": "required_file_id",
                "user_id": "optional_user_id",
                "bucket_id": "optional_bucket_id",
                "path": "optional_path",
                "connectionId": "optional — same value as POST /process (connection_id alias accepted)"
            }
            Query fallback (either method): ?fileId=&userId=&bucketId=&path=&connectionId=
            Deletes only within the caller's tenant (from the API key, or userId for service keys).
            """
            _require(principal, "uploader")
            try:
                body: Dict[str, Any] = {}
                try:
                    ct = (request.headers.get("content-type") or "").lower()
                    if "application/json" in ct:
                        body = await request.json()
                        if body is None:
                            body = {}
                except Exception:
                    body = {}

                qp = request.query_params
                file_id = (
                    body.get("file_id")
                    or body.get("fileId")
                    or qp.get("file_id")
                    or qp.get("fileId")
                )
                if not file_id:
                    raise HTTPException(
                        status_code=400,
                        detail="file_id is required (JSON body or ?fileId= / ?file_id=)",
                    )

                user_id = _tenant(
                    principal,
                    body.get("user_id") or body.get("userId") or qp.get("user_id") or qp.get("userId"),
                    "user_id is required to delete a document",
                )
                bucket_id = (
                    body.get("bucket_id")
                    or body.get("bucketId")
                    or qp.get("bucket_id")
                    or qp.get("bucketId")
                )
                path = body.get("path") or qp.get("path")
                connection_id = (
                    body.get("connectionId")
                    or body.get("connection_id")
                    or qp.get("connectionId")
                    or qp.get("connection_id")
                )

                vi = ui_processor.get_vector_integration()
                if not vi:
                    raise HTTPException(status_code=500, detail="Vector integration not available")

                logger.info(
                    f"[DELETE] Deleting document: file_id={file_id}, "
                    f"user_id={user_id}, bucket_id={bucket_id}, path={path}, connection_id={connection_id}"
                )

                result = vi.delete_document(
                    file_id=file_id,
                    user_id=user_id,
                    bucket_id=bucket_id,
                    path=path,
                    connection_id=connection_id,
                )
                
                if result.get("success"):
                    n_del = int(result.get("chunks_deleted") or 0)
                    if n_del == 0:
                        raise HTTPException(
                            status_code=404,
                            detail={
                                "message": "No Milvus entities matched; nothing deleted.",
                                "file_id": file_id,
                                "hint": "Use the Milvus file_id from search results (often upload_…), not only the "
                                "display filename (.pdf). Set userId to the same value as search. Omit bucketId, path, "
                                "connectionId (or connection_id) unless those were set at ingest.",
                            },
                        )
                    _CONTENT_SCAN_CACHE.pop(user_id, None)
                    # Invalidate metadata index + disk cache so deleted file
                    # never surfaces in future metadata-first routing.
                    try:
                        if user_id:
                            _eng = get_semantic_engine()
                            _rm = list(result.get("distinct_file_ids_removed") or [])
                            _fids = [x for x in _rm if x] or [file_id]
                            for _fid in _fids:
                                _eng.invalidate_document_cache(user_id, str(_fid))
                        else:
                            logger.warning(
                                "[DELETE] user_id missing; metadata disk cache not invalidated for file_id=%s",
                                file_id[:48],
                            )
                    except Exception as _inv_err:
                        logger.warning(f"[DELETE] Cache invalidation non-fatal error: {_inv_err}")
                    n_files = int(result.get("files_deleted") or 0)
                    logger.info(
                        f"[DELETE] fileId={file_id} userId={user_id} removed "
                        f"files={n_files} chunks={n_del}"
                    )
                    return {
                        "success": True,
                        "message": f"Removed {n_files} file(s), {n_del} vector chunk(s) for {file_id}",
                        "file_id": file_id,
                        "files_deleted": n_files,
                        "chunks_deleted": n_del,
                        "document_chunks_deleted": int(result.get("document_chunks_deleted") or 0),
                        "image_chunks_deleted": int(result.get("image_chunks_deleted") or 0),
                        "document_files_deleted": int(result.get("document_files_deleted") or 0),
                        "image_files_deleted": int(result.get("image_files_deleted") or 0),
                        "distinct_file_ids_removed": result.get("distinct_file_ids_removed") or [],
                    }
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Failed to delete document")
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Delete document failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
        
        # Global lock for lazy initialization to prevent concurrent loading in multi-worker scenarios
        _init_lock = asyncio.Lock()

        @app.on_event("startup")
        async def startup_event():
            """Warm up engines on startup to prevent timeout on first request."""
            logger.info("Ultimate UI search service starting up...")
            
            try:
                def warm_up():
                    logger.info("Pre-warming semantic and vector engines...")
                    try:
                        engine = get_semantic_engine()
                        vi = ui_processor.get_vector_integration()
                        try:
                            if engine.doc_db is None:
                                engine._init_milvus_connections()
                                logger.info("[PREWARM] Semantic pipeline Milvus connection initialized")
                        except Exception as _me:
                            logger.warning(f"[PREWARM] Semantic Milvus init non-fatal: {_me}")
                        logger.info("Engine pre-warm complete. Models loaded and Milvus connected.")
                    except Exception as inner_e:
                        logger.warning(f"Error during engine pre-warming: {inner_e}")

                    # ── Disk-cache pre-load ────────────────────────────────────────
                    # IMPORTANT: Only pre-load the PRIMARY production user (not all
                    # cached users). Loading multiple users into the shared single-tenant
                    # MetadataIndex instance causes each load to overwrite the previous
                    # user's data — the last loaded user's index would be active for ALL
                    # subsequent queries, returning wrong results for other users.
                    #
                    # Instead we only pre-load the one user most likely to be queried
                    # (configurable via PRIMARY_USER_ID env var, falling back to the
                    # most recently-modified cache file). All other users are loaded
                    # on first query via the disk-cache fast-path in
                    # _ensure_metadata_index_for_user (typically <500ms per load).
                    try:
                        import glob as _glob
                        cache_dir = os.environ.get("METADATA_CACHE_DIR", "data/metadata_cache")
                        primary_uid = os.environ.get("PRIMARY_USER_ID", "").strip()
                        engine_obj = get_semantic_engine()

                        if primary_uid:
                            # Load explicitly-configured primary user
                            try:
                                engine_obj._ensure_metadata_index_for_user(primary_uid)
                                logger.info(f"[PREWARM] Loaded primary user metadata index: {primary_uid}")
                                # Start background watcher so this server auto-reloads
                                # when the processing server pushes a newer index to Redis.
                                engine_obj._start_metadata_redis_watcher(primary_uid)
                            except Exception as _ue:
                                logger.warning(f"[PREWARM] Primary user load failed: {_ue}")
                        else:
                            # Fallback: load the most-recently modified cache file
                            pattern = os.path.join(cache_dir, "metadata_index_*.json")
                            cache_files = sorted(_glob.glob(pattern), key=lambda p: -__import__("os").path.getmtime(p))
                            if cache_files:
                                fname = __import__("os").path.basename(cache_files[0])
                                uid = fname[len("metadata_index_"):-len(".json")]
                                if uid:
                                    try:
                                        engine_obj._ensure_metadata_index_for_user(uid)
                                        logger.info(f"[PREWARM] Loaded most-recent metadata index: {uid}")
                                        engine_obj._start_metadata_redis_watcher(uid)
                                    except Exception as _ue:
                                        logger.warning(f"[PREWARM] Most-recent user load failed: {_ue}")
                            else:
                                logger.info(f"[PREWARM] No disk caches found in {cache_dir} — skipping")
                    except Exception as _ce:
                        logger.warning(f"[PREWARM] Disk-cache pre-load error: {_ce}")

                # Run warm_up BLOCKING (awaited) so the disk cache is fully
                # loaded before the first request is accepted.  Without await,
                # a race condition exists: an inbound request may trigger a
                # Milvus rebuild WHILE the pre-warm is still loading from disk,
                # causing the correct on-disk by_full_date / by_location / by_person
                # data to be overwritten by the (potentially incomplete) Milvus
                # rebuild.  Blocking startup adds ≤5s on a cache-hit path and
                # prevents the stale-index bug entirely.
                import concurrent.futures as _cf
                loop = asyncio.get_event_loop()
                executor = _cf.ThreadPoolExecutor(max_workers=1)
                await loop.run_in_executor(executor, warm_up)

            except Exception as e:
                logger.warning(f"Engine pre-warming registration failed: {e}")
        
        @app.get("/health/live")
        async def health_live():
            """Liveness: the process serves HTTP. No dependency calls (used by the container healthcheck)."""
            return {"status": "alive", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        def _readiness_response():
            from src.health import run_checks, summarize

            code, body = summarize(run_checks())
            body["processor_available"] = ui_processor.available
            body["vector_integration_loaded"] = ui_processor.vector_integration is not None
            return JSONResponse(status_code=code, content=body)

        @app.get("/health/ready")
        def health_ready():
            """Readiness: 200 when Redis, Milvus and the embedder answer (workers only degrade); else 503."""
            return _readiness_response()

        @app.get("/health")
        def health_check():
            """Dependency report (same checks and status code as /health/ready)."""
            return _readiness_response()
        
        return app
        
    except ImportError:
        logger.error("FastAPI not available. Please install: pip install fastapi uvicorn")
        return None

# Create app at module level so uvicorn (run from Docker CMD) can import it.
app = create_fastapi_app()
