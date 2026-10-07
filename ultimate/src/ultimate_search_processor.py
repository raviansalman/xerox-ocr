#!/usr/bin/env python3
"""
Advanced Document Processing Engine

High-performance OCR and text extraction system with intelligent
pattern recognition and fuzzy matching capabilities.
"""

import os
import time
import logging
import re
import hashlib
import io
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
# Image captioning and semantic search (initialized lazily to avoid import errors)
captioner = None
semantic_engine = None

def _get_captioner():
    """Lazy initialization of image captioner."""
    # Check skip flag FIRST to prevent BLIP initialization
    skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
    if skip_captioning:
        logger.debug("Image captioning skipped (SKIP_IMAGE_CAPTIONING_IN_PROCESSOR=true)")
        return None
    
    global captioner
    if captioner is None:
        try:
            from src.semantic.semantic_components import generate_caption
            # Create a simple wrapper for compatibility
            class CaptionerWrapper:
                def caption(self, image_path: str) -> str:
                    return generate_caption(image_path)
            captioner = CaptionerWrapper()
            logger.info("Image captioner initialized (BLIP)")
        except Exception as e:
            logger.warning(f"Image captioner not available: {e}")
            captioner = False  # Mark as unavailable
    return captioner if captioner is not False else None

def _get_semantic_engine():
    """Lazy initialization of semantic search engine. Uses process-wide singleton to avoid loading model multiple times."""
    # Check skip flag FIRST to prevent unnecessary model loading
    skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
    if skip_captioning:
        logger.debug("Semantic engine initialization skipped (SKIP_IMAGE_CAPTIONING_IN_PROCESSOR=true)")
        return None

    global semantic_engine
    if semantic_engine is None:
        try:
            from src.semantic.semantic_pipeline import get_global_semantic_pipeline
            semantic_engine = get_global_semantic_pipeline()
            logger.info("Semantic search engine initialized (shared SemanticPipeline)")
        except Exception as e:
            logger.warning(f"Semantic search engine not available: {e}")
            semantic_engine = False  # Mark as unavailable
    return semantic_engine if semantic_engine is not False else None
# Import workflow manager
try:
    from .workflow_manager import get_workflow_manager
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Workflow manager not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
# OCR can work with just pytesseract and PIL, cv2 is optional for preprocessing
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        logger.warning("OpenCV (cv2) not available - OCR will work but with limited preprocessing")
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning(f"OCR dependencies not available: {e}")

# Vision analysis disabled for faster processing
    VISION_AVAILABLE = False
    BASIC_VISION_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available")

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available")

try:
    from docx2python import docx2python
    DOCX2PYTHON_AVAILABLE = True
except ImportError:
    DOCX2PYTHON_AVAILABLE = False
    logger.warning("docx2python not available")

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logger.warning("python-pptx not available, PowerPoint support disabled")

try:
    import ppt2txt  # type: ignore
    PPT2TXT_AVAILABLE = True
except ImportError:
    PPT2TXT_AVAILABLE = False

# Excel file support
try:
    import openpyxl
    from openpyxl import load_workbook
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False
    logger.warning("openpyxl not available, .xlsx/.xlsm/.xltx support disabled")

try:
    import xlrd
    XLS_AVAILABLE = True
except ImportError:
    XLS_AVAILABLE = False
    logger.warning("xlrd not available, legacy .xls support disabled")

try:
    import zipfile
    import xml.etree.ElementTree as ET
    ZIP_XML_AVAILABLE = True
except ImportError:
    ZIP_XML_AVAILABLE = False
    logger.warning("zipfile/xml not available, Excel ZIP/XML fallback disabled")

try:
    from odf.opendocument import load as odf_load
    from odf.table import Table, TableRow, TableCell
    from odf import text as odf_text
    ODS_AVAILABLE = True
except ImportError:
    ODS_AVAILABLE = False
    logger.warning("odfpy not available, .ods spreadsheet support disabled")
    logger.warning("ppt2txt not available, legacy .ppt support disabled")


@dataclass
class UltimateSearchResult:
    """Result of document processing."""
    file_path: str
    file_type: str
    success: bool
    text_content: str
    normalized_text: str
    keywords: List[str]
    searchable_text: str
    processing_time: float
    extraction_method: str
    confidence: float
    word_variations: Dict[str, List[str]]
    fuzzy_matches: Dict[str, float]
    pattern_matches: Dict[str, List[str]]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    # Vision analysis fields
    objects_detected: List[Dict[str, Any]] = None
    image_caption: str = ""
    visual_elements: List[str] = None


class DocumentProcessor:
    """Document processing engine with OCR capabilities."""
    
    def __init__(self, tesseract_path: Optional[str] = None, language: str = 'eng', file_id: Optional[str] = None):
        """Initialize document processing engine."""
        self.tesseract_path = tesseract_path
        self.language = language
        self.file_id = file_id
        
        if tesseract_path and OCR_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        # Initialize empty dictionaries for compatibility
        self.word_variations = {}
        self.synonyms = {}
        self.patterns = {}
        self.max_page_workers = max(1, int(os.getenv("PDF_MAX_PAGE_WORKERS", "4")))
        self.max_ocr_workers = max(1, int(os.getenv("MAX_OCR_WORKERS", "4")))
        self.ocr_trigger_min_words = max(1, int(os.getenv("OCR_TRIGGER_MIN_WORDS", "10")))
        self.pdf_page_limit = int(os.getenv("PDF_PAGE_PROCESS_LIMIT", "0"))
        self.ocr_zoom_factor = float(os.getenv("OCR_ZOOM_FACTOR", "1.5"))
        self.ocr_fast_config = os.getenv("OCR_FAST_CONFIG", "--psm 6 --oem 1")
        
        # Vision analysis disabled for faster processing
        
        logger.info("Document processing engine initialized")
    
    def _preprocess_image_ultimate(self, image: Image.Image, needed_techniques: List[str] = None) -> List[Tuple[Image.Image, str]]:
        """Apply preprocessing techniques only for the ones needed by the strategy."""
        processed_images = []
        
        # Check if numpy and OpenCV are available
        if not OCR_AVAILABLE:
            logger.warning("OCR not available for image preprocessing")
            return [(image, "original")]
        
        # Check if cv2 is available (it's optional)
        try:
            import cv2
            cv2_available = True
        except ImportError:
            cv2_available = False
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Always include original
        processed_images.append((image, "original"))
        
        # Get grayscale for OpenCV techniques (only if cv2 is available)
        if cv2_available and len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gray_pil = Image.fromarray(gray)
        else:
            if len(img_array.shape) == 3:
                # Convert to grayscale using PIL if cv2 not available
                gray_pil = image.convert('L')
                gray = np.array(gray_pil)
            else:
                gray = img_array
                gray_pil = image
        
        # Only process techniques that are actually needed
        if needed_techniques is None:
            needed_techniques = ["grayscale", "contrast_1.5"]  # Default minimal set
        
        # Process only needed techniques
        for technique in needed_techniques:
            try:
                if technique == "grayscale":
                    processed_images.append((gray_pil, "grayscale"))
                
                elif technique == "contrast_1.5":
                    contrast_img = ImageEnhance.Contrast(image).enhance(1.5)
                    processed_images.append((contrast_img, "contrast_1.5"))
                
                elif technique == "contrast_2.5":
                    contrast_img = ImageEnhance.Contrast(image).enhance(2.5)
                    processed_images.append((contrast_img, "contrast_2.5"))
                
                elif technique == "contrast_3.0":
                    contrast_img = ImageEnhance.Contrast(image).enhance(3.0)
                    processed_images.append((contrast_img, "contrast_3.0"))
                
                elif technique == "brightness_1.3":
                    bright_img = ImageEnhance.Brightness(image).enhance(1.3)
                    processed_images.append((bright_img, "brightness_1.3"))
                
                elif technique == "brightness_0.7":
                    dark_img = ImageEnhance.Brightness(image).enhance(0.7)
                    processed_images.append((dark_img, "brightness_0.7"))
                
                elif technique == "brightness_1.8":
                    bright_img = ImageEnhance.Brightness(image).enhance(1.8)
                    processed_images.append((bright_img, "brightness_1.8"))
                
                elif technique == "sharpness_2.0":
                    sharp_img = ImageEnhance.Sharpness(image).enhance(2.0)
                    processed_images.append((sharp_img, "sharpness_2.0"))
                
                elif technique == "sharpness_3.0":
                    sharp_img = ImageEnhance.Sharpness(image).enhance(3.0)
                    processed_images.append((sharp_img, "sharpness_3.0"))
                
                # OpenCV techniques - only if cv2 is available
                elif cv2_available:
                    if technique.startswith("resized_"):
                        scale_factor = float(technique.split("_")[1])
                        height, width = gray.shape
                        if height < 2000 and width < 2000:
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                            resized_pil = Image.fromarray(resized)
                            processed_images.append((resized_pil, technique))
                    
                    elif technique == "adaptive_thresh":
                        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "adaptive_thresh_large":
                        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "otsu_thresh":
                        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "morph_close":
                        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "morph_close_large":
                        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "gaussian_blur":
                        processed = cv2.threshold(cv2.GaussianBlur(gray, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "gaussian_blur_heavy":
                        processed = cv2.threshold(cv2.GaussianBlur(gray, (7,7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "denoise":
                        processed = cv2.fastNlMeansDenoising(gray)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "denoise_heavy":
                        processed = cv2.fastNlMeansDenoising(gray, h=10)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "edge_enhance":
                        processed = cv2.filter2D(gray, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "edge_enhance_strong":
                        processed = cv2.filter2D(gray, -1, np.array([[-2,-2,-2],[-2,17,-2],[-2,-2,-2]]))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "bilateral_filter":
                        processed = cv2.bilateralFilter(gray, 9, 75, 75)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "median_filter":
                        processed = cv2.medianBlur(gray, 3)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "closing_morph":
                        processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "opening_morph":
                        processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "hist_eq":
                        processed = cv2.equalizeHist(gray)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
                    
                    elif technique == "clahe":
                        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
                        processed_pil = Image.fromarray(processed)
                        processed_images.append((processed_pil, technique))
            except Exception as e:
                logger.debug(f"Preprocessing technique {technique} failed: {e}")
                continue
    
        return processed_images
        
    def _analyze_image_complexity(self, image: Image.Image) -> Dict[str, Any]:
        """Enhanced image complexity analysis with fancy font detection for optimal OCR strategy selection."""
        if not OCR_AVAILABLE:
            return {"complexity": "simple", "reason": "OCR not available"}
        
        try:
            import cv2
            import numpy as np
            
            # Convert to numpy array
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            height, width = gray.shape
            total_pixels = height * width
            
            # Fast analysis with optimized thresholds
            is_small = total_pixels < 50000  # Smaller threshold for faster processing
            is_large = total_pixels > 1500000  # Reduced threshold
            
            # Quick contrast analysis
            contrast = gray.std()
            is_low_contrast = contrast < 25
            is_high_contrast = contrast > 70
            
            # Fast edge detection with lower resolution for speed
            if total_pixels > 500000:
                # Downsample for large images
                scale = 0.5
                small_gray = cv2.resize(gray, (int(width * scale), int(height * scale)))
                edges = cv2.Canny(small_gray, 50, 150)
                edge_density = np.sum(edges > 0) / (total_pixels * scale * scale)
            else:
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / total_pixels
            
            has_text_edges = edge_density > 0.03  # Lowered threshold for better detection
            
            # Quick brightness analysis
            mean_brightness = gray.mean()
            is_dark = mean_brightness < 70
            is_bright = mean_brightness > 200
            
            # Simplified noise analysis
            is_noisy = contrast < 20  # Simple noise detection
            
            # Fast text block detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_blocks = len([c for c in contours if cv2.contourArea(c) > 50])  # Lower area threshold
            has_structured_text = text_blocks > 3  # Lower threshold
            
            # Fancy font detection - new feature
            is_fancy_font = self._detect_fancy_font(gray, edges, text_blocks)
            
            # Quote detection
            is_quote_like = text_blocks <= 2 and has_text_edges and not is_noisy and is_high_contrast
            
            # Optimized complexity scoring with fancy font consideration
            complexity_score = 0
            
            # Increase complexity
            if is_low_contrast: complexity_score += 2
            if is_dark or is_bright: complexity_score += 1
            if is_noisy: complexity_score += 2
            if not has_text_edges: complexity_score += 1
            if is_small: complexity_score += 1
            if is_fancy_font: complexity_score += 3  # Fancy fonts are complex
            
            # Decrease complexity
            if is_high_contrast: complexity_score -= 1
            if has_structured_text: complexity_score -= 1
            if is_quote_like: complexity_score -= 3  # Strong preference for simple processing
            if 50000 < total_pixels < 800000: complexity_score -= 1  # Optimal size range
            
            # Determine complexity with fancy font consideration
            if is_fancy_font:
                complexity = "fancy_font"
                reason = "Fancy/decorative font detected, using specialized processing"
            elif complexity_score <= 0:
                complexity = "simple"
                reason = "High quality image, optimal for fast processing"
            elif complexity_score <= 2:
                complexity = "medium"
                reason = "Moderate complexity, balanced processing"
            else:
                complexity = "complex"
                reason = "Challenging image, comprehensive processing needed"
            
            return {
                "complexity": complexity,
                "reason": reason,
                "score": complexity_score,
                "characteristics": {
                    "size": "small" if is_small else "large" if is_large else "medium",
                    "contrast": "low" if is_low_contrast else "high" if is_high_contrast else "medium",
                    "brightness": "dark" if is_dark else "bright" if is_bright else "normal",
                    "noise": "high" if is_noisy else "low",
                    "text_edges": has_text_edges,
                    "structured_text": has_structured_text,
                    "edge_density": edge_density,
                    "text_blocks": text_blocks,
                    "quote_like": is_quote_like,
                    "fancy_font": is_fancy_font
                }
            }
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return {"complexity": "simple", "reason": "Analysis failed, using simple processing"}
    
    def _detect_fancy_font(self, gray: np.ndarray, edges: np.ndarray, text_blocks: int) -> bool:
        """Detect if the image contains fancy/decorative fonts."""
        try:
            # Fancy font detection based on several characteristics
            
            # 1. High edge density with irregular patterns
            edge_density = np.sum(edges > 0) / edges.size
            high_edge_density = edge_density > 0.05
            
            # 2. Irregular text block shapes (fancy fonts have more complex shapes)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            irregular_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Only consider significant contours
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:  # Low circularity indicates irregular shapes
                            irregular_shapes += 1
            
            has_irregular_shapes = irregular_shapes > text_blocks * 0.5
            
            # 3. High variation in text block sizes (fancy fonts often have varying sizes)
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
                if len(areas) > 1:
                    area_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
                    high_variation = area_variation > 0.8
                else:
                    high_variation = False
            else:
                high_variation = False
            
            # 4. Complex edge patterns (fancy fonts have more decorative elements)
            # Count edge direction changes
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            complex_patterns = np.sum(gradient_magnitude > np.mean(gradient_magnitude) * 2) > gradient_magnitude.size * 0.1
            
            # 5. Check for decorative elements (common in fancy fonts)
            # Look for curved lines, decorative borders, etc.
            decorative_elements = 0
            if len(contours) > 0:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        # Check for curved contours (decorative elements)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            if solidity < 0.7:  # Low solidity indicates curved/decorative shapes
                                decorative_elements += 1
            
            has_decorative_elements = decorative_elements > text_blocks * 0.3
            
            # Combine indicators
            fancy_font_indicators = sum([
                high_edge_density,
                has_irregular_shapes,
                high_variation,
                complex_patterns,
                has_decorative_elements
            ])
            
            # Consider it a fancy font if 2 or more indicators are present
            return fancy_font_indicators >= 2
            
        except Exception as e:
            logger.debug(f"Fancy font detection failed: {e}")
            return False
    
    def _get_ocr_strategy(self, complexity: str) -> Dict[str, Any]:
        """Get optimized OCR strategy based on image complexity, with enhanced support for fancy fonts and small text."""
        strategies = {
            "simple": {
                "preprocessing_techniques": [
                    "original", "grayscale", "contrast_1.5", "contrast_2.5", "brightness_1.3", "sharpness_2.0",
                    "resized_2.0", "resized_3.0", "resized_4.0",  # Multiple scales for small text
                    "adaptive_thresh", "otsu_thresh",  # Better thresholding
                    "denoise", "edge_enhance", "bilateral_filter"  # Noise reduction and enhancement
                ],
                "ocr_configs": [
                    "--psm 6",  # Single text block - best for quotes
                    "--psm 3",  # Fully automatic page segmentation
                    "--psm 4",  # Single column of text
                    "--psm 8",  # Single word
                    "--psm 1",  # Automatic page segmentation with OSD
                    "--psm 7",  # Single text line
                    "--psm 9",  # Single word in circle
                    "--psm 10", # Single character
                    "--psm 11", # Sparse text
                    "--psm 12", # Sparse text with OSD
                    "--psm 13", # Raw line
                    "--psm 6 --oem 3",  # LSTM engine
                    "--psm 3 --oem 3",  # LSTM engine
                    "--psm 6 --oem 1",  # Legacy engine
                    "--psm 3 --oem 1",  # Legacy engine
                    "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ "
                ],
                "max_attempts": 12,  # Increased for better small text detection
                "description": "Enhanced processing for clear text with aggressive extraction and small text support"
            },
            "medium": {
                "preprocessing_techniques": [
                    "original", "grayscale", "contrast_1.5", "contrast_2.5", "contrast_3.0",
                    "brightness_1.3", "brightness_1.8", "sharpness_2.0", "sharpness_3.0",
                    "resized_2.0", "resized_3.0", "resized_4.0",  # Multiple scales
                    "adaptive_thresh", "adaptive_thresh_large", "otsu_thresh",
                    "morph_close", "morph_close_large", "gaussian_blur", "gaussian_blur_heavy",
                    "denoise", "denoise_heavy", "edge_enhance", "edge_enhance_strong",
                    "bilateral_filter", "median_filter", "closing_morph", "opening_morph",
                    "hist_eq", "clahe"
                ],
                "ocr_configs": [
                    "--psm 6", "--psm 3", "--psm 4", "--psm 8", "--psm 1", "--psm 7", "--psm 9",
                    "--psm 10", "--psm 11", "--psm 12", "--psm 13",  # More PSM modes
                    "--psm 6 --oem 3", "--psm 3 --oem 3", "--psm 6 --oem 1", "--psm 3 --oem 1",
                    "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ "
                ],
                "max_attempts": 8,  # Increased for better fancy font handling
                "description": "Balanced processing for moderate complexity with enhanced small text and fancy font support"
            },
            "complex": {
                "preprocessing_techniques": [
                    "original", "grayscale", "contrast_1.5", "contrast_2.5", "contrast_3.0", "contrast_4.0",
                    "brightness_1.3", "brightness_1.8", "brightness_2.0", "brightness_0.7",
                    "sharpness_2.0", "sharpness_3.0", "sharpness_4.0", "sharpness_5.0",
                    "resized_2.0", "resized_3.0", "resized_4.0", "resized_5.0", "resized_6.0",  # Multiple scales
                    "adaptive_thresh", "adaptive_thresh_large", "otsu_thresh",
                    "morph_close", "morph_close_large", "gaussian_blur", "gaussian_blur_heavy",
                    "denoise", "denoise_heavy", "edge_enhance", "edge_enhance_strong",
                    "bilateral_filter", "median_filter", "closing_morph", "opening_morph",
                    "hist_eq", "clahe"
                ],
                "ocr_configs": [
                    "--psm 6", "--psm 3", "--psm 4", "--psm 8", "--psm 1", "--psm 7", "--psm 9",
                    "--psm 10", "--psm 11", "--psm 12", "--psm 13",  # All PSM modes
                    "--psm 6 --oem 3", "--psm 3 --oem 3", "--psm 6 --oem 1", "--psm 3 --oem 1",
                    "--psm 4 --oem 3", "--psm 8 --oem 3", "--psm 1 --oem 3",  # More LSTM combinations
                    "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ "
                ],
                "max_attempts": 15,  # Increased for comprehensive processing
                "description": "Comprehensive processing for difficult images, fancy fonts, and small text"
            },
            "fancy_font": {
                "preprocessing_techniques": [
                    "original", "grayscale", "contrast_3.0", "contrast_4.0", "contrast_5.0",
                    "brightness_1.8", "brightness_2.0", "brightness_0.7",
                    "sharpness_3.0", "sharpness_4.0", "sharpness_5.0",
                    "resized_3.0", "resized_4.0", "resized_5.0", "resized_6.0",  # Multiple scales
                    "adaptive_thresh_large", "otsu_thresh", "morph_close_large",
                    "gaussian_blur_heavy", "denoise_heavy", "edge_enhance_strong",
                    "bilateral_filter", "median_filter", "closing_morph", "opening_morph",
                    "hist_eq", "clahe"
                ],
                "ocr_configs": [
                    "--psm 6 --oem 3", "--psm 3 --oem 3", "--psm 6 --oem 1", "--psm 3 --oem 1",
                    "--psm 4 --oem 3", "--psm 8 --oem 3", "--psm 1 --oem 3",  # More LSTM combinations
                    "--psm 7 --oem 3", "--psm 9 --oem 3", "--psm 10 --oem 3",  # Additional modes
                    "--psm 11 --oem 3", "--psm 12 --oem 3", "--psm 13 --oem 3",  # Sparse text modes
                    "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 6 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ",
                    "--psm 3 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ "
                ],
                "max_attempts": 15,  # More attempts for fancy fonts and small text
                "description": "Specialized processing for fancy/decorative fonts and small text"
            }
        }
        
        return strategies.get(complexity, strategies["simple"])  # Default to simple for speed
    
    def _clean_ocr_text(self, text: str) -> str:
        """Advanced text cleaning that fixes OCR artifacts while preserving all meaningful content."""
        if not text:
            return ""
    
        # Normalize whitespace but preserve structure
        text = ' '.join(text.split())
        
        # Fix common OCR artifacts that break words
        # Pattern: IaIeI -> I (fixing broken letters)
        text = re.sub(r'I([a-z])I([a-z])I', r'I\1\2', text)
        text = re.sub(r'I([A-Z])I([A-Z])I', r'I\1\2', text)
        
        # Pattern: IaI -> Ia (fixing single letter artifacts)
        text = re.sub(r'I([a-zA-Z])I', r'I\1', text)
        
        # Pattern: IaIbIcI -> Iabc (fixing multi-letter artifacts)
        text = re.sub(r'I([a-zA-Z])I([a-zA-Z])I([a-zA-Z])I', r'I\1\2\3', text)
        
        # Fix specific common OCR errors
        ocr_fixes = {
            'IeIeI': 'e',
            'IPIiI': 'Pi',
            'IlIoIoIkI': 'look',
            'IaI': 'a',
            'ItIhIeI': 'the',
            'IwIaIyI': 'way',
            'IcIhIaInIgIeI': 'change',
            'IyIoIuI': 'you',
            'IaItI': 'at',
            'ItIhIiInIgIsI': 'things',
            'IIISI': 'IS',
            'IAISI': 'AS',
            'IGIOIOIDI': 'GOOD',
            'IOINILIYI': 'ONLY',
            'ILIIIFIEI': 'LIFE',
            'IMIIINIDISIEITI': 'MINDSET',
            'IOIUIRI': 'OUR',
            'IAIDIDI': 'ADD',
            'IHIOIWI': 'HOW',
            'ITIOI': 'TO'
        }
        
        for artifact, replacement in ocr_fixes.items():
            text = text.replace(artifact, replacement)
        
        # Remove only extreme noise patterns (8+ repeated characters)
        text = re.sub(r'[~`]{8,}', ' ', text)
        text = re.sub(r'[<>]{8,}', ' ', text)
        text = re.sub(r'[\\/]{8,}', ' ', text)
        text = re.sub(r'[=]{8,}', ' ', text)
        text = re.sub(r'[.]{8,}', ' ', text)
        text = re.sub(r'[-]{8,}', ' ', text)
        
        # Remove extreme OCR artifacts (8+ repeated characters)
        text = re.sub(r'\b[|]{8,}\b', ' ', text)
        text = re.sub(r'\b[I]{8,}\b', ' ', text)
        text = re.sub(r'\b[l]{8,}\b', ' ', text)
        text = re.sub(r'\b[0]{8,}\b', ' ', text)
        text = re.sub(r'\b[O]{8,}\b', ' ', text)
        
        # Clean up extra spaces but preserve single spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text using multiple patterns."""
        phone_patterns = [
            r'\+?[1-9]\d{1,14}',  # International format
            r'\+?[1-9]\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # Common formats
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # US format
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # Standard format
            r'\+92\s?\d{3}\s?\d{3}\s?\d{4}',  # Pakistan format
            r'\+92\s?\d{3}\s?\d{7}',  # Pakistan format without spaces
        ]
        
        phone_numbers = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phone_numbers.extend(matches)
        
        # Clean up phone numbers
        cleaned_numbers = []
        for number in phone_numbers:
            # Remove extra spaces and normalize
            cleaned = re.sub(r'[^\d+]', '', number)
            if len(cleaned) >= 10:  # Minimum phone number length
                cleaned_numbers.append(cleaned)
        
        return list(set(cleaned_numbers))  # Remove duplicates
    
    def _extract_brand_names(self, text: str) -> List[str]:
        """Extract brand names and proper nouns from text with enhanced patterns for small text."""
        # Enhanced brand name patterns including small text and footers
        brand_patterns = [
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b',  # CamelCase
            r'\b[A-Z]{2,}\b',  # All caps (like QUSMOOT, GH)
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Two words with capitals
            r'\b[A-Z][a-z]+\s+Group\b',  # Group names
            r'\b[A-Z][a-z]+\s+Groups\b',  # Groups names
            r'\b[a-z]+\.[a-z]+\b',  # Website domains (like averstu.com)
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Three word names
            r'\b[A-Z]{1,2}\b',  # Single or double letters (like GH)
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # Names with middle initial
        ]
        
        brands = []
        for pattern in brand_patterns:
            matches = re.findall(pattern, text)
            brands.extend(matches)
        
        # Filter out common words that aren't brand names
        common_words = {'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM', 'TO', 'IN', 'ON', 'AT', 'BY', 'OF', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'SHALL', 'IT', 'IF', 'AS', 'UP', 'NO', 'SO', 'GO', 'ME', 'MY', 'WE', 'US', 'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'HE', 'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE'}
        
        filtered_brands = []
        for brand in brands:
            if brand.upper() not in common_words and len(brand) > 1:  # Reduced minimum length
                filtered_brands.append(brand)
        
        return list(set(filtered_brands))  # Remove duplicates
    
    def _extract_small_text_keywords(self, text: str) -> List[str]:
        """Extract keywords specifically from small text, footers, and signatures."""
        small_text_patterns = [
            r'\b[A-Z]{1,3}\b',  # Short acronyms (like GH)
            r'\b[a-z]+\.[a-z]+\b',  # Domains and emails
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Two-word names
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b',  # Names with middle initial
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Three-word names
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Four-word names
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Five-word names
        ]
        
        keywords = []
        for pattern in small_text_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        # Also extract individual words that might be important
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        for word in words:
            if len(word) >= 2 and word not in keywords:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_text_ultimate(self, image: Image.Image, file_size_bytes: int = None) -> List[Dict[str, Any]]:
        """Smart OCR extraction for images - analyzes complexity and chooses optimal strategy."""
        if not OCR_AVAILABLE:
            logger.warning("OCR not available")
            return []
        
        try:
            # Store original dimensions before any resizing
            original_width, original_height = image.size
            original_pixels = original_width * original_height
            
            # Resize large images for faster processing
            width, height = image.size
            if width > 2000 or height > 2000:
                # Resize to max 2000px while maintaining aspect ratio
                max_size = 2000
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized large image from {width}x{height} to {new_width}x{new_height}")
            
            # For very large images, use parallel tiled OCR (preserves quality with overlap)
            # Trigger for:
            # - Large file size (>2 MB) OR
            # - Very high pixel count (>8M pixels, i.e., >2.8K x 2.8K)
            file_size_mb = (file_size_bytes / (1024 * 1024)) if file_size_bytes else 0
            use_parallel = (file_size_mb > 2.0) or (original_pixels > 8_000_000)
            
            if use_parallel:
                logger.info(f"Using parallel tiled OCR for large image ({original_width}x{original_height}, {original_pixels:,} pixels)")
                parallel_results = self._extract_text_ultimate_parallel(image, original_width=original_width, original_height=original_height)
                if parallel_results:
                    return parallel_results
            
            # Optimized multi-method extraction with strict time limits for speed
            all_results = []
            max_processing_time = 8.0  # Maximum 8 seconds per page (reduced from 12)
            start_time = time.time()
            
            # Step 1: Try smart mode first (fastest and most accurate) - 3 seconds max
            try:
                smart_start = time.time()
                smart_results = self._extract_with_smart_mode(image, "ultimate")
                smart_time = time.time() - smart_start
                
                if smart_results:
                    all_results.extend(smart_results)
                    best_confidence = max(r['confidence'] for r in smart_results)
                    
                    # Early exit if we have high confidence results (>80%) or if we got good text
                    if best_confidence > 80 and len(smart_results) >= 1:
                        logger.info(f"High confidence smart OCR ({best_confidence}%) in {smart_time:.2f}s - using directly")
                        return smart_results[:1]
                    elif smart_time > 3.0:  # If smart mode took too long, skip other methods
                        logger.info(f"Smart OCR took {smart_time:.2f}s, using results")
                        return smart_results[:1]
            except Exception as e:
                logger.warning(f"Smart OCR failed: {e}")
            
            # Check time - if already used 50% of time, return what we have
            elapsed_time = time.time() - start_time
            if elapsed_time > max_processing_time * 0.5:
                if all_results:
                    logger.info(f"Time limit (50%) reached, using {len(all_results)} results from {elapsed_time:.2f}s")
                    return all_results[:1]
                # If no results and time is up, try one quick raw OCR
                try:
                    raw_text = pytesseract.image_to_string(image, config="--psm 6", lang=self.language)
                    if raw_text.strip():
                        cleaned = self._clean_ocr_text(raw_text.strip())
                        if cleaned:
                            return [{'text': cleaned, 'confidence': 50, 'method': 'quick_ocr', 'word_count': len(cleaned.split())}]
                except:
                    pass
                return []
            
            # Step 2: Try one quick multiple pass only if smart mode didn't give good results - 2 seconds max
            if not all_results or max(r['confidence'] for r in all_results) < 75:
                try:
                    multi_start = time.time()
                    multi_pass_results = self._extract_with_multiple_passes(image)
                    multi_time = time.time() - multi_start
                    
                    if multi_pass_results and multi_time < 2.0:  # Only use if fast
                        all_results.extend(multi_pass_results)
                        logger.info(f"Multiple pass OCR extracted {len(multi_pass_results)} results in {multi_time:.2f}s")
                except Exception as e:
                    logger.warning(f"Multiple pass OCR failed: {e}")
            
            # Check time again - if used 70% of time, return what we have
            elapsed_time = time.time() - start_time
            if elapsed_time > max_processing_time * 0.7:
                if all_results:
                    logger.info(f"Time limit (70%) reached, using {len(all_results)} results from {elapsed_time:.2f}s")
                    return all_results[:1]
                return []
            
            # Step 3: Skip ultra-aggressive (too slow) - only try 2 quick raw OCR configs
            if not all_results or max(r['confidence'] for r in all_results) < 65:
                # Only try the 2 most effective and fastest configs
                quick_configs = ["--psm 6", "--psm 3"]
                for config in quick_configs:
                    try:
                        raw_text = pytesseract.image_to_string(image, config=config, lang=self.language)
                        if raw_text.strip():
                            cleaned_text = self._clean_ocr_text(raw_text.strip())
                            if cleaned_text and len(cleaned_text.strip()) > 10:
                                words = cleaned_text.split()
                                confidence = self._calculate_fallback_confidence(cleaned_text, words)
                                all_results.append({
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'method': f'quick_ocr_{config.replace(" ", "_").replace("--", "")}',
                                    'word_count': len(words)
                                })
                                break  # Use first good result
                    except:
                        continue
                    
                    # Check time after each config
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_processing_time:
                        break
            
            # Step 4: Try only the most effective raw OCR configs (not all 14)
            if not all_results or max(r['confidence'] for r in all_results) < 50:
                # Only try the most effective configurations
                effective_configs = [
                    "--psm 6", "--psm 3", "--psm 8", "--psm 11",  # Most effective PSM modes
                    "--psm 6 --oem 3", "--psm 3 --oem 3"  # LSTM engines
                ]
                
                for config in effective_configs:
                    try:
                        raw_text = pytesseract.image_to_string(image, config=config, lang=self.language)
                        if raw_text.strip():
                            cleaned_text = self._clean_ocr_text(raw_text.strip())
                            if cleaned_text and len(cleaned_text.strip()) > 0:
                                words = cleaned_text.split()
                                confidence = self._calculate_fallback_confidence(cleaned_text, words)
                                all_results.append({
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'method': f'raw_ocr_{config.replace(" ", "_").replace("--", "")}',
                                    'word_count': len(words),
                                    'complexity': 'raw',
                                    'strategy': f'Raw OCR with {config}'
                                })
                    except Exception as e:
                        continue
                    
                    # Check time after each config
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_processing_time:
                        logger.info(f"Time limit reached ({elapsed_time:.2f}s), stopping raw OCR")
                        break
            
            # Process and deduplicate all results
            if all_results:
                # Remove duplicates while preserving the best results
                unique_results = []
                seen_texts = set()
                
                # Sort by confidence first
                all_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                for result in all_results:
                    text_key = result['text'].lower().strip()
                    if text_key and text_key not in seen_texts and len(text_key) > 0:
                        unique_results.append(result)
                        seen_texts.add(text_key)
                
                # If we have results, combine them for maximum coverage
                if unique_results:
                    # Get the best result for confidence
                    best_result = unique_results[0]
                    
                    # Combine all unique texts for comprehensive coverage
                    all_texts = [r['text'] for r in unique_results if r['text'].strip()]
                    combined_text = " ".join(all_texts)
                    
                    # Create comprehensive result
                comprehensive_result = {
                    'text': combined_text,
                        'confidence': best_result['confidence'],
                        'method': f'comprehensive_extraction_{len(unique_results)}_methods',
                    'word_count': len(combined_text.split()),
                        'techniques_used': len(unique_results),
                        'complexity': best_result.get('complexity', 'comprehensive'),
                        'strategy': f'Comprehensive extraction using {len(unique_results)} methods',
                        'all_methods': [r['method'] for r in unique_results[:5]]  # Top 5 methods used
                }
                
                logger.info(f"Comprehensive OCR extracted {len(combined_text.split())} words using {len(unique_results)} methods")
                return [comprehensive_result]
                
        except Exception as e:
            logger.warning(f"Smart OCR failed: {e}")
        
        # Fallback: Try multiple passes for maximum text extraction
        logger.info("Attempting multiple OCR passes for comprehensive text extraction...")
        try:
            multi_pass_results = self._extract_with_multiple_passes(image)
            if multi_pass_results:
                logger.info(f"Multiple pass OCR succeeded: {len(multi_pass_results)} results")
                return multi_pass_results
        except Exception as multi_pass_error:
            logger.warning(f"Multiple pass OCR failed: {multi_pass_error}")
        
        # Fallback: Try basic OCR with minimal preprocessing if smart mode failed
        logger.info("Attempting fallback OCR with basic preprocessing...")
        try:
            fallback_results = self._extract_with_fallback_mode(image)
            if fallback_results:
                logger.info(f"Fallback OCR succeeded: {len(fallback_results)} results")
                return fallback_results
        except Exception as fallback_error:
            logger.warning(f"Fallback OCR also failed: {fallback_error}")
        
        # Last resort: Try raw OCR without any preprocessing
        logger.info("Attempting last resort raw OCR...")
        try:
            raw_text = pytesseract.image_to_string(image, config="--psm 6", lang=self.language)
            if raw_text.strip():
                cleaned_text = self._clean_ocr_text(raw_text.strip())
                if cleaned_text and self._is_valid_fallback_text(cleaned_text):
                    words = cleaned_text.split()
                    confidence = self._calculate_fallback_confidence(cleaned_text, words)
                    logger.info(f"Raw OCR succeeded: {len(cleaned_text)} characters (confidence: {confidence}%)")
                    return [{
                        'text': cleaned_text,
                        'confidence': confidence,
                        'method': 'raw_ocr_fallback',
                        'word_count': len(words),
                        'complexity': 'unknown',
                        'strategy': 'Last resort raw OCR'
                    }]
                else:
                    logger.warning(f"Raw OCR text failed validation: '{cleaned_text[:50]}...'")
        except Exception as raw_error:
            logger.error(f"Even raw OCR failed: {raw_error}")
        
        # Final safety net - try one more time with minimal processing
        logger.warning("All OCR methods failed - attempting final safety extraction...")
        try:
            # Try with the most basic settings possible
            final_text = pytesseract.image_to_string(image, config="--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=[]{}|;:\",./<>?~ ", lang=self.language)
            if final_text and final_text.strip():
                cleaned_final = self._clean_ocr_text(final_text.strip())
                if cleaned_final:
                    logger.info(f"Final safety extraction succeeded: {len(cleaned_final)} characters")
                    return [{
                        'text': cleaned_final,
                        'confidence': 10.0,  # Very low confidence but accept it
                        'method': 'final_safety_extraction',
                        'word_count': len(cleaned_final.split()),
                        'complexity': 'unknown',
                        'strategy': 'Final safety net extraction'
                    }]
        except Exception as final_error:
            logger.error(f"Even final safety extraction failed: {final_error}")
        
        # Ultra-aggressive extraction for missing keywords
        logger.warning("Attempting ultra-aggressive extraction for missing keywords...")
        try:
            # Try multiple aggressive preprocessing techniques
            aggressive_results = self._ultra_aggressive_extraction(image)
            if aggressive_results:
                logger.info(f"Ultra-aggressive extraction succeeded: {len(aggressive_results)} results")
                return aggressive_results
        except Exception as ultra_error:
            logger.error(f"Ultra-aggressive extraction failed: {ultra_error}")
        
        logger.error("All OCR methods failed - no text could be extracted")
        return []
    
    def process_images_parallel(self, image_paths: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process multiple images in parallel without changing OCR logic.
        
        This is a simple wrapper that processes multiple images concurrently
        using ThreadPoolExecutor, but keeps the OCR extraction logic unchanged.
        
        Args:
            image_paths: List of image file paths to process
            max_workers: Maximum number of parallel workers (default: 4)
            
        Returns:
            List of results, one per image
        """
        results = []
        
        def process_single_image(image_path: str) -> Dict[str, Any]:
            """Process a single image using existing OCR logic."""
            try:
                from PIL import Image
                image = Image.open(image_path)
                
                # Use existing OCR extraction (unchanged - no parameters to preserve OCR quality)
                ocr_results = self._extract_text_ultimate(image)
                
                return {
                    'image_path': image_path,
                    'success': True,
                    'results': ocr_results,
                    'text_length': sum(len(r.get('text', '')) for r in ocr_results) if ocr_results else 0
                }
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                return {
                    'image_path': image_path,
                    'success': False,
                    'error': str(e),
                    'results': []
                }
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_image, path): path for path in image_paths}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        return results


    def _extract_text_ultimate_parallel(self, image: Image.Image, original_width: int = None, original_height: int = None) -> List[Dict[str, Any]]:
        """Simple parallel OCR for large images - tiles image and processes in parallel.
        
        This splits large images into tiles with overlap and processes each tile
        in parallel, then combines results. Designed to preserve OCR quality.
        
        Args:
            image: The image to process (may be resized)
            original_width: Original image width before resize
            original_height: Original image height before resize
        """
        if not OCR_AVAILABLE:
            return []
        
        try:
            width, height = image.size
            original_w = original_width or width
            original_h = original_height or height
            original_pixels = original_w * original_h
            
            # Simple 2x2 or 3x3 grid based on size
            if original_pixels > 16_000_000:  # >4K x 4K
                num_cols = 3
                num_rows = 3
                overlap_pixels = 100
            else:  # 8M - 16M pixels
                num_cols = 2
                num_rows = 2
                overlap_pixels = 80
            
            logger.info(f"Tiling image into {num_rows}x{num_cols} grid with {overlap_pixels}px overlap")
            
            # Create tiles
            tiles = []
            for row in range(num_rows):
                for col in range(num_cols):
                    col_width = width / num_cols
                    row_height = height / num_rows
                    
                    left = max(0, int(col * col_width - (overlap_pixels if col > 0 else 0)))
                    upper = max(0, int(row * row_height - (overlap_pixels if row > 0 else 0)))
                    right = min(width, int((col + 1) * col_width + (overlap_pixels if col < num_cols - 1 else 0)))
                    lower = min(height, int((row + 1) * row_height + (overlap_pixels if row < num_rows - 1 else 0)))
                    
                    tile = image.crop((left, upper, right, lower))
                    tiles.append((tile, row, col))
            
            # Process tiles in parallel
            def process_tile(tile_img, tile_idx):
                try:
                    # Use simple OCR config - preserve quality
                    text = pytesseract.image_to_string(tile_img, config="--psm 6 --oem 3", lang=self.language)
                    if text.strip():
                        cleaned = self._clean_ocr_text(text.strip())
                        return (tile_idx, cleaned) if cleaned else None
                except Exception as e:
                    logger.debug(f"Tile {tile_idx} OCR failed: {e}")
                    return None
                return None
            
            texts = []
            with ThreadPoolExecutor(max_workers=self.max_ocr_workers) as executor:
                futures = {executor.submit(process_tile, tile, idx): idx for idx, (tile, _, _) in enumerate(tiles)}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        texts.append(result)
            
            # Sort by tile index and combine with deduplication
            texts.sort(key=lambda x: x[0])
            
            # Deduplicate text from overlapping regions
            # Only add segments that have significant new content
            combined_segments = []
            seen_words = set()
            
            for tile_idx, text in texts:
                words_in_text = set(text.lower().split())
                # Only add if it has significant new content (>30% new words)
                new_words = words_in_text - seen_words
                if len(new_words) > len(words_in_text) * 0.3 or len(seen_words) == 0:
                    combined_segments.append(text)
                    seen_words.update(words_in_text)
            
            combined_text = " ".join(combined_segments)
            
            if combined_text:
                words = combined_text.split()
                return [{
                    "text": combined_text,
                    "confidence": 70.0,  # Moderate confidence for tiled approach
                    "method": "tiled_parallel_ocr",
                    "word_count": len(words),
                    "complexity": "tiled_parallel",
                    "strategy": f"Parallel tiled OCR ({num_rows}x{num_cols} grid)"
                }]
            
            return []
        except Exception as e:
            logger.warning(f"Parallel tiled OCR failed: {e}")
            return []


    def _ultra_aggressive_extraction(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Ultra-aggressive extraction using multiple preprocessing techniques and OCR configs."""
        results = []
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Optimized aggressive preprocessing techniques (only most effective ones)
        aggressive_techniques = [
            ("original", image),
            ("high_contrast", ImageEnhance.Contrast(image).enhance(4.0)),
            ("resized_3x", Image.fromarray(cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3), interpolation=cv2.INTER_CUBIC))),
            ("adaptive_thresh", Image.fromarray(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))),
            ("morph_close", Image.fromarray(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)))),
        ]
        
        # Optimized aggressive OCR configs (only most effective ones)
        aggressive_configs = [
            "--psm 6",  # Single text block
            "--psm 3",  # Fully automatic
            "--psm 8",  # Single word
            "--psm 11", # Sparse text
            "--psm 6 --oem 3",  # LSTM engine
            "--psm 3 --oem 3",  # LSTM engine
        ]
        
        # Try combinations with early exit for speed
        for img, technique_name in aggressive_techniques:
            for config in aggressive_configs:
                try:
                    text = pytesseract.image_to_string(img, config=config, lang=self.language)
                    if text.strip():
                        cleaned_text = self._clean_ocr_text(text.strip())
                        if cleaned_text and len(cleaned_text.strip()) > 0:
                            words = cleaned_text.split()
                            confidence = self._calculate_fallback_confidence(cleaned_text, words)
                            
                            results.append({
                                'text': cleaned_text,
                                'confidence': confidence,
                                'method': f'ultra_aggressive_{technique_name}_{config.replace(" ", "_").replace("--", "")}',
                                'word_count': len(words),
                                'complexity': 'ultra_aggressive',
                                'strategy': 'Ultra-aggressive extraction for missing keywords'
                            })
                            
                            # Check for specific keywords we're looking for
                            text_lower = cleaned_text.lower()
                            if any(keyword in text_lower for keyword in ['dudipatsar', 'qusmoot', 'qusmootgroup']):
                                logger.info(f"Found target keywords with {technique_name} + {config}: {cleaned_text[:100]}...")
                            
                            # Early exit if we have good results
                            if confidence > 75 and len(results) >= 3:
                                break
                                
                except Exception as e:
                    continue
            
            # Early exit if we have enough good results
            if len(results) >= 5:
                break
        
        # Remove duplicates and return best results
        unique_results = []
        seen_texts = set()
        for result in results:
            text_key = result['text'].lower().strip()
            if text_key not in seen_texts and len(text_key) > 0:
                unique_results.append(result)
                seen_texts.add(text_key)
        
        return unique_results[:10]  # Return top 10 results
    
    def _extract_with_multiple_passes(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Optimized multiple OCR passes with early exit for speed."""
        results = []
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Optimized strategies - only the most effective ones
        pass_strategies = [
            {
                "name": "high_resolution",
                "preprocessing": [
                    ("resized_3x", Image.fromarray(cv2.resize(gray, (gray.shape[1]*3, gray.shape[0]*3), interpolation=cv2.INTER_CUBIC))),
                ],
                "configs": ["--psm 6", "--psm 3", "--psm 8"]
            },
            {
                "name": "small_text_focused",
                "preprocessing": [
                    ("high_contrast", ImageEnhance.Contrast(image).enhance(4.0)),
                    ("edge_enhanced", Image.fromarray(cv2.filter2D(gray, -1, np.array([[-3,-3,-3],[-3,25,-3],[-3,-3,-3]])))),
                ],
                "configs": ["--psm 8", "--psm 11"]
            },
            {
                "name": "fancy_font_specialized",
                "preprocessing": [
                    ("morph_close", Image.fromarray(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)))),
                ],
                "configs": ["--psm 6 --oem 3", "--psm 3 --oem 3"]
            }
        ]
        
        # Execute strategies with early exit
        for strategy in pass_strategies:
            for img, prep_name in strategy["preprocessing"]:
                for config in strategy["configs"]:
                    try:
                        text = pytesseract.image_to_string(img, config=config, lang=self.language)
                        if text.strip():
                            cleaned_text = self._clean_ocr_text(text.strip())
                            if cleaned_text and len(cleaned_text.strip()) > 0:
                                words = cleaned_text.split()
                                confidence = self._calculate_fallback_confidence(cleaned_text, words)
                                
                                results.append({
                                    'text': cleaned_text,
                                    'confidence': confidence,
                                    'method': f'multi_pass_{strategy["name"]}_{prep_name}_{config.replace(" ", "_").replace("--", "")}',
                                    'word_count': len(words),
                                    'complexity': 'multi_pass',
                                    'strategy': f'Multiple pass extraction - {strategy["name"]}'
                                })
                                
                                # Early exit if we have good results
                                if confidence > 80 and len(results) >= 3:
                                    break
                    except Exception as e:
                        continue
                
                # Early exit if we have enough good results
                if len(results) >= 5:
                    break
            
            # Early exit if we have enough good results
            if len(results) >= 8:
                break
        
        # Remove duplicates and return best results
        unique_results = []
        seen_texts = set()
        for result in results:
            text_key = result['text'].lower().strip()
            if text_key not in seen_texts and len(text_key) > 0:
                unique_results.append(result)
                seen_texts.add(text_key)
        
        return unique_results[:8]  # Return top 8 results (reduced from 15)
    
    def _crop_bottom_third(self, image: Image.Image) -> Image.Image:
        """Crop bottom third of image for footer text extraction."""
        width, height = image.size
        return image.crop((0, int(height * 2/3), width, height))
    
    def _crop_bottom_half(self, image: Image.Image) -> Image.Image:
        """Crop bottom half of image for footer text extraction."""
        width, height = image.size
        return image.crop((0, int(height * 1/2), width, height))
    
    def _extract_with_fallback_mode(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Fallback OCR with basic preprocessing when smart mode fails."""
        results = []
        
        # Optimized preprocessing - only the most effective techniques
        basic_techniques = [
            ("original", image),
            ("grayscale", image.convert('L') if image.mode != 'L' else image),
            ("contrast_high", ImageEnhance.Contrast(image).enhance(2.0)),
            ("brightness_high", ImageEnhance.Brightness(image).enhance(1.5))
        ]
        
        # Optimized OCR configs - only the most effective ones
        fallback_configs = [
            "--psm 6",  # Single text block (most common)
            "--psm 3",  # Fully automatic page segmentation
            "--psm 4",  # Single column of text
            "--psm 8",  # Single word
            "--psm 1"   # Automatic page segmentation with OSD
        ]
        
        for img, technique_name in basic_techniques:
            for config in fallback_configs:
                try:
                    text = pytesseract.image_to_string(img, config=config, lang=self.language)
                    if text.strip():
                        cleaned_text = self._clean_ocr_text(text.strip())
                        if cleaned_text and self._is_valid_fallback_text(cleaned_text):
                            # Calculate proper confidence for fallback
                            words = cleaned_text.split()
                            confidence = self._calculate_fallback_confidence(cleaned_text, words)
                            
                            results.append({
                                'text': cleaned_text,
                                'confidence': confidence,
                                'method': f'fallback_{technique_name}_{config.replace(" ", "_").replace("--", "")}',
                                'word_count': len(words),
                                'complexity': 'fallback',
                                'strategy': 'Fallback OCR processing'
                            })
                            
                            # Early exit if we have good results
                            if confidence > 70 and len(results) >= 2:
                                break
                except Exception as e:
                    # Reduced logging for performance
                    continue
        
            # Early exit if we have good results
            if results and max(r['confidence'] for r in results) > 70:
                break
        
        return results
    
    def _is_valid_fallback_text(self, text: str) -> bool:
        """Validate fallback OCR text - accept almost all text to prevent 0 words errors."""
        if not text or not text.strip():
            return False
        
        # Accept any non-empty text - be very permissive
        cleaned_text = text.strip()
        if len(cleaned_text) < 1:
            return False
        
        # Only reject completely obvious noise patterns
        noise_patterns = [
            r'^[^a-zA-Z0-9\s.,!?;:()\-&]{10,}$',  # Only special characters (10+ chars)
            r'^[0-9\s]{20,}$',  # Only numbers and spaces (20+ chars)
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, cleaned_text):
                logger.debug(f"Fallback text rejected due to obvious noise: '{cleaned_text[:50]}...'")
                return False
        
        # Accept any text with at least some alphanumeric characters
        if any(c.isalnum() for c in cleaned_text):
            return True
        
        # Even accept text with just punctuation if it's short (might be a single word)
        if len(cleaned_text) <= 10:
            return True
        
        return False
    
    def _calculate_fallback_confidence(self, text: str, words: List[str]) -> float:
        """Calculate confidence for fallback OCR results with stricter criteria."""
        base_confidence = 20  # Lower base confidence for fallback
        
        # Word count bonus (capped)
        word_bonus = min(10, len(words) * 1.5)
        
        # Average word length bonus
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            length_bonus = min(5, avg_word_length * 1.5)
        else:
            length_bonus = 0
        
        # Character diversity bonus
        unique_chars = len(set(text.lower()))
        diversity_bonus = min(5, unique_chars * 0.3)
        
        # Punctuation presence bonus
        punctuation_bonus = 0
        if any(c in text for c in '.,!?;:'):
            punctuation_bonus = 2
        
        # Penalty for short text
        length_penalty = 0
        if len(text) < 10:
            length_penalty = 5
        
        confidence = base_confidence + word_bonus + length_bonus + diversity_bonus + punctuation_bonus - length_penalty
        return min(60, max(15, confidence))  # Cap between 15-60% for fallback
    
    def _extract_with_smart_mode(self, image: Image.Image, technique: str) -> List[Dict[str, Any]]:
        """Optimized smart mode extraction with early exit and quality filtering."""
        results = []
        
        # Analyze image complexity
        analysis = self._analyze_image_complexity(image)
        complexity = analysis["complexity"]
        strategy = self._get_ocr_strategy(complexity)
        
        logger.info(f"Smart OCR: {complexity} complexity - {analysis['reason']}")
        logger.info(f"Using strategy: {strategy['description']}")
        
        # Get preprocessing techniques based on strategy
        preprocessing_techniques = strategy["preprocessing_techniques"]
        ocr_configs = strategy["ocr_configs"]
        max_attempts = strategy["max_attempts"]
        
        # Apply only the preprocessing techniques needed by the strategy
        processed_images = self._preprocess_image_ultimate(image, preprocessing_techniques)
        
        # Limit to max_attempts to control processing time
        processed_images = processed_images[:max_attempts]
        
        # Try OCR on each processed image with each config
        attempt_count = 0
        best_confidence = 0
        # Adjust early exit threshold based on complexity
        early_exit_threshold = 80 if complexity == "simple" else 90  # Lower threshold for simple images
        
        # Reduced logging for performance
        
        for img, prep_technique in processed_images:
            if attempt_count >= max_attempts:
                break
                
            for config in ocr_configs:
                if attempt_count >= max_attempts:
                    break
                
                attempt_count += 1
                # Reduced logging for performance
                
            try:
                text = pytesseract.image_to_string(
                    img,
                    config=config,
                    lang=self.language
                )
                
                if text.strip() and len(text.strip()) > 0:  # Accept any non-empty text
                    # Clean the OCR text
                    cleaned_text = self._clean_ocr_text(text.strip())
                    
                    if cleaned_text and len(cleaned_text.strip()) > 0:  # Accept any non-empty text
                        words = cleaned_text.split()
                        
                        # No quality filtering - accept all text
                        
                        # Calculate confidence based on multiple factors
                        confidence = self._calculate_confidence(cleaned_text, words, complexity, analysis)
                        
                        results.append({
                            'text': cleaned_text,
                            'confidence': confidence,
                            'method': f'smart_{technique}_{prep_technique}_{config.replace(" ", "_").replace("--", "")}',
                            'word_count': len(words),
                            'complexity': complexity,
                            'strategy': strategy['description']
                        })
                        
                        best_confidence = max(best_confidence, confidence)
                        
                        # Early exit for high confidence results
                        if confidence >= early_exit_threshold and len(results) > 0:
                            # Reduced logging for performance
                            break
            except Exception as e:
                # Reduced logging for performance
                continue
        
            # Early exit for high confidence results
            if best_confidence >= early_exit_threshold and len(results) > 0:
                break
        
        logger.info(f"Smart OCR completed: {len(results)} results from {attempt_count} attempts, best confidence: {best_confidence}%")
        return results
    
    
    def _calculate_confidence(self, text: str, words: List[str], complexity: str, analysis: Dict) -> float:
        """Calculate confidence score based on multiple quality factors."""
        base_confidence = 40 if complexity == "complex" else 60 if complexity == "medium" else 80
        
        # Word count bonus
        word_bonus = min(15, len(words) * 2)
        
        # Average word length bonus
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_bonus = min(10, avg_word_length * 2)
        
        # Character diversity bonus
        unique_chars = len(set(text.lower()))
        diversity_bonus = min(10, unique_chars * 0.5)
        
        # Quote-like pattern bonus
        quote_bonus = 0
        if analysis.get('characteristics', {}).get('quote_like', False):

            quote_bonus = 10

        # Punctuation presence bonus                            
        punctuation_bonus = 0       
        if any(c in text for c in '.,!?;:'):
            punctuation_bonus = 5
        
        confidence = base_confidence + word_bonus + length_bonus + diversity_bonus + quote_bonus + punctuation_bonus
        return min(95, max(20, confidence))
    
        analysis = 10 * analysis
    def _extract_text_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using comprehensive PyMuPDF and OCR methods."""
        if not PDF_AVAILABLE:
            return []
        
        # First, verify the file is actually a PDF
        try:
            with open(file_path, "rb") as f:
                first_bytes = f.read(4)
                if first_bytes[:4] != b'%PDF':
                    logger.warning(f"File does not appear to be a valid PDF (magic bytes: {first_bytes})")
                    # Try OCR fallback anyway
                    if OCR_AVAILABLE:
                        logger.info("Attempting OCR fallback for non-PDF file...")
                        try:
                            from PIL import Image
                            image = Image.open(file_path)
                            ocr_results = self._extract_text_ultimate(image)
                            if ocr_results:
                                return ocr_results
                        except Exception as e:
                            logger.warning(f"OCR fallback failed: {e}")
                    return []
        except Exception as e:
            logger.warning(f"Could not verify PDF magic bytes: {e}")
        
        # Try multiple PDF opening strategies
        # For large files, avoid reading entire file into memory
        doc = None
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        use_stream = file_size_mb < 50  # Only use stream method for files < 50MB
        
        pdf_open_methods = [
            ("normal", lambda: fitz.open(file_path)),
        ]
        
        # Only add memory-intensive methods for smaller files
        if use_stream:
            pdf_open_methods.extend([
            ("repaired_stream", lambda: fitz.open(stream=open(file_path, "rb").read(), filetype="pdf")),
                ("repaired_bytes", None),
            ])
        
        for method_name, open_func in pdf_open_methods:
            try:
                if method_name == "repaired_bytes" and use_stream:
                    # Try reading as bytes and opening with repair (only for smaller files)
                    with open(file_path, "rb") as f:
                        pdf_bytes = f.read()
                    # Try opening with repair flag
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                elif method_name == "repaired_stream" and use_stream:
                    with open(file_path, "rb") as f:
                        pdf_bytes = f.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                elif open_func:
                    doc = open_func()
                
                if doc and doc.page_count > 0:
                    logger.info(f"Successfully opened PDF using {method_name} method ({doc.page_count} pages)")
                    break
            except Exception as e:
                logger.debug(f"Failed to open PDF with {method_name} method: {e}")
                if doc:
                    try:
                        doc.close()
                    except:
                        pass
                doc = None
                continue
        
        # If PDF opening failed, try OCR as fallback
        if not doc:
            logger.warning(f"All PDF opening methods failed for: {file_path}, trying OCR fallback")
            if OCR_AVAILABLE:
                try:
                    # First check if file is actually HTML (common with download pages)
                    with open(file_path, "rb") as f:
                        first_1k = f.read(1024)
                        if first_1k.startswith(b'<!DOCTYPE') or first_1k.startswith(b'<html') or first_1k.startswith(b'<!doctype'):
                            logger.warning(f"File is HTML, not PDF. Attempting OCR on HTML content as image...")
                            # Try to treat HTML as an image and OCR it (might extract text from rendered page)
                            try:
                                from PIL import Image
                                # This won't work well, but let's try to extract any text from HTML
                                html_text = first_1k.decode('utf-8', errors='ignore')
                                # Look for text content in HTML
                                import re
                                text_in_html = re.sub(r'<[^>]+>', ' ', html_text)
                                text_in_html = ' '.join(text_in_html.split())
                                if len(text_in_html) > 50:  # If we found substantial text
                                    logger.info(f"Extracted {len(text_in_html)} characters from HTML")
                                    return [{
                                        'text': text_in_html,
                                        'confidence': 30.0,
                                        'method': 'html_text_extraction',
                                        'word_count': len(text_in_html.split()),
                                        'page': 1
                                    }]
                            except Exception as html_err:
                                logger.debug(f"HTML text extraction failed: {html_err}")
                    
                    # Try to convert PDF pages to images and use OCR
                    # This is a last resort - try opening one more time with repair
                    try:
                        with open(file_path, "rb") as f:
                            pdf_bytes = f.read()
                        # Check if it's actually a PDF
                        if pdf_bytes[:4] != b'%PDF':
                            logger.warning(f"File does not have PDF magic bytes, attempting to open as image for OCR...")
                            try:
                                from PIL import Image
                                image = Image.open(file_path)
                                ocr_results = self._extract_text_ultimate(image)
                                if ocr_results:
                                    logger.info(f"OCR on non-PDF file extracted {len(ocr_results)} text results")
                                    return ocr_results
                            except Exception as img_err:
                                logger.debug(f"Could not open as image: {img_err}")
                        
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        if doc and doc.page_count > 0:
                            logger.info(f"Successfully opened PDF after repair attempt ({doc.page_count} pages)")
                    except Exception as repair_err:
                        # If still failing, try OCR on first page as image
                        logger.info(f"PDF repair failed ({repair_err}), attempting to render first page as image for OCR...")
                        try:
                            # Try one more time to open
                            temp_doc = fitz.open(file_path)
                            if temp_doc.page_count > 0:
                                page = temp_doc[0]
                                mat = fitz.Matrix(2.0, 2.0)
                                pix = page.get_pixmap(matrix=mat)
                                img_data = pix.tobytes("png")
                                from PIL import Image
                                import io
                                image = Image.open(io.BytesIO(img_data))
                                ocr_results = self._extract_text_ultimate(image)
                                temp_doc.close()
                                if ocr_results:
                                    logger.info(f"OCR fallback extracted {len(ocr_results)} text results")
                                    return ocr_results
                            temp_doc.close()
                        except Exception as ocr_err:
                            logger.warning(f"OCR fallback also failed: {ocr_err}")
                            # Last resort: try opening the file as an image directly
                            try:
                                from PIL import Image
                                image = Image.open(file_path)
                                ocr_results = self._extract_text_ultimate(image)
                                if ocr_results:
                                    logger.info(f"Direct image OCR extracted {len(ocr_results)} text results")
                                    return ocr_results
                            except Exception as final_err:
                                logger.debug(f"Final image OCR attempt failed: {final_err}")
                            return []
                except Exception as e:
                    logger.error(f"OCR fallback failed: {e}")
                    return []
            else:
                logger.error(f"PDF cannot be opened and OCR not available")
                return []
        
        if not doc:
            logger.error(f"All PDF opening methods and OCR fallback failed for: {file_path}")
            return []
        
        try:
            results = []
            all_texts = set()
            
            # Process all pages to ensure complete document extraction
            pages_to_process = len(doc)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            page_limit = self.pdf_page_limit
            effective_pages = pages_to_process
            if page_limit > 0 and pages_to_process > page_limit:
                effective_pages = page_limit
                logger.warning(f"Page processing limited to {page_limit} pages by configuration (total pages: {pages_to_process})")
            
            # Quick sample to determine if PDF already has rich text
            sample_pages = min(3, effective_pages)
            total_sample_words = 0
            for sample_idx in range(sample_pages):
                sample_page = doc[sample_idx]
                sample_text = sample_page.get_text()
                total_sample_words += len(sample_text.split()) if sample_text.strip() else 0
            
            avg_words_per_page = total_sample_words / sample_pages if sample_pages > 0 else 0
            is_text_based_pdf = avg_words_per_page > 30  # If average >30 words/page, likely text-based
            if is_text_based_pdf:
                logger.info(f"Detected text-based PDF (avg {avg_words_per_page:.0f} words/page). OCR only when needed.")
            
            logger.info(f"Processing PDF: {effective_pages} pages, file size: {file_size_mb:.2f} MB, text-based: {is_text_based_pdf}")
            
            results_by_page: Dict[int, List[Dict[str, Any]]] = {}
            combined_segments: List[str] = []
            
            # PHASE 1: Parallel extraction from all pages (Direct text + Render for OCR)
            def process_pdf_page(p_num):
                try:
                    p = doc.load_page(p_num)
                    # 1. Direct text extraction (fast)
                    t = p.get_text()
                    t_clean = t.strip()
                    w_count = len(t_clean.split()) if t_clean else 0
                    
                    page_result = None
                    if t_clean:
                        page_result = {
                            'text': t_clean,
                            'confidence': 100.0,
                            'method': 'pymupdf_direct',
                            'word_count': w_count,
                            'page': p_num + 1
                        }
                    
                    # 2. Check if OCR is needed
                    ocr_img_bytes = None
                    needs_ocr = OCR_AVAILABLE and not is_text_based_pdf and w_count < self.ocr_trigger_min_words
                    if needs_ocr:
                        # Rendering is slow, so we do it in parallel too
                        matrix = fitz.Matrix(self.ocr_zoom_factor, self.ocr_zoom_factor)
                        px = p.get_pixmap(matrix=matrix)
                        ocr_img_bytes = px.tobytes("png")
                        
                    return p_num, page_result, ocr_img_bytes
                except Exception as e:
                    logger.debug(f"Parallel page processing failed for page {p_num + 1}: {e}")
                    return p_num, None, None

            logger.info(f"Extracting text from {effective_pages} pages in parallel using {self.max_page_workers} workers...")
            ocr_tasks: List[Tuple[int, bytes]] = []
            
            # Using ThreadPoolExecutor because fitz is often GIL-friendly enough for I/O and rendering
            with ThreadPoolExecutor(max_workers=self.max_page_workers) as executor:
                page_futures = [executor.submit(process_pdf_page, i) for i in range(effective_pages)]
                for future in as_completed(page_futures):
                    p_num, page_result, ocr_img_bytes = future.result()
                    if page_result:
                        results_by_page.setdefault(p_num, []).append(page_result)
                        combined_segments.append(page_result['text'])
                    if ocr_img_bytes:
                        ocr_tasks.append((p_num, ocr_img_bytes))

            # PHASE 2: Parallel OCR for pages that needed it
            if ocr_tasks and OCR_AVAILABLE:
                logger.info(f"Running OCR on {len(ocr_tasks)} pages using {self.max_ocr_workers} workers")
                with ThreadPoolExecutor(max_workers=self.max_ocr_workers) as executor:
                    future_map = {
                        executor.submit(self._run_fast_ocr_on_image, img_bytes): p_num
                        for p_num, img_bytes in ocr_tasks
                    }
                    for future in as_completed(future_map):
                        p_num = future_map[future]
                        try:
                            ocr_text = future.result()
                        except Exception as ocr_error:
                            logger.debug(f"OCR future failed for page {p_num + 1}: {ocr_error}")
                            continue
                        if ocr_text:
                            page_entry = {
                                'text': ocr_text,
                                'confidence': 70.0,
                                'method': 'tesseract_parallel',
                                'word_count': len(ocr_text.split()),
                                'page': p_num + 1
                            }
                            results_by_page.setdefault(p_num, []).append(page_entry)
                            combined_segments.append(ocr_text)
            
            # Flatten results preserving page order
            ordered_results: List[Dict[str, Any]] = []
            for page_num in sorted(results_by_page.keys()):
                ordered_results.extend(results_by_page[page_num])
            
            if ordered_results:
                combined_text = " ".join(combined_segments) if combined_segments else " ".join(
                    [entry['text'] for entry in ordered_results]
                )
                combined_confidence = max(entry['confidence'] for entry in ordered_results)
                comprehensive_result = {
                    'text': combined_text,
                    'confidence': combined_confidence,
                    'method': 'comprehensive_pdf_parallel',
                    'word_count': len(combined_text.split()),
                    'pages_processed': effective_pages,
                    'extraction_methods': len(set(entry['method'] for entry in ordered_results))
                }
                logger.info(
                    f"Comprehensive PDF extraction: {len(combined_text.split())} words "
                    f"from {effective_pages} pages using {comprehensive_result['extraction_methods']} methods"
                )
                return [comprehensive_result]
            
            return []
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return []
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
    
    def _extract_text_pptx(self, file_path: str) -> str:
        """
        Extract text from PowerPoint files (.pptx, .pptm, .ppsx, .potx).
        
        Note: python-pptx library supports .pptx files, but .ppsx (slideshow) files
        have a different content type. This method tries python-pptx first, then
        falls back to direct ZIP/XML extraction for unsupported formats.
        
        Extracts text from:
        - Slide titles and content from all shapes
        - Text from tables
        - Notes pages (if available)
        
        Args:
            file_path: Path to the PowerPoint file (.pptx, .pptm, .ppsx, or .potx)
            
        Returns:
            Combined text content from all slides
        """
        if not PPTX_AVAILABLE:
            raise ImportError("python-pptx library not available")
        
        # Try python-pptx first (works for .pptx and some .pptm files)
        try:
            prs = Presentation(file_path)
            all_text = []
            
            # Extract text from each slide
            for slide_num, slide in enumerate(prs.slides, start=1):
                slide_texts = []
                
                # Extract text from shapes (text boxes, placeholders, etc.)
                for shape in slide.shapes:
                    # Check if shape has text
                    if hasattr(shape, "text") and shape.text:
                        text = shape.text.strip()
                        if text:
                            slide_texts.append(text)
                    
                    # Extract text from tables
                    if shape.has_table:
                        table = shape.table
                        table_rows = []
                        for row in table.rows:
                            row_cells = []
                            for cell in row.cells:
                                cell_text = cell.text.strip()
                                if cell_text:
                                    row_cells.append(cell_text)
                            if row_cells:
                                table_rows.append('\t'.join(row_cells))
                        
                        if table_rows:
                            slide_texts.append('\n'.join(table_rows))
                
                # Combine all text from this slide
                if slide_texts:
                    slide_content = '\n\n'.join(slide_texts)
                    all_text.append(f"--- Slide {slide_num} ---\n{slide_content}")
                
                # Try to extract notes (speaker notes) if available
                try:
                    if hasattr(slide, 'notes_slide') and slide.notes_slide:
                        notes_text = []
                        for shape in slide.notes_slide.shapes:
                            if hasattr(shape, "text") and shape.text:
                                text = shape.text.strip()
                                if text:
                                    notes_text.append(text)
                        if notes_text:
                            all_text.append(f"--- Notes for Slide {slide_num} ---\n" + '\n'.join(notes_text))
                except Exception as notes_error:
                    # Notes extraction is optional, log but don't fail
                    logger.debug(f"Could not extract notes from slide {slide_num}: {notes_error}")
            
            # Combine all slides into one text
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {len(prs.slides)} slides in PowerPoint file using python-pptx")
            
            return full_text
            
        except ValueError as e:
            # python-pptx raises ValueError for unsupported formats (.ppsx, some .pptm)
            # Fall back to direct ZIP/XML extraction
            if "is not a PowerPoint file" in str(e) or "content type" in str(e).lower():
                logger.info(f"python-pptx doesn't support this format, trying ZIP/XML extraction: {e}")
                return self._extract_text_pptx_from_zip(file_path)
            else:
                raise
        except Exception as e:
            logger.warning(f"python-pptx extraction failed, trying ZIP/XML fallback: {e}")
            return self._extract_text_pptx_from_zip(file_path)
    
    def _extract_text_pptx_from_zip(self, file_path: str) -> str:
        """
        Extract text from PowerPoint files by directly parsing ZIP/XML structure.
        This works for .ppsx, .pptm, .potx files that python-pptx doesn't support.
        
        Args:
            file_path: Path to the PowerPoint file
            
        Returns:
            Combined text content from all slides
        """
        import zipfile
        import xml.etree.ElementTree as ET
        
        all_text = []
        slide_count = 0
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Find all slide files
                slide_files = sorted([f for f in zip_ref.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')])
                
                if not slide_files:
                    logger.warning(f"No slide files found in {file_path}")
                    return ""
                
                # Namespaces for PowerPoint XML
                namespaces = {
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
                    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                }
                
                # Extract text from each slide
                for slide_file in slide_files:
                    slide_count += 1
                    slide_texts = []
                    
                    try:
                        xml_content = zip_ref.read(slide_file)
                        root = ET.fromstring(xml_content)
                        
                        # Find all text elements (a:t tags)
                        text_elements = root.findall('.//a:t', namespaces)
                        for elem in text_elements:
                            if elem.text and elem.text.strip():
                                slide_texts.append(elem.text.strip())
                        
                        # Also check for table text
                        table_cells = root.findall('.//a:tc', namespaces)
                        for cell in table_cells:
                            cell_texts = cell.findall('.//a:t', namespaces)
                            cell_text = ' '.join([t.text for t in cell_texts if t.text and t.text.strip()])
                            if cell_text.strip():
                                slide_texts.append(cell_text.strip())
                        
                        if slide_texts:
                            slide_content = '\n\n'.join(slide_texts)
                            all_text.append(f"--- Slide {slide_count} ---\n{slide_content}")
                    
                    except Exception as slide_error:
                        logger.debug(f"Error extracting text from {slide_file}: {slide_error}")
                        continue
                
                # Try to extract notes if available
                notes_files = sorted([f for f in zip_ref.namelist() if f.startswith('ppt/notesSlides/notesSlide') and f.endswith('.xml')])
                for notes_file in notes_files:
                    try:
                        xml_content = zip_ref.read(notes_file)
                        root = ET.fromstring(xml_content)
                        notes_texts = []
                        
                        text_elements = root.findall('.//a:t', namespaces)
                        for elem in text_elements:
                            if elem.text and elem.text.strip():
                                notes_texts.append(elem.text.strip())
                        
                        if notes_texts:
                            # Try to match notes to slide number
                            slide_num = slide_count  # Approximate
                            all_text.append(f"--- Notes for Slide {slide_num} ---\n" + '\n'.join(notes_texts))
                    except Exception:
                        pass  # Notes are optional
            
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {slide_count} slides using ZIP/XML extraction")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PowerPoint ZIP/XML {file_path}: {e}")
            raise
    def _extract_text_ppt(self, file_path: str) -> str:
        """
        Extract text from legacy PowerPoint files (.ppt - PowerPoint 97-2003 format).
        
        Uses multiple methods: ppt2txt, OLE stream extraction, and fallback to placeholder.
        
        Args:
            file_path: Path to the .ppt file
            
        Returns:
            Combined text content from all slides, or minimal placeholder if extraction fails
        """
        # Method 1: Try ppt2txt
        if PPT2TXT_AVAILABLE:
            try:
                result = ppt2txt.process(file_path)
                
                all_text = []
                if "content" in result and isinstance(result["content"], dict):
                    # result["content"] is a dict mapping slide numbers to text
                    for slide_num in sorted(result["content"].keys()):
                        slide_text = result["content"][slide_num]
                        if slide_text and slide_text.strip():
                            all_text.append(f"--- Slide {slide_num} ---\n{slide_text.strip()}")
                
                full_text = "\n\n".join(all_text)
                slide_count = len(result.get("content", {})) if isinstance(result.get("content"), dict) else 0
                
                if full_text.strip():
                    logger.info(f"Extracted text from {slide_count} slides in legacy .ppt file using ppt2txt")
                    return full_text
                else:
                    logger.warning("ppt2txt returned empty text, trying OLE stream extraction")
            except Exception as e:
                logger.warning(f"ppt2txt extraction failed: {e}, trying OLE stream extraction")
        
        # Method 2: Try OLE stream extraction
        try:
            import olefile
            if olefile.isOleFile(file_path):
                ole = olefile.OleFileIO(file_path)
                all_text = []
                
                # Try to read from PowerPoint Document stream
                if ole.exists('PowerPoint Document'):
                    try:
                        stream = ole.openstream('PowerPoint Document')
                        data = stream.read()
                        # Extract readable ASCII text (basic heuristic)
                        import re
                        text_matches = re.findall(rb'[\x20-\x7E]{4,}', data)
                        if text_matches:
                            extracted = []
                            for match in text_matches[:200]:  # Limit to avoid too much noise
                                try:
                                    text = match.decode('ascii', errors='ignore').strip()
                                    if len(text) >= 4 and text not in extracted:
                                        extracted.append(text)
                                except:
                                    pass
                            if extracted:
                                full_text = "\n".join(extracted)
                                logger.info(f"Extracted {len(full_text)} characters from legacy .ppt file using OLE stream extraction")
                                ole.close()
                                return full_text
                    except Exception as stream_error:
                        logger.debug(f"Could not read PowerPoint Document stream: {stream_error}")
                
                ole.close()
        except ImportError:
            logger.debug("olefile not available for OLE stream extraction")
        except Exception as e:
            logger.debug(f"OLE stream extraction failed: {e}")
        
        # Method 3: Return minimal placeholder to prevent "No text extracted" error
        logger.warning(f"All extraction methods failed for .ppt file {file_path}, returning placeholder text")
        return f"PowerPoint presentation file: {Path(file_path).name}\n[Text extraction from legacy .ppt format was not possible. File may contain only images or use unsupported format.]"


    def _extract_text_xlsx(self, file_path: str) -> str:
        """
        Extract text from Excel files (.xlsx, .xlsm, .xltx, .xlsb).
        
        Uses openpyxl to extract text from all sheets and cells.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Combined text content from all sheets
        """
        if not XLSX_AVAILABLE:
            raise ImportError("openpyxl library not available for Excel files")
        
        try:
            # Prefer minimal workbook features for speed/memory
            try:
                workbook = load_workbook(
                    file_path,
                    data_only=True,
                    read_only=True,
                    keep_links=False
                )
            except TypeError:
                # Older openpyxl versions may not support keep_links
                workbook = load_workbook(file_path, data_only=True, read_only=True)
            all_text = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_texts = []
                
                # Extract all cell values
                for row in sheet.iter_rows(values_only=True):
                    # OPTIMIZED: Use local bindings and skip empty values quickly
                    row_cells = []
                    append_cell = row_cells.append
                    for val in row:
                        if val is None:
                            continue
                        if isinstance(val, str):
                            stripped = val.strip()
                            if stripped:
                                append_cell(stripped)
                        else:
                            append_cell(str(val))
                    
                    
                    if row_cells:
                        sheet_texts.append('\t'.join(row_cells))
                
                if sheet_texts:
                    sheet_content = '\n'.join(sheet_texts)
                    all_text.append(f"--- Sheet: {sheet_name} ---\n{sheet_content}")
            
            workbook.close()
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {len(workbook.sheetnames)} sheets in Excel file using openpyxl")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from Excel file {file_path}: {e}")
            raise

    def _extract_text_xlsx_from_zip(self, file_path: str) -> str:
        """
        Extract text from Excel files by directly parsing ZIP/XML structure.
        This works for .xlsm, .xlsb, .xltx files that openpyxl might not fully support.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Combined text content from all sheets
        """
        if not ZIP_XML_AVAILABLE:
            raise ImportError("zipfile/xml not available for Excel ZIP/XML extraction")
        
        all_text = []
        sheet_count = 0
        
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            from io import BytesIO
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Find shared strings (common text values)
                shared_strings = []
                try:
                    if 'xl/sharedStrings.xml' in zip_ref.namelist():
                        strings_xml = zip_ref.read('xl/sharedStrings.xml')
                        ns_main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
                        tag_si = f'{{{ns_main}}}si'
                        tag_t = f'{{{ns_main}}}t'
                        for _, elem in ET.iterparse(BytesIO(strings_xml), events=('end',)):
                            if elem.tag == tag_si:
                                text_parts = [t.text for t in elem.iter(tag_t) if t.text]
                                if text_parts:
                                    shared_strings.append(''.join(text_parts))
                                elem.clear()
                except Exception as e:
                    logger.debug(f"Could not read shared strings: {e}")
                
                # Get sheet names from workbook.xml
                sheet_names = {}
                try:
                    if 'xl/workbook.xml' in zip_ref.namelist():
                        workbook_xml = zip_ref.read('xl/workbook.xml')
                        workbook_root = ET.fromstring(workbook_xml)
                        wb_ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main', 
                                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
                        sheets = workbook_root.findall('.//main:sheet', wb_ns)
                        relationships = {}
                        # Get relationships to map sheet IDs to names
                        if 'xl/_rels/workbook.xml.rels' in zip_ref.namelist():
                            rels_xml = zip_ref.read('xl/_rels/workbook.xml.rels')
                            rels_root = ET.fromstring(rels_xml)
                            rels_ns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
                            for rel in rels_root.findall('.//r:Relationship', rels_ns):
                                rel_id = rel.get('Id')
                                target = rel.get('Target')
                                if target and 'worksheets' in target:
                                    relationships[rel_id] = target
                        # Map sheet IDs to names
                        for sheet in sheets:
                            sheet_id = sheet.get('sheetId')
                            sheet_name = sheet.get('name', f'Sheet{sheet_id}')
                            r_id = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                            if r_id and r_id in relationships:
                                # Extract sheet number from relationship target
                                target = relationships[r_id]
                                sheet_num = target.split('sheet')[1].split('.')[0] if 'sheet' in target else None
                                if sheet_num:
                                    sheet_names[f'xl/worksheets/sheet{sheet_num}.xml'] = sheet_name
                except Exception as e:
                    logger.debug(f"Could not read sheet names: {e}")
                
                # Find all worksheet files
                worksheet_files = sorted([f for f in zip_ref.namelist() 
                                         if f.startswith('xl/worksheets/sheet') and f.endswith('.xml')])
                
                if not worksheet_files:
                    logger.warning(f"No worksheet files found in {file_path}")
                    return ""
                
                # Extract text from each worksheet
                for sheet_file in worksheet_files:
                    sheet_count += 1
                    sheet_name = sheet_names.get(sheet_file, f'Sheet {sheet_count}')
                    sheet_texts = []
                    
                    try:
                        xml_content = zip_ref.read(sheet_file)
                        ns_main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
                        tag_row = f'{{{ns_main}}}row'
                        tag_cell = f'{{{ns_main}}}c'
                        tag_v = f'{{{ns_main}}}v'
                        
                        # Extract cell values using iterparse for speed/memory
                        for _, row in ET.iterparse(BytesIO(xml_content), events=('end',)):
                            if row.tag != tag_row:
                                continue
                            row_cells = []
                            append_cell = row_cells.append
                            for cell in row.findall(tag_cell):
                                value_elem = cell.find(tag_v)
                                if value_elem is None or value_elem.text is None:
                                    continue
                                cell_value = value_elem.text
                                # Check if it's a shared string reference
                                if cell.get('t') == 's' and shared_strings:
                                    try:
                                        idx = int(cell_value)
                                        if 0 <= idx < len(shared_strings):
                                            cell_value = shared_strings[idx]
                                    except (ValueError, IndexError):
                                        pass
                                append_cell(cell_value)
                            
                            if row_cells:
                                sheet_texts.append('\t'.join(row_cells))
                            row.clear()
                        
                        if sheet_texts:
                            sheet_content = '\n'.join(sheet_texts)
                            all_text.append(f"--- Sheet: {sheet_name} ---\n{sheet_content}")
                    
                    except Exception as sheet_error:
                        logger.debug(f"Error extracting text from {sheet_file}: {sheet_error}")
                        continue
            
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {sheet_count} sheets using ZIP/XML extraction")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from Excel ZIP/XML {file_path}: {e}")
            raise

    def _extract_text_xls(self, file_path: str) -> str:
        """
        Extract text from legacy Excel files (.xls - Excel 97-2003 format).
        
        Uses xlrd library to extract text from binary .xls format.
        
        Args:
            file_path: Path to the .xls file
            
        Returns:
            Combined text content from all sheets
        """
        if not XLS_AVAILABLE:
            raise ImportError("xlrd library not available for legacy .xls files")
        
        try:
            import xlrd
            workbook = xlrd.open_workbook(file_path, on_demand=True)
            all_text = []
            
            for sheet_idx, sheet in enumerate(workbook.sheets(), 1):
                sheet_texts = []
                
                for row_idx in range(sheet.nrows):
                    # OPTIMIZED: Use row_values() for much faster bulk access (10-50x faster)
                    row_values = sheet.row_values(row_idx)
                    row_types = sheet.row_types(row_idx)
                    
                    row_cells = []
                    for col_idx, (cell_value, cell_type) in enumerate(zip(row_values, row_types)):
                        if cell_value:
                            # Handle dates efficiently
                            if cell_type == xlrd.XL_CELL_DATE:
                                try:
                                    date_tuple = xlrd.xldate_as_tuple(cell_value, workbook.datemode)
                                    row_cells.append(f"{date_tuple[0]}-{date_tuple[1]:02d}-{date_tuple[2]:02d}")
                                except:
                                    row_cells.append(str(cell_value))
                            elif isinstance(cell_value, str):
                                row_cells.append(cell_value.strip())
                            else:
                                row_cells.append(str(cell_value))
                    
                    if row_cells:
                        sheet_texts.append('\t'.join(row_cells))
                
                if sheet_texts:
                    sheet_content = '\n'.join(sheet_texts)
                    all_text.append(f"--- Sheet: {sheet.name} ---\n{sheet_content}")
            
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {len(workbook.sheets())} sheets in legacy .xls file using xlrd")
            
            workbook.release_resources()
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from legacy .xls file {file_path}: {e}")
            raise

    def _extract_text_ods(self, file_path: str) -> str:
        """
        Extract text from OpenDocument Spreadsheet files (.ods).
        
        Uses odfpy library to extract text from .ods format.
        
        Args:
            file_path: Path to the .ods file
            
        Returns:
            Combined text content from all sheets
        """
        if not ODS_AVAILABLE:
            raise ImportError("odfpy library not available for .ods files")
        
        try:
            from odf.opendocument import load as odf_load
            from odf.table import Table, TableRow, TableCell
            from odf import text as odf_text
            
            doc = odf_load(file_path)
            all_text = []
            sheet_count = 0
            
            # Extract text from each table (sheet)
            tables = doc.getElementsByType(Table)
            for table in tables:
                sheet_count += 1
                sheet_texts = []
                
                for row in table.getElementsByType(TableRow):
                    row_cells = []
                    for cell in row.getElementsByType(TableCell):
                        # Get text content from cell
                        text_parts = []
                        for paragraph in cell.getElementsByType(odf_text.P):
                            if paragraph.textContent:
                                text_parts.append(paragraph.textContent.strip())
                        
                        cell_text = ' '.join(text_parts) if text_parts else ''
                        if cell_text:
                            row_cells.append(cell_text)
                    
                    if row_cells:
                        sheet_texts.append('\t'.join(row_cells))
                
                if sheet_texts:
                    sheet_content = '\n'.join(sheet_texts)
                    all_text.append(f"--- Sheet {sheet_count} ---\n{sheet_content}")
            
            full_text = '\n\n'.join(all_text)
            logger.info(f"Extracted text from {sheet_count} sheets in ODS file using odfpy")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from ODS file {file_path}: {e}")
            raise

    def _extract_text_csv(self, file_path: str) -> str:
        """
        Extract text from CSV files (.csv).
        
        Uses Python's built-in csv module to extract text from CSV format.
        Handles various delimiters (comma, semicolon, tab) and encodings.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Combined text content from all rows
        """
        import csv
        import os
        
        # Reset warning flags for this extraction call (instance-level flags)
        if hasattr(self, '_csv_row_limit_warned'):
            delattr(self, '_csv_row_limit_warned')
        if hasattr(self, '_csv_char_limit_warned'):
            delattr(self, '_csv_char_limit_warned')
        
        try:
            # Fast mode for large files: avoid repeated passes over the file
            fast_mode_min_bytes = int(os.getenv("CSV_FAST_MODE_MIN_BYTES", "0"))
            file_size = os.path.getsize(file_path)
            fast_mode = file_size >= fast_mode_min_bytes
            # Hard caps to prevent large CSVs from exhausting memory during extraction.
            # Defaults align with embedding limits but can be overridden independently.
            extract_row_limit = int(
                os.getenv("CSV_EXTRACT_ROW_LIMIT", os.getenv("CSV_EMBED_ROW_LIMIT", "200000"))
            )
            extract_char_limit = int(
                os.getenv("CSV_EXTRACT_CHAR_LIMIT", os.getenv("CSV_EMBED_CHAR_LIMIT", "2000000"))
            )
            # Hard safety caps: keep CSV extraction bounded regardless of env values.
            extract_row_limit = min(extract_row_limit, 25000)
            extract_char_limit = min(extract_char_limit, 250000)
            
            # Try different encodings; detect delimiter once per encoding
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            delimiters = [',', ';', '\t', '|']
            
            def _choose_delimiter(sample: str) -> str:
                sniffer = csv.Sniffer()
                try:
                    return sniffer.sniff(sample, delimiters=delimiters).delimiter
                except csv.Error:
                    counts = {d: sample.count(d) for d in delimiters}
                    return max(counts, key=counts.get) if any(counts.values()) else ','
            
            if fast_mode:
                # Single-pass decode with replacement to avoid expensive retries
                with open(file_path, 'r', encoding='utf-8-sig', errors='replace', newline='') as csvfile:
                    sample = csvfile.read(4096)
                    csvfile.seek(0)
                    detected_delimiter = _choose_delimiter(sample)
                    
                    reader = csv.reader(csvfile, delimiter=detected_delimiter)
                    all_rows = []
                    append_row = all_rows.append
                    row_count = 0
                    char_count = 0
                    
                    for row in reader:
                        row_cells = [cell.strip() for cell in row if cell and cell.strip()]
                        if row_cells:
                            line = '\t'.join(row_cells)
                            # Bound CSV extraction so very large files do not exhaust worker memory.
                            if row_count >= extract_row_limit and not hasattr(self, '_csv_row_limit_warned'):
                                logger.warning(
                                    f"[CSV EXTRACTION] Hit row limit ({extract_row_limit}), "
                                    "stopping extraction for stability"
                                )
                                self._csv_row_limit_warned = True
                                break
                            if char_count + len(line) > extract_char_limit and not hasattr(self, '_csv_char_limit_warned'):
                                logger.warning(
                                    f"[CSV EXTRACTION] Hit char limit ({extract_char_limit}), "
                                    "stopping extraction for stability"
                                )
                                self._csv_char_limit_warned = True
                                break
                            append_row(line)
                            row_count += 1
                            char_count += len(line)
                
                full_text = '\n'.join(all_rows)
                logger.info(f"Extracted text from CSV file: {len(all_rows)} rows (fast mode)")
                return full_text
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
                        sample = csvfile.read(4096)
                        csvfile.seek(0)
                        detected_delimiter = _choose_delimiter(sample)
                        
                        reader = csv.reader(csvfile, delimiter=detected_delimiter)
                        all_rows = []
                        append_row = all_rows.append
                        row_count = 0
                        char_count = 0
                        
                        for row in reader:
                            # Filter out empty cells and convert to strings
                            row_cells = [cell.strip() for cell in row if cell and cell.strip()]
                            if row_cells:
                                line = '\t'.join(row_cells)
                                # Bound CSV extraction so very large files do not exhaust worker memory.
                                if row_count >= extract_row_limit and not hasattr(self, '_csv_row_limit_warned'):
                                    logger.warning(
                                        f"[CSV EXTRACTION] Hit row limit ({extract_row_limit}), "
                                        "stopping extraction for stability"
                                    )
                                    self._csv_row_limit_warned = True
                                    break
                                if char_count + len(line) > extract_char_limit and not hasattr(self, '_csv_char_limit_warned'):
                                    logger.warning(
                                        f"[CSV EXTRACTION] Hit char limit ({extract_char_limit}), "
                                        "stopping extraction for stability"
                                    )
                                    self._csv_char_limit_warned = True
                                    break
                                append_row(line)
                                row_count += 1
                                char_count += len(line)
                    
                    full_text = '\n'.join(all_rows)
                    logger.info(f"Extracted text from CSV file: {len(all_rows)} rows")
                    return full_text
                except UnicodeDecodeError:
                    continue
                except csv.Error:
                    continue
            
            # Final fallback: decode with replacement to avoid failures on bad encodings
            with open(file_path, 'r', encoding='utf-8', errors='replace', newline='') as csvfile:
                sample = csvfile.read(4096)
                csvfile.seek(0)
                detected_delimiter = _choose_delimiter(sample)
                reader = csv.reader(csvfile, delimiter=detected_delimiter)
                all_rows = []
                append_row = all_rows.append
                row_count = 0
                char_count = 0
                
                for row in reader:
                    row_cells = [cell.strip() for cell in row if cell and cell.strip()]
                    if row_cells:
                        line = '\t'.join(row_cells)
                        # Bound CSV extraction so very large files do not exhaust worker memory.
                        if row_count >= extract_row_limit and not hasattr(self, '_csv_row_limit_warned'):
                            logger.warning(
                                f"[CSV EXTRACTION] Hit row limit ({extract_row_limit}), "
                                "stopping extraction for stability"
                            )
                            self._csv_row_limit_warned = True
                            break
                        if char_count + len(line) > extract_char_limit and not hasattr(self, '_csv_char_limit_warned'):
                            logger.warning(
                                f"[CSV EXTRACTION] Hit char limit ({extract_char_limit}), "
                                "stopping extraction for stability"
                            )
                            self._csv_char_limit_warned = True
                            break
                        append_row(line)
                        row_count += 1
                        char_count += len(line)
            
            full_text = '\n'.join(all_rows)
            logger.info(f"Extracted text from CSV file: {len(all_rows)} rows (fallback)")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from CSV file {file_path}: {e}")
            raise


    def _normalize_text(self, text: str) -> str:
        """Intelligent text normalization for maximum searchability."""
        if not text:
            return ""
        
        # Keep original case for better exact matching
        # Don't convert to lowercase immediately - preserve case information
        
        # Remove extra whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors that might affect searchability
        ocr_fixes = {
            '|': 'I',  # Common OCR error
            '0': 'O',  # In certain contexts
            '1': 'I',  # In certain contexts
            '5': 'S',  # In certain contexts
            '8': 'B',  # In certain contexts
        }
        
        # Apply OCR fixes only in specific contexts to avoid over-correction
        for wrong, correct in ocr_fixes.items():
            # Only replace if it's likely an OCR error (surrounded by letters)
            text = re.sub(f'(?<=[a-zA-Z]){wrong}(?=[a-zA-Z])', correct, text)
        
        # Preserve original case and punctuation for better matching
        # This helps with exact word matching and maintains text integrity
        return text.strip()
    
    def _find_ultimate_matches(self, text: str, target_words: List[str]) -> Dict[str, float]:
        """Find matches using all available methods."""
        if not text:
            return {}
        
        matches = {}
        normalized_text = self._normalize_text(text)
        
        for word in target_words:
            max_confidence = 0
            
            # Direct matching
            if word.lower() in normalized_text:
                max_confidence = 100
            
            # Fuzzy matching
            if FUZZY_AVAILABLE:
                ratio = fuzz.partial_ratio(word.lower(), normalized_text)
                max_confidence = max(max_confidence, ratio)
            
            # Word variations
            if word in self.word_variations:
                for variation in self.word_variations[word]:
                    if variation.lower() in normalized_text:
                        max_confidence = max(max_confidence, 90)
                    
                    if FUZZY_AVAILABLE:
                        ratio = fuzz.partial_ratio(variation.lower(), normalized_text)
                        max_confidence = max(max_confidence, ratio)
            
            # Synonyms
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    if synonym.lower() in normalized_text:
                        max_confidence = max(max_confidence, 85)
                    
                    if FUZZY_AVAILABLE:
                        ratio = fuzz.partial_ratio(synonym.lower(), normalized_text)
                        max_confidence = max(max_confidence, ratio)
            
            # Regex patterns
            if word in self.patterns:
                for pattern in self.patterns[word]:
                    if re.search(pattern, normalized_text, re.IGNORECASE):
                        max_confidence = max(max_confidence, 80)
            
            if max_confidence > 60:  # Threshold for matching
                matches[word] = max_confidence
        
        return matches
    
    def _run_fast_ocr_on_image(self, image_bytes: bytes) -> str:
        """Run OCR on in-memory image bytes (used for parallel OCR)."""
        if not OCR_AVAILABLE:
            return ""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.ocr_fast_config
            )
            return text.strip()
        except Exception as exc:
            logger.debug(f"Fast OCR worker failed: {exc}")
            return ""


    def process_document(self, file_path: str, target_words: List[str] = None, file_type: str = None) -> UltimateSearchResult:
        """Process document with advanced OCR and text extraction."""
        start_time = time.time()
        
        # If file_type is not provided, we'll detect from filename extension below
        # This allows fallback when file_type is None, empty, or application/octet-stream
        if not file_type:
            logger.info(f"No file_type provided for file: {file_path}, will attempt filename-based detection")
        
        
        # Map MIME types and generic types to file extensions
        mime_to_extension = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-word': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml': '.docx',
            'application/vnd.ms-powerpoint': '.ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/vnd.openxmlformats-officedocument.presentationml': '.pptx',
            'application/vnd.ms-powerpoint.presentation.macroEnabled.12': '.pptm',
            'application/vnd.ms-powerpoint.presentation.macroenabled.12': '.pptm',  # Case variation
            'application/vnd.ms-powerpoint.template.macroEnabled.12': '.potm',
            'application/vnd.ms-powerpoint.template.macroenabled.12': '.potm',  # Case variation
            'application/vnd.ms-powerpoint.slideshow.macroEnabled.12': '.ppsm',
            'application/vnd.ms-powerpoint.slideshow.macroenabled.12': '.ppsm',  # Case variation
            'application/vnd.openxmlformats-officedocument.presentationml.slideshow': '.ppsx',
            'application/vnd.openxmlformats-officedocument.presentationml.template': '.potx',
            'application/mspowerpoint': '.ppt',  # Alternative MIME type for PPT
            'application/powerpoint': '.ppt',  # Alternative MIME type for PPT
            'application/vnd.oasis.opendocument.presentation': '.odp',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml': '.xlsx',
            'application/vnd.ms-excel.sheet.macroEnabled.12': '.xlsm',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.template': '.xltx',
            'application/vnd.ms-excel.template.macroEnabled.12': '.xltm',
            'application/vnd.ms-excel.sheet.binary.macroEnabled.12': '.xlsb',
            'application/vnd.oasis.opendocument.spreadsheet': '.ods',
            'image/jpeg': '.jpeg',
            'image/jpg': '.jpg', 
            'image/png': '.png',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/webp': '.webp',
            'image/tiff': '.tiff',
            'image/svg+xml': '.svg',
            'text/plain': '.txt',
            'text/html': '.html',
            'text/markdown': '.md',
            'text/csv': '.csv',
            'application/csv': '.csv',
            # Generic types
            'pdf': '.pdf',
            'doc': '.doc',
            'docx': '.docx',
            'ppt': '.ppt',
            'pptx': '.pptx',
            'pptm': '.pptm',
            'potm': '.potm',
            'ppsm': '.ppsm',
            'ppsx': '.ppsx',
            'potx': '.potx',
            'odp': '.odp',
            'xls': '.xls',
            'xlsx': '.xlsx',
            'xlsm': '.xlsm',
            'xltx': '.xltx',
            'xltm': '.xltm',
            'xlsb': '.xlsb',
            'ods': '.ods',
            'png': '.png',
            'jpg': '.jpg',
            'jpeg': '.jpeg',
            'gif': '.gif',
            'bmp': '.bmp',
            'webp': '.webp',
            'tiff': '.tiff',
            'svg': '.svg',
            'txt': '.txt',
            'html': '.html',
            'md': '.md',
            'csv': '.csv',
            'image': '.jpg'
        }
        
        file_extension = mime_to_extension.get(file_type.lower() if file_type else '', None)
        
        # Always check filename extension - it's more reliable than MIME type
        # This is especially important for PowerPoint formats where generic MIME types
        # (like application/vnd.ms-powerpoint) don't distinguish between .ppt, .pptm, .ppsm, etc.
        filename_ext = Path(file_path).suffix.lower()
        if filename_ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.pptm', '.ppsx', '.potx', '.potm', '.ppsm', '.odp', '.xls', '.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsb', '.ods', '.txt', '.md', '.html', '.rtf', '.csv', 
                          '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.svg']:
            # Prioritize filename extension over MIME type mapping
            # This ensures .pptm files are correctly identified even with generic MIME types
            if file_extension and filename_ext != file_extension:
                logger.info(f"Using filename extension '{filename_ext}' instead of MIME-mapped '{file_extension}' (MIME type was '{file_type}')")
            file_extension = filename_ext
        
        # If MIME type is unknown or application/octet-stream and filename didn't help, try file signature
        if not file_extension or (file_type and file_type.lower() == 'application/octet-stream' and not filename_ext):
                # Try to detect PDF by file signature
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(4)
                        if header.startswith(b'%PDF'):
                            file_extension = '.pdf'
                            logger.info(f"Detected PDF from file signature (MIME type was '{file_type}')")
                        else:
                            file_extension = '.txt'  # Default fallback
                            logger.warning(f"Unknown file type '{file_type}', defaulting to .txt")
                except Exception as e:
                    logger.warning(f"Could not read file signature: {e}, defaulting to .txt")
                    file_extension = '.txt'
        
        if not file_extension:
            logger.error(f"Unsupported file type: {file_type}")
            return UltimateSearchResult(
                file_path=file_path,
                file_type=file_type,
                success=False,
                text_content="",
                normalized_text="",
                keywords=[],
                searchable_text="",
                processing_time=time.time() - start_time,
                extraction_method="none",
                confidence=0.0,
                word_variations={},
                fuzzy_matches={},
                pattern_matches={},
                metadata={},
                error=f"Unsupported file type: {file_type}"
            )
        
        logger.info(f"Processing file with type '{file_type}' -> extension '{file_extension}' for file: {file_path}")
        
        # Add timeout protection - increased for large files
        # Check file size first
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 10:
            MAX_PROCESSING_TIME = 1800  # 30 minutes for very large files (>10MB)
        elif file_size_mb > 4:
            MAX_PROCESSING_TIME = 1200  # 20 minutes for large files (>4MB)
        elif file_size_mb > 3:
            MAX_PROCESSING_TIME = 900  # 15 minutes for medium-large files (>3MB)
        else:
            MAX_PROCESSING_TIME = 600  # 10 minutes for normal files
        
        # Initialize workflow manager
        workflow_manager = None
        if WORKFLOW_AVAILABLE and self.file_id:
            workflow_manager = get_workflow_manager()
            workflow_manager.update_processing(self.file_id)
        
        try:
            all_text_results = []
            
            if file_extension == '.pdf':
                # Process PDF
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "PDF_PROCESSING")
                pdf_results = self._extract_text_pdf(file_path)
                all_text_results.extend(pdf_results)
            
            elif file_extension in ['.doc', '.docx']:
                # Process DOC/DOCX files
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "DOCX_PROCESSING")
                
                docx_text = ""
                extraction_method = "docx_extraction"

                # Fast path for legacy .doc using antiword
                if file_extension == '.doc':
                    try:
                        antiword_result = subprocess.run(
                            ["antiword", file_path],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                        if antiword_result.returncode == 0 and antiword_result.stdout and antiword_result.stdout.strip():
                            docx_text = antiword_result.stdout
                            extraction_method = "antiword"
                            logger.info(f"Extracted {len(docx_text)} characters from DOC using antiword (fast path)")
                        elif antiword_result.stderr:
                            logger.warning(f"antiword returned code {antiword_result.returncode}: {antiword_result.stderr[:200]}")
                    except FileNotFoundError:
                        logger.warning("antiword not found, falling back to other DOC extractors")
                    except Exception as e:
                        logger.warning(f"antiword extraction failed: {e}")
                
                # Try python-docx first (better for .docx)
                if DOCX_AVAILABLE and file_extension == '.docx':
                    try:
                        doc = DocxDocument(file_path)
                        paragraphs = []
                        for para in doc.paragraphs:
                            if para.text.strip():
                                paragraphs.append(para.text)
                        
                        # Also extract text from tables
                        for table in doc.tables:
                            for row in table.rows:
                                for cell in row.cells:
                                    if cell.text.strip():
                                        paragraphs.append(cell.text)
                        
                        docx_text = '\n'.join(paragraphs)
                        if docx_text.strip():
                            extraction_method = "python_docx"
                            logger.info(f"Extracted {len(docx_text)} characters from DOCX using python-docx")
                    except Exception as e:
                        logger.warning(f"python-docx extraction failed: {e}, trying docx2python")
                
                # Try docx2python as fallback (works for both .doc and .docx)
                if not docx_text.strip() and DOCX2PYTHON_AVAILABLE:
                    try:
                        docx_content = docx2python(file_path)
                        docx_text = docx_content.text
                        if docx_text.strip():
                            extraction_method = "docx2python"
                            logger.info(f"Extracted {len(docx_text)} characters from DOC/DOCX using docx2python")
                    except Exception as e:
                        logger.warning(f"docx2python extraction failed: {e}")
                
                # If both methods failed, try OCR as last resort
                if not docx_text.strip() and OCR_AVAILABLE:
                    logger.info("DOC/DOCX text extraction failed, trying OCR as fallback")
                    try:
                        # For now, just log the error
                        logger.warning("OCR fallback for DOC/DOCX not implemented - file may need conversion to PDF first")
                    except Exception as e:
                        logger.error(f"OCR fallback failed: {e}")
                
                if docx_text.strip():
                    all_text_results.append({
                        'text': docx_text.strip(),
                        'confidence': 95.0 if extraction_method != "ocr" else 70.0,
                        'method': extraction_method,
                        'word_count': len(docx_text.split()),
                        'page_number': 1
                    })
                    logger.info(f"DOC/DOCX file processed: {len(docx_text)} characters, {len(docx_text.split())} words")
                else:
                    logger.error(f"Failed to extract text from DOC/DOCX file: {file_path}")
            
            elif file_extension in ['.ppt', '.pptx', '.pptm', '.ppsx', '.potx', '.potm', '.ppsm', '.odp']:
                # Process PowerPoint files
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "PPTX_PROCESSING")
                
                pptx_text = ""
                extraction_method = "pptx_extraction"
                
                # Try python-pptx for OpenXML PowerPoint formats (.pptx, .pptm, .ppsx, .potx)
                # python-pptx can handle all OpenXML-based PowerPoint formats
                if PPTX_AVAILABLE and file_extension in ['.pptx', '.pptm', '.ppsx', '.potx', '.potm', '.ppsm']:
                    try:
                        pptx_text = self._extract_text_pptx(file_path)
                        if pptx_text.strip():
                            extraction_method = "python_pptx"
                            logger.info(f"Extracted {len(pptx_text)} characters from PPTX using python-pptx")
                        else:
                            # python-pptx returned empty, try ZIP/XML extraction
                            logger.info("python-pptx returned empty, trying ZIP/XML extraction")
                            pptx_text = self._extract_text_pptx_from_zip(file_path)
                            if pptx_text.strip():
                                extraction_method = "zip_xml"
                                logger.info(f"Extracted {len(pptx_text)} characters using ZIP/XML extraction")
                    except Exception as e:
                        logger.warning(f"PPTX extraction failed: {e}, trying ZIP/XML fallback")
                        try:
                            pptx_text = self._extract_text_pptx_from_zip(file_path)
                            if pptx_text.strip():
                                extraction_method = "zip_xml"
                                logger.info(f"Extracted {len(pptx_text)} characters using ZIP/XML extraction")
                            else:
                                logger.warning("ZIP/XML extraction returned empty, using fallback")
                                pptx_text = f"PowerPoint presentation file: {Path(file_path).name}\n[Text extraction from {file_extension} format returned no readable text. File may contain only images or use unsupported format.]"
                                extraction_method = "fallback"
                        except Exception as zip_error:
                            logger.error(f"ZIP/XML extraction also failed: {zip_error}")
                            # Return minimal placeholder to prevent failure
                            pptx_text = f"PowerPoint presentation file: {Path(file_path).name}\n[Text extraction failed: {str(zip_error)}]"
                            extraction_method = "fallback"
                
                # For .ppt (legacy format), use ppt2txt FIRST (before OpenXML methods)
                if file_extension == '.ppt':
                    try:
                        pptx_text = self._extract_text_ppt(file_path)
                        if pptx_text.strip():
                            extraction_method = "ppt2txt"
                            logger.info(f"Extracted {len(pptx_text)} characters from legacy .ppt")
                        # _extract_text_ppt now always returns text (even if placeholder), so we continue
                    except Exception as e:
                        logger.error(f"Legacy .ppt extraction failed: {e}")
                        # Return minimal placeholder to prevent failure
                        pptx_text = f"PowerPoint presentation file: {Path(file_path).name}\n[Text extraction failed: {str(e)}]"
                        extraction_method = "fallback"
                
                
                # For .odp (OpenDocument format), try to extract if odfpy is available
                elif file_extension == '.odp':
                    try:
                        # Try using odfpy for OpenDocument format
                        try:
                            from odf.opendocument import load
                            from odf.text import P
                            from odf.draw import Page
                            odp_doc = load(file_path)
                            odp_text = []
                            slide_count = 0
                            
                            # Extract text from each page/slide
                            pages = odp_doc.getElementsByType(Page)
                            for page_num, page in enumerate(pages, 1):
                                slide_count += 1
                                page_texts = []
                                
                                # Get all text elements in this page
                                for elem in page.getElementsByType(P):
                                    text = elem.textContent
                                    if text and text.strip():
                                        page_texts.append(text.strip())
                                
                                if page_texts:
                                    odp_text.append(f"--- Slide {slide_num} ---\n" + '\n\n'.join(page_texts))
                            
                            # Also get any standalone paragraphs (fallback)
                            if not odp_text:
                                for paragraph in odp_doc.getElementsByType(P):
                                    text = paragraph.textContent
                                    if text.strip():
                                        odp_text.append(text.strip())
                            
                            if odp_text:
                                pptx_text = '\n\n'.join(odp_text)
                                extraction_method = "odfpy_odp"
                                logger.info(f"Extracted {len(pptx_text)} characters from {slide_count} slides in ODP using odfpy")
                        except ImportError:
                            logger.warning(f"odfpy library not available for .odp files. Install with: pip install odfpy")
                            pptx_text = ""
                    except Exception as e:
                        logger.warning(f"ODP extraction failed: {e}")
                        pptx_text = ""
                
                # Always add result, even if minimal (prevents "No text extracted" error)
                if pptx_text.strip():
                    all_text_results.append({
                        'text': pptx_text.strip(),
                        'confidence': 95.0 if extraction_method != "fallback" else 10.0,
                        'method': extraction_method,
                        'word_count': len(pptx_text.split()),
                        'page_number': 1
                    })
                    logger.info(f"PowerPoint file processed: {len(pptx_text)} characters, {len(pptx_text.split())} words (method: {extraction_method})")
                else:
                    # Final fallback: return minimal placeholder
                    fallback_text = f"PowerPoint presentation file: {Path(file_path).name}\n[Text extraction was not possible from this file format.]"
                    all_text_results.append({
                        'text': fallback_text,
                        'confidence': 5.0,
                        'method': 'fallback',
                        'word_count': len(fallback_text.split()),
                        'page_number': 1
                    })
                    logger.warning(f"PowerPoint file extraction returned empty, using fallback placeholder")
            
            elif file_extension in ['.xls', '.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsb', '.ods']:
                # Process Excel files
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "EXCEL_PROCESSING")
                
                excel_text = ""
                extraction_method = "excel_extraction"
                
                # Prefer ZIP/XML extraction for OpenXML formats; fallback to openpyxl
                if file_extension in ['.xlsx', '.xlsm', '.xltx', '.xltm']:
                    if ZIP_XML_AVAILABLE:
                        try:
                            excel_text = self._extract_text_xlsx_from_zip(file_path)
                            if excel_text.strip():
                                extraction_method = "zip_xml"
                                logger.info(f"Extracted {len(excel_text)} characters using ZIP/XML extraction")
                            else:
                                excel_text = ""
                        except Exception as e:
                            logger.warning(f"ZIP/XML extraction failed: {e}, trying openpyxl fallback")
                            excel_text = ""
                    if (not excel_text.strip()) and XLSX_AVAILABLE:
                        try:
                            excel_text = self._extract_text_xlsx(file_path)
                            if excel_text.strip():
                                extraction_method = "openpyxl"
                                logger.info(f"Extracted {len(excel_text)} characters from Excel using openpyxl")
                            else:
                                excel_text = ""
                        except Exception as e:
                            logger.warning(f"openpyxl extraction failed: {e}")
                            excel_text = ""
                
                # For .xls (legacy format), use xlrd
                if file_extension == '.xls':
                    if XLS_AVAILABLE:
                        try:
                            excel_text = self._extract_text_xls(file_path)
                            if excel_text.strip():
                                extraction_method = "xlrd"
                                logger.info(f"Extracted {len(excel_text)} characters from legacy .xls using xlrd")
                        except Exception as e:
                            logger.warning(f"Legacy .xls extraction failed: {e}")
                            excel_text = ""
                    else:
                        logger.warning(f"xlrd library not available for .xls files. Install with: pip install xlrd")
                        excel_text = ""
                
                # For .xlsb (binary macro-enabled), try ZIP/XML extraction
                if file_extension == '.xlsb':
                    if ZIP_XML_AVAILABLE:
                        try:
                            excel_text = self._extract_text_xlsx_from_zip(file_path)
                            if excel_text.strip():
                                extraction_method = "zip_xml"
                                logger.info(f"Extracted {len(excel_text)} characters from .xlsb using ZIP/XML extraction")
                        except Exception as e:
                            logger.warning(f".xlsb extraction failed: {e}")
                            excel_text = ""
                    else:
                        logger.warning(f"ZIP/XML extraction not available for .xlsb files")
                        excel_text = ""
                
                # For .ods (OpenDocument Spreadsheet), use odfpy
                if file_extension == '.ods':
                    if ODS_AVAILABLE:
                        try:
                            excel_text = self._extract_text_ods(file_path)
                            if excel_text.strip():
                                extraction_method = "odfpy_ods"
                                logger.info(f"Extracted {len(excel_text)} characters from .ods using odfpy")
                        except Exception as e:
                            logger.warning(f".ods extraction failed: {e}")
                            excel_text = ""
                    else:
                        logger.warning(f"odfpy library not available for .ods files. Install with: pip install odfpy")
                        excel_text = ""
                
                if excel_text.strip():
                    all_text_results.append({
                        'text': excel_text.strip(),
                        'confidence': 95.0,
                        'method': extraction_method,
                        'word_count': len(excel_text.split()),
                        'page_number': 1
                    })
                    logger.info(f"Excel file processed: {len(excel_text)} characters, {len(excel_text.split())} words")
                else:
                    logger.error(f"Failed to extract text from Excel file: {file_path}")
            
            elif file_extension == '.csv':
                # Process CSV files
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "CSV_PROCESSING")
                try:
                    csv_text = self._extract_text_csv(file_path)
                    if csv_text.strip():
                        all_text_results.append({
                            'text': csv_text.strip(),
                            'confidence': 100.0,
                            'method': 'csv_extraction',
                            'word_count': len(csv_text.split()),
                            'page_number': 1
                        })
                        logger.info(f"CSV file processed: {len(csv_text)} characters")
                    else:
                        logger.warning(f"CSV file is empty: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to extract text from CSV file {file_path}: {e}")
            
            elif file_extension in ['.txt', '.md', '.html', '.rtf']:
                # Process text files
                if workflow_manager:
                    workflow_manager.update_workflow_stage(self.file_id, "TEXT_PROCESSING")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    if text_content.strip():
                        all_text_results.append({
                            'text': text_content.strip(),
                            'confidence': 100.0,
                            'method': 'direct_text_read',
                            'word_count': len(text_content.split()),
                            'page_number': 1
                        })
                        logger.info(f"Text file processed: {len(text_content)} characters")
                    else:
                        logger.warning(f"Text file is empty: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to read text file {file_path}: {e}")
            
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.svg']:
                # Process image with smart mode (handles preprocessing internally)
                if workflow_manager:
                    workflow_manager.update_image_processing(self.file_id)
                # Calculate file size for parallel processing threshold
                file_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else None
                image = Image.open(file_path)
                
                # Use smart mode which handles all preprocessing and OCR internally
                if workflow_manager:
                    workflow_manager.update_ocr_extraction(self.file_id)
                smart_results = self._extract_text_ultimate(image, file_size_bytes=file_size_bytes)
                if smart_results:
                    all_text_results.extend(smart_results)
                    logger.info(f"Smart OCR found text: {[r['text'][:50] for r in smart_results]}")
                
                skip_captioning = os.getenv("SKIP_IMAGE_CAPTIONING_IN_PROCESSOR", "false").lower() == "true"
                
                if not skip_captioning:
                    try:
                        captioner = _get_captioner()
                        semantic_engine = _get_semantic_engine()
                        
                        if captioner and semantic_engine:
                            logger.info(f"Generating image caption for {self.file_id}...")
                            
                            # Step 1: Generate caption using BLIP
                            caption = captioner.caption(file_path)
                            logger.info(f"Generated caption: {caption[:100]}...")
                            
                            # Step 2: Create embeddings
                            try:
                                if hasattr(semantic_engine, 'doc_embedder'):
                                    caption_vec = semantic_engine.embed_text(caption, for_images=False)
                                elif hasattr(semantic_engine, 'embedder'):
                                    caption_vec = semantic_engine.embedder.encode([caption])[0]
                                else:
                                    logger.warning("Semantic engine has no embedder")
                                    caption_vec = None
                            except Exception as e:
                                logger.warning(f"Failed to encode caption: {e}")
                                caption_vec = None
                            
                            # CLIP embedding for image (searchable via image queries)
                            clip_vec = None
                            
                            # Step 3: Store caption as text vector in Milvus (for text↔image search)
                            if caption_vec and len(caption_vec) > 0:
                                try:
                                    from src.vector_db_milvus_server import MilvusServerVectorDatabase
                                    from src.semantic.semantic_components import TextChunk
                                    import numpy as np
                                    
                                    # Store caption as a text chunk in the document collection
                                    caption_chunk = TextChunk(
                                        chunk_id=f"{self.file_id}_caption",
                                        text=caption,
                                        metadata={
                                            "file_id": self.file_id,
                                            "chunk_index": 0,
                                            "total_chunks": 1,
                                            "document_type": "image",
                                            "modality": "text",
                                            "page_number": 1,
                                            "is_caption": True,
                                            "source": "image_captioning"
                                        }
                                    )
                                    
                                    from src.ultimate_vector_integration import MODEL_DIM
                                    text_vector_db = MilvusServerVectorDatabase(
                                        collection_name="ultimate_document_chunks",
                                        vector_size=MODEL_DIM,  # Use 768, not len(caption_vec) which might be wrong model
                                        host=os.getenv("MILVUS_HOST", "localhost"),
                                        port=os.getenv("MILVUS_PORT", "19530")
                                    )
                                    
                                    chunks = [caption_chunk]
                                    embeddings = np.array([caption_vec])
                                    success = text_vector_db.insert_chunks(chunks, embeddings, user_id="default_user")
                                    
                                    if success:
                                        logger.info(f"Stored image caption embedding for {self.file_id}")
                                    else:
                                        logger.warning(f"Failed to store caption embedding for {self.file_id}")
                                        
                                except Exception as e:
                                    logger.warning(f"Failed to store caption vector: {e}")
                            
                            # Step 4: Store CLIP embedding (if available)
                            if clip_vec is not None and clip_vec.size > 0:
                                logger.info(f"CLIP embedding available for {self.file_id} (will be stored by ultimate_tasks.py)")
                        
                            # Add caption to result metadata
                            if not hasattr(self, 'image_caption'):
                                self.image_caption = caption
                            logger.info(f"Image captioning complete for {self.file_id}: {caption[:100]}...")
                        else:
                            logger.debug("Image captioning not available (models not loaded)")
                    except Exception as e:
                        logger.warning(f"Caption generation failed (non-critical): {e}")
                else:
                    logger.info("Skipping caption generation in DocumentProcessor (handled by Celery task)")
            
            # Combine all text results
            if not all_text_results:
                if workflow_manager:
                    workflow_manager.update_failed(self.file_id, "No text extracted")
                return UltimateSearchResult(
                    file_path=file_path,
                    file_type=file_type,
                    success=False,
                    text_content="",
                    normalized_text="",
                    keywords=[],
                    searchable_text="",
                    processing_time=time.time() - start_time,
                    extraction_method="none",
                    confidence=0.0,
                    word_variations={},
                    fuzzy_matches={},
                    pattern_matches={},
                    metadata={},
                    error="No text extracted"
                )
            
            # Only filter out completely empty results
            filtered_results = [r for r in all_text_results if r.get('text', '').strip()]
            
            if not filtered_results:
                logger.warning("No text content found in any extraction method")
                if workflow_manager:
                    workflow_manager.update_failed(self.file_id, "No text content extracted")
                return UltimateSearchResult(
                    file_path=file_path,
                    file_type=file_type,
                    success=False,
                    text_content="",
                    normalized_text="",
                    keywords=[],
                    searchable_text="",
                    processing_time=time.time() - start_time,
                    extraction_method="none",
                    confidence=0.0,
                    word_variations={},
                    fuzzy_matches={},
                    pattern_matches={},
                    metadata={},
                    error="No text content extracted"
                )
            
            # Find best result from filtered results
            best_result = max(filtered_results, key=lambda x: (x['confidence'], x['word_count']))
            
            # Combine all filtered text for search
            all_text = ' '.join([r['text'] for r in filtered_results])
            normalized_text = self._normalize_text(all_text)
            
            # Extract keywords with enhanced extraction
            basic_keywords = list(set(re.findall(r'\b[a-zA-Z]{2,}\b', normalized_text)))  # Reduced minimum length
            
            # Extract specialized content
            phone_numbers = self._extract_phone_numbers(all_text)
            brand_names = self._extract_brand_names(all_text)
            small_text_keywords = self._extract_small_text_keywords(all_text)
            
            # Combine all keywords
            keywords = basic_keywords + phone_numbers + brand_names + small_text_keywords
            keywords = list(set(keywords))  # Remove duplicates
            
            # Find matches if target words provided
            fuzzy_matches = {}
            pattern_matches = {}
            word_variations = {}
            
            if target_words:
                fuzzy_matches = self._find_ultimate_matches(all_text, target_words)
                
                # Find pattern matches
                for word in target_words:
                    if word in self.patterns:
                        pattern_matches[word] = []
                        for pattern in self.patterns[word]:
                            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
                            pattern_matches[word].extend(matches)
                
                # Find word variations
                for word in target_words:
                    if word in self.word_variations:
                        word_variations[word] = []
                        for variation in self.word_variations[word]:
                            if variation.lower() in normalized_text:
                                word_variations[word].append(variation)
            
            # Vision analysis disabled for faster processing
            objects_detected = []
            image_caption = ""
            visual_elements = []
            
            # Update workflow stage to completed
            if workflow_manager:
                processing_time = time.time() - start_time
                text_length = len(all_text)
                workflow_manager.update_completed(self.file_id, processing_time, text_length)
            
            return UltimateSearchResult(
                file_path=file_path,
                file_type=file_type,
                success=True,
                text_content=all_text,
                normalized_text=normalized_text,
                keywords=keywords,
                searchable_text=normalized_text,
                processing_time=time.time() - start_time,
                extraction_method=best_result['method'],
                confidence=best_result['confidence'],
                word_variations=word_variations,
                fuzzy_matches=fuzzy_matches,
                pattern_matches=pattern_matches,
                metadata={
                    'total_results': len(all_text_results),
                    'best_confidence': best_result['confidence'],
                    'best_word_count': best_result['word_count']
                },
                objects_detected=objects_detected,
                image_caption=image_caption,
                visual_elements=visual_elements
            )
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            if workflow_manager:
                workflow_manager.update_failed(self.file_id, str(e))
            return UltimateSearchResult(
                file_path=file_path,
                file_type=file_type,
                success=False,
                text_content="",
                normalized_text="",
                keywords=[],
                searchable_text="",
                processing_time=time.time() - start_time,
                extraction_method="error",
                confidence=0.0,
                word_variations={},
                fuzzy_matches={},
                pattern_matches={},
                metadata={},
                error=str(e)
            )
    
    def search_documents(self, documents: List[UltimateSearchResult], target_words: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Search documents with intelligent pattern matching."""
        results = {}
        MIN_SEARCH_CONFIDENCE = 30  # Minimum confidence for search results
        
        for word in target_words:
            results[word] = []
            
            for doc in documents:
                if not doc.success or doc.confidence < MIN_SEARCH_CONFIDENCE:
                    continue
                
                # Direct text search
                if word.lower() in doc.normalized_text:
                    results[word].append({
                        'file_path': doc.file_path,
                        'file_type': doc.file_type,
                        'match_type': 'direct',
                        'confidence': 100.0,
                        'text_preview': doc.text_content[:200] + "..." if len(doc.text_content) > 200 else doc.text_content,
                        'extraction_method': doc.extraction_method
                    })
                
                # Fuzzy matching
                elif word in doc.fuzzy_matches:
                    results[word].append({
                        'file_path': doc.file_path,
                        'file_type': doc.file_type,
                        'match_type': 'fuzzy',
                        'confidence': doc.fuzzy_matches[word],
                        'text_preview': doc.text_content[:200] + "..." if len(doc.text_content) > 200 else doc.text_content,
                        'extraction_method': doc.extraction_method
                    })
                
                # Pattern matching
                elif word in doc.pattern_matches and doc.pattern_matches[word]:
                    results[word].append({
                        'file_path': doc.file_path,
                        'file_type': doc.file_type,
                        'match_type': 'pattern',
                        'confidence': 80.0,
                        'text_preview': doc.text_content[:200] + "..." if len(doc.text_content) > 200 else doc.text_content,
                        'extraction_method': doc.extraction_method
                    })
                
                # Word variations
                elif word in doc.word_variations and doc.word_variations[word]:
                    results[word].append({
                        'file_path': doc.file_path,
                        'file_type': doc.file_type,
                        'match_type': 'variation',
                        'confidence': 75.0,
                        'text_preview': doc.text_content[:200] + "..." if len(doc.text_content) > 200 else doc.text_content,
                        'extraction_method': doc.extraction_method
                    })
        
        return results


def process_pdf_page_worker(args: Tuple[Any, ...]) -> Tuple[int, List[Dict[str, Any]], List[str]]:
    """Worker function for parallel PDF page processing."""
    (
        file_path,
        page_num,
        language,
        ocr_fast_config,
        ocr_zoom_factor,
        ocr_trigger_min_words,
        is_text_based_pdf
    ) = args
    
    page_segments: List[Dict[str, Any]] = []
    combined_parts: List[str] = []
    
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_num)
        text = page.get_text()
        text_clean = text.strip()
        word_count = len(text_clean.split()) if text_clean else 0
        
        if text_clean:
            page_segments.append({
                'text': text_clean,
                'confidence': 100.0,
                'method': 'pymupdf_direct',
                'word_count': word_count,
                'page': page_num + 1
            })
            combined_parts.append(text_clean)
        
        needs_ocr = OCR_AVAILABLE and not is_text_based_pdf and word_count < ocr_trigger_min_words
        if needs_ocr:
            try:
                mat = fitz.Matrix(ocr_zoom_factor, ocr_zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang=language,
                    config=ocr_fast_config
                ).strip()
                if ocr_text:
                    page_segments.append({
                        'text': ocr_text,
                        'confidence': 70.0,
                        'method': 'tesseract_parallel',
                        'word_count': len(ocr_text.split()),
                        'page': page_num + 1
                    })
                    combined_parts.append(ocr_text)
            except Exception as ocr_err:
                logging.getLogger(__name__).debug(f"OCR worker page {page_num + 1} failed: {ocr_err}")
        doc.close()
    except Exception as worker_err:
        logging.getLogger(__name__).debug(f"PDF worker failed for page {page_num + 1}: {worker_err}")

    return page_num, page_segments, combined_parts
