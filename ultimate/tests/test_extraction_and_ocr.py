"""Characterize text extraction and OCR in src/ultimate_search_processor.py."""
import re

import pytest

import src.ultimate_search_processor as usp
from tests.conftest import requires_tesseract, requires_tesseract_ara
from tests.support import documents


@pytest.fixture(scope="module")
def dp():
    return usp.DocumentProcessor()


def test_ocr_defaults(dp):
    assert dp.language == "eng"
    assert dp.ocr_zoom_factor == 1.5  # scanned PDF pages are rendered at 108 DPI for OCR
    assert dp.ocr_fast_config == "--psm 6 --oem 1"


def test_normalize_text_is_currently_corrupting(dp):
    # Pins today's behaviour: the '|' key in ocr_fixes is used as a raw regex, i.e. alternation,
    # so an 'I' is inserted around every letter. Harmless only because ingestion indexes
    # result.text_content, never result.normalized_text. Do not start indexing normalized_text.
    assert dp._normalize_text("Lisa Riordan") == "ILIiIsIaI IRIiIoIrIdIaInI"


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-07: '|' in ocr_fixes is an unescaped regex alternation")
@pytest.mark.parametrize("raw,expected", [
    ("C0MPANY  REP0RT", "COMPANY REPORT"),   # digit-for-letter fix only between letters
    ("Room 101, Floor 5", "Room 101, Floor 5"),
    ("a\n\n b\t c", "a b c"),
])
def test_normalize_text(dp, raw, expected):
    assert dp._normalize_text(raw) == expected


def test_clean_ocr_text_keeps_arabic(dp):
    assert dp._clean_ocr_text("فاتورة   رقم 2024") == "فاتورة رقم 2024"


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-01: 'I?I' artifact rule rewrites real uppercase words")
@pytest.mark.parametrize("word", ["DIGITAL INVOICE", "DEFINITIONS", "CIVIL LIABILITY", "MINIMUM PAYMENT"])
def test_clean_ocr_text_preserves_uppercase_words(dp, word):
    assert dp._clean_ocr_text(word) == word


def test_text_pdf_extraction(dp, tmp_path):
    pdf = documents.text_pdf(tmp_path / "pr.pdf", documents.PRESS_RELEASE_PAGES)
    r = dp.process_document(str(pdf), target_words=[], file_type="application/pdf")
    assert r.success
    assert r.extraction_method == "comprehensive_pdf_parallel"  # all pages collapse into one result
    for phrase in ("FOR IMMEDIATE RELEASE", "StorageChain", "Lisa Riordan", "press release"):
        assert phrase in r.text_content
    assert r.confidence == 100.0


def test_docx_extraction_includes_tables(dp, tmp_path):
    path = documents.invoice_docx(tmp_path / "inv.docx")
    r = dp.process_document(str(path), target_words=[],
                            file_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert r.success and r.extraction_method == "python_docx"
    assert "Gulf Trading LLC" in r.text_content and "USD 12,500" in r.text_content


def test_plain_text_and_html_are_read_verbatim(dp, tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<p>Hello</p><img src=x onerror=alert(1)>", encoding="utf-8")
    r = dp.process_document(str(p), target_words=[], file_type="text/html")
    # Markup is indexed as-is; the UI renders result text via innerHTML (KD-SEC-06).
    assert "onerror=alert(1)" in r.text_content


@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-03: page text is joined in thread-completion order")
def test_pdf_page_order_does_not_depend_on_thread_completion(dp, tmp_path, monkeypatch):
    pdf = documents.numbered_report_pdf(tmp_path / "report.pdf", 6)
    # Simulate pages finishing out of order (what happens under real load).
    monkeypatch.setattr(usp, "as_completed", lambda fs: list(reversed(list(fs))))
    r = dp.process_document(str(pdf), target_words=[], file_type="application/pdf")
    order = [int(n) for n in re.findall(r"report page (\d+)", r.text_content)]
    assert order == sorted(order)


@requires_tesseract
def test_scanned_image_ocr_english(dp, tmp_path):
    img = documents.render_page(["FOR IMMEDIATE RELEASE", "Approved by Lisa Riordan", "Reference INV-2024-0042"])
    path = tmp_path / "scan.png"
    img.save(path, dpi=(300, 300))
    r = dp.process_document(str(path), target_words=[], file_type="image/png")
    assert r.success
    for w in ("IMMEDIATE", "RELEASE", "Lisa", "Riordan", "INV-2024-0042"):
        assert w in r.text_content


@requires_tesseract
@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-05: OCR is skipped for every page once the first 3 pages have text")
def test_mixed_pdf_scanned_page_is_ocrd(dp, tmp_path):
    pdf = documents.mixed_digital_plus_scan_pdf(tmp_path / "mixed.pdf", ["SCANNED APPENDIX", "ZEBRA QUANTUM 7781"])
    r = dp.process_document(str(pdf), target_words=[], file_type="application/pdf")
    assert "ZEBRA" in r.text_content.upper()


@requires_tesseract_ara
@pytest.mark.known_defect
@pytest.mark.xfail(strict=True, reason="KD-OCR-06: OCR language is hardcoded to 'eng'")
def test_arabic_scan_yields_arabic_text(dp, tmp_path):
    img = documents.render_page(["فاتورة رقم ٢٠٢٤", "شركة الخليج للتجارة"], size_pt=16, rtl=True)
    path = tmp_path / "ar.png"
    img.save(path, dpi=(300, 300))
    r = dp.process_document(str(path), target_words=[], file_type="image/png")
    assert len(re.findall(r"[؀-ۿ]", r.text_content or "")) >= 5
