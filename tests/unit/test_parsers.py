"""Every supported format, scanned pages, mixed PDFs and failure modes."""
import shutil
import subprocess

import pytest

from docintel.processing.chunking import chunk_pages
from docintel.processing.detect import detect
from docintel.processing.parsers import ParseError, parse_file
from tests.fixtures import corpus as C

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
needs_soffice = pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not installed")


def test_text_pdf_keeps_pages_in_order(tmp_path):
    p = C.text_pdf(tmp_path / "r.pdf", [[f"Report page {i}", f"Section {i} body text."] for i in range(1, 13)])
    doc = parse_file(p, p.name)
    assert doc.kind == "native" and [pg.number for pg in doc.pages] == list(range(1, 13))
    assert all(f"Report page {pg.number}" in pg.text for pg in doc.pages)


@needs_tesseract
def test_scanned_pdf_is_ocrd_with_words_boxes_and_confidence(tmp_path):
    p = C.scanned_pdf(tmp_path / "s.pdf", [["SERVICE TICKET 48213", "Fuser unit replaced on VersaLink C405", "Technician: Priya Raman"]])
    doc = parse_file(p, p.name)
    page = doc.pages[0]
    assert doc.kind == "scanned" and page.kind == "ocr"
    assert "48213" in page.text and "Priya Raman" in page.text
    assert page.words and 0.5 < page.ocr_confidence <= 1.0
    w = next(w for w in page.words if w.text == "48213")
    assert w.x1 > w.x0 and w.y1 > w.y0


@needs_tesseract
def test_mixed_pdf_ocrs_only_the_scanned_pages(tmp_path):
    import fitz
    native = C.text_pdf(tmp_path / "n.pdf", [[f"Digital page {i} clause text for the contract terms." * 3] for i in range(3)])
    scan = C.scanned_pdf(tmp_path / "s.pdf", [["SCANNED APPENDIX", "ZEBRA QUANTUM 7781"]])
    out = fitz.open(native)
    out.insert_pdf(fitz.open(scan))
    out.save(tmp_path / "mixed.pdf")
    doc = parse_file(tmp_path / "mixed.pdf", "mixed.pdf")
    assert doc.kind == "mixed" and [p.kind for p in doc.pages] == ["native", "native", "native", "ocr"]
    assert "ZEBRA QUANTUM 7781" in doc.pages[3].text


@needs_tesseract
def test_image_with_signature_mark(tmp_path):
    p = C.png(tmp_path / "a.png", ["APPROVAL LETTER", "The renewal is approved.", "Approved by John Smith"], signature=True)
    doc = parse_file(p, p.name)
    assert doc.kind == "image" and "Approved by John Smith" in doc.pages[0].text
    assert [r.type for r in doc.pages[0].regions] == ["signature"]


@needs_tesseract
def test_no_signature_mark_on_plain_text_scan(tmp_path):
    p = C.png(tmp_path / "b.png", ["APPROVAL LETTER", "The renewal is approved.", "Approved by John Smith"])
    assert parse_file(p, p.name).pages[0].regions == []


def test_docx_paragraphs_and_tables(tmp_path):
    p = C.docx_file(tmp_path / "i.docx", "INVOICE 2025-0042", ["Bill to: Gulf Trading LLC"], [["Item", "Amount"], ["Print", "USD 12,500"]])
    doc = parse_file(p, p.name)
    assert doc.kind == "office" and "Gulf Trading LLC" in doc.text and "Print | USD 12,500" in doc.text


def test_xlsx_rows_with_sheet_names(tmp_path):
    p = C.xlsx_file(tmp_path / "x.xlsx", {"Summary": [["Employees paid", 42], ["Total", 3150000.0]], "Other": [["a", "b"]]})
    doc = parse_file(p, p.name)
    assert doc.kind == "spreadsheet" and "Sheet: Summary" in doc.text and "Employees paid | 42" in doc.text
    assert "3150000" in doc.text and "Sheet: Other" in doc.text


def test_csv(tmp_path):
    p = C.txt(tmp_path / "po.csv", "Purchase Order,PO-2025-0193\nItem,Qty\nPrinter,20\n")
    doc = parse_file(p, p.name)
    assert doc.kind == "spreadsheet" and "Purchase Order | PO-2025-0193" in doc.text


def test_pptx_slides_are_pages(tmp_path):
    p = C.pptx_file(tmp_path / "d.pptx", [("Fleet review", "Uptime 99.2 percent"), ("Next", "Replace 12 printers")])
    doc = parse_file(p, p.name)
    assert doc.kind == "presentation" and len(doc.pages) == 2 and "Replace 12 printers" in doc.pages[1].text


def test_html_drops_scripts_and_keeps_title(tmp_path):
    p = C.txt(tmp_path / "t.html", "<html><head><title>Toner</title><script>alert(1)</script></head><body><h1>Toner</h1><p>Open the door.</p></body></html>")
    doc = parse_file(p, p.name)
    assert doc.title == "Toner" and "Open the door." in doc.text and "alert" not in doc.text


def test_rtf(tmp_path):
    p = C.txt(tmp_path / "r.rtf", r"{\rtf1\ansi REMOTE WORK POLICY\par Employees may work from home.\par}")
    assert "Employees may work from home." in parse_file(p, p.name).text


def test_email_with_attachment(tmp_path):
    p = C.eml_file(tmp_path / "m.eml", "Order PO-2025-0193", "Schedule attached.", ("schedule.txt", b"20 printers on 15 March 2026"))
    doc = parse_file(p, p.name)
    assert doc.kind == "email" and doc.title == "Order PO-2025-0193"
    assert "Subject: Order PO-2025-0193" in doc.pages[0].text
    assert doc.pages[1].text.startswith("Attachment: schedule.txt") and "15 March 2026" in doc.pages[1].text


def test_plain_text_encodings(tmp_path):
    (tmp_path / "a.txt").write_bytes("Résumé café".encode("cp1252"))
    assert "Résumé café" in parse_file(tmp_path / "a.txt", "a.txt").text


@needs_soffice
def test_legacy_doc_via_libreoffice(tmp_path):
    src = C.docx_file(tmp_path / "legacy.docx", "LEGACY CONTRACT", ["Governed by the laws of the State of Texas."])
    subprocess.run(["soffice", "--headless", "--convert-to", "doc", "--outdir", str(tmp_path), str(src)], check=True,
                   capture_output=True, timeout=120)
    doc = parse_file(tmp_path / "legacy.doc", "legacy.doc")
    assert "State of Texas" in doc.text


def test_password_protected_pdf_fails_clearly(tmp_path):
    import fitz
    d = fitz.open()
    d.new_page().insert_text((72, 72), "secret")
    d.save(tmp_path / "enc.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(ParseError, match="password"):
        parse_file(tmp_path / "enc.pdf", "enc.pdf")


def test_unsupported_and_corrupted_files(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\x00\x01\x02binary\x00")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "x.bin", "x.bin")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "bad.pdf", "bad.pdf")


def test_detection_uses_content_not_extension(tmp_path):
    p = C.text_pdf(tmp_path / "looks_like.txt", [["hello"]])
    assert detect(p, "looks_like.txt") == "pdf"


def test_chunks_respect_pages_headings_and_size():
    from docintel.models import Page
    body = " ".join(f"Sentence number {i} describes the service level for printers." for i in range(60))
    pages = [Page(1, "ARTICLE 5 SERVICE LEVELS\n\n" + body, "native"), Page(2, "Short second page.", "native")]
    chunks = chunk_pages(pages, 500)
    assert all(c.page_start == c.page_end for c in chunks)
    assert {c.page_start for c in chunks} == {1, 2}
    assert all(len(c.text) <= 625 for c in chunks)          # target 500, hard limit 1.25x
    assert chunks[0].heading == "ARTICLE 5 SERVICE LEVELS" and chunks[1].heading == "ARTICLE 5 SERVICE LEVELS"
    # overlap: the start of a chunk repeats the tail of the previous one
    assert chunks[1].text.split(".")[0] in chunks[0].text
