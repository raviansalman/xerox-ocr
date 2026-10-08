"""Canonical model invariants on every supported format: blocks join to the page text, units and spans are exact
slices of it, tables keep their cells."""
import shutil

import pytest

from docintel.models import BLOCK_SEP, Block, Page, Table
from docintel.packs import default_domain
from docintel.processing.chunking import chunk_pages
from docintel.processing.parsers import parse_file
from docintel.understanding import analyze
from tests.fixtures import corpus as C

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def _formats(tmp_path):
    yield "pdf", C.text_pdf(tmp_path / "a.pdf", [["SERVICE AGREEMENT", "Contract No. 17/2024", "Either party may terminate "
                                                   "this agreement with 60 days written notice."], ["Page two text."]])
    yield "docx", C.docx_file(tmp_path / "b.docx", "INVOICE 2025-0042", ["Bill to: Gulf Office LLC", "Total due USD 12,500"],
                              [["Item", "Amount"], ["Printing", "USD 12,500"]])
    yield "xlsx", C.xlsx_file(tmp_path / "c.xlsx", {"Summary": [["Name", "Paid"], ["Employees paid", 42]]})
    yield "pptx", C.pptx_file(tmp_path / "d.pptx", [("Fleet review", "Uptime 99.2 percent"), ("Next", "Replace printers")])
    yield "html", C.txt(tmp_path / "e.html", "<html><head><title>T</title></head><body><h1>Toner</h1><p>Open the door.</p>"
                                              "<table><tr><th>Step</th><th>Action</th></tr><tr><td>1</td><td>Lift</td></tr>"
                                              "</table></body></html>")
    yield "txt", C.txt(tmp_path / "f.txt", "MEMO\n\nThe window is Saturday.\n\n- item one\n- item two\n")
    yield "eml", C.eml_file(tmp_path / "g.eml", "Order PO-77", "Schedule attached.", ("s.txt", b"20 printers on 15 March 2026"))
    yield "csv", C.txt(tmp_path / "h.csv", "Purchase Order,PO-2025-0193\nItem,Qty\nPrinter,20\n")


def test_every_format_satisfies_the_invariants(tmp_path):
    for name, path in _formats(tmp_path):
        doc = parse_file(path, path.name)
        assert doc.parser, name
        for page in doc.pages:
            assert page.text == BLOCK_SEP.join(b.text for b in page.blocks), name
            for (s, e), b in zip(page.block_offsets(), page.blocks, strict=True):
                assert page.text[s:e] == b.text and b.type, name
                if b.type == "table":
                    assert b.table is not None and page.tables[b.table].text() == b.text, name
        for unit in chunk_pages(doc.pages, 400):
            page = doc.pages[unit.page_start - 1]
            assert page.text[unit.char_start:unit.char_end] == unit.text, (name, unit)
        u = analyze(doc, path.name)
        for item in [*u.fields, *u.entities, *u.clauses, *u.relations]:
            if item.span is None:
                continue
            page = next(p for p in doc.pages if p.number == item.span.page)
            assert 0 <= item.span.char_start <= item.span.char_end <= len(page.text), (name, item)
        for e in u.entities:
            if e.type in ("person", "organization", "identifier", "email") and e.span:
                page = next(p for p in doc.pages if p.number == e.span.page)
                assert page.text[e.span.char_start:e.span.char_end] == e.value, (name, e)


def test_structure_is_recognized(tmp_path):
    docs = {name: parse_file(p, p.name) for name, p in _formats(tmp_path)}
    pdf = docs["pdf"].pages[0]
    assert pdf.blocks[0].type == "heading" and pdf.blocks[0].bbox is not None
    docx_tables = docs["docx"].pages[0].tables
    assert docx_tables and docx_tables[0].rows == [["Item", "Amount"], ["Printing", "USD 12,500"]]
    sheet = docs["xlsx"].pages[0]
    assert sheet.blocks[0].text == "Sheet: Summary" and sheet.tables[0].labelled_row(1) == "Name: Employees paid; Paid: 42"
    assert docs["pptx"].pages[0].blocks[0] == Block("heading", "Fleet review", docs["pptx"].pages[0].blocks[0].bbox)
    html = docs["html"].pages[0]
    assert [b.type for b in html.blocks] == ["heading", "paragraph", "table"] and html.tables[0].header_rows == 1
    assert [b.type for b in docs["txt"].pages[0].blocks] == ["heading", "paragraph", "list_item", "list_item"]
    eml = docs["eml"]
    assert eml.pages[0].blocks[0].type == "key_value" and eml.pages[1].blocks[0].text == "Attachment: s.txt"


def test_relations_from_patterns_and_key_values(tmp_path):
    p = C.text_pdf(tmp_path / "x.pdf", [["SERVICE AGREEMENT", "Technician: Jane Roe",
                                          "Either party may terminate this agreement with sixty (60) days written notice.",
                                          "Signed by John Doe"]])
    doc = parse_file(p, p.name)
    u = analyze(doc, p.name, domain=default_domain())
    rel = {(r.predicate, r.subject.lower()): r for r in u.relations}
    term = rel[("may_terminate", "either party")]
    assert "60" in term.qualifiers.get("notice", "") and term.span is not None
    assert rel[("attribute", "technician")].object == "Jane Roe"
    assert ("signed", "john doe") in rel


def test_table_rows_become_units_with_header_context():
    t = Table([["Item", "Qty"], ["Toner", "20"]])
    page = Page.from_blocks(1, "sheet", [Block("table", t.text(), table=0)], [t])
    rows = [c for c in chunk_pages([page], 500) if c.unit_type == "table_row"]
    assert len(rows) == 1 and rows[0].text == "Toner | 20" and rows[0].context.endswith("Item: Toner; Qty: 20")


def test_language_detection_is_honest():
    from docintel.understanding.language import detect
    assert detect("This agreement is made between the parties and is governed by the law of the state.") == "en"
    assert detect("Der Vertrag ist nicht gültig und wird von den Parteien mit der Unterschrift bestätigt.") == "de"
    assert detect("هذا العقد موقع من الطرفين ويخضع لقوانين المملكة") == "ar"
    assert detect("INV-2026 12 34 99") == "und"
