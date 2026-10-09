"""Format parsers. Each returns a ``ParsedDocument`` whose pages hold typed blocks in reading order (headings,
paragraphs, list items, key-value lines, tables with cells), with positions where the format has them.

Pages that need OCR are rendered and recognized in parallel (``ocr_pages``); text layers are used directly when
they are complete. Every parser applies the resource limits in ``docintel.processing.safety`` before handing the
file to a third-party library, and fails with ``ParseError`` (never a partial success) when it cannot read it.
"""
from __future__ import annotations

import contextlib
import csv
import email
import io
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from email import policy
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from docintel import text as T
from docintel.config import get_settings
from docintel.models import Block, Page, ParsedDocument, Table, text_blocks
from docintel.processing import safety
from docintel.processing.chunking import is_heading
from docintel.processing.detect import detect

logger = logging.getLogger(__name__)

ROWS_PER_PAGE = 200
TEXT_PAGE_CHARS = 12000
HEADER_FOOTER_BAND = 0.06          # top and bottom share of a PDF page where short blocks are headers/footers


class ParseError(Exception):
    """The document cannot be processed (corrupted, encrypted, unsupported, or over a resource limit)."""


def _refine(blocks: list[Block]) -> list[Block]:
    """Assign heading, list item and key-value types to untyped paragraphs from their text; a heading line at the
    top of a multi-line paragraph becomes its own block."""
    split: list[Block] = []
    for b in blocks:
        if b.type == "paragraph" and "\n" in b.text:
            first, rest = b.text.split("\n", 1)
            if is_heading(first) and rest.strip():
                split += [Block("heading", first, b.bbox, b.confidence), Block("paragraph", rest, b.bbox, b.confidence)]
                continue
        split.append(b)
    blocks = split
    for b in blocks:
        if b.type != "paragraph":
            continue
        single = "\n" not in b.text
        if single and is_heading(b.text):
            b.type = "heading"
        elif single and re.match(r"^\s*(?:[-*•▪◦·]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S", b.text):
            b.type = "list_item"
        elif single and re.match(r"^[A-Z][\w /&.'-]{1,40}:\s+\S", b.text):
            b.type = "key_value"
    return blocks


def _table_block(table: Table, index: int) -> Block:
    return Block("table", table.text(), table.bbox, None, index)


# --------------------------------------------------------------------------------------------------- OCR helpers

def ocr_pages(jobs: list[tuple[int, Callable[[], Image.Image]]]) -> dict[int, Page]:
    """Run OCR for (page_number, render) jobs concurrently. Rendering happens in the calling thread (renderers
    such as PyMuPDF are not thread-safe); recognition runs in a bounded pool."""
    from docintel.processing.ocr import get_ocr

    engine = get_ocr()
    workers = max(1, get_settings().ocr_workers)
    detect_marks = get_settings().signature_detection

    def work(number: int, img: Image.Image) -> Page:
        r = engine.recognize(img)
        regions = []
        if detect_marks:
            from docintel.processing.signatures import detect_signatures
            try:
                regions = detect_signatures(img, r.words)
            except Exception:                      # detection is best effort; never fail OCR because of it
                logger.warning("signature detection failed", exc_info=True)
        return Page.from_blocks(number, "ocr", _refine(r.blocks), width=r.width, height=r.height,
                                ocr_confidence=r.confidence, words=r.words, regions=regions)

    results: dict[int, Page] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for number, render in jobs:
            futures[number] = pool.submit(work, number, render())
            if len(futures) >= workers * 2:          # bound memory: wait for the oldest page
                n = min(futures)
                results[n] = futures.pop(n).result()
        for n, fut in futures.items():
            results[n] = fut.result()
    return results


# ------------------------------------------------------------------------------------------------------- PDF

def _pdf_line_text(line: dict) -> str:
    out, prev_x1 = "", None
    for span in line["spans"]:
        t = span["text"]
        if not t:
            continue
        if prev_x1 is not None and span["bbox"][0] - prev_x1 > 1.0 and not out.endswith(" ") and not t.startswith(" "):
            out += " "
        out += t
        prev_x1 = span["bbox"][2]
    return out


def _pdf_native_blocks(page) -> tuple[list[Block], list[Table]]:
    """Text blocks (with font-size based heading detection and header/footer bands) and ruled tables."""
    tables: list[Table] = []
    table_boxes = []
    try:
        for tb in page.find_tables().tables:
            rows = [[(c or "").strip() for c in row] for row in tb.extract()]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2 or max(len(r) for r in rows) < 2:
                continue
            header = 0 if getattr(tb.header, "external", False) else 1
            tables.append(Table(rows, header, tuple(round(v, 1) for v in tb.bbox)))
            table_boxes.append(tb.bbox)
    except Exception:                               # table detection is an enhancement; text is still extracted
        logger.warning("PDF table detection failed", exc_info=True)

    data = page.get_text("dict", sort=True)
    sizes = [s["size"] for b in data["blocks"] if b.get("type") == 0 for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0
    height = page.rect.height or 1.0
    items: list[tuple[float, float, Block]] = []
    for b in data["blocks"]:
        if b.get("type") != 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for tx0, ty0, tx1, ty1 in table_boxes):
            continue                                # the table's cells carry this text
        text = "\n".join(t for t in (_pdf_line_text(ln) for ln in b["lines"]) if t.strip())
        if not text.strip():
            continue
        span_sizes = [s["size"] for ln in b["lines"] for s in ln["spans"] if s["text"].strip()]
        size = max(span_sizes) if span_sizes else body_size
        kind = "paragraph"
        if len(text) <= 80 and (y1 <= height * HEADER_FOOTER_BAND or y0 >= height * (1 - HEADER_FOOTER_BAND)):
            kind = "header" if y1 <= height * HEADER_FOOTER_BAND else "footer"
        elif "\n" not in text and len(text) <= 120 and body_size and size >= body_size * 1.2:
            kind = "heading"
        items.append((y0, x0, Block(kind, text, (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)))))
    for i, t in enumerate(tables):
        items.append((t.bbox[1], t.bbox[0], _table_block(t, i)))
    items.sort(key=lambda it: (round(it[0] / 3), it[1]))
    blocks = _refine([b for _, _, b in items])
    ordered_tables = []                             # renumber table indices in reading order
    for b in blocks:
        if b.type == "table" and b.table is not None:
            ordered_tables.append(tables[b.table])
            b.table = len(ordered_tables) - 1
    return blocks, ordered_tables


def parse_pdf(path: Path) -> ParsedDocument:
    import fitz

    s = get_settings()
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ParseError(f"cannot open PDF: {e}") from e
    try:
        if doc.needs_pass:
            raise ParseError("password-protected PDF")
        if doc.page_count == 0:                   # a damaged file that PyMuPDF "repaired" into nothing
            raise ParseError("corrupted PDF: no readable pages")
        if doc.page_count > s.max_pages:
            raise ParseError(f"PDF has {doc.page_count} pages; limit is {s.max_pages}")
        pages: dict[int, Page] = {}
        ocr_jobs = []
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text", sort=True).strip()
            area = max(1.0, page.rect.width * page.rect.height)
            img_cover = sum(max(0.0, (x1 - x0) * (y1 - y0)) / area for x0, y0, x1, y1 in
                            (info["bbox"] for info in page.get_image_info()))
            # an Arabic text layer that was extracted garbled is replaced by OCR (when Arabic OCR is configured)
            garbled = "ara" in s.ocr_languages.split("+") and T.garbled_arabic(txt)
            needs_ocr = len(txt) < 30 or (img_cover > 0.5 and len(txt) < 400) or garbled
            native = None
            if txt:
                blocks, tables = _pdf_native_blocks(page)
                native = Page.from_blocks(i, "native", blocks, tables, width=page.rect.width, height=page.rect.height)
            if needs_ocr:
                zoom = safety.render_scale(page.rect.width, page.rect.height, s.ocr_dpi)

                def render(page=page, zoom=zoom):
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
                    return Image.frombytes("L", (pix.width, pix.height), pix.samples)
                ocr_jobs.append((i, render, None if garbled else native))
            else:
                pages[i] = native or Page.from_blocks(i, "native", [], width=page.rect.width, height=page.rect.height)
        if ocr_jobs:
            recognized = ocr_pages([(n, r) for n, r, _ in ocr_jobs])
            for n, _r, native in ocr_jobs:
                p = recognized[n]
                if native is not None and len(native.text) > len(p.text):   # keep the richer text layer
                    p = native
                pages[n] = p                      # OCR pages keep width/height in rendered pixels (word box unit)
        title = (doc.metadata or {}).get("title") or None
        ordered = [pages[n] for n in sorted(pages)]
    finally:
        doc.close()
    kinds = {p.kind for p in ordered}
    kind = "native" if kinds <= {"native"} else ("scanned" if kinds == {"ocr"} else "mixed")
    return ParsedDocument(kind=kind, pages=ordered, title=title if title and len(title) > 3 else None,
                          parser=f"pymupdf-{fitz.VersionBind}")


# ------------------------------------------------------------------------------------------------------- images

def _image_frame(img: Image.Image, index: int) -> Image.Image:
    img.seek(index)
    safety.check_image_size(*img.size)
    return img.copy()


def parse_image(path: Path) -> ParsedDocument:
    """Every frame is size-checked before it is decoded, and frames are decoded one at a time as OCR needs them."""
    s = get_settings()
    safety.configure_image_limits()
    warnings = []
    try:
        with Image.open(path) as img:
            count = getattr(img, "n_frames", 1)
            if count > s.max_pages:
                warnings.append(f"only the first {s.max_pages} of {count} frames were processed (DOCINTEL_MAX_PAGES)")
            numbers = range(1, min(count, s.max_pages) + 1)
            recognized = ocr_pages([(n, (lambda n=n: _image_frame(img, n - 1))) for n in numbers])
    except (Image.DecompressionBombError, safety.UnsafeFileError) as e:
        raise ParseError(f"image exceeds the pixel limit ({s.max_image_pixels}): {e}") from e
    except Exception as e:
        raise ParseError(f"cannot open image: {e}") from e
    return ParsedDocument(kind="image", pages=[recognized[n] for n in sorted(recognized)], parser="pillow+ocr",
                          warnings=warnings)


def _rows_warning(sheet: str, limit: int) -> str:
    return f"sheet {sheet!r} was cut at {limit} rows (DOCINTEL_MAX_SHEET_ROWS)" if sheet else \
        f"cut at {limit} rows (DOCINTEL_MAX_SHEET_ROWS)"


# ------------------------------------------------------------------------------------------------------- Word

def _docx_items(document) -> list[tuple[Block | Table, bool]]:
    """(block or table, page_break_before) in body order."""
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    out: list[tuple[Block | Table, bool]] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, document)
            xml = child.xml
            brk = 'w:type="page"' in xml or "w:pageBreakBefore" in xml
            text = p.text.strip()
            style = (p.style.name or "").lower() if p.style is not None else ""
            kind = "heading" if style.startswith("heading") or style == "title" else \
                "list_item" if "list" in style else "paragraph"
            out.append((Block(kind, text), brk))
        elif tag == "tbl":
            t = DocxTable(child, document)
            rows = []
            for row in t.rows:
                cells, prev = [], None
                for c in row.cells:
                    if c._tc is prev:                # merged cells repeat their element
                        continue
                    prev = c._tc
                    cells.append(c.text.strip())
                if any(cells):
                    rows.append(cells)
            if rows:
                out.append((Table(rows, 1 if len(rows) > 1 else 0), False))
    return out


def parse_docx(path: Path, parser: str = "python-docx") -> ParsedDocument:
    import docx

    try:
        safety.check_container(path)
        d = docx.Document(str(path))
    except safety.UnsafeFileError as e:
        raise ParseError(str(e)) from e
    except Exception as e:
        raise ParseError(f"cannot open DOCX: {e}") from e
    pages: list[Page] = []
    blocks: list[Block] = []
    tables: list[Table] = []

    def flush() -> None:
        nonlocal blocks, tables
        pages.append(Page.from_blocks(len(pages) + 1, "section", _refine(blocks), tables))
        blocks, tables = [], []

    for item, brk in _docx_items(d):
        if brk and (blocks or tables):
            flush()
        if isinstance(item, Table):
            tables.append(item)
            blocks.append(_table_block(item, len(tables) - 1))
        elif item.text:
            blocks.append(item)
    if blocks or tables or not pages:
        flush()
    title = (d.core_properties.title or "").strip() or None
    return ParsedDocument(kind="office", pages=pages, title=title, parser=parser)


def _kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):     # the group already exited
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, ValueError):
        proc.communicate(timeout=5)


def _convert(path: Path, target: str, out_dir: Path) -> Path:
    """Convert legacy/OpenDocument formats with LibreOffice (headless, private profile, time limit)."""
    s = get_settings()
    cmd = [s.soffice_path, "--headless", "--norestore", f"-env:UserInstallation=file://{out_dir}/profile",
           "--convert-to", target, "--outdir", str(out_dir), str(path)]
    try:
        # own process group: LibreOffice starts helper processes, and a timeout must stop all of them
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)  # noqa: S603
    except FileNotFoundError as e:
        raise ParseError("LibreOffice is required for this format but is not installed") from e
    try:
        _, err = proc.communicate(timeout=s.convert_timeout_sec)
    except subprocess.TimeoutExpired as e:
        _kill_group(proc)
        raise ParseError("document conversion timed out") from e
    finally:
        _kill_group(proc)                     # leftovers of a finished conversion, if any
    if proc.returncode != 0:
        raise ParseError(f"document conversion failed: {err.decode(errors='ignore')[:200]}")
    produced = list(out_dir.glob(f"*.{target}"))
    if not produced:
        raise ParseError("document conversion produced no output")
    return produced[0]


def _converted(path: Path, target: str, parse: Callable[[Path], ParsedDocument]) -> ParsedDocument:
    work = Path(tempfile.mkdtemp(prefix="docintel-convert-"))
    try:
        doc = parse(_convert(path, target, work))
        doc.parser = f"libreoffice->{doc.parser}"
        return doc
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------- sheets

def _sheet_pages(sheets: list[tuple[str, list[list[str]]]], kind_label: str) -> list[Page]:
    """One page per sheet (or per ROWS_PER_PAGE rows), each a heading block and a table with the header row."""
    pages: list[Page] = []
    for name, rows in sheets:
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = rows[0]
        for start in range(0, len(rows), ROWS_PER_PAGE):
            part = rows[start:start + ROWS_PER_PAGE]
            if start > 0:
                part = [header, *part]
            table = Table(part, 1 if len(part) > 1 else 0)
            blocks = [Block("heading", f"Sheet: {name}")] if name else []
            blocks.append(_table_block(table, 0))
            pages.append(Page.from_blocks(len(pages) + 1, kind_label, blocks, [table]))
    return pages or [Page.from_blocks(1, kind_label, [])]


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def parse_xlsx(path: Path, parser: str = "openpyxl") -> ParsedDocument:
    import openpyxl

    limit = get_settings().max_sheet_rows
    try:
        safety.check_container(path)
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except safety.UnsafeFileError as e:
        raise ParseError(str(e)) from e
    except Exception as e:
        raise ParseError(f"cannot open spreadsheet: {e}") from e
    sheets, warnings = [], []
    try:
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= limit:
                    warnings.append(_rows_warning(ws.title, limit))
                    break
                rows.append([_cell(v) for v in row])
            sheets.append((ws.title, rows))
    finally:
        wb.close()
    return ParsedDocument(kind="spreadsheet", pages=_sheet_pages(sheets, "sheet"), parser=parser, warnings=warnings)


def parse_xls(path: Path) -> ParsedDocument:
    import xlrd

    limit = get_settings().max_sheet_rows
    try:
        wb = xlrd.open_workbook(str(path))
    except Exception as e:
        raise ParseError(f"cannot open spreadsheet: {e}") from e
    sheets, warnings = [], []
    for sh in wb.sheets():
        rows = [[_cell(sh.cell_value(r, c)) for c in range(sh.ncols)] for r in range(min(sh.nrows, limit))]
        sheets.append((sh.name, rows))
        if sh.nrows > limit:
            warnings.append(_rows_warning(sh.name, limit))
    return ParsedDocument(kind="spreadsheet", pages=_sheet_pages(sheets, "sheet"), parser="xlrd", warnings=warnings)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_csv(path: Path) -> ParsedDocument:
    text = _decode(path.read_bytes())
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    limit = get_settings().max_sheet_rows
    rows, warnings = [], []
    for i, r in enumerate(csv.reader(io.StringIO(text), dialect)):
        if i >= limit:
            warnings.append(_rows_warning("", limit))
            break
        rows.append([c.strip() for c in r])
    return ParsedDocument(kind="spreadsheet", pages=_sheet_pages([("", rows)], "sheet"), parser="csv", warnings=warnings)


# ------------------------------------------------------------------------------------------------------- slides

def parse_pptx(path: Path, parser: str = "python-pptx") -> ParsedDocument:
    from pptx import Presentation
    from pptx.util import Emu

    try:
        safety.check_container(path)
        prs = Presentation(str(path))
    except safety.UnsafeFileError as e:
        raise ParseError(str(e)) from e
    except Exception as e:
        raise ParseError(f"cannot open presentation: {e}") from e
    pages = []
    for i, slide in enumerate(prs.slides, start=1):
        blocks: list[Block] = []
        tables: list[Table] = []
        title_shape = slide.shapes.title
        for shape in slide.shapes:
            bbox = None
            if None not in (shape.left, shape.top, shape.width, shape.height):
                bbox = tuple(round(Emu(v).pt, 1) for v in (shape.left, shape.top, shape.left + shape.width,
                                                           shape.top + shape.height))
            if shape.has_text_frame:
                paras = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                if not paras:
                    continue
                if title_shape is not None and shape.shape_id == title_shape.shape_id:
                    blocks.append(Block("heading", " ".join(paras), bbox))
                else:
                    blocks.extend(Block("list_item" if len(paras) > 1 else "paragraph", p, bbox) for p in paras)
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[c.text.strip() for c in row.cells] for row in shape.table.rows]
                rows = [r for r in rows if any(r)]
                if rows:
                    tables.append(Table(rows, 1 if len(rows) > 1 else 0, bbox))
                    blocks.append(_table_block(tables[-1], len(tables) - 1))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                blocks.append(Block("notes", f"Notes: {notes}"))
        pages.append(Page.from_blocks(i, "slide", blocks, tables))
    return ParsedDocument(kind="presentation", pages=pages or [Page.from_blocks(1, "slide", [])], parser=parser)


# ------------------------------------------------------------------------------------------------------- text

def _split_text_pages(text: str, kind: str) -> list[Page]:
    if "\f" in text:
        parts = text.split("\f")
    elif len(text) > TEXT_PAGE_CHARS:
        parts, cur = [], ""
        for para in text.split("\n\n"):
            if cur and len(cur) + len(para) > TEXT_PAGE_CHARS:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{para}" if cur else para
        parts.append(cur)
    else:
        parts = [text]
    return [Page.from_blocks(i, kind, text_blocks(p)) for i, p in enumerate(parts, start=1)]


def parse_txt(path: Path) -> ParsedDocument:
    return ParsedDocument(kind="text", pages=_split_text_pages(_decode(path.read_bytes()), "section"), parser="text")


class _HTMLBlocks(HTMLParser):
    """HTML to typed blocks: headings, paragraphs, list items and tables (rows of cells); scripts and styles are
    dropped. The document is never rendered or executed."""
    SKIP = ("script", "style", "noscript", "template", "head")
    BREAK = frozenset({"p", "div", "br", "section", "article", "header", "footer", "main", "nav", "aside",
                       "blockquote", "pre", "ul", "ol", "dl", "dt", "dd", "hr", "form", "fieldset"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.tables: list[Table] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._buf: list[str] = []
        self._kind = "paragraph"
        self._table_stack: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        if text:
            self.blocks.append(Block(self._kind, text))
        self._buf, self._kind = [], "paragraph"

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
            return
        if tag in self.SKIP:
            self._skip += 1
            return
        if tag == "table":
            self._flush()
            self._table_stack.append([])
        elif tag == "tr" and self._table_stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._kind = "heading"
        elif tag == "li":
            self._flush()
            self._kind = "list_item"
        elif tag in self.BREAK and self._cell is None:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            return
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table_stack:
            if any(self._row):
                self._table_stack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._table_stack:
            rows = self._table_stack.pop()
            if rows:
                self.tables.append(Table(rows, 1 if len(rows) > 1 else 0))
                self.blocks.append(_table_block(self.tables[-1], len(self.tables) - 1))
        elif (re.fullmatch(r"h[1-6]", tag) or tag == "li" or tag in self.BREAK) and self._cell is None:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip:
            return
        elif self._cell is not None:
            self._cell.append(data)
        else:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def html_to_blocks(html: str) -> tuple[list[Block], list[Table], str]:
    p = _HTMLBlocks()
    p.feed(html)
    p.close()
    return _refine(p.blocks), p.tables, " ".join(p.title.split())


def html_to_text(html: str) -> tuple[str, str]:
    blocks, _tables, title = html_to_blocks(html)
    return "\n\n".join(b.text for b in blocks), title


def parse_html(path: Path) -> ParsedDocument:
    blocks, tables, title = html_to_blocks(_decode(path.read_bytes()))
    return ParsedDocument(kind="text", pages=[Page.from_blocks(1, "section", blocks, tables)], title=title or None,
                          parser="html")


def parse_rtf(path: Path) -> ParsedDocument:
    from striprtf.striprtf import rtf_to_text

    text = rtf_to_text(_decode(path.read_bytes()), errors="ignore")
    return ParsedDocument(kind="office", pages=_split_text_pages(text, "section"), parser="striprtf")


# ------------------------------------------------------------------------------------------------------- e-mail

def parse_eml(path: Path, depth: int = 0) -> ParsedDocument:
    s = get_settings()
    msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    header = [Block("key_value", f"{h}: {msg[h]}") for h in ("From", "To", "Cc", "Date", "Subject") if msg[h]]
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body_blocks: list[Block] = []
    tables: list[Table] = []
    if body_part is not None:
        content = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            body_blocks, tables, _ = html_to_blocks(content)
        else:
            body_blocks = text_blocks(content)
    pages = [Page.from_blocks(1, "section", header + body_blocks, tables)]
    warnings = []
    for part in msg.iter_attachments():
        name = part.get_filename() or "attachment"
        data = part.get_payload(decode=True) or b""
        if not data:
            continue
        if depth >= s.max_attachment_depth:
            warnings.append(f"attachment {name} skipped: nested deeper than {s.max_attachment_depth} levels")
            continue
        if len(data) > s.max_upload_bytes:
            warnings.append(f"attachment {name} skipped: larger than the upload limit")
            continue
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / (Path(name.replace("\\", "/")).name or "attachment")
            p.write_bytes(data)
            try:
                child = parse_file(p, name, depth + 1)
            except (ParseError, ValueError) as e:
                warnings.append(f"attachment {name} skipped: {e}")
                continue
        for cp in child.pages:
            blocks = [Block("heading", f"Attachment: {name}"), *cp.blocks]
            pages.append(Page.from_blocks(len(pages) + 1, cp.kind, blocks, cp.tables, width=cp.width, height=cp.height,
                                          ocr_confidence=cp.ocr_confidence, words=cp.words, regions=cp.regions))
        warnings += child.warnings
    return ParsedDocument(kind="email", pages=pages, title=str(msg["Subject"] or "") or None, warnings=warnings,
                          parser="email")


# ------------------------------------------------------------------------------------------------------- dispatch

def parse_file(path: Path, filename: str, depth: int = 0) -> ParsedDocument:
    if path.stat().st_size == 0:
        raise ParseError("empty file")
    try:
        fmt = detect(path, filename)
    except ValueError as e:
        raise ParseError(str(e)) from e
    try:
        if fmt == "pdf":
            doc = parse_pdf(path)
        elif fmt == "image":
            doc = parse_image(path)
        elif fmt == "docx":
            doc = parse_docx(path)
        elif fmt in ("doc", "odt"):
            if fmt == "odt":
                safety.check_container(path)
            doc = _converted(path, "docx", parse_docx)
        elif fmt == "rtf":
            doc = parse_rtf(path)
        elif fmt == "xlsx":
            doc = parse_xlsx(path)
        elif fmt == "xls":
            doc = parse_xls(path)
        elif fmt == "ods":
            safety.check_container(path)
            doc = _converted(path, "xlsx", parse_xlsx)
        elif fmt == "csv":
            doc = parse_csv(path)
        elif fmt == "pptx":
            doc = parse_pptx(path)
        elif fmt in ("ppt", "odp"):
            if fmt == "odp":
                safety.check_container(path)
            doc = _converted(path, "pptx", parse_pptx)
        elif fmt == "html":
            doc = parse_html(path)
        elif fmt == "eml":
            doc = parse_eml(path, depth)
        else:
            doc = parse_txt(path)
    except safety.UnsafeFileError as e:
        raise ParseError(str(e)) from e
    doc.canonicalize()
    return doc
