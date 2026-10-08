"""Canonical in-memory document model produced by processing and persisted by ``docintel.storage.repo``.

A page is a list of typed blocks in reading order. The page's canonical text is its blocks' texts joined by a blank
line (``BLOCK_SEP``), so a character offset in the page text identifies exactly one block and position; every
extracted value, entity, clause, relation and retrievable unit refers to the page text by such offsets.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from datetime import date

BLOCK_SEP = "\n\n"
BlockType = str   # heading | paragraph | list_item | table | key_value | caption | header | footer | notes | figure | other
BBox = tuple[float, float, float, float]


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float


@dataclass
class Region:
    type: str                      # signature
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float


@dataclass
class Block:
    type: BlockType
    text: str
    bbox: BBox | None = None
    confidence: float | None = None          # OCR confidence of the block, 0..1
    table: int | None = None                 # index into Page.tables for table blocks


@dataclass
class Table:
    rows: list[list[str]]
    header_rows: int = 1                     # leading rows that label the columns (0 when unknown)
    bbox: BBox | None = None
    caption: str | None = None

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def row_text(self, i: int) -> str:
        return " | ".join(c for c in self.rows[i] if c)

    def text(self) -> str:
        return "\n".join(self.row_text(i) for i in range(len(self.rows)) if any(self.rows[i]))

    def labelled_row(self, i: int) -> str:
        """A data row with its column headers: "Item: Toner; Qty: 20" (used to search and embed table rows)."""
        if self.header_rows <= 0 or i < self.header_rows:
            return self.row_text(i)
        header = self.rows[self.header_rows - 1]
        parts = []
        for j, v in enumerate(self.rows[i]):
            if not v:
                continue
            h = header[j] if j < len(header) else ""
            parts.append(f"{h}: {v}" if h and h != v else v)
        return "; ".join(parts)


@dataclass
class Page:
    number: int                    # 1-based
    text: str
    kind: str                      # native | ocr | sheet | slide | section
    width: float | None = None
    height: float | None = None
    ocr_confidence: float | None = None
    words: list[Word] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    @classmethod
    def from_blocks(cls, number: int, kind: str, blocks: list[Block], tables: list[Table] | None = None,
                    **kw) -> Page:
        blocks = [b for b in blocks if b.text.strip()]
        for b in blocks:
            b.text = _clean_block_text(b.text)
        return cls(number, BLOCK_SEP.join(b.text for b in blocks), kind, blocks=blocks, tables=tables or [], **kw)

    def ensure_blocks(self) -> None:
        """Give a page built from plain text its blocks (paragraphs, headings, list items) and make ``text`` the
        canonical join of them."""
        if not self.blocks:
            self.blocks = text_blocks(self.text)
            for b in self.blocks:
                b.text = _clean_block_text(b.text)
            self.blocks = [b for b in self.blocks if b.text]
        self.text = BLOCK_SEP.join(b.text for b in self.blocks)

    def block_offsets(self) -> list[tuple[int, int]]:
        """(start, end) of every block in ``text``."""
        out, pos = [], 0
        for b in self.blocks:
            out.append((pos, pos + len(b.text)))
            pos += len(b.text) + len(BLOCK_SEP)
        return out

    def block_at(self, offset: int) -> int | None:
        """Index of the block containing a character offset of ``text`` (separators belong to no block)."""
        offsets = self.block_offsets()
        starts = [s for s, _ in offsets]
        i = bisect.bisect_right(starts, offset) - 1
        if 0 <= i < len(offsets) and offsets[i][0] <= offset < max(offsets[i][1], offsets[i][0] + 1):
            return i
        return None


_LIST_ITEM = re.compile(r"^\s*(?:[-*•▪◦·]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+\S")


def _clean_block_text(text: str) -> str:
    lines = [" ".join(ln.split()) for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def text_blocks(text: str) -> list[Block]:
    """Blocks from plain text: blank lines separate paragraphs; headings, list items and table-like lines
    ("a | b | c") become their own blocks."""
    from docintel.processing.chunking import is_heading

    blocks: list[Block] = []
    for para in re.split(r"\n\s*\n|\f", text or ""):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        buf: list[str] = []
        kind = "paragraph"

        def flush() -> None:
            nonlocal buf, kind
            if buf:
                blocks.append(Block(kind, "\n".join(buf)))
            buf, kind = [], "paragraph"

        for ln in lines:
            if is_heading(ln):
                flush()
                blocks.append(Block("heading", ln))
            elif _LIST_ITEM.match(ln):
                flush()
                blocks.append(Block("list_item", ln))
            elif " | " in ln:
                if kind != "table_text":
                    flush()
                    kind = "table_text"
                buf.append(ln)
            else:
                if kind != "paragraph":
                    flush()
                buf.append(ln)
        flush()
    for b in blocks:
        if b.type == "table_text":
            b.type = "table"
        elif b.type == "paragraph" and re.match(r"^[A-Z][\w /&.'-]{1,40}:\s+\S", b.text) and "\n" not in b.text:
            b.type = "key_value"
    return blocks


@dataclass
class ParsedDocument:
    kind: str                      # native | scanned | mixed | image | office | text | spreadsheet | presentation | email
    pages: list[Page]
    title: str | None = None
    warnings: list[str] = field(default_factory=list)
    parser: str = ""               # parser name and version, for provenance

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def canonicalize(self) -> None:
        for p in self.pages:
            p.ensure_blocks()


@dataclass
class Chunk:
    """A retrievable unit: an exact span of one page's canonical text (``text == page.text[char_start:char_end]``),
    or, for ``unit_type == "document"``, a synthetic summary of the document (title, type, key fields)."""
    ordinal: int
    page_start: int
    page_end: int
    text: str
    heading: str | None = None
    unit_type: str = "passage"               # passage | table_row | document
    char_start: int | None = None
    char_end: int | None = None
    block_from: int | None = None
    block_to: int | None = None
    context: str = ""                        # added for search and embedding (section path, table headers)


@dataclass
class Span:
    page: int
    char_start: int
    char_end: int
    block: int | None = None


@dataclass
class FieldValue:
    name: str
    value_text: str | None = None
    value_num: float | None = None
    value_date: date | None = None
    unit: str | None = None
    page: int | None = None
    snippet: str | None = None
    confidence: float = 0.8
    method: str = "rule"
    span: Span | None = None


@dataclass
class Entity:
    type: str                      # person | organization | email | phone | identifier | jurisdiction
    value: str
    value_norm: str
    role: str | None = None
    page: int | None = None
    snippet: str | None = None
    confidence: float = 0.8
    method: str = "rule"
    span: Span | None = None


@dataclass
class Clause:
    clause_type: str
    text: str
    ref: str | None = None
    heading: str | None = None
    page: int | None = None
    confidence: float = 0.8
    span: Span | None = None


@dataclass
class Relation:
    """subject —predicate→ object, with qualifiers ({"notice": "30 days"}) and the sentence that states it."""
    predicate: str
    subject_type: str              # person | organization | party | document | text
    subject: str
    object_type: str
    object: str
    qualifiers: dict[str, str] = field(default_factory=dict)
    snippet: str | None = None
    confidence: float = 0.8
    extractor: str = "rule"
    span: Span | None = None


@dataclass
class Classification:
    label: str
    confidence: float
    method: str


@dataclass
class Understanding:
    classification: Classification
    fields: list[FieldValue]
    entities: list[Entity]
    clauses: list[Clause]
    has_signature: bool
    language: str
    title: str | None
    relations: list[Relation] = field(default_factory=list)
