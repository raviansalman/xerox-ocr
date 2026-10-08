"""Retrievable units built from canonical blocks.

* passage: consecutive blocks of one page, about ``target_chars`` long, never crossing a page; long blocks are split
  at sentence boundaries; consecutive passages overlap by a short tail starting at a sentence boundary. A passage's
  text is exactly ``page.text[char_start:char_end]``, so evidence offsets can always be checked against the source.
* table_row: one data row of a table, its text the row's line in the page text, with the column headers as context
  ("Item: Toner; Qty: 20") for search and embedding.

Headings become the context of the passages that follow them.
"""
from __future__ import annotations

import re

from docintel.models import Chunk, Page

_HEADING = re.compile(
    r"^(?:(?:article|section|clause|part|chapter|schedule|appendix|annex|exhibit)\s+[\dIVXLC]+(?:\.\d+)*\b.*"
    r"|\d+(?:\.\d+)*\.?\s+[A-Z][^.]{2,70}"
    r"|[A-Z][A-Z0-9 &/,'()-]{3,80})$", re.IGNORECASE)
_SENT_END = re.compile(r"[.!?;:](?=\s+[A-Z0-9\"'(])")
MAX_ROW_UNITS_PER_DOCUMENT = 5000


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or s.endswith((".", ",", ";")):
        return False
    if s.isupper() and sum(c.isalpha() for c in s) >= 4:
        return True
    return bool(_HEADING.match(s)) and not s.isupper() and len(s.split()) <= 12 and any(c.isdigit() for c in s[:15])


def _sentence_starts(text: str, start: int, end: int) -> list[int]:
    """Offsets in ``text[start:end]`` where a sentence begins (after punctuation and whitespace)."""
    out = []
    for m in _SENT_END.finditer(text, start, end):
        pos = m.end()
        while pos < end and text[pos].isspace():
            pos += 1
        if pos < end:
            out.append(pos)
    return out


def _pieces(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into spans no longer than ``limit``, at sentence boundaries where possible,
    otherwise at whitespace."""
    if end - start <= limit:
        return [(start, end)]
    out = []
    cur = start
    while end - cur > limit:
        bounds = [b for b in _sentence_starts(text, cur, cur + limit) if b - cur >= limit // 3]
        if bounds:
            cut = bounds[-1]
        else:
            ws = text.rfind(" ", cur + limit // 2, cur + limit)
            cut = ws + 1 if ws > 0 else cur + limit
        piece_end = cut
        while piece_end > cur and text[piece_end - 1].isspace():
            piece_end -= 1
        out.append((cur, piece_end))
        cur = cut
    if cur < end:
        out.append((cur, end))
    return out


def _overlap_start(text: str, start: int, end: int, overlap: int) -> int | None:
    """A sentence start within the last ``overlap`` characters of ``text[start:end]``, to begin the next unit."""
    starts = [s for s in _sentence_starts(text, max(start, end - overlap), end) if s > start]
    return starts[0] if starts else None


def chunk_pages(pages: list[Page], target_chars: int = 1000) -> list[Chunk]:
    limit = int(target_chars * 1.25)
    overlap = max(80, target_chars // 7)
    chunks: list[Chunk] = []
    heading: str | None = None
    row_units = 0
    for page in pages:
        if not page.blocks:
            page.ensure_blocks()
        text = page.text
        if not text.strip():
            continue
        offsets = page.block_offsets()
        cur_start: int | None = None
        cur_end = 0
        cur_blocks: list[int] = []
        unit_heading = heading

        def emit() -> None:
            nonlocal cur_start, cur_blocks
            if cur_start is not None and cur_end > cur_start and text[cur_start:cur_end].strip():
                chunks.append(Chunk(len(chunks), page.number, page.number, text[cur_start:cur_end], unit_heading,
                                    "passage", cur_start, cur_end, cur_blocks[0], cur_blocks[-1],
                                    unit_heading or ""))
            cur_start, cur_blocks = None, []

        for bi, block in enumerate(page.blocks):
            b_start, b_end = offsets[bi]
            if block.type == "heading":
                if cur_start is not None and cur_end - cur_start > target_chars // 3:
                    emit()
                heading = block.text.split("\n")[0][:200]
                if cur_start is None:
                    unit_heading = heading
            for p_start, p_end in _pieces(text, b_start, b_end, target_chars):
                if cur_start is not None and (p_end - cur_start > limit or
                                              (p_end - cur_start > target_chars and cur_end - cur_start >= target_chars // 3)):
                    prev_start, prev_end = cur_start, cur_end
                    emit()
                    unit_heading = heading
                    tail = _overlap_start(text, prev_start, prev_end, overlap)
                    if tail is not None and p_end - tail <= limit:
                        cur_start = tail
                        cur_blocks = [page.block_at(tail) or bi]
                if cur_start is None:
                    cur_start = p_start
                cur_end = p_end
                if bi not in cur_blocks:
                    cur_blocks.append(bi)
        emit()
        # one unit per table data row, with the column headers as context
        for bi, block in enumerate(page.blocks):
            if block.type != "table" or block.table is None or block.table >= len(page.tables):
                continue
            table = page.tables[block.table]
            b_start, _ = offsets[bi]
            pos = b_start
            line_starts = {}
            for i in range(len(table.rows)):
                line = table.row_text(i)
                if not line:
                    continue
                found = text.find(line, pos)
                if found < 0:
                    continue
                line_starts[i] = (found, found + len(line))
                pos = found + len(line)
            for i, (s, e) in line_starts.items():
                if i < table.header_rows or row_units >= MAX_ROW_UNITS_PER_DOCUMENT:
                    continue
                context = " · ".join(x for x in (heading if heading and heading != block.text else None,
                                                  table.caption, table.labelled_row(i)) if x)
                chunks.append(Chunk(len(chunks), page.number, page.number, text[s:e], heading, "table_row", s, e,
                                    bi, bi, context))
                row_units += 1
    return chunks


def document_unit(ordinal: int, title: str, filename: str, type_label: str | None, key_fields: list[str]) -> Chunk:
    """A synthetic unit describing the whole document, for document-level semantic and file-name search."""
    parts = [p for p in (title, type_label, filename, "; ".join(key_fields)) if p]
    return Chunk(ordinal, 1, 1, "\n".join(parts), None, "document", None, None, None, None, "")
