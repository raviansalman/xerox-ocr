"""Render a page of an original document as PNG for evidence display."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageSequence

from docintel.processing import safety
from docintel.processing.detect import detect

MAX_SIDE = 1600


def render_page_png(path: Path, filename: str, page: int, dpi: int = 110) -> bytes | None:
    try:
        fmt = detect(path, filename)
    except ValueError:
        return None
    if fmt == "pdf":
        import fitz
        with fitz.open(path) as doc:
            if not 1 <= page <= doc.page_count:
                return None
            p = doc[page - 1]
            zoom = safety.render_scale(p.rect.width, p.rect.height, dpi)    # bounded like OCR rendering
            return p.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    if fmt == "image":
        safety.configure_image_limits()
        try:
            img = Image.open(path)
        except Image.DecompressionBombError:
            return None
        with img:
            for i, frame in enumerate(ImageSequence.Iterator(img), start=1):
                if i == page:
                    try:
                        safety.check_image_size(*frame.size)
                    except safety.UnsafeFileError:
                        return None
                    im = frame.convert("RGB")
                    im.thumbnail((MAX_SIDE, MAX_SIDE))
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    return buf.getvalue()
    return None
