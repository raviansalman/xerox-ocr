"""Resource limits for untrusted files: container (zip) bombs, image decompression bombs, rendering size.

Parsers call these before handing a file to a third-party library, so a hostile file fails with a clear
``ParseError`` instead of exhausting memory or disk.
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

from PIL import Image

from docintel.config import get_settings


class UnsafeFileError(ValueError):
    """The file exceeds a resource limit or is structured to exhaust resources."""


MAX_ZIP_ENTRIES = 20000


def check_container(path: Path) -> None:
    """Reject zip-based documents (DOCX, XLSX, PPTX, OpenDocument) whose uncompressed size or compression ratio
    exceeds the configured limits, or that contain absolute or parent-relative member paths."""
    s = get_settings()
    compressed = max(1, path.stat().st_size)
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except zipfile.BadZipFile as e:
        raise UnsafeFileError(f"corrupted container: {e}") from e
    if len(infos) > MAX_ZIP_ENTRIES:
        raise UnsafeFileError(f"container has {len(infos)} entries; limit is {MAX_ZIP_ENTRIES}")
    total = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            raise UnsafeFileError("container has an unsafe member path")
        total += info.file_size
    if total > s.max_archive_uncompressed_bytes:
        raise UnsafeFileError(f"container expands to {total} bytes; limit is {s.max_archive_uncompressed_bytes}")
    if total / compressed > s.max_archive_ratio:
        raise UnsafeFileError(f"container compression ratio {total // compressed}:1 exceeds {s.max_archive_ratio}:1")


def configure_image_limits() -> None:
    """Make Pillow refuse images larger than the configured pixel count (it raises DecompressionBombError)."""
    Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels


def check_image_size(width: int, height: int) -> None:
    """Refuse a frame above the pixel limit before it is decoded (Pillow itself only refuses at twice the limit)."""
    limit = get_settings().max_image_pixels
    if width * height > limit:
        raise UnsafeFileError(f"image of {width}x{height} pixels exceeds the limit of {limit}")


def render_scale(width_pt: float, height_pt: float, dpi: int) -> float:
    """Zoom factor for rendering a page at ``dpi`` without exceeding the pixel limit."""
    limit = get_settings().max_image_pixels
    zoom = dpi / 72.0
    pixels = width_pt * zoom * height_pt * zoom
    if pixels > limit:
        zoom *= (limit / pixels) ** 0.5
        while math.ceil(width_pt * zoom) * math.ceil(height_pt * zoom) > limit:   # renderers round up
            zoom *= 0.995
    return zoom
