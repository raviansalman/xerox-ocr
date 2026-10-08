"""Hostile files fail clearly instead of exhausting resources."""
import io
import zipfile

import pytest
from PIL import Image

from docintel.config import get_settings
from docintel.processing.parsers import ParseError, parse_file
from tests.fixtures import corpus as C


def _zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_zip_bomb_office_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_archive_ratio", 50)
    p = _zip(tmp_path / "bomb.docx", {"[Content_Types].xml": b"<x/>", "word/document.xml": b"\0" * (20 * 1024 * 1024)})
    with pytest.raises(ParseError, match="compression ratio"):
        parse_file(p, p.name)


def test_oversized_container_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_archive_uncompressed_bytes", 1024)
    p = C.docx_file(tmp_path / "big.docx", "TITLE", ["x" * 5000])
    with pytest.raises(ParseError, match="expands to"):
        parse_file(p, p.name)


def test_unsafe_member_path_is_rejected(tmp_path):
    p = _zip(tmp_path / "evil.xlsx", {"xl/workbook.xml": b"<x/>", "../../etc/passwd": b"x"})
    with pytest.raises(ParseError, match="unsafe member path"):
        parse_file(p, p.name)


def test_plain_zip_archives_are_not_supported(tmp_path):
    p = _zip(tmp_path / "files.zip", {"a.txt": b"hello"})
    with pytest.raises(ParseError, match="unsupported"):
        parse_file(p, p.name)


def test_image_decompression_bomb_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_image_pixels", 1_000_000)
    buf = io.BytesIO()
    Image.new("1", (3000, 3000), 1).save(buf, format="PNG")
    (tmp_path / "bomb.png").write_bytes(buf.getvalue())
    with pytest.raises(ParseError, match="pixel limit"):
        parse_file(tmp_path / "bomb.png", "bomb.png")


def test_nested_email_attachments_stop_at_the_depth_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_attachment_depth", 1)
    inner = C.eml_file(tmp_path / "inner.eml", "Inner", "inner body", ("deep.txt", b"deepest text"))
    outer = C.eml_file(tmp_path / "outer.eml", "Outer", "outer body", ("inner.eml", inner.read_bytes()))
    doc = parse_file(outer, outer.name)
    assert "deepest text" not in doc.text and any("nested deeper" in w for w in doc.warnings)


def test_empty_and_binary_files_fail(tmp_path):
    (tmp_path / "empty.pdf").write_bytes(b"")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "empty.pdf", "empty.pdf")
    (tmp_path / "x.bin").write_bytes(b"\x00\x01\x02\x00binary")
    with pytest.raises(ParseError):
        parse_file(tmp_path / "x.bin", "x.bin")


def test_outlook_msg_is_rejected_with_a_clear_reason(tmp_path):
    from docintel.processing.detect import detect
    p = tmp_path / "message.msg"
    p.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600)
    with pytest.raises(ValueError, match=r"\.eml"):
        detect(p, "message.msg")
    assert detect(p, "legacy.doc") == "doc"


def test_ocr_is_bounded_in_time():
    from PIL import Image

    from docintel.processing.ocr import OcrTimeout, TesseractOcr
    ocr = TesseractOcr("eng", timeout_sec=1)

    class Slow:
        class Output:
            DICT = "dict"

        @staticmethod
        def image_to_data(*a, timeout=None, **k):
            assert timeout == 1                      # the limit reaches pytesseract
            raise RuntimeError("Tesseract process timeout")

    ocr._pt = Slow
    with pytest.raises(OcrTimeout):
        ocr.recognize(Image.new("L", (100, 100), 255))


def test_images_between_the_limit_and_twice_the_limit_are_rejected(tmp_path, monkeypatch):
    """Pillow only raises at twice MAX_IMAGE_PIXELS; the engine refuses anything over the limit itself."""
    monkeypatch.setattr(get_settings(), "max_image_pixels", 1_000_000)
    Image.new("1", (1200, 1200), 1).save(tmp_path / "big.png")           # 1.44 million pixels
    with pytest.raises(ParseError, match="pixel limit"):
        parse_file(tmp_path / "big.png", "big.png")


def test_frames_beyond_the_page_limit_are_reported_not_dropped_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_pages", 2)
    frames = [Image.new("L", (200, 100), 255) for _ in range(4)]
    frames[0].save(tmp_path / "multi.tiff", save_all=True, append_images=frames[1:])
    parsed = parse_file(tmp_path / "multi.tiff", "multi.tiff")
    assert len(parsed.pages) == 2
    assert parsed.warnings == ["only the first 2 of 4 frames were processed (DOCINTEL_MAX_PAGES)"]


@pytest.mark.parametrize("kind", ["csv", "xlsx"])
def test_spreadsheet_rows_beyond_the_limit_are_reported(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(get_settings(), "max_sheet_rows", 3)
    rows = [["Item", "Qty"]] + [[f"item {i}", i] for i in range(10)]
    if kind == "csv":
        p = tmp_path / "rows.csv"
        p.write_text("\n".join(f"{a},{b}" for a, b in rows))
    else:
        p = C.xlsx_file(tmp_path / "rows.xlsx", {"Stock": rows})
    parsed = parse_file(p, p.name)
    assert "item 1" in parsed.text and "item 5" not in parsed.text
    assert len(parsed.warnings) == 1 and "cut at 3 rows" in parsed.warnings[0]


def test_page_images_are_rendered_within_the_pixel_limit(tmp_path, monkeypatch):
    import fitz

    from docintel.processing.render import render_page_png
    monkeypatch.setattr(get_settings(), "max_image_pixels", 400_000)
    doc = fitz.open()
    doc.new_page(width=5000, height=5000)                 # about 7,600 x 7,600 pixels at 110 dpi unbounded
    doc.save(tmp_path / "huge.pdf")
    png = Image.open(io.BytesIO(render_page_png(tmp_path / "huge.pdf", "huge.pdf", 1)))
    assert png.width * png.height <= 400_000
    Image.new("L", (1000, 1000), 255).save(tmp_path / "big.png")
    assert render_page_png(tmp_path / "big.png", "big.png", 1) is None


def test_a_conversion_timeout_kills_every_process_it_started(tmp_path, monkeypatch):
    """LibreOffice spawns helpers; a timed-out conversion must not leave any of them running."""
    import os
    import time

    from docintel.processing.parsers import _convert
    script = tmp_path / "fake_soffice.sh"
    pidfile = tmp_path / "child.pid"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pidfile}\nwait\n")
    script.chmod(0o755)
    monkeypatch.setattr(get_settings(), "soffice_path", str(script))
    monkeypatch.setattr(get_settings(), "convert_timeout_sec", 1)
    started = time.monotonic()
    with pytest.raises(ParseError, match="timed out"):
        _convert(tmp_path / "in.doc", "docx", tmp_path)
    assert time.monotonic() - started < 10       # a helper holding the output pipe must not stretch the limit
    child = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the converter's child process survived the timeout")
