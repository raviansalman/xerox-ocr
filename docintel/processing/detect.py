"""File type detection from content (magic bytes), with the file extension only as a tie-breaker."""
from __future__ import annotations

import zipfile
from pathlib import Path

TEXT_EXTS = {".txt", ".md", ".markdown", ".log", ".json", ".xml", ".yaml", ".yml"}

SUPPORTED = {
    "pdf", "image", "docx", "doc", "odt", "rtf", "xlsx", "xls", "ods", "csv", "pptx", "ppt", "odp",
    "txt", "html", "eml",
}


def detect(path: Path, filename: str) -> str:
    """Return a format key from SUPPORTED, or raise ValueError for unsupported content."""
    ext = Path(filename).suffix.lower()
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*", b"BM")) or \
            (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                if "mimetype" in names:
                    mt = z.read("mimetype").decode("ascii", "ignore")
                    if "opendocument.text" in mt:
                        return "odt"
                    if "opendocument.spreadsheet" in mt:
                        return "ods"
                    if "opendocument.presentation" in mt:
                        return "odp"
                if any(n.startswith("word/") for n in names):
                    return "docx"
                if any(n.startswith("xl/") for n in names):
                    return "xlsx"
                if any(n.startswith("ppt/") for n in names):
                    return "pptx"
        except zipfile.BadZipFile:
            pass
        raise ValueError("unsupported archive or corrupted Office file")
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):     # legacy OLE2 Office
        if ext == ".msg":
            raise ValueError("unsupported file type (.msg): save the message as .eml")
        return {".xls": "xls", ".ppt": "ppt", ".pps": "ppt"}.get(ext, "doc")
    if head.lstrip().startswith(b"{\\rtf"):
        return "rtf"
    lowered = head.lower().lstrip()
    if ext in (".html", ".htm") or lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    if ext in (".eml",) or (head.startswith((b"Received:", b"From:", b"Return-Path:", b"MIME-Version:", b"Message-ID:"))
                            and b"\n\n" in head.replace(b"\r\n", b"\n")):
        return "eml"
    if ext in (".csv", ".tsv"):
        return "csv"
    if b"\x00" not in head:
        return "txt"
    raise ValueError(f"unsupported file type ({ext or 'no extension'})")


MIME = {
    "pdf": "application/pdf", "image": "image/*",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword", "odt": "application/vnd.oasis.opendocument.text", "rtf": "application/rtf",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xls": "application/vnd.ms-excel",
    "ods": "application/vnd.oasis.opendocument.spreadsheet", "csv": "text/csv",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint", "odp": "application/vnd.oasis.opendocument.presentation",
    "txt": "text/plain", "html": "text/html", "eml": "message/rfc822",
}
