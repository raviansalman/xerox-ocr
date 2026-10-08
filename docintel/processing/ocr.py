"""OCR behind a small interface. The default engine is Tesseract (words, boxes and real confidences)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageOps

from docintel.models import Block, Word

# One Tesseract thread per call: pages are parallelized by the pipeline instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


@dataclass
class OcrResult:
    text: str
    words: list[Word]
    confidence: float | None      # mean word confidence, 0..1
    width: int
    height: int
    blocks: list[Block]           # one per recognized paragraph, in reading order, with box and confidence


class OcrEngine(Protocol):
    """Recognizes printed text in an image. Implementations report word boxes and real confidences; they do not
    claim handwriting recognition."""
    name: str

    def recognize(self, image: Image.Image) -> OcrResult: ...


class OcrTimeout(RuntimeError):
    """OCR of one page exceeded its time limit (a hostile or pathological image); the document fails."""


class TesseractOcr:
    def __init__(self, languages: str = "eng", psm: int = 3, timeout_sec: int = 180):
        import pytesseract

        self._pt = pytesseract
        self.languages = languages
        self.psm = psm
        self.timeout_sec = timeout_sec
        self.name = f"tesseract-{pytesseract.get_tesseract_version()}:{languages}"

    @staticmethod
    def preprocess(image: Image.Image) -> tuple[Image.Image, float]:
        img = ImageOps.exif_transpose(image)
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        img = ImageOps.autocontrast(img.convert("L"))
        scale = 1.0
        if max(img.size) < 1400:          # small images: upscale so glyphs are tall enough for OCR
            scale = 2.0
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        return img, scale

    def recognize(self, image: Image.Image) -> OcrResult:
        img, scale = self.preprocess(image)
        try:
            data = self._pt.image_to_data(img, lang=self.languages, config=f"--oem 1 --psm {self.psm}",
                                          output_type=self._pt.Output.DICT, timeout=self.timeout_sec)
        except RuntimeError as e:                    # pytesseract kills the process and raises on timeout
            if "timeout" in str(e).lower():
                raise OcrTimeout(f"OCR of a page took longer than {self.timeout_sec} s") from e
            raise
        words: list[Word] = []
        lines: dict[tuple, list[Word]] = {}
        order: list[tuple] = []
        for i, txt in enumerate(data["text"]):
            txt = (txt or "").strip()
            conf = float(data["conf"][i])
            if not txt or conf < 0:
                continue
            x, y, w, h = (data[k][i] / scale for k in ("left", "top", "width", "height"))
            word = Word(txt, x, y, x + w, y + h, conf)
            words.append(word)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in lines:
                lines[key] = []
                order.append(key)
            lines[key].append(word)
        blocks: list[Block] = []
        paragraphs: dict[tuple, list[tuple]] = {}
        for key in order:
            paragraphs.setdefault(key[:2], []).append(key)
        for keys in paragraphs.values():
            par_words = [w for k in keys for w in lines[k]]
            text = "\n".join(" ".join(w.text for w in lines[k]) for k in keys)
            bbox = (min(w.x0 for w in par_words), min(w.y0 for w in par_words),
                    max(w.x1 for w in par_words), max(w.y1 for w in par_words))
            blocks.append(Block("paragraph", text, bbox, round(sum(w.conf for w in par_words) / len(par_words) / 100.0, 3)))
        conf = sum(w.conf for w in words) / len(words) / 100.0 if words else None
        return OcrResult(text="\n\n".join(b.text for b in blocks), words=words, confidence=conf,
                         width=image.width, height=image.height, blocks=blocks)


_engine: OcrEngine | None = None


def get_ocr() -> OcrEngine:
    global _engine
    if _engine is None:
        from docintel.config import get_settings
        s = get_settings()
        _engine = TesseractOcr(s.ocr_languages, timeout_sec=s.ocr_page_timeout_sec)
    return _engine
