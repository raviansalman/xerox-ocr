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


# A page recognized with lower mean confidence than this is checked for a sideways or upside-down scan.
ORIENTATION_RETRY_CONF = 0.70
# Clockwise correction reported by Tesseract OSD -> the PIL transpose that applies it.
_UPRIGHT = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}


def _unrotate(result: OcrResult, rotate: int, width: int, height: int) -> OcrResult:
    """Map word and block boxes recognized on the upright image back to the original (as scanned) image, so they
    still line up with the stored page image."""
    def point(x: float, y: float) -> tuple[float, float]:
        if rotate == 90:            # upright = original turned 90 degrees clockwise
            return y, height - x
        if rotate == 180:
            return width - x, height - y
        return width - y, x         # 270

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        (ax, ay), (bx, by) = point(x0, y0), point(x1, y1)
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

    words = [Word(w.text, *box(w.x0, w.y0, w.x1, w.y1), w.conf) for w in result.words]
    blocks = [Block(b.type, b.text, box(*b.bbox) if b.bbox else None, b.confidence) for b in result.blocks]
    return OcrResult(result.text, words, result.confidence, width, height, blocks)


class TesseractOcr:
    def __init__(self, languages: str = "eng", psm: int = 3, timeout_sec: int = 180):
        import pytesseract

        self._pt = pytesseract
        self.languages = languages
        self.psm = psm
        self.timeout_sec = timeout_sec
        self.name = f"tesseract-{pytesseract.get_tesseract_version()}:{languages}:osd"

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
        """Recognize a page. A page read with low confidence is checked with Tesseract's orientation detection; a
        sideways or upside-down scan is turned upright and read again, and the better reading is kept (boxes are
        reported on the image as scanned)."""
        result = self._recognize(image)
        if result.confidence is not None and result.confidence >= ORIENTATION_RETRY_CONF:
            return result
        rotate = self._orientation(image)
        # orientation detection declines sparse pages ("too few characters"): try the two sideways turns instead
        for turn in ([rotate] if rotate is not None else [90, 270]):
            if not turn:
                continue
            upright = self._recognize(ImageOps.exif_transpose(image).transpose(_UPRIGHT[turn]))
            if (upright.confidence or 0) * len(upright.words) > (result.confidence or 0) * len(result.words) * 1.2:
                result = _unrotate(upright, turn, image.width, image.height)
        return result

    def _orientation(self, image: Image.Image) -> int | None:
        """Clockwise rotation (90, 180, 270) that makes the page upright; 0 when it is upright or the detector is
        unsure; None when the detector could not judge the page (too little text) or is not installed."""
        img, _ = self.preprocess(image)
        try:
            osd = self._pt.image_to_osd(img, config="--psm 0", output_type=self._pt.Output.DICT, timeout=60)
        except Exception:
            return None
        rotate = int(osd.get("rotate", 0)) % 360
        return rotate if rotate in _UPRIGHT and float(osd.get("orientation_conf", 0)) >= 2.0 else 0

    def _recognize(self, image: Image.Image) -> OcrResult:
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
