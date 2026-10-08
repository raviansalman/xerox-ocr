"""Signature-like mark detection on scanned pages (classical image analysis, no model download).

Printed text is removed using the OCR word boxes, ruled lines are removed morphologically, and the remaining ink is
grouped into blobs. A blob counts as a signature-like mark when its size, aspect ratio and ink density look like
handwriting; confidence rises when it sits next to a signature cue ("signed", "signature", "approved by", ...).
The result is reported as a *signature-like mark* with a confidence, never as proof that a document is signed.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

from docintel.models import Region, Word

CUES = re.compile(r"^(?:sign(?:ed|ature|atory)?|approved|authori[sz]ed|executed|witness|by|/s/)[:.,]?$", re.I)
MIN_CONFIDENCE = 0.6


def detect_signatures(image: Image.Image, words: list[Word]) -> list[Region]:
    import cv2

    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    if h < 200 or w < 200:
        return []
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    heights = sorted(wd.y1 - wd.y0 for wd in words if wd.text.strip())
    line_h = heights[len(heights) // 2] if heights else max(12.0, h / 60)
    # 1. remove printed words
    pad = max(2, int(line_h * 0.15))
    for wd in words:
        if wd.conf >= 60 and any(c.isalnum() for c in wd.text):    # misread scribbles have low confidence
            cv2.rectangle(ink, (int(wd.x0) - pad, int(wd.y0) - pad), (int(wd.x1) + pad, int(wd.y1) + pad), 0, -1)
    # 2. remove ruled lines (long horizontal/vertical runs)
    for k in ((max(40, w // 15), 1), (1, max(40, h // 15))):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        ink = cv2.subtract(ink, lines)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))       # speckle
    if cv2.countNonZero(ink) < 50:
        return []
    # 3. group strokes into blobs
    kx, ky = max(9, int(line_h * 1.2)), max(5, int(line_h * 0.6))
    merged = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    cue_boxes = [wd for wd in words if CUES.match(wd.text.strip())]
    out = []
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        if bw < max(0.05 * w, 3 * line_h) or bw > 0.6 * w or bh < 0.9 * line_h or bh > 8 * line_h:
            continue
        aspect = bw / max(1, bh)
        if not 1.4 <= aspect <= 18:
            continue
        roi = ink[y:y + bh, x:x + bw]
        density = cv2.countNonZero(roi) / float(bw * bh)
        if not 0.02 <= density <= 0.35:                         # handwriting is sparse; logos and stamps are dense
            continue
        # strokes cross many columns with few pixels each (thin, continuous lines)
        cols = (roi > 0).sum(axis=0)
        coverage = float((cols > 0).mean())
        if coverage < 0.5:
            continue
        conf = 0.5 + min(0.15, 0.6 * coverage * 0.25)
        cx, cy = x + bw / 2, y + bh / 2
        for c in cue_boxes:
            near_v = abs(((c.y0 + c.y1) / 2) - cy) <= 4 * line_h
            near_h = c.x0 - 2 * line_h <= cx <= c.x1 + 0.6 * w
            if near_v and near_h:
                conf += 0.25
                break
        if cy > 0.4 * h:
            conf += 0.05
        conf = round(min(conf, 0.95), 2)
        if conf >= MIN_CONFIDENCE:
            out.append(Region("signature", float(x), float(y), float(x + bw), float(y + bh), conf))
    return out
