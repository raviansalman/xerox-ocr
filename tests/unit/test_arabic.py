"""Arabic text handling: normalization, identifiers, light stemming, stop words, language and OCR orientation."""
import shutil

import pytest

from docintel import text as T
from docintel.models import Block, Word
from docintel.processing.ocr import OcrResult, _unrotate
from docintel.understanding.language import detect


@pytest.mark.parametrize("written,plain", [
    ("الْمَكْتَبَةُ", "المكتبة"),             # diacritics
    ("كتـــاب", "كتاب"),                     # tatweel
    ("إدارة", "ادارة"), ("أمن", "امن"), ("آخر", "اخر"), ("ٱلله", "الله"),   # alef variants
    ("مستشفى", "مستشفي"),                    # alef maqsura
    ("مسؤول", "مسوول"), ("رئيس", "رييس"),   # hamza seats
    ("کتاب", "كتاب"), ("فارسی", "فارسي"),    # Persian keheh and yeh
    ("\u200fعقد\u200e", "عقد"),               # bidi marks
])
def test_normalize_folds_arabic_spelling_variants(written, plain):
    assert T.normalize(written) == T.normalize(plain)


def test_normalize_maps_arabic_indic_digits_and_punctuation():
    assert T.normalize("رقم ٢٠٢٤،٥٫٥ ۱۲؟") == "رقم 2024,5.5 12?"


def test_identifiers_written_with_arabic_indic_digits_match_ascii():
    assert T.identifiers("فاتورة رقم INV-٢٠٢٤-٠٤٥٧") == T.identifiers("Invoice INV-2024-0457") == {"inv20240457"}


@pytest.mark.parametrize("forms", [
    ["المكتبة", "مكتبة", "المكتبات", "والمكتبة", "بالمكتبة"],
    ["الطابعة", "طابعة", "الطابعات", "للطابعة"],
    ["الموظفين", "موظفون", "موظف", "والموظفين"],
    ["العقد", "عقد", "للعقد", "بالعقد"],
])
def test_light_stems_unite_article_prefix_and_suffix_forms(forms):
    assert len({T.stem_text(f) for f in forms}) == 1, {f: T.stem_text(f) for f in forms}


def test_stem_text_leaves_latin_words_and_identifiers_to_the_english_configuration():
    assert T.stem_text("Invoices INV-2024-0457 for printers") == T.search_text("Invoices INV-2024-0457 for printers")
    assert T.stem_text("عقود الطابعات Xerox") == "عقود طابع xerox"     # broken plurals are left to semantic search


def test_short_arabic_words_are_not_stemmed_away():
    assert T.arabic_stem("هو") == "هو" and T.arabic_stem("ان") == "ان" and len(T.arabic_stem("بيت")) >= 2


def test_arabic_stop_words_are_dropped_but_the_name_ali_is_kept():
    assert T.content_words("ما هي شروط إنهاء العقد في هذه الاتفاقية") == ["شروط", "انهاء", "العقد", "الاتفاقيه"]
    assert "علي" in T.content_words("تقرير علي حسن")


def test_language_of_arabic_text():
    assert detect("تقرير مشروع محطة تحلية المياه في مدينة الواحة يهدف إلى إزالة الأملاح من مياه البحر") == "ar"


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_boxes_read_on_the_upright_page_map_back_to_the_scanned_image(rotate):
    W, H = 1000, 600                                   # the image as scanned
    up_w, up_h = (H, W) if rotate in (90, 270) else (W, H)
    # a word at the top left of the upright page
    res = OcrResult("x", [Word("x", 10, 20, 110, 60, 90.0)], 0.9, up_w, up_h, [Block("paragraph", "x", (10, 20, 110, 60), 0.9)])
    w = _unrotate(res, rotate, W, H).words[0]
    expected = {90: (20, H - 110, 60, H - 10), 180: (W - 110, H - 60, W - 10, H - 20), 270: (W - 60, 10, W - 20, 110)}[rotate]
    assert (w.x0, w.y0, w.x1, w.y1) == expected
    assert 0 <= w.x0 < w.x1 <= W and 0 <= w.y0 < w.y1 <= H


def _tesseract_has(lang: str) -> bool:
    if not shutil.which("tesseract"):
        return False
    import pytesseract
    return lang in pytesseract.get_languages(config="")


@pytest.mark.skipif(not _tesseract_has("ara"), reason="Tesseract Arabic model not installed")
def test_arabic_scan_is_read_in_reading_order_and_sideways_pages_are_turned_upright():
    from PIL import features

    from docintel.processing.ocr import TesseractOcr
    from tests.fixtures.corpus import render_arabic_lines
    if not features.check("raqm"):
        pytest.skip("Pillow without raqm cannot shape Arabic test text")
    lines = ["محضر اجتماع لجنة الجودة", "عقدت اللجنة اجتماعها الدوري في مقر الإدارة", "Ref: QA-2019/044"]
    img = render_arabic_lines(lines)
    ocr = TesseractOcr("ara+eng")
    for page in (img, img.rotate(90, expand=True)):
        got = T.normalize(ocr.recognize(page).text)
        assert T.normalize("محضر اجتماع لجنة الجودة") in got, got
        assert T.identifiers(got) >= {"qa2019044"}, got


CLEAN = ("تهدف محطة تحلية المياه بالتناضح العكسي في مدينة الواحة إلى إزالة الأملاح من مياه البحر وتوفير مياه الشرب "
         "النقية لنحو مئتين وخمسين ألف نسمة وقد بدأت أعمال الإنشاء في شهر مارس ويشرف على التنفيذ فريق إدارة الموارد المائية")


def test_clean_arabic_text_layer_is_trusted():
    assert not T.garbled_arabic(CLEAN) and not T.garbled_arabic("Invoice INV-2024-0457 " * 30)


def test_fragmented_arabic_text_layer_is_detected():
    # what glyph-positioned PDFs produce: words broken into letters and short pieces
    fragmented = " ".join(" ".join(w[i:i + 2] if i % 3 else w[i] for i in range(0, len(w), 2)) for w in CLEAN.split())
    assert T.garbled_arabic(fragmented)


def test_lam_alef_order_corruption_is_detected():
    # "الأمم المتحدة" decoded as "األمم املتحدة": every al- word starts with an alef pair
    swapped = CLEAN.replace("الأ", "األ").replace("الإ", "اإل").replace(" ال", " اال")
    assert T.garbled_arabic(swapped)


def test_is_arabic_word():
    assert T.is_arabic_word("طابعه") and not T.is_arabic_word("printer") and not T.is_arabic_word("inv2024")


@pytest.mark.parametrize("order,expected", [("dmy", "2024-09-03"), ("mdy", "2024-03-09")])
def test_ambiguous_numeric_dates_follow_the_configured_order(order, expected, monkeypatch):
    from docintel.config import get_settings
    from docintel.understanding.dates import find_dates
    monkeypatch.setattr(get_settings(), "date_order", order)
    assert [d.value.isoformat() for d in find_dates("Date: 03/09/2024")] == [expected]
    assert [d.value.isoformat() for d in find_dates("Date: 25/09/2024")] == ["2024-09-25"]   # unambiguous either way
