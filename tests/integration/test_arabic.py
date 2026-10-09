"""Arabic documents end to end: OCR of an Arabic scan, a native Arabic Word file, and search that tolerates
diacritics, letter variants, inflection and Arabic-Indic digits. Runs in its own tenant (tenantarabic), so the shared
corpus and its counts are unchanged. The scan needs the Tesseract Arabic model; the meaning tests need a multilingual
embedding model."""
import time

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration

KEY = "arabic_uploader"
REPORT = [
    "تقرير صيانة الطابعات",
    "Report No: MR-2024/118",
    "تم استبدال وحدة الصهر في الطابعة الرئيسية",
    "الفني المسؤول: أحمد الزهراني",
    "التاريخ: 15 مارس 2024",
]
CONTRACT = [
    "عقد إيجار الطابعات",
    "مدة العقد ثلاث سنوات تبدأ من تاريخ التوقيع.",
    "يحق لأي من الطرفين إنهاء الْعَقْدِ بإشعار كتابي مدته تسعون يوماً.",
    "يدفع المستأجر رسوماً شهرية قدرها 1,250 ريالاً.",
]


def _multilingual() -> bool:
    from docintel.config import get_settings
    from docintel.model_registry import model_spec
    s = get_settings()
    return s.semantic_enabled and "ar" in model_spec(s.embedding_model).languages


def _has_arabic_ocr() -> bool:
    from docintel.config import get_settings
    return "ara" in get_settings().ocr_languages.split("+")


@pytest.fixture(scope="module")
def arabic_docs(client, tmp_path_factory):
    from docx import Document

    from tests.fixtures.corpus import scanned_arabic_pdf, text_pdf
    d = tmp_path_factory.mktemp("arabic")
    files = []
    if _has_arabic_ocr():
        files.append(scanned_arabic_pdf(d / "تقرير_صيانة_مسح.pdf", [REPORT]))
    doc = Document()
    for line in CONTRACT:
        doc.add_paragraph(line)
    doc.save(str(d / "عقد_ايجار.docx"))
    files.append(d / "عقد_ايجار.docx")
    files.append(text_pdf(d / "Printer_Repair_Visit.pdf", [["Field service visit: the fuser unit of the main printer was replaced.",
                                                            "Technician: Daniel Brooks. Reference FS-7731."]]))
    r = client.post("/api/v1/documents", headers=headers(KEY), files=[("files", (p.name, p.read_bytes())) for p in files])
    assert r.status_code == 201, r.text
    ids = {}
    for p, out in zip(files, r.json()["documents"]):
        assert out.get("status") != "rejected", out
        ids[p.name] = out["id"]
    for name, doc_id in ids.items():
        for _ in range(600):
            st = client.get(f"/api/v1/documents/{doc_id}", headers=headers(KEY)).json()
            if st["status"] in ("indexed", "failed"):
                break
            time.sleep(0.1)
        assert st["status"] == "indexed", (name, st)
    return ids


def ask(client, q):
    r = client.post("/api/v1/query", headers=headers(KEY), json={"q": q, "limit": 10})
    assert r.status_code == 200, r.text
    b = r.json()
    return [x["filename"] for x in b["results"]], b


def test_arabic_documents_are_detected_as_arabic(client, arabic_docs):
    d = client.get(f"/api/v1/documents/{arabic_docs['عقد_ايجار.docx']}", headers=headers(KEY)).json()
    assert d["language"] == "ar"


@pytest.mark.parametrize("q", [
    "إنهاء العقد",            # as written
    "انهاء العقد",            # without hamza
    "إِنْهَاء الْعَقْد",        # with diacritics
    "العقد إنهاء",            # word order
])
def test_word_file_found_with_any_spelling(client, arabic_docs, q):
    files, _ = ask(client, q)
    assert files and files[0] == "عقد_ايجار.docx", (q, files)


def test_inflected_forms_meet_through_light_stems(client, arabic_docs):
    files, b = ask(client, "طابعة مستأجر")                # singular, no article: the text has الطابعات and المستأجر
    assert "عقد_ايجار.docx" in files[:2], files
    assert "all_terms" in {m for r in b["results"] for m in r["match_types"]} or files[0] == "عقد_ايجار.docx"


def test_single_arabic_word_finds_its_inflected_forms(client, arabic_docs):
    files, _ = ask(client, "طابعة")                        # the text only has الطابعات
    assert "عقد_ايجار.docx" in files[:2], files


@pytest.mark.skipif(not _has_arabic_ocr(), reason="Arabic OCR not configured")
class TestArabicScan:
    def test_scan_is_ocrd_in_reading_order(self, client, arabic_docs):
        doc_id = arabic_docs["تقرير_صيانة_مسح.pdf"]
        page = client.get(f"/api/v1/documents/{doc_id}/pages/1", headers=headers(KEY)).json()
        from docintel import text as T
        assert T.normalize("تم استبدال وحدة الصهر في الطابعة الرئيسية") in T.normalize(page["text"]), page["text"]
        assert page["ocr_confidence"] and page["ocr_confidence"] > 0.6

    @pytest.mark.parametrize("q", ["وحدة الصهر", "احمد الزهراني", "MR-2024/118", "MR-٢٠٢٤/١١٨", "\"الطابعة الرئيسية\""])
    def test_scan_found_by_phrase_name_and_identifier(self, client, arabic_docs, q):
        files, b = ask(client, q)
        assert files and files[0] == "تقرير_صيانة_مسح.pdf", (q, files)
        ev = b["results"][0]["evidence"][0]
        assert ev["page"] == 1 and ev["char_end"] > ev["char_start"]


@pytest.mark.semantic
@pytest.mark.skipif(not _multilingual(), reason="needs a multilingual embedding model")
class TestArabicMeaning:
    def test_arabic_paraphrase(self, client, arabic_docs):
        files, _ = ask(client, "كم تستمر مدة الإيجار")      # "how long does the rental last"
        assert "عقد_ايجار.docx" in files[:3], files

    def test_english_question_finds_arabic_document(self, client, arabic_docs):
        files, _ = ask(client, "printer lease contract termination notice")
        assert "عقد_ايجار.docx" in files[:3], files

    def test_arabic_question_finds_english_document(self, client, arabic_docs):
        files, _ = ask(client, "زيارة فني لإصلاح الطابعة")   # "technician visit to repair the printer"
        assert "Printer_Repair_Visit.pdf" in files[:3], files

    def test_unrelated_arabic_question_returns_nothing(self, client, arabic_docs):
        files, b = ask(client, "وصفة كعكة الشوكولاتة بالفراولة")  # "chocolate strawberry cake recipe"
        assert not files and b["answer"]["kind"] == "none", files


def test_other_tenants_do_not_see_arabic_documents(client, arabic_docs):
    r = client.post("/api/v1/query", headers=headers("acme_reader"), json={"q": "إنهاء العقد", "limit": 10})
    assert r.status_code == 200 and not any(x["filename"] in arabic_docs for x in r.json()["results"])

