import pytest

from docintel import text as T


def test_normalize_unifies_unicode_quotes_dashes_and_case():
    assert T.normalize("  “Contract” — No. 17 ") == '"contract" - no. 17'


def test_search_text_splits_punctuation_and_adds_camel_case_variants():
    st = T.search_text("FOR IMMEDIATE RELEASE: StorageChain #INV-2026-00481, Article 12.4")
    assert st.startswith("for immediate release storagechain inv 2026 00481 article 12 4")
    assert st.endswith("storage chain")


@pytest.mark.parametrize("ocr,clean", [("Califomia", "California"), ("C0NTRACT", "contract"), ("rnaintenance", "maintenance"),
                                       ("IMMEDlATE", "IMMEDIATE"), ("vvarranty", "warranty")])
def test_fold_makes_ocr_confusions_meet_clean_text(ocr, clean):
    assert T.fold(ocr) == T.fold(clean)


def test_fold_keeps_identifiers_intact():
    assert T.fold("INV-2026-00481 C8170") == "inv 2026 00481 c8170"


def test_identifiers_canonical_forms():
    ids = T.identifiers("Invoice #INV-2026-00481, Contract No. 17/2024, model C8170, total 418,750.00, PO-2025-0193")
    assert ids == {"inv202600481", "172024", "c8170", "po20250193"}


def test_identifiers_ignore_plain_words_and_amounts():
    assert T.identifiers("the total is 1,250.00 and 12.4 percent") == set()


def test_join_variants():
    assert T.join_variants(["storage", "chain"]) == ["storagechain"]
    assert "versalink c405" in T.join_variants(["versa", "link", "c405"])
    assert T.join_variants(["single"]) == []


def test_content_words_drop_stopwords():
    assert T.content_words("Show me all the contracts with Acme") == ["contracts", "acme"]


def test_edit_distance_counts_typos():
    from docintel.retrieval.lexical import edit_distance
    assert edit_distance("northwnd", "northwind") == 1          # dropped letter
    assert edit_distance("fabrikm", "fabrikam") == 1
    assert edit_distance("hartwlel", "hartwell") == 1          # adjacent transposition
    assert edit_distance("contract", "contrast") == 1
    assert edit_distance("invoice", "invoice") == 0 and edit_distance("kitten", "sitting") == 3
