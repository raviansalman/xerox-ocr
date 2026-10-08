import numpy as np
import pytest

from docintel.models import Page, ParsedDocument
from docintel.understanding import analyze
from docintel.understanding.classify import classify


@pytest.mark.parametrize("filename,text,label", [
    ("Mutual_NDA_Acme.pdf", "MUTUAL NON-DISCLOSURE AGREEMENT\nThe receiving party shall protect confidential information.", "nda"),
    ("a.pdf", "TAX INVOICE\nInvoice #INV-2026-00481\nBill to: X", "invoice"),
    ("Service_Contract.pdf", "SERVICE CONTRACT Contract No. 17/2024", "service_agreement"),
    ("p.pdf", "FOR IMMEDIATE RELEASE\nCompany announces", "press_release"),
    ("Employment_Agreement.docx", "EMPLOYMENT AGREEMENT\nEmployment is at will.", "employment_agreement"),
    ("Purchase_Order.csv", "PURCHASE ORDER PO-2025-0193", "purchase_order"),
    ("lease.pdf", "EQUIPMENT LEASE\nMonthly payment of USD 12,000", "lease_agreement"),
    ("Payroll_Summary_2024.xlsx", "PAYROLL SUMMARY 2024", "payroll"),
    ("memo.txt", "Internal memo: maintenance window", "memo"),
])
def test_rule_classification(filename, text, label):
    c = classify(filename, text)
    assert c.label == label and c.method == "rules" and c.confidence >= 0.8


def test_specific_contract_type_beats_generic_contract():
    c = classify("x.pdf", "SUPPLY AGREEMENT\nThis agreement covers the supply of goods.")
    assert c.label == "supply_agreement"


def test_unknown_document_is_other_without_embeddings():
    assert classify("notes.txt", "random words about birds").label == "other"


def test_prototype_fallback_uses_embeddings():
    from docintel.packs import default_domain
    labels = [label for label, t in default_domain().types.items() if t.description]
    target = labels.index("invoice")

    def embed(texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0 if ("invoice" in t.lower() or "payment for goods" in t.lower()) else 1] = 1.0
        return out
    c = classify("scan.txt", "request for payment for goods delivered", embed=embed, embed_key="unit-test")
    assert c.label == labels[target] and c.method == "embedding_prototype"


def test_analyze_produces_fields_entities_clauses_and_signature():
    doc = ParsedDocument("native", [Page(1, (
        "SERVICE CONTRACT    Contract No. 17/2024\n"
        "This service contract is made between Riyadh Logistics Co. and Gulf Office Systems.\n"
        "Article 12.4 Termination for convenience: either party may terminate this contract with sixty (60) days written notice.\n"
        "Governing law: Kingdom of Saudi Arabia. This contract expires on 31 December 2027.\n"
        "Total contract value: USD 250,000.\nSigned by Omar Haddad, Chief Operating Officer"), "native")])
    u = analyze(doc, "Service_Contract_17-2024.pdf")
    fields = {(f.name, f.value_text) for f in u.fields}
    assert ("contract_number", "17/2024") in fields and ("expiry_date", "2027-12-31") in fields
    assert ("jurisdiction", "Saudi Arabia") in fields and ("total_amount", "USD 250,000.00") in fields
    assert ("signer", "Omar Haddad") in fields and u.has_signature
    assert u.classification.label == "service_agreement" and u.language == "en"
    assert any(c.clause_type == "termination" and c.ref == "12.4" for c in u.clauses)
    assert all(f.page == 1 and f.snippet for f in u.fields)
