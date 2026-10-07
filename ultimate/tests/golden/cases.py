"""Golden search dataset: a small deterministic corpus, the queries, and what a correct answer looks like.

This is a measurement baseline for the search forensics milestone, not a statement that the
current search meets it. `tests/integration/test_golden_search.py` runs every case in every
search mode, grades it, and compares the pass/fail map with a recorded baseline per embedder.

Tenants:
  acme      main corpus (English, Arabic, OCR scan, OCR-noisy text, boundary/order probes)
  globex    a second tenant whose documents deliberately repeat acme's key terms
  haystack  60 near-duplicate distractors plus one exact-phrase "needle" (tests the vector cap)
  carol     no documents
  xdemo     synthetic Xerox-style documents (invoice ids, contract numbers, articles, Arabic, OCR errors)
  xother    one document that repeats xdemo's identifiers (count/aggregation isolation later)
"""
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tests.support import documents

PDF = "application/pdf"
TXT = "text/plain"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass
class Doc:
    tenant: str
    file_id: str
    filename: str
    file_type: str
    build: object  # callable(path) -> path
    bucket_id: Optional[str] = None
    path: Optional[str] = None


def _txt(content: str):
    def build(p: Path) -> Path:
        p.write_text(content, encoding="utf-8")
        return p
    return build


def _pdf(pages: Sequence[Sequence[str]]):
    return lambda p: documents.text_pdf(p, pages)


def _scan_pdf(lines: List[str]):
    def build(p: Path) -> Path:
        import fitz

        img = documents.render_page(lines, size_pt=14, height_in=4.0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buf.getvalue())
        doc.save(str(p))
        return p
    return build


HERON = ("Field note {i}: the great blue heron colony followed the nesting survey protocol again this week. "
         "Observers logged blue heron sightings, wading behaviour and protocol deviations along the marsh, "
         "and the heron protocol checklist was reviewed by volunteer team {i}.")

CORPUS: List[Doc] = [
    Doc("acme", "press_release", "Press_Release_StorageChain.pdf", PDF, _pdf(documents.PRESS_RELEASE_PAGES),
        bucket_id="b-ops", path="comms"),
    Doc("acme", "altalink_agreement", "Xerox_AltaLink_Service_Agreement.pdf", PDF, _pdf([[
        "MASTER SERVICE AGREEMENT",
        "This agreement is made between Northwind Logistics and the service provider",
        "for twelve AltaLink C8170 multifunction printers.",
        "Term: 36 months starting March 1, 2025. Governing law: State of New York.",
        "Signed by Marcus Webb, Procurement Director, Northwind Logistics.",
    ]]), bucket_id="b-legal", path="contracts"),
    Doc("acme", "toner_procedure", "Toner_Replacement_Procedure.txt", TXT, _txt(
        "Procedure: open the front door of the printer, pull out the empty cartridge, shake the new "
        "cartridge five times, slide it in until it clicks and close the door. Recycle the old cartridge.\n"),
        bucket_id="b-ops", path="howto"),
    Doc("acme", "remote_work", "Remote_Work_Policy.pdf", PDF, _pdf([[
        "Employees may work from home up to three days per week with manager approval.",
        "A home office equipment stipend of USD 300 is paid once per year.",
        "Staff working remotely must use the company VPN.",
    ]]), bucket_id="b-hr", path="policies"),
    Doc("acme", "invoice", "invoice.docx", DOCX, documents.invoice_docx, bucket_id="b-finance", path="ap"),
    Doc("acme", "arabic_invoice", "arabic_invoice.txt", TXT, _txt("فاتورة رقم 2024 مقدمة إلى شركة الخليج للتجارة\n"),
        bucket_id="b-finance", path="ap"),
    Doc("acme", "scanned_ticket", "Scanned_Service_Ticket.pdf", PDF, _scan_pdf([
        "SERVICE TICKET 48213",
        "Fuser unit replaced on VersaLink C405",
        "Technician: Priya Raman",
    ]), bucket_id="b-ops", path="tickets"),
    Doc("acme", "ocr_noisy_contract", "maintenance_scan_0007.txt", TXT, _txt(
        "MAINTENANCE C0NTRACT. Preventive rnaintenance visits every quarter for the entire Xer0x fleet. "
        "Response tirne for urgent repairs is four hours.\n"), bucket_id="b-legal", path="contracts"),
    Doc("acme", "ink_usage", "Ink_Usage.txt", TXT, _txt(
        "Ink usage report: ink levels on the inkjet units dropped faster than planned this quarter.\n"),
        bucket_id="b-ops", path="reports"),
    Doc("acme", "network_status", "Network_Status.txt", TXT, _txt(
        "Network status: the uplink was restored, the AltaLink network link is stable and the blinking "
        "lights stopped.\n"), bucket_id="b-ops", path="reports"),
    Doc("acme", "release_schedule", "Firmware_Schedule.txt", TXT, _txt(
        "Immediate action is required for the release of the new firmware on all devices.\n"),
        bucket_id="b-ops", path="reports"),
    Doc("globex", "globex_release", "Globex_Release.txt", TXT, _txt(
        "FOR IMMEDIATE RELEASE Globex confidential merger. Contact Lisa Riordan. StorageChain partner. "
        "AltaLink C8170 ticket 48213. فاتورة\n")),
    Doc("haystack", "needle", "Fleet_Audit_Appendix.txt", TXT, _txt(
        "Fleet audit appendix. Printer counts by site, toner spend per department and uptime targets. "
        "Reference code: blue heron protocol 7781. Next audit in the second quarter.\n")),
    # Synthetic Xerox-style corpus (placeholder until real Xerox documents and queries are provided).
    Doc("xdemo", "x_invoice", "Tax_Invoice_INV-2026-00481.pdf", PDF, _pdf([[
        "TAX INVOICE",
        "Invoice #INV-2026-00481    Date: 14 January 2026",
        "Bill to: Saudi Aramco, Dhahran, Kingdom of Saudi Arabia",
        "Managed print services, quarter 1. Amount due: SAR 418,750.00 including VAT 15%.",
    ]])),
    Doc("xdemo", "x_contract", "Service_Contract_17-2024.pdf", PDF, _pdf([[
        "SERVICE CONTRACT    Contract No. 17/2024",
        "Between Riyadh Logistics Co. and Gulf Office Systems.",
        "Article 12.4 Termination for convenience: either party may terminate this contract",
        "with sixty (60) days written notice. Governing law: Kingdom of Saudi Arabia.",
        "This contract expires on 31 December 2027.",
    ]])),
    Doc("xdemo", "x_nda_ca", "Mutual_NDA_Acme.pdf", PDF, _pdf([[
        "MUTUAL NON-DISCLOSURE AGREEMENT",
        "This agreement is governed by the laws of the State of California.",
        "Signed: John Smith, Chief Executive Officer, Acme Corp. Date: 3 March 2025.",
    ]])),
    Doc("xdemo", "x_nda_tx", "Mutual_NDA_Lone_Star.pdf", PDF, _pdf([[
        "MUTUAL NON-DISCLOSURE AGREEMENT",
        "This agreement is governed by the laws of the State of Texas.",
        "Signed: Maria Lopez, General Counsel, Lone Star Imaging. Date: 9 July 2024.",
    ]])),
    Doc("xdemo", "x_employment_ca", "Employment_Agreement_Field_Engineer.pdf", PDF, _pdf([[
        "EMPLOYMENT AGREEMENT",
        "Employment is at will under the California Labor Code.",
        "Either the employee or the company may end the employment relationship at any time.",
    ]])),
    Doc("xdemo", "x_ocr_califomia", "scan_agreement_0412.txt", TXT, _txt(
        "SUPPLY AGREEMENT. This agreement shall be governed by the laws of the State of Califomia. "
        "Paym ent terms: net 45 days.\n")),
    Doc("xdemo", "x_payroll", "Payroll_Summary_2024.txt", TXT, _txt(
        "PAYROLL SUMMARY 2024. Employees paid: 42. Total gross pay: SAR 3,150,000. Paid monthly by bank transfer.\n")),
    Doc("xdemo", "x_po", "Purchase_Order_PO-2025-0193.txt", TXT, _txt(
        "PURCHASE ORDER PO-2025-0193. Quantity 20 VersaLink C7130 colour printers for the Jeddah office.\n")),
    Doc("xdemo", "x_lease", "Equipment_Lease_Northwind.txt", TXT, _txt(
        "EQUIPMENT LEASE. Monthly payment of USD 12,000 for 36 months, total contract value USD 432,000.\n")),
    Doc("xdemo", "x_arabic_contract", "عقد_صيانة.txt", TXT, _txt(
        "عقد صيانة رقم 17/2024. المادة 12.4: يجوز لأي من الطرفين إنهاء العقد بإشعار كتابي مدته ستون يوما.\n")),
    Doc("xdemo", "x_scanned_letter", "Scanned_Approval_Letter.pdf", PDF, _scan_pdf([
        "APPROVAL LETTER",
        "The renewal is approved.",
        "Approved by John Smith",
    ])),
    Doc("xdemo", "x_policy", "Records_Retention_Policy.txt", TXT, _txt(
        "Records retention policy: invoices, contracts, agreements, purchase orders and payroll records are "
        "kept for ten years. Termination of the policy requires board approval.\n")),
    Doc("xother", "xo_nda_ca", "Mutual_NDA_Other.pdf", PDF, _pdf([[
        "MUTUAL NON-DISCLOSURE AGREEMENT",
        "This agreement is governed by the laws of the State of California. Invoice #INV-2026-00481.",
        "Signed: John Smith.",
    ]])),
] + [
    Doc("haystack", f"heron_{i:02d}", f"heron_note_{i:02d}.txt", TXT, _txt(HERON.format(i=i) + "\n"))
    for i in range(60)
]


@dataclass
class Case:
    id: str
    category: str
    query: str
    tenant: str = "acme"
    top1: Optional[List[str]] = None          # acceptable first results
    in_top: Optional[List[object]] = None     # [file_id, k]
    before: Optional[List[str]] = None        # [a, b]: a must rank above b (b may be absent)
    absent: List[str] = field(default_factory=list)
    empty: bool = False
    extra: Dict[str, object] = field(default_factory=dict)  # extra request fields
    note: str = ""


CASES: List[Case] = [
    # exact phrase
    Case("phrase_fir", "exact_phrase", "FOR IMMEDIATE RELEASE", top1=["press_release"],
         before=["press_release", "release_schedule"]),
    Case("phrase_fir_lower", "exact_phrase", "for immediate release", top1=["press_release"]),
    Case("phrase_press_release", "exact_phrase", "press release", top1=["press_release"]),
    Case("phrase_needle", "exact_phrase", "blue heron protocol 7781", tenant="haystack", top1=["needle"],
         note="exact phrase in 1 doc; 60 near-duplicates are closer in vector space"),
    Case("phrase_governing_law", "exact_phrase", "Governing law: State of New York", top1=["altalink_agreement"]),
    # exact keyword / identifiers
    Case("kw_storagechain", "exact_keyword", "StorageChain", top1=["press_release"]),
    Case("kw_ticket_number", "exact_keyword", "48213", top1=["scanned_ticket"]),
    Case("kw_model_c8170", "exact_keyword", "C8170", top1=["altalink_agreement"]),
    Case("kw_amount", "exact_keyword", "USD 12,500", top1=["invoice"]),
    Case("kw_needle_code", "exact_keyword", "7781", tenant="haystack", top1=["needle"]),
    # joined / split forms
    Case("join_storage_chain", "joined_form", "Storage Chain", top1=["press_release"]),
    Case("join_alta_link", "joined_form", "Alta Link", top1=["altalink_agreement", "network_status"]),
    Case("join_versa_link", "joined_form", "Versa Link C405", top1=["scanned_ticket"]),
    # word boundary
    Case("boundary_ink", "word_boundary", "ink", top1=["ink_usage"], before=["ink_usage", "network_status"],
         note="'ink' must not be satisfied by 'link'/'blinking'/'inkjet' alone"),
    # person / entity
    Case("person_lisa", "person_entity", "Lisa Riordan", top1=["press_release"]),
    Case("person_marcus", "person_entity", "Marcus Webb", top1=["altalink_agreement"]),
    Case("person_priya_ocr", "person_entity", "Priya Raman", top1=["scanned_ticket"],
         note="name only exists in OCR text of a scanned PDF"),
    Case("org_northwind", "person_entity", "Northwind Logistics", top1=["altalink_agreement"]),
    Case("org_gulf", "person_entity", "Gulf Trading", top1=["invoice"]),
    # semantic (paraphrase, no shared keywords where possible)
    Case("sem_toner", "semantic", "how do I change the toner", top1=["toner_procedure"]),
    Case("sem_wfh", "semantic", "rules for working from home", top1=["remote_work"]),
    Case("sem_repair_visit", "semantic", "record of a printer repair visit", top1=["scanned_ticket"]),
    Case("sem_media_contact", "semantic", "who handles questions from journalists", top1=["press_release"]),
    Case("sem_bill", "semantic", "bill for managed printing", top1=["invoice"]),
    Case("sem_contract_length", "semantic", "how long does the printer lease last", top1=["altalink_agreement"]),
    # OCR-corrupted queries and OCR-corrupted documents
    Case("ocrq_riordon", "ocr_fuzzy", "Lisa Riordon", top1=["press_release"]),
    Case("ocrq_st0rage", "ocr_fuzzy", "St0rageChain", top1=["press_release"]),
    Case("ocrq_immediate", "ocr_fuzzy", "FOR IMMEDlATE RELEASE", top1=["press_release"]),
    Case("ocrq_fuser", "ocr_fuzzy", "fuser unit replacd", top1=["scanned_ticket"]),
    Case("ocrd_maintenance", "ocr_fuzzy", "maintenance contract", top1=["ocr_noisy_contract"],
         note="document text is OCR-corrupted (C0NTRACT, rnaintenance)"),
    # filename
    Case("fn_altalink", "filename", "Xerox AltaLink Service Agreement", top1=["altalink_agreement"]),
    Case("fn_remote_work", "filename", "Remote_Work_Policy", top1=["remote_work"]),
    Case("fn_with_ext", "filename", "Toner_Replacement_Procedure.txt", top1=["toner_procedure"]),
    Case("fn_scan_name", "filename", "maintenance_scan_0007", top1=["ocr_noisy_contract"]),
    # Arabic
    Case("ar_invoice", "arabic", "فاتورة", top1=["arabic_invoice"]),
    Case("ar_company", "arabic", "شركة الخليج للتجارة", top1=["arabic_invoice"]),
    # no-result behaviour
    Case("unknown_term", "no_result", "zzqxunknownzzq", empty=True),
    # Synthetic Xerox-style retrieval cases
    Case("x_invoice_id", "exact_keyword", "INV-2026-00481", tenant="xdemo", top1=["x_invoice"]),
    Case("x_invoice_id_spaced", "exact_keyword", "INV 2026 00481", tenant="xdemo", top1=["x_invoice"]),
    Case("x_contract_no", "exact_keyword", "Contract No. 17/2024", tenant="xdemo", top1=["x_contract"]),
    Case("x_article", "exact_phrase", "Article 12.4", tenant="xdemo", top1=["x_contract"]),
    Case("x_aramco", "person_entity", "Saudi Aramco", tenant="xdemo", top1=["x_invoice"]),
    Case("x_po", "exact_keyword", "PO-2025-0193", tenant="xdemo", top1=["x_po"]),
    Case("x_expiry", "exact_phrase", "expires on 31 December 2027", tenant="xdemo", top1=["x_contract"]),
    Case("x_sem_early_exit", "semantic", "agreements that let either side end the contract early", tenant="xdemo",
         top1=["x_contract", "x_employment_ca"]),
    Case("x_ca_law", "semantic", "contracts governed by California law", tenant="xdemo",
         top1=["x_nda_ca", "x_employment_ca", "x_ocr_califomia"], before=["x_nda_ca", "x_nda_tx"]),
    Case("x_ocr_token", "ocr_fuzzy", "Califomia", tenant="xdemo", top1=["x_ocr_califomia"]),
    Case("x_john_smith", "person_entity", "John Smith", tenant="xdemo", top1=["x_nda_ca", "x_scanned_letter"]),
    Case("x_ar_article", "arabic", "المادة 12.4", tenant="xdemo", top1=["x_arabic_contract"]),
    Case("x_ar_termination", "arabic", "إنهاء العقد", tenant="xdemo", top1=["x_arabic_contract"]),
    Case("x_payroll", "semantic", "how many people were paid in 2024", tenant="xdemo", top1=["x_payroll"]),
    Case("x_lease_amount", "exact_phrase", "monthly payment of USD 12,000", tenant="xdemo", top1=["x_lease"]),
    Case("x_other_tenant", "tenant", "INV-2026-00481", tenant="xother", top1=["xo_nda_ca"], absent=["x_invoice"]),
    # tenant isolation and scoping (every case is also checked for foreign-tenant ids)
    Case("tenant_globex", "tenant", "FOR IMMEDIATE RELEASE", tenant="globex", top1=["globex_release"],
         absent=["press_release"]),
    Case("tenant_carol", "tenant", "FOR IMMEDIATE RELEASE", tenant="carol", empty=True),
    Case("scope_bucket", "tenant", "service agreement", extra={"bucketId": "b-ops"},
         absent=["altalink_agreement", "ocr_noisy_contract", "remote_work", "invoice", "arabic_invoice"]),
    Case("scope_path", "tenant", "printer", extra={"path": "howto"}, top1=["toner_procedure"],
         absent=["altalink_agreement", "press_release", "scanned_ticket", "ink_usage"]),
    Case("inject_bucket_list", "tenant", "FOR IMMEDIATE RELEASE",
         extra={"bucketId": ['x" or user_id != "acme']}, absent=["globex_release"]),
    Case("inject_bucket_number", "tenant", "FOR IMMEDIATE RELEASE",
         extra={"bucketId": 1}, absent=["globex_release"]),
    Case("inject_bucket_dict", "tenant", "FOR IMMEDIATE RELEASE",
         extra={"bucketId": {"or user_id != ''": 1}}, absent=["globex_release"]),
    Case("inject_connection_str", "tenant", "FOR IMMEDIATE RELEASE",
         extra={"connectionId": 'c" or user_id != "acme'}, absent=["globex_release"]),
]

MODES = ["both", "vector", "semantic"]

TENANT_OF = {d.file_id: d.tenant for d in CORPUS}


def grade(case: Case, ids: List[str]) -> List[str]:
    """Return the list of failed expectations (empty list = pass)."""
    fails = []
    foreign = [i for i in ids if TENANT_OF.get(i, case.tenant) != case.tenant]
    if foreign:
        fails.append(f"FOREIGN-TENANT documents returned: {foreign}")
    if case.empty and ids:
        fails.append(f"expected no results, got {len(ids)}")
    if case.top1 and (not ids or ids[0] not in case.top1):
        fails.append(f"top1 {ids[0] if ids else None!r} not in {case.top1}")
    if case.in_top:
        fid, k = case.in_top
        if fid not in ids[:k]:
            fails.append(f"{fid} not in top {k}")
    if case.before:
        a, b = case.before
        if a not in ids:
            fails.append(f"{a} missing")
        elif b in ids and ids.index(b) < ids.index(a):
            fails.append(f"{b} ranked above {a}")
    for fid in case.absent:
        if fid in ids:
            fails.append(f"{fid} must be absent")
    return fails
