"""Deterministic test corpus covering every supported format, scanned pages, OCR noise and several tenants.

All names, companies and identifiers are fictitious test data. Production code never imports this module.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def text_pdf(path: Path, pages: list[list[str]]) -> Path:
    import fitz
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 72
        for line in lines:
            page.insert_text((60, y), line, fontsize=11)
            y += 16
    doc.save(str(path))
    return path


def render_lines(lines: list[str], size_pt: float = 13, dpi: int = 200, width_in: float = 8.27, height_in: float = 5.0,
                 signature: bool = False):
    from PIL import Image, ImageDraw, ImageFont
    w, h = int(width_in * dpi), int(height_in * dpi)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, int(size_pt * dpi / 72))
    y = int(0.4 * dpi)
    for line in lines:
        d.text((int(0.7 * dpi), y), line, font=font, fill=0)
        y += int(size_pt * dpi / 72 * 1.7)
    if signature:                         # a hand-drawn looking stroke above the signer line
        import math
        pts = [(int(0.9 * dpi + i * 4), int(y + 40 + 18 * math.sin(i / 6.0) + 8 * math.sin(i / 2.3))) for i in range(110)]
        d.line(pts, fill=0, width=4)
    return img


def scanned_pdf(path: Path, pages: list[list[str]], signature_on_last: bool = False) -> Path:
    import fitz
    doc = fitz.open()
    for i, lines in enumerate(pages):
        img = render_lines(lines, signature=signature_on_last and i == len(pages) - 1)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page = doc.new_page(width=595, height=842)
        page.insert_image(fitz.Rect(0, 0, 595, 360), stream=buf.getvalue())
    doc.save(str(path))
    return path


def png(path: Path, lines: list[str], signature: bool = False) -> Path:
    render_lines(lines, signature=signature).save(path)
    return path


def docx_file(path: Path, title: str, paragraphs: list[str], table: list[list[str]] | None = None) -> Path:
    import docx
    d = docx.Document()
    d.add_heading(title, level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    d.save(str(path))
    return path


def xlsx_file(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(str(path))
    return path


def pptx_file(path: Path, slides: list[tuple[str, str]]) -> Path:
    from pptx import Presentation
    prs = Presentation()
    for title, body in slides:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        s.placeholders[1].text = body
    prs.save(str(path))
    return path


def eml_file(path: Path, subject: str, body: str, attachment: tuple[str, bytes] | None = None) -> Path:
    m = EmailMessage()
    m["From"] = "Procurement Desk <procurement@example.com>"
    m["To"] = "Finance Team <finance@example.com>"
    m["Date"] = "Mon, 02 Feb 2026 09:30:00 +0000"
    m["Subject"] = subject
    m.set_content(body)
    if attachment:
        m.add_attachment(attachment[1], maintype="text", subtype="plain", filename=attachment[0])
    path.write_bytes(bytes(m))
    return path


def txt(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


@dataclass
class Doc:
    tenant: str
    key: str
    filename: str
    build: Callable[[Path], Path]


HERON = ("Field note {i}: the great blue heron colony followed the nesting survey protocol again this week. "
         "Observers logged blue heron sightings, wading behaviour and protocol deviations along the marsh, "
         "and the heron protocol checklist was reviewed by volunteer team {i}.")

PRESS = [["FOR IMMEDIATE RELEASE", "Northbridge Data Announces Archive Service for Regulated Industries",
          "Austin, Texas - June 3, 2024 - Northbridge Data today announced a new archive",
          "service for enterprise records. The press release outlines pricing and availability.",
          "Media contact: Nadia Hartwell, Director of Communications"],
         ["About Northbridge Data", "Northbridge Data builds secure storage for regulated industries.",
          "For more information contact Nadia Hartwell at the press office."]]


def corpus() -> list[Doc]:
    a = "acme"
    docs = [
        Doc(a, "press_release", "Press_Release_Northbridge.pdf", lambda p: text_pdf(p, PRESS)),
        Doc(a, "firmware", "Firmware_Schedule.txt", lambda p: txt(p, "Immediate action is required for the release of the new firmware on all devices.\n")),
        Doc(a, "invoice", "Tax_Invoice_INV-2026-00481.pdf", lambda p: text_pdf(p, [[
            "TAX INVOICE", "Invoice #INV-2026-00481    Invoice Date: 14 January 2026",
            "Bill to: Saudi Aramco, Dhahran, Kingdom of Saudi Arabia",
            "Managed print services, quarter 1. Amount due: SAR 418,750.00 including VAT 15%.",
            "Payment due 13 February 2026."]])),
        Doc(a, "invoice_docx", "Invoice_Gulf_Trading.docx", lambda p: docx_file(p, "INVOICE 2025-0042", [
            "Invoice Date: 3 November 2025", "Bill to: Gulf Trading LLC, Dubai", "Payment due within 30 days."],
            [["Item", "Amount"], ["Managed print services", "USD 12,500"], ["Total", "USD 12,500"]])),
        Doc(a, "contract", "Service_Contract_17-2024.pdf", lambda p: text_pdf(p, [[
            "SERVICE CONTRACT    Contract No. 17/2024",
            "This service contract is made between Riyadh Logistics Co. and Gulf Office Systems.",
            "Article 12.4 Termination for convenience: either party may terminate this contract",
            "with sixty (60) days written notice.",
            "Governing law: Kingdom of Saudi Arabia.", "This contract expires on 31 December 2027.",
            "Total contract value: USD 250,000.", "Signed by Omar Haddad, Chief Operating Officer"]])),
        Doc(a, "nda_ca", "Mutual_NDA_Acme.pdf", lambda p: text_pdf(p, [[
            "MUTUAL NON-DISCLOSURE AGREEMENT",
            "This agreement is made between Acme Corp and Bluefin Analytics Inc.",
            "The receiving party shall protect confidential information of the disclosing party.",
            "This agreement is governed by the laws of the State of California.",
            "Signed: John Smith, Chief Executive Officer, Acme Corp. Date: 3 March 2025."]])),
        Doc(a, "nda_tx", "Mutual_NDA_Lone_Star.pdf", lambda p: text_pdf(p, [[
            "MUTUAL NON-DISCLOSURE AGREEMENT",
            "The receiving party shall protect confidential information of the disclosing party.",
            "This agreement is governed by the laws of the State of Texas.",
            "Signed: Maria Lopez, General Counsel, Lone Star Imaging. Date: 9 July 2024."]])),
        Doc(a, "employment", "Employment_Agreement_Field_Engineer.docx", lambda p: docx_file(p, "EMPLOYMENT AGREEMENT", [
            "This employment agreement is made between Acme Corp (the employer) and the employee.",
            "Employment is at will under the California Labor Code.",
            "Either the employee or the company may end the employment relationship at any time.",
            "Annual salary: USD 96,000."])),
        Doc(a, "ocr_noisy", "scan_agreement_0412.txt", lambda p: txt(p,
            "SUPPLY AGREEMENT. This agreement shall be governed by the laws of the State of Califomia. "
            "Preventive rnaintenance visits every quarter for the entire printer fleet. Paym ent terms: net 45 days.\n")),
        Doc(a, "payroll", "Payroll_Summary_2024.xlsx", lambda p: xlsx_file(p, {"Summary": [
            ["PAYROLL SUMMARY 2024"], ["Employees paid", 42], ["Total gross pay (SAR)", 3150000], ["Pay frequency", "Monthly"]]})),
        Doc(a, "po", "Purchase_Order_PO-2025-0193.csv", lambda p: txt(p,
            "Purchase Order,PO-2025-0193\nVendor,Gulf Office Systems\nItem,Quantity,Unit price\n"
            "VersaLink C7130 colour printer,20,USD 1450\nShip to,Jeddah office\n")),
        Doc(a, "lease", "Equipment_Lease_Northwind.pdf", lambda p: text_pdf(p, [[
            "EQUIPMENT LEASE", "This lease is made between Northwind Leasing Ltd (the lessor) and Acme Corp (the lessee).",
            "Monthly payment of USD 12,000 for 36 months, total contract value USD 432,000.",
            "Effective date: 1 April 2025. The lease expires on 31 March 2028."]])),
        Doc(a, "toner", "Toner_Replacement_Procedure.html", lambda p: txt(p,
            "<html><head><title>Toner replacement procedure</title><script>var x=1;</script></head><body>"
            "<h1>Toner replacement procedure</h1><p>Open the front door of the printer, pull out the empty cartridge, "
            "shake the new cartridge five times and slide it in until it clicks.</p><p>Recycle the old cartridge.</p></body></html>")),
        Doc(a, "remote", "Remote_Work_Policy.rtf", lambda p: txt(p,
            r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard REMOTE WORK POLICY\par "
            r"Employees may work from home up to three days per week with manager approval.\par "
            r"A home office equipment stipend of USD 300 is paid once per year.\par}")),
        Doc(a, "ticket", "Scanned_Service_Ticket.pdf", lambda p: scanned_pdf(p, [[
            "SERVICE TICKET 48213", "Fuser unit replaced on VersaLink C405", "Technician: Priya Raman",
            "Customer: Acme Corp, Riyadh"]])),
        Doc(a, "approval", "Signed_Approval_Letter.png", lambda p: png(p, [
            "APPROVAL LETTER", "Dear Procurement Team,", "The renewal of the printer fleet contract is approved.",
            "Approved by John Smith"], signature=True)),
        Doc(a, "deck", "Fleet_Review_Q3.pptx", lambda p: pptx_file(p, [
            ("Fleet review Q3 2025", "Uptime 99.2 percent across 140 devices"),
            ("Recommendations", "Replace 12 legacy printers before the end of the year")])),
        Doc(a, "email", "Order_Confirmation.eml", lambda p: eml_file(p, "Order confirmation PO-2025-0193",
            "Hello,\n\nPlease find the delivery schedule for purchase order PO-2025-0193 attached.\n\nRegards,\nProcurement",
            ("delivery_schedule.txt", b"Delivery schedule: 20 printers to the Jeddah office on 15 March 2026.\n"))),
        Doc(a, "memo", "memo.txt", lambda p: txt(p, "Internal memo: printer maintenance window is Saturday.\n")),
        Doc(a, "ink", "Ink_Usage_Report.txt", lambda p: txt(p, "Ink usage report: ink levels on the inkjet units dropped faster than planned this quarter.\n")),
        Doc(a, "network", "Network_Status.txt", lambda p: txt(p, "Network status: the uplink was restored, the AltaLink network link is stable and the blinking lights stopped.\n")),
        # second tenant repeats acme's identifiers and names: must never leak
        Doc("globex", "globex_release", "Globex_Release.txt", lambda p: txt(p,
            "FOR IMMEDIATE RELEASE Globex confidential merger. Contact Nadia Hartwell. Invoice #INV-2026-00481. "
            "MUTUAL NON-DISCLOSURE AGREEMENT governed by the laws of the State of California. Signed: John Smith.\n")),
        Doc("haystack", "needle", "Fleet_Audit_Appendix.txt", lambda p: txt(p,
            "Fleet audit appendix. Printer counts by site, toner spend per department and uptime targets. "
            "Reference code: blue heron protocol 7781. Next audit in the second quarter.\n")),
    ] + [Doc("haystack", f"heron_{i:02d}", f"heron_note_{i:02d}.txt", (lambda i: lambda p: txt(p, HERON.format(i=i) + "\n"))(i))
         for i in range(60)]
    return docs


def build_all(folder: Path) -> list[tuple[Doc, Path]]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for d in corpus():
        sub = folder / d.tenant
        sub.mkdir(exist_ok=True)
        out.append((d, d.build(sub / d.filename)))
    return out
