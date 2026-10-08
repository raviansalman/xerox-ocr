from datetime import date

import pytest

from docintel.understanding import entities as E
from docintel.understanding.clauses import find_clauses
from docintel.understanding.dates import find_dates
from docintel.understanding.fields import find_amounts, find_identifiers


def test_dates_with_roles():
    t = ("Invoice Date: 14 January 2026\nThis contract expires on 31 December 2027. Payment due 2026-02-13.\n"
         "Effective date: 1 April 2025. Signed on 3rd March, 2025.")
    got = {(d.value, d.role) for d in find_dates(t)}
    assert got == {(date(2026, 1, 14), "issue_date"), (date(2027, 12, 31), "expiry_date"), (date(2026, 2, 13), "due_date"),
                   (date(2025, 4, 1), "effective_date"), (date(2025, 3, 3), "signature_date")}


def test_ambiguous_numeric_date_is_low_confidence_and_day_first():
    (d,) = find_dates("Valid until 05/04/2026")
    assert d.value == date(2026, 4, 5) and d.confidence == 0.6 and d.role == "expiry_date"


def test_invalid_dates_are_skipped():
    assert find_dates("31 February 2026 and 2026-13-45") == []


def test_month_year_has_month_precision():
    (d,) = find_dates("The lease expires in June 2028")
    assert d.value == date(2028, 6, 1) and d.precision == "month"


def test_amounts_currency_and_roles():
    t = "Amount due: SAR 418,750.00. Monthly payment of USD 12,000. Price $1.5 million. Fee 250 EUR."
    got = [(a.value, a.currency, a.role) for a in find_amounts(t)]
    assert got == [(418750.0, "SAR", "total_amount"), (12000.0, "USD", "periodic_amount"), (1500000.0, "USD", "amount"),
                   (250.0, "EUR", "amount")]


def test_numbers_without_currency_are_not_amounts():
    assert find_amounts("Quantity 20, ticket 48213, year 2026") == []


def test_labelled_identifiers():
    t = "Invoice #INV-2026-00481. Contract No. 17/2024. PURCHASE ORDER PO-2025-0193. Ticket 48213."
    got = {(n, v) for n, v, _, _ in find_identifiers(t)}
    assert {("invoice_number", "INV-2026-00481"), ("contract_number", "17/2024"), ("po_number", "PO-2025-0193"),
            ("ticket_number", "48213")} <= got


def test_people_by_role_on_one_line():
    t = "Signed: John Smith, Chief Executive Officer\nApproved by Maria Lopez\nBill to: X\nTechnician: Priya Raman\n/s/ Robert Chen"
    got = {(p.value, p.role) for p in E.find_people(t)}
    assert got == {("John Smith", "signer"), ("Maria Lopez", "approver"), ("Priya Raman", "technician"), ("Robert Chen", "signer")}


def test_organizations_from_cues_parties_and_suffixes():
    t = "This agreement is made between Northwind Logistics and Acme Corp. Bill to: Saudi Aramco, Dhahran. Paid to Microsoft Corporation."
    got = {o.value for o in E.find_organizations(t)}
    assert {"Northwind Logistics", "Acme Corp", "Saudi Aramco", "Microsoft Corporation"} <= got


@pytest.mark.parametrize("phrase,expected", [
    ("governed by the laws of the State of California.", "California"),
    ("governed by the laws of the State of Califomia.", "California"),
    ("Governing law: Kingdom of Saudi Arabia.", "Saudi Arabia"),
    ("This agreement is governed by and construed in accordance with the laws of England and Wales.", "England and Wales"),
    ("The laws of New York shall govern this agreement.", "New York"),
])
def test_jurisdictions_are_normalized_against_the_gazetteer(phrase, expected):
    assert [j.value for j in E.find_jurisdictions(phrase)] == [expected]


def test_unknown_jurisdiction_is_not_invented():
    assert E.find_jurisdictions("governed by the laws of Atlantis.") == []


def test_emails_and_phones():
    got = {(e.type, e.value) for e in E.find_contacts("Email lisa@example.com or call +1 512 555 0199. Date 20260114.")}
    assert got == {("email", "lisa@example.com"), ("phone", "+1 512 555 0199")}


def test_numbered_and_titled_clauses():
    t = ("Article 12.4 Termination for convenience: either party may terminate this contract with sixty (60) days notice. "
         "Governing law: Kingdom of Saudi Arabia.\n\n5.2 Payment Terms. Invoices are payable within 30 days.\n\n"
         "Confidentiality: Each party shall keep the other party's information secret.\n")
    got = [(c.clause_type, c.ref) for c in find_clauses(t)]
    assert ("termination", "12.4") in got and ("payment", "5.2") in got and ("confidentiality", None) in got
    assert ("governing_law", None) in got


def test_end_of_employment_is_a_termination_clause():
    assert [c.clause_type for c in find_clauses("Either the employee or the company may end the employment relationship at any time.")] == ["termination"]
