"""The web UI in a real browser: connecting with a key, every kind of question, the document drawer, upload and delete.

Runs against a live server sharing the integration test database, so the uploaded corpus is searchable here.
"""
import os
import re

import pytest

from tests.conftest import _free_port, _serve, headers, raw_key

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest.fixture(scope="session")
def base_url(client, ingested):
    from docintel.api.app import app
    port = _free_port()
    _serve(app, port)
    return f"http://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        # DOCINTEL_TEST_CHROMIUM points at a system Chromium when Playwright's own build is not installed
        b = pw.chromium.launch(executable_path=os.environ.get("DOCINTEL_TEST_CHROMIUM") or None)
        yield b
        b.close()


@pytest.fixture
def page(browser, base_url):
    ctx = browser.new_context(base_url=base_url, accept_downloads=True)
    pg = ctx.new_page()
    errors: list[str] = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto("/")
    pg.get_by_test_id("key-input").fill(raw_key("acme_uploader"))
    pg.get_by_test_id("key-save").click()
    pg.get_by_test_id("tenant").filter(has_text="acme").wait_for()
    yield pg
    ctx.close()
    assert not errors, errors                       # no script errors and no CSP violations


def ask(page, q):
    page.get_by_test_id("query-input").fill(q)
    page.get_by_test_id("query-submit").click()
    page.get_by_test_id("answer").wait_for(state="visible")
    return page


def titles(page):
    return page.get_by_test_id("result-title").all_inner_texts()


def test_wrong_key_is_refused(browser, base_url):
    ctx = browser.new_context(base_url=base_url)
    pg = ctx.new_page()
    pg.goto("/")
    pg.get_by_test_id("key-input").fill("not-a-key")
    pg.get_by_test_id("key-save").click()
    pg.get_by_text("That key was not accepted.").wait_for()
    assert pg.get_by_test_id("key-dialog").is_visible()
    ctx.close()


EXAMPLES = [
    {"label": "Exact phrase", "q": "\"for immediate release\""},
    {"label": "Identifier", "q": "INV-2026-00481"},
    {"label": "Meaning", "q": "rules for working from home"},
    {"label": "Count", "q": "How many NDAs do we have?"},
    {"label": "Percentage", "q": "What percentage of our contracts are NDAs?"},
    {"label": "Jurisdiction", "q": "contracts governed by California law"},
    {"label": "Dates", "q": "Which agreements expire in 2027?"},
    {"label": "Amounts", "q": "invoices above SAR 400,000"},
    {"label": "Total", "q": "What is the total value of all invoices?"},
    {"label": "Clauses", "q": "Which contracts have termination clauses?"},
    {"label": "Signatures", "q": "documents signed by John Smith"},
    {"label": "Lookup", "q": "What is the invoice number of the Saudi Aramco invoice?"},
    {"label": "Breakdown", "q": "How many documents per type?"},
]


@pytest.fixture
def configured_examples(client):
    """Example questions are tenant configuration (PUT /api/v1/settings, admin role), never part of the UI code."""
    r = client.put("/api/v1/settings", headers=headers("acme_admin"), json={"examples": EXAMPLES})
    assert r.status_code == 200, r.text
    yield
    client.put("/api/v1/settings", headers=headers("acme_admin"), json={"examples": []})


def test_guidance_without_configured_examples(page):
    assert page.get_by_test_id("guidance").is_visible()
    assert page.get_by_test_id("examples").locator("button").count() == 0


def test_settings_require_the_admin_role(client):
    r = client.put("/api/v1/settings", headers=headers("acme_reader"), json={"examples": EXAMPLES})
    assert r.status_code == 403


def test_every_example_question_answers(configured_examples, page):
    page.reload()
    page.get_by_test_id("tenant").filter(has_text="acme").wait_for()
    chips = page.get_by_test_id("examples").locator("button")
    chips.first.wait_for()
    assert chips.count() == len(EXAMPLES)
    for i in range(chips.count()):
        q = chips.nth(i).get_attribute("data-q")
        page.get_by_test_id("answer").evaluate("e => e.hidden = true")
        chips.nth(i).click()
        page.get_by_test_id("answer").wait_for(state="visible")
        text = page.get_by_test_id("answer-text").inner_text()
        assert text and "failed" not in text.lower(), q
        shown = sum(page.get_by_test_id(t).count() for t in ("result", "answer-value", "answer-table"))
        assert shown > 0, q


@pytest.mark.parametrize("q,check", [
    ("How many NDAs do we have?", lambda p: p.get_by_test_id("answer-value").inner_text() == "2"),
    ("What percentage of our contracts are NDAs?", lambda p: p.get_by_test_id("answer-value").inner_text() == "33.3%"),
    ("INV-2026-00481", lambda p: "Tax_Invoice_INV-2026-00481.pdf" in p.get_by_test_id("result").first.inner_text()),
    ('"for immediate release"', lambda p: p.get_by_test_id("result").count() == 1),
    ("Califomia", lambda p: p.get_by_test_id("result").count() >= 1),
    ("What is the invoice number of the Saudi Aramco invoice?", lambda p: p.get_by_test_id("answer-value").inner_text() == "INV-2026-00481"),
    ("What is the total value of all invoices?", lambda p: "418,750" in p.get_by_test_id("answer-table").inner_text()),
    ("How many documents per type?", lambda p: p.get_by_test_id("answer-table").is_visible()),
    ("How many employees were paid in 2024?", lambda p: p.get_by_test_id("answer-value").inner_text().startswith("42")),
    ("Which contracts are unsigned?", lambda p: p.get_by_test_id("result").count() == 3),
    ("zzqxunknownzzq", lambda p: p.get_by_test_id("result").count() == 0),
])
def test_typed_questions(page, q, check):
    ask(page, q)
    assert check(page), (q, page.get_by_test_id("answer").inner_text(), titles(page))


def test_identifier_is_highlighted_and_explain_shows_the_plan(page):
    page.get_by_test_id("explain-toggle").check()
    ask(page, "INV-2026-00481")
    marks = page.locator("[data-testid=result] mark").all_inner_texts()
    assert "INV-2026-00481" in marks and not any(m.lower() == "inv" for m in marks), marks
    assert '"intent": "search"' in page.get_by_test_id("plan").inner_text()


def test_drawer_shows_fields_and_page_image(page):
    ask(page, "INV-2026-00481")
    page.get_by_test_id("result-title").first.click()
    drawer = page.get_by_test_id("drawer")
    drawer.wait_for(state="visible")
    fields = page.get_by_test_id("fields-table")
    assert "INV-2026-00481" in fields.inner_text() and "SAR 418,750.00" in fields.inner_text()
    img = page.get_by_test_id("page-view").locator("img")
    img.wait_for()
    page.wait_for_function("() => document.querySelector('[data-testid=page-view] img').naturalWidth > 0")
    with page.expect_download() as dl:
        drawer.get_by_text("Download original").click()
    assert dl.value.suggested_filename.endswith(".pdf")
    page.get_by_test_id("drawer-close").click()
    assert drawer.is_hidden()


def test_upload_search_and_delete(page, tmp_path):
    f = tmp_path / "Kiosk_Audit_KX-5521.txt"
    f.write_text("Kiosk printer audit. Reference KX-5521. The lobby kiosk printer jams on glossy paper.\n")
    page.get_by_test_id("tab-documents").click()
    page.get_by_test_id("file-input").set_input_files(str(f))
    page.get_by_test_id("upload-status").filter(has_text="1 uploaded").wait_for()
    row = page.get_by_test_id("doc-row").filter(has_text=f.name)
    row.wait_for()
    page.wait_for_function("n => [...document.querySelectorAll('[data-testid=doc-row]')]"
                           ".some(r => r.textContent.includes(n) && r.dataset.status === 'indexed')", arg=f.name)

    page.get_by_test_id("doc-filter").fill("Kiosk")
    page.wait_for_function("() => document.querySelectorAll('[data-testid=doc-row]').length === 1")
    assert int(page.get_by_test_id("stat-indexed").inner_text().split()[0].replace(",", "")) >= 22

    page.get_by_test_id("tab-ask").click()
    ask(page, "KX-5521")
    assert titles(page) and re.search("Kiosk", titles(page)[0])

    page.get_by_test_id("tab-documents").click()
    page.once("dialog", lambda d: d.accept())
    row.get_by_test_id("delete").click()
    row.wait_for(state="detached")
    page.get_by_test_id("tab-ask").click()
    ask(page, "KX-5521")
    assert page.get_by_test_id("result").count() == 0


def test_other_tenant_sees_none_of_acme(browser, base_url):
    ctx = browser.new_context(base_url=base_url)
    pg = ctx.new_page()
    pg.goto("/")
    pg.get_by_test_id("key-input").fill(raw_key("carol_reader"))
    pg.get_by_test_id("key-save").click()
    pg.get_by_test_id("tenant").filter(has_text="carol").wait_for()
    ask(pg, "INV-2026-00481")
    assert pg.get_by_test_id("result").count() == 0
    ask(pg, "How many documents per type?")
    pg.get_by_test_id("tab-documents").click()
    pg.get_by_test_id("stat-indexed").filter(has_text="0").wait_for()
    assert pg.get_by_test_id("doc-row").count() == 0
    ctx.close()
