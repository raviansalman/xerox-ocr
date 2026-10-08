"""End-to-end verification of a fresh deployment: 22 representative workflows plus observability.

    python scripts/verify_deployment.py http://localhost:8000 keys.json -p docintel -f deploy/docker-compose.yml --env-file deploy/.env

keys.json maps two tenants, "alpha" and "beta", to admin keys of an otherwise empty deployment. The script uploads
generated documents (fictitious content), queries, restarts the data services through docker compose, and deletes
one document. Run it against a test deployment only.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.fixtures.corpus import docx_file, png, scanned_pdf, text_pdf, xlsx_file

BASE, KEYS, COMPOSE = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), sys.argv[3:]
A = httpx.Client(base_url=BASE, headers={"X-API-Key": KEYS["alpha"]}, timeout=120)
B = httpx.Client(base_url=BASE, headers={"X-API-Key": KEYS["beta"]}, timeout=120)
results: list[tuple[int, str, bool, str]] = []


def step(n, name, ok, detail=""):
    results.append((n, name, bool(ok), detail))
    print(f"{n:2d} {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def compose(*args, capture=True):
    return subprocess.run(["docker", "compose", *COMPOSE, *args], capture_output=capture, text=True, check=False)


def sql(q):
    return compose("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "docintel", "-tAc", q).stdout.strip()


def upload(client, path):
    r = client.post("/api/v1/documents", files=[("files", (path.name, path.read_bytes()))])
    r.raise_for_status()
    return r.json()["documents"][0]


def wait_indexed(client, doc_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/v1/documents/{doc_id}").json()
        if d["status"] in ("indexed", "failed"):
            return d
        time.sleep(1)
    return client.get(f"/api/v1/documents/{doc_id}").json()


def ask(client, q, **kw):
    r = client.post("/api/v1/query", json={"q": q, "limit": 10, **kw})
    r.raise_for_status()
    return r.json()


def names(out):
    return [r["filename"] for r in out["results"]]


tmp = Path(tempfile.mkdtemp())
files = {
    "native": text_pdf(tmp / "Service_Agreement_SA-7741.pdf", [
        ["SERVICE AGREEMENT    Agreement No. SA-7741",
         "This agreement is made between Harbor Freight Systems Ltd and Meridian Analytics Inc.",
         "Either party may terminate this agreement with ninety (90) days written notice.",
         "Governing law: State of New York. This agreement expires on 30 June 2028."],
        ["Schedule A: Pricing", "The annual fee is USD 84,000 payable quarterly in advance.",
         "Signed by Elena Marsh, Chief Operating Officer"]]),
    "scanned": scanned_pdf(tmp / "Scanned_Invoice_INV-5512.pdf", [[
        "TAX INVOICE", "Invoice No: INV-5512    Invoice Date: 12 March 2025", "Bill to: Northgate Utilities",
        "Total Amount Due: USD 18,400.00"]]),
    "image": png(tmp / "Delivery_Note_DN-3307.png", [
        "DELIVERY NOTE DN-3307", "Delivered to: Northgate Utilities warehouse", "Items: 40 replacement filters"]),
    "docx": docx_file(tmp / "Remote_Work_Guidelines.docx", "REMOTE WORK GUIDELINES", [
        "Employees may perform their duties remotely up to three days per week.",
        "A home office allowance of USD 250 is paid once per year."]),
    "xlsx": xlsx_file(tmp / "Invoice_Register_2025.xlsx", {"Invoices": [
        ["Invoice", "Customer", "Amount"], ["INV-5512", "Northgate Utilities", 18400], ["INV-5513", "Harbor Freight", 9200]]}),
}
ids = {}
for i, (key, path) in enumerate(files.items(), start=1):
    d = wait_indexed(A, upload(A, path)["id"])
    ids[key] = d["id"]
    labels = {"native": "Upload a native PDF", "scanned": "Upload a scanned PDF", "image": "Upload an image",
              "docx": "Upload DOCX", "xlsx": "Upload XLSX"}
    step(i, labels[key], d["status"] == "indexed", f"kind={d.get('kind')} pages={d.get('page_count')} type={d.get('doc_type')}")

d = A.get(f"/api/v1/documents/{ids['native']}").json()
p2 = A.get(f"/api/v1/documents/{ids['native']}/pages/2").json()
step(6, "Process multiple pages", d["page_count"] == 2 and "84,000" in p2["text"], f"pages={d['page_count']}")

out = ask(A, '"ninety (90) days written notice"')
step(7, "Exact phrase", names(out)[:1] == [files["native"].name], str(names(out)[:3]))
out = ask(A, "INV 5512")
step(8, "Identifier", files["scanned"].name in names(out)[:2] and out["results"][0]["tier"] == 1, str(names(out)[:3]))
out = ask(A, "Northgate Utilitis")
step(9, "Typo / OCR variant", files["scanned"].name in names(out) or files["image"].name in names(out), str(names(out)[:3]))
out = ask(A, "work from home policy")
step(10, "Semantic meaning", names(out)[:1] == [files["docx"].name], str(names(out)[:3]))
out = ask(A, "Can either party cancel the agreement early?", explain=True)
step(11, "Contextual (concept)", names(out)[:1] == [files["native"].name] and "contextual" in out["results"][0]["retrievers"],
     str(out["plan"].get("concepts")))
out = ask(A, "agreements involving Meridian Analytics")
step(12, "Entity", names(out)[:1] == [files["native"].name], str(out["results"][0]["retrievers"] if out["results"] else []))
out = ask(A, "agreements governed by New York law")
step(13, "Metadata filter", names(out) == [files["native"].name], str(names(out)))
out = ask(A, "How many invoices do we have?")
step(14, "Count", out["answer"].get("kind") == "number", json.dumps({k: out["answer"].get(k) for k in ("value", "text")}))
out = ask(A, "What is the total value of all invoices?")
rows = out["answer"].get("rows") or []
step(15, "Sum", any(r.get("currency") == "USD" and r.get("value") == 18400.0 for r in rows), out["answer"].get("text", ""))
out = ask(A, "invoices issued in 2025")
step(16, "Date range", files["scanned"].name in names(out) and files["native"].name not in names(out), str(names(out)))
out = ask(A, '"annual fee is USD 84,000"')
ev = out["results"][0]["evidence"][0] if out["results"] else {}
page = A.get(f"/api/v1/documents/{out['results'][0]['document_id']}/pages/{ev.get('page')}").json() if ev else {}
step(17, "Evidence with span", ev and page["text"][ev["char_start"]:ev["char_end"]] == ev["text"],
     f"page={ev.get('page')} span={ev.get('char_start')}-{ev.get('char_end')}")
out = ask(A, "How many days notice is needed to terminate the service agreement?", answer=True)
g = out.get("generated_answer") or {}
step(18, "Grounded answer (extractive; no LLM key configured)", g.get("status") in ("answered", "computed") and
     ("90" in json.dumps(g) or "ninety" in json.dumps(g)), f"provider={g.get('provider')} status={g.get('status')}")

leak = []
for q in ("SA-7741", "Northgate Utilities", "How many invoices do we have?", "work from home policy"):
    o = ask(B, q)
    leak += [r["filename"] for r in o["results"]]
    if o["answer"].get("kind") == "number" and o["answer"].get("value"):
        leak.append("count")
direct = [B.get(f"/api/v1/documents/{i}").status_code for i in ids.values()]
spoof = B.post("/api/v1/query", headers={"X-Tenant-Id": "alpha"}, json={"q": "SA-7741"}).status_code
step(19, "Cross-tenant access", not leak and set(direct) == {404} and spoof in (400, 403), f"leak={leak} direct={set(direct)} spoof={spoof}")

compose("restart", "postgres", "milvus", "redis", "embedder")
deadline = time.time() + 300
while time.time() < deadline:
    try:
        if httpx.get(f"{BASE}/health/ready", timeout=5).status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(3)
ok_query = False
for _ in range(20):
    try:
        ok_query = names(ask(A, "SA-7741"))[:1] == [files["native"].name]
        if ok_query:
            break
    except httpx.HTTPError:
        pass
    time.sleep(3)
late = tmp / "After_Restart_Note.txt"
late.write_text("Post-restart verification note: chiller unit CH-9921 inspected.")
late_doc = wait_indexed(A, upload(A, late)["id"])
step(20, "Restart dependencies and recover", ok_query and late_doc["status"] == "indexed", f"query={ok_query} new upload={late_doc['status']}")

before = A.get(f"/api/v1/documents/{ids['native']}").json()
A.post(f"/api/v1/documents/{ids['native']}/reprocess").raise_for_status()
after = wait_indexed(A, ids["native"])
time.sleep(2)
after = A.get(f"/api/v1/documents/{ids['native']}").json()
again = upload(A, files["native"])
step(21, "Reprocess the same document", after["status"] == "indexed" and len(after["versions"]) == len(before["versions"]) + 1
     and names(ask(A, "SA-7741"))[:1] == [files["native"].name] and again["duplicate"] and again["id"] == ids["native"],
     f"versions {len(before['versions'])}->{len(after['versions'])}, re-upload duplicate={again['duplicate']}")

victim = ids["docx"]
A.delete(f"/api/v1/documents/{victim}").raise_for_status()
tables = ["pages", "blocks", "chunks", "fields", "entities", "clauses", "relations", "unit_terms", "document_versions"]
rows = {t: sql(f"SELECT count(*) FROM {t} WHERE document_id = '{victim}'") for t in tables}
vec = compose("exec", "-T", "api", "python", "-c",
              "from docintel.indexing.vectors import get_vector_store as v; from docintel.indexing.embeddings import get_embedder as e; "
              f"print(len(v().search('alpha', e().embed_query('remote work'), 10, document_ids=['{victim}'])))").stdout.strip()
gone = A.get(f"/api/v1/documents/{victim}").status_code == 404 and files["docx"].name not in names(ask(A, "work from home policy"))
step(22, "Delete from every index", gone and set(rows.values()) == {"0"} and vec == "0", f"rows={rows} vectors={vec}")

wm = compose("exec", "-T", "worker", "curl", "-fs", "http://127.0.0.1:9100/metrics").stdout
ready = httpx.get(f"{BASE}/health/ready", timeout=10).json()
step(23, "Extra: worker metrics and readiness", "docintel_documents_processed_total" in wm and ready["status"] == "ready"
     and ready["checks"]["queue"]["stalled"] == 0, f"stalled={ready['checks']['queue'].get('stalled')}")

failed = [r for r in results if not r[2]]
print(json.dumps({"passed": len(results) - len(failed), "total": len(results), "failed": [r[:2] for r in failed]}))
sys.exit(1 if failed else 0)
