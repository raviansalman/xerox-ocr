"""The evaluation harness over the synthetic fixtures: every gate passes (exact recall 100%, no leakage, computed
answers exact, no false positives on unanswerable questions), and the v1 engine is compared with v2."""
import json
import os
from pathlib import Path

import pytest

from tests.conftest import headers

pytestmark = pytest.mark.integration
DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "fixtures.yaml"
READERS = {"acme": "acme_reader", "globex": "globex_reader", "haystack": "haystack_uploader"}


class ClientTarget:
    def __init__(self, client):
        self.client = client

    def query(self, tenant, payload):
        r = self.client.post("/api/v1/query", headers=headers(READERS[tenant]), json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    def documents(self, tenant):
        r = self.client.get("/api/v1/documents", headers=headers(READERS[tenant]), params={"limit": 500})
        return {d["id"]: d["filename"] for d in r.json()["documents"]}

    def semantic_enabled(self):
        """Paraphrase questions need a real embedding model; the deterministic stand-in only hashes words."""
        from docintel.config import get_settings
        s = get_settings()
        return s.semantic_enabled and s.embedding_model != "test-hash-768"


@pytest.fixture(scope="module")
def reports(client, ingested, tmp_path_factory):
    from docintel.config import get_settings
    from docintel.evaluation import evaluate, load_dataset
    from docintel.query import reset_engines
    dataset, out = load_dataset(DATASET), {}
    s = get_settings()
    for engine in ("v1", "v2"):
        s.retrieval_engine = engine
        reset_engines()
        try:
            out[engine] = evaluate(dataset, ClientTarget(client), isolation=(engine == "v2"))
        finally:
            s.retrieval_engine = "v2"
            reset_engines()
    folder = Path(os.environ.get("DOCINTEL_EVAL_REPORT_DIR") or tmp_path_factory.mktemp("eval"))
    folder.mkdir(parents=True, exist_ok=True)
    for engine, report in out.items():
        (folder / f"{engine}.json").write_text(json.dumps(report, indent=2, default=str))
    return out


def test_v2_passes_every_gate(reports):
    from docintel.evaluation.runner import render_summary
    r = reports["v2"]
    print(render_summary(r))
    assert r["engine"] == "v2" and r["source"] == "synthetic" and "Synthetic" in r["note"]
    assert r["passed"], render_summary(r)
    assert r["summary"]["leakage"] == 0 and r["isolation_queries"] >= 150
    assert r["summary"]["exact_recall"] == 1.0 and r["summary"]["abstention_fp_rate"] == 0.0


def test_every_question_passes(reports):
    failed = [(q["id"], q["question"], q["results"][:3]) for q in reports["v2"]["questions"] if not q["passed"]]
    assert failed == []


def test_v2_does_not_regress_against_v1(reports):
    from docintel.evaluation import compare
    from docintel.evaluation.runner import render_summary
    print(render_summary(reports["v1"]))
    c = compare(reports["v1"], reports["v2"])
    print(json.dumps(c["metrics"], indent=1))
    assert not [k for k, m in c["metrics"].items() if m["regression"] and k in ("exact_recall", "computed_accuracy", "leakage")]
    assert c["metrics"]["mrr"]["delta"] >= 0 and c["metrics"]["abstention_fp_rate"]["candidate"] == 0.0
