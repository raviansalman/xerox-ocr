"""Configuration profiles fail closed; domain knowledge comes only from packs."""
import pytest

from docintel.config import ConfigError, Settings
from docintel.packs import CORE, PackError, available_packs, get_domain, reset_packs
from docintel.search.planner import plan_query
from docintel.understanding.classify import classify


@pytest.fixture(autouse=True)
def _no_docintel_env(monkeypatch):
    """Settings in these tests come only from their arguments, not from the test session's environment."""
    import os
    for k in [k for k in os.environ if k.startswith("DOCINTEL_") and k != "DOCINTEL_PACKS_DIR"]:
        monkeypatch.delenv(k)


def settings(**kw) -> Settings:
    base = {"database_url": "postgresql://u@h/db", "milvus_uri": "http://m:1", "embedder_url": "http://e:1"}
    return Settings(_env_file=None, **{**base, **kw})


def test_required_settings_have_no_defaults():
    s = Settings(_env_file=None)
    assert s.database_url is None and s.milvus_uri is None and s.embedder_url is None and s.redis_url is None
    with pytest.raises(ConfigError, match="DOCINTEL_DATABASE_URL"):
        s.validate_for_startup("api")
    with pytest.raises(ConfigError, match="DOCINTEL_DATABASE_URL"):
        s.require("database_url")


def test_production_profile_rejects_unsafe_settings():
    with pytest.raises(ConfigError) as e:
        settings(environment="production", auth_mode="dev", vector_backend="memory", task_mode="inline").validate_for_startup()
    msg = str(e.value)
    assert "AUTH_MODE=dev" in msg and "memory" in msg and "inline" in msg and "API keys" in msg
    settings(environment="production", api_keys_file="/run/secrets/keys", metrics_token="scrape").validate_for_startup()
    settings(environment="production", api_keys_file="/run/secrets/keys", metrics_enabled=False).validate_for_startup()
    with pytest.raises(ConfigError, match="METRICS_TOKEN"):
        settings(environment="production", api_keys_file="/run/secrets/keys").validate_for_startup()
    with pytest.raises(ConfigError, match="REDIS_URL"):
        settings(task_mode="celery").validate_for_startup("worker")
    for field, var in (("embedding_model", "EMBEDDING_MODEL"), ("reranker_model", "RERANKER_MODEL")):
        with pytest.raises(ConfigError, match=f"{var}=test-.* tests only"):
            settings(environment="production", api_keys_file="/run/secrets/keys", metrics_token="scrape",
                     **{field: "test-hash-768" if field == "embedding_model" else "test-overlap"}).validate_for_startup()
    # the Anthropic SDK resolves credentials itself (key setting, ANTHROPIC_API_KEY or a profile); only the SDK is required
    settings(answer_provider="anthropic").validate_for_startup()


def test_semantic_search_can_be_disabled_without_an_embedder():
    s = settings(vector_backend="disabled", embedder_url=None, milvus_uri=None)
    s.validate_for_startup("api")
    assert s.semantic_enabled is False


def test_url_ingestion_is_off_by_default():
    assert Settings(_env_file=None).url_fetch_enabled is False


def test_core_pack_is_always_enabled_and_generic():
    core = get_domain(())
    assert core.packs == (CORE,)
    assert "invoice" not in core.types and "nda" not in core.types and not core.clause_types
    assert {"letter", "memo", "email", "report", "form"} <= set(core.types)
    assert core.jurisdictions                      # public reference data


def test_packs_add_domain_knowledge():
    d = get_domain(("legal", "finance"))
    assert d.packs[0] == CORE and {"legal", "finance"} <= set(d.packs)
    assert "termination" in d.clause_types and "termination" in d.concepts
    assert d.types["invoice"].date_field == "issue_date" and d.types["contract"].generic
    assert any(r.predicate == "may_terminate" for r in d.relation_patterns)


def test_engine_works_without_domain_packs():
    core = get_domain(())
    plan = plan_query("How many documents mention Jeddah?", domain=core)
    assert plan.intent == "count" and plan.filters.doc_types == [] and plan.text == "Jeddah"
    # without the legal pack "NDA" is not a document type, so the question asks for a figure, not a document count
    assert plan_query("How many NDAs do we have?", domain=core).intent == "fact"
    assert classify("x.pdf", "TAX INVOICE Invoice No: 1", domain=core).label != "invoice"
    assert plan_query("INV-1024-77", domain=core).identifiers == ["inv102477"]


def test_unknown_or_invalid_packs_are_rejected():
    with pytest.raises(PackError):
        get_domain(("no_such_pack",))
    with pytest.raises(PackError):
        get_domain(("../etc",))


def test_deployment_pack_from_packs_dir(tmp_path, monkeypatch):
    pack = tmp_path / "shipping"
    pack.mkdir()
    (pack / "pack.yaml").write_text("name: shipping\nversion: 0.1.0\ndepends_on: [core]\n")
    (pack / "taxonomy.yaml").write_text(
        "types:\n  bill_of_lading:\n    display: Bill of lading\n    query_terms: [bill of lading, bills of lading]\n"
        "    title_patterns: ['\\bbill\\s+of\\s+lading\\b']\n")
    monkeypatch.setenv("DOCINTEL_PACKS_DIR", str(tmp_path))
    reset_packs()
    try:
        assert "shipping" in available_packs()
        d = get_domain(("shipping",))
        assert classify("bl.pdf", "BILL OF LADING\nShipper: X", domain=d).label == "bill_of_lading"
        assert plan_query("How many bills of lading do we have?", domain=d).filters.doc_types == ["bill_of_lading"]
    finally:
        monkeypatch.delenv("DOCINTEL_PACKS_DIR")
        reset_packs()


def test_tenant_settings_validate_pack_names():
    from pydantic import ValidationError

    from docintel.settings_store import TenantSettings
    assert TenantSettings(packs=["legal", "legal"]).packs == ["legal"]
    with pytest.raises(ValidationError):
        TenantSettings(packs=["unknown"])
    with pytest.raises(ValidationError):
        TenantSettings(examples=[{"label": "", "q": "x"}])


def test_unknown_or_malformed_default_packs_fail_at_startup(tmp_path, monkeypatch):
    from docintel.packs import reset_packs
    with pytest.raises(ConfigError, match="DEFAULT_PACKS"):
        settings(default_packs="business,nonexistent").validate_for_startup("api")
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "pack.yaml").write_text("name: broken\nversion: 1\ndepends_on: [core]\n")
    (bad / "relations.yaml").write_text("relations:\n  - predicate: x\n    pattern: '(?P<subject>unclosed'\n")
    monkeypatch.setenv("DOCINTEL_PACKS_DIR", str(tmp_path))
    reset_packs()
    try:
        with pytest.raises(ConfigError, match="DEFAULT_PACKS"):
            settings(default_packs="broken").validate_for_startup("worker")
    finally:
        monkeypatch.delenv("DOCINTEL_PACKS_DIR")
        reset_packs()


def test_every_setting_is_documented():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[2] / "docs" / "CONFIGURATION.md").read_text()
    missing = [f"DOCINTEL_{n.upper()}" for n in Settings.model_fields if f"DOCINTEL_{n.upper()}" not in doc]
    assert missing == []
