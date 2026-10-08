"""Production code holds no fixture names, customer names, addresses, credentials or paths (scripts/check_hardcoded.py)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _scanner():
    spec = importlib.util.spec_from_file_location("check_hardcoded", ROOT / "scripts" / "check_hardcoded.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_production_code_is_free_of_hardcoded_data():
    assert _scanner().scan() == []


def test_scanner_detects_violations(tmp_path, monkeypatch):
    mod = _scanner()
    (tmp_path / "bad.py").write_text('URL = "postgresql://user:pw@10.1.2.3/db"\nNAME = "John Smith"\nKEY = "AKIAABCDEFGHIJKLMNOP"\n'
                                     'P = "/home/someone/models"\nok = "/opt/x"  # hardcoded-ok(absolute-path): reviewed\n'
                                     'mixed = "/opt/x 10.9.8.7"  # hardcoded-ok(absolute-path): reviewed\n'
                                     'bare = "/opt/y"  # hardcoded-ok: no rule named\n'
                                     'local = "http://127.0.0.1:8000"\n')
    monkeypatch.setattr(mod, "SCAN", tmp_path)
    monkeypatch.setattr(mod, "ROOT", ROOT)
    findings = mod.scan()
    rules = {f.split(": ", 2)[1].split(" ")[0] for f in findings}
    assert {"url", "url-credential", "ip-address", "fixture/customer", "key-shaped", "absolute-path"} <= rules
    assert not any("bad.py:5:" in f for f in findings)
    assert [f.split(": ")[1] for f in findings if "bad.py:6:" in f] == ["ip-address"]      # waivers are per rule
    assert any("bad.py:7:" in f for f in findings)                                         # a waiver must name rules
    assert any("bad.py:8:" in f for f in findings)                                         # the application: no hosts


def test_the_scanner_covers_deployment_files_scripts_and_ci():
    mod = _scanner()
    scanned = {str(p.relative_to(ROOT)) for p in mod.files()}
    assert {"deploy/Dockerfile", "deploy/Dockerfile.embedder", "deploy/docker-compose.yml", "deploy/.env.example",
            "deploy/postgres-init.sql", ".github/workflows/ci.yml", "scripts/verify_deployment.py", ".env.example"} <= scanned
    assert not any(p.startswith(("tests/", "docs/")) for p in scanned)


def test_passwords_in_urls_must_be_placeholders(tmp_path, monkeypatch):
    mod = _scanner()
    (tmp_path / "compose.yml").write_text("a: postgresql://app:${DB_PASSWORD:?set it}@db:5432/x\n"
                                          "b: redis://:CHANGE_ME@cache:6379/0\n"
                                          "c: postgresql://app:hunter22@db:5432/x\n"
                                          "d: postgresql://app:${DB_PASSWORD:-hunter22}@db:5432/x\n")
    monkeypatch.setattr(mod, "SCAN", tmp_path)
    monkeypatch.setattr(mod, "ROOT", ROOT)
    found = sorted(f.split(": ")[0] for f in mod.scan() if "url-credential" in f)
    assert found == [str(tmp_path / "compose.yml") + ":3", str(tmp_path / "compose.yml") + ":4"]


def test_secrets_in_configuration_files_are_found_in_every_usual_form(tmp_path, monkeypatch):
    mod = _scanner()
    (tmp_path / "deploy.env").write_text("POSTGRES_PASSWORD=hunter22xx\nDOCINTEL_ANSWER_API_KEY=sk-live-value\n"
                                         "DOCINTEL_METRICS_TOKEN=${DOCINTEL_METRICS_TOKEN:?set it}\n"
                                         "DOCINTEL_API_KEYS_FILE=./api_keys.json\nmax_tokens: 100000\n")
    (tmp_path / "compose.yml").write_text('env: { POSTGRES_PASSWORD: "hunter22xx" }\nX_SECRET: ${X_SECRET:-}\n')
    (tmp_path / "code.py").write_text('token = request.headers.get("x")\nDB_PASSWORD = "hunter22xx"\n')
    monkeypatch.setattr(mod, "SCAN", tmp_path)
    monkeypatch.setattr(mod, "ROOT", ROOT)
    found = sorted(f.rsplit("/", 1)[-1].split(": ")[0] for f in mod.scan() if ": credential:" in f)
    assert found == ["code.py:2", "compose.yml:1", "deploy.env:1", "deploy.env:2"]
