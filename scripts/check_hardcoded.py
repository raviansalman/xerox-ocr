"""Fail when production code contains data that belongs in configuration, secrets or test fixtures.

    python scripts/check_hardcoded.py            # exit 1 and list findings, or exit 0

Scans every file tracked by git except tests and documentation: the application, packs and static UI, deployment
files (Dockerfiles, compose, environment templates, SQL), scripts and CI. Checks: names and identifiers from
``tests/fixtures/denylist.txt``, IP addresses and localhost, URLs other than the documented public defaults,
credential-like assignments and key-shaped strings, absolute filesystem paths. A line can waive named rules with a
trailing comment ``hardcoded-ok(<rule>[,<rule>]): <reason>`` (reviewed in code review); the rule names and the
reason are mandatory, and every other rule still applies to that line.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN: Path | None = None                            # a directory to scan instead of the git-tracked files (tests)
EXCLUDE = ("tests/", "docs/",
           "scripts/load_test.py")                  # generates a synthetic corpus: fixture data, like tests/
APP = "docintel/"                                   # the application: no addresses or paths at all
LOCAL_HOSTS = {"127.0.0.1", "0.0.0.0", "localhost"}  # outside the application: health checks, binds, examples
# a reference to a variable, never a value: ${VAR}, ${VAR:?message} (required) or ${VAR:-} (empty default)
PLACEHOLDER_SECRETS = re.compile(r"^(?:CHANGE_ME|\$\{[A-Z0-9_]+(?::\?[^}]*|:-)?\}|\$\{\{[^}]*\}\})$")
SECRET_NAME = r"[A-Za-z0-9_]*(?:password|passwd|secret|api[_-]?key|token)"   # holds a secret (not *_FILE, max_tokens)
# configuration files (YAML, env, shell, SQL, Dockerfiles): NAME=value or NAME: value, quoted or not
CONFIG_CREDENTIAL = re.compile(rf"(?i)(?:^|[^A-Za-z0-9_]){SECRET_NAME}\s*[:=]\s*[\"']?([^\"'\s#]{{6,}})")
CODE_SUFFIXES = {".py", ".js", ".html", ".css"}
SUFFIXES = {".py", ".yaml", ".yml", ".js", ".html", ".css", ".sql", ".json", ".toml", ".sh", ".svg", ".cfg", ".ini",
            ".env", ".example", ".txt", ".conf"}
NAMES = ("Dockerfile", ".dockerignore", ".gitignore")
ALLOWED_URLS = (
    "http://www.w3.org/2000/svg",                   # SVG namespace in the favicon
    "https://download.pytorch.org/whl/cpu",         # public CPU wheel index for the embedder image
    "https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/ara.traineddata",   # Arabic OCR model (checksum pinned)
)
RULES = {
    "ip-address": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "localhost": re.compile(r"\blocalhost\b", re.I),
    "url": re.compile(r"\b(?:https?|postgres(?:ql)?|redis|amqp|mongodb)://[^\s\"'`)<>]+"),
    "credential": re.compile(rf"(?i)(?:^|[^A-Za-z0-9_]){SECRET_NAME}\s*[:=]\s*[\"']([^\"'\s]{{6,}})[\"']"),
    "key-shaped": re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|dik_[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "url-credential": re.compile(r"\b[a-z]+://[^\s:/@\"']*:([^\s@/\"']+)@"),
    "absolute-path": re.compile(r"[\"'](?:/home|/root|/tmp|/opt|/var|/data|/models|/mnt|/Users|C:\\\\)[/\\\\\w.\-]*[\"']"),
}
COMPOSE_REQUIRED = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")
OPT_OUT = re.compile(r"hardcoded-ok\(([\w\-, /]+)\):\s*\S+")


def denylist() -> list[str]:
    path = ROOT / "tests" / "fixtures" / "denylist.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def files() -> list[Path]:
    if SCAN is not None:
        return sorted(p for p in SCAN.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [ROOT / f for f in sorted(out.split("\0")) if f and not f.startswith(EXCLUDE) and not f.endswith(".md")]


def _local(rule: str, m: re.Match) -> bool:
    """Loopback and bind-all addresses, and URLs to them or to single-label service names (compose services)."""
    if rule in ("ip-address", "localhost"):
        return m.group(0).lower() in LOCAL_HOSTS
    if rule == "url":
        host = re.sub(r"^[a-z]+://(?:[^@/]*@)?", "", m.group(0)).split("/")[0].rsplit(":", 1)[0]
        return host.lower() in LOCAL_HOSTS or (host.isidentifier() and "." not in host)
    return False


def scan() -> list[str]:
    names = [(n, re.compile(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", re.I)) for n in denylist()]
    findings = []
    for path in files():
        if (path.suffix not in SUFFIXES and not path.name.startswith(NAMES)) or not path.exists():
            continue
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        ops = SCAN is None and not str(rel).startswith(APP)
        deploy = ops and str(rel).startswith("deploy/")
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            line = COMPOSE_REQUIRED.sub(r"${\1}", line)   # ${VAR:?message} -> ${VAR}
            m = OPT_OUT.search(line)
            waived = {r.strip() for r in m.group(1).split(",")} if m else set()
            for n, rx in names:
                if "name" not in waived and rx.search(line):
                    findings.append(f"{rel}:{i}: fixture/customer name {n!r}")
            if "credential" not in waived and path.suffix not in CODE_SUFFIXES:
                for m in CONFIG_CREDENTIAL.finditer(line):
                    if not PLACEHOLDER_SECRETS.match(m.group(1).rstrip(",;")):
                        findings.append(f"{rel}:{i}: credential: {m.group(0).strip()[:80]}")
            for rule, rx in RULES.items():
                if rule in waived or (rule == "credential" and path.suffix not in CODE_SUFFIXES):
                    continue
                for m in rx.finditer(line):
                    if rule == "url" and m.group(0).rstrip(".,;}") in ALLOWED_URLS:
                        continue
                    if ops and _local(rule, m):
                        continue
                    if rule == "url-credential" and PLACEHOLDER_SECRETS.match(m.group(1)):
                        continue
                    if rule == "absolute-path" and deploy:  # deployment files define the image layout
                        continue
                    findings.append(f"{rel}:{i}: {rule}: {m.group(0)[:80]}")
    return findings


def main() -> int:
    findings = scan()
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s)" if findings else "no hardcoded data found")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
