"""Thread-mode dispatch: a document submitted again while it is processing is processed once more afterwards."""
import threading

from docintel.config import get_settings
from docintel.ingest import dispatch


def test_a_submit_during_a_run_is_not_dropped(monkeypatch):
    monkeypatch.setattr(get_settings(), "task_mode", "thread")
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    runs = []

    def process(tenant_id, document_id):
        runs.append(document_id)
        if len(runs) == 1:
            started.set()
            release.wait(5)
        else:
            done.set()

    monkeypatch.setattr(dispatch.pipeline, "process", process)
    dispatch.submit("t1", "doc-a")
    assert started.wait(5)
    dispatch.submit("t1", "doc-a")              # reprocess requested while the first run is still going
    dispatch.submit("t1", "doc-a")              # repeated requests collapse into one more run
    release.set()
    assert done.wait(5)
    for _ in range(50):
        with dispatch._lock:
            if "doc-a" not in dispatch._inflight:
                break
        threading.Event().wait(0.02)
    assert runs == ["doc-a", "doc-a"]
    assert "doc-a" not in dispatch._inflight and "doc-a" not in dispatch._again


def test_stale_detection_covers_the_ocr_engine_and_the_pdf_parser():
    from docintel.ingest.pipeline import stale_reasons
    current = {"pipeline": "2", "packs": ["core"], "embedding_model": "m", "ocr": "tesseract-5.3:eng",
               "pdf_parser": "pymupdf-1.26.0"}
    same = {"pipeline": "2", "packs": ["core"], "embedding_model": "m", "parser": "pymupdf-1.26.0", "ocr": "tesseract-5.3:eng"}
    assert stale_reasons(same, current) == []
    assert stale_reasons({**same, "ocr": "tesseract-5.3:eng+deu"}, current) == ["ocr changed"]
    assert stale_reasons({**same, "parser": "pymupdf-1.24.0"}, current) == ["parser changed"]
    native = {k: v for k, v in same.items() if k != "ocr"}                     # never OCRed: OCR changes do not matter
    assert stale_reasons(native, {**current, "ocr": "tesseract-6.0:eng"}) == []
    assert stale_reasons({**same, "parser": "openpyxl"}, current) == []


def test_logs_go_to_the_real_stdout_even_when_stdout_is_redirected(monkeypatch):
    """Celery replaces sys.stdout in worker children with a proxy that drops log records written back through it."""
    import io
    import logging
    import sys

    from docintel.logging_setup import configure_logging
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    try:
        configure_logging()
        assert [h.stream for h in root.handlers] == [sys.__stdout__]
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])


def test_thread_mode_refuses_several_api_processes(monkeypatch, capsys):
    """Each process would reap the other's queued backlog as stalled and process it again (seen at 50,000 documents)."""
    import argparse

    from docintel import cli
    monkeypatch.setattr(get_settings(), "task_mode", "thread")
    assert cli.cmd_serve(argparse.Namespace(workers=2, host=None, port=0)) == 2
    assert "one process" in capsys.readouterr().err
