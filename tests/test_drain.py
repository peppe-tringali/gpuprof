"""Regression tests for the offline drain CLI.

Pre-refactor, `drain.py` predated `comm_events` and `trace_windows`
and silently dropped them — the utility marked a buffer file "fully
drained" while comm/window events never made it to the server. Cover
the invariant explicitly so it can't regress again.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gpuprof.drain import drain_file
from gpuprof._batch import BATCH_KINDS


class _RecordingHandler(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        type(self).received.append((self.path, body))
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *a, **kw): pass


@pytest.fixture
def mock_server():
    _RecordingHandler.received = []
    srv = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"http://127.0.0.1:{port}"}
    srv.shutdown()


def test_drain_pushes_all_kinds(mock_server, tmp_path):
    """Buffer file with every kind of event must round-trip end to end
    — none may silently vanish."""
    buf = tmp_path / "run-42.jsonl"
    buf.write_text(
        json.dumps({
            "samples":       [{"t": 1.0, "gpu_index": 0, "sm_util": 0.5}],
            "steps":         [{"step": 0}],
            "traces":        [{"step": 0, "kernels": []}],
            "comm_events":   [{"step": 0, "bucket_id": 0,
                                "t_start_rel": 0.0, "t_end_rel": 0.001}],
            "trace_windows": [{"t_start_rel": 0.0, "t_end_rel": 1.0,
                                "kernels": []}],
            "host_samples":  [{"t": 1.0, "cpu_percent": 42.0,
                                "n_cpus": 8}],
        }) + "\n"
    )
    drained, pushed = drain_file(buf, mock_server["url"], api_key=None)
    assert drained
    assert pushed == len(BATCH_KINDS)   # one of each kind
    assert not buf.exists()          # empty file removed

    # Server got the batch in one call to /api/runs/42/ingest with all
    # five kinds present.
    assert len(_RecordingHandler.received) == 1
    path, body = _RecordingHandler.received[0]
    assert path == "/api/runs/42/ingest"
    for kind in BATCH_KINDS:
        assert kind in body and len(body[kind]) == 1, kind


def test_drain_skips_non_run_files(mock_server, tmp_path):
    junk = tmp_path / "not-a-run.jsonl"
    junk.write_text("{}\n")
    drained, pushed = drain_file(junk, mock_server["url"], api_key=None)
    assert not drained and pushed == 0
    assert junk.exists()             # left alone


def test_drain_preserves_file_on_post_failure(tmp_path):
    """A dead server should leave the file exactly as it was."""
    buf = tmp_path / "run-7.jsonl"
    payload = json.dumps({"samples": [{"t": 1.0, "gpu_index": 0}]}) + "\n"
    buf.write_text(payload)
    drained, pushed = drain_file(buf, "http://127.0.0.1:1",  # nothing listening
                                  api_key=None)
    assert not drained
    assert buf.exists()
    assert buf.read_text() == payload
