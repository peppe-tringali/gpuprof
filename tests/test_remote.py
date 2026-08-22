"""Remote pusher tests — mock HTTP server + explicit outage."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from gpuprof.remote import Remote
from gpuprof.sampler import Sample


class _RecordingHandler(BaseHTTPRequestHandler):
    posts: list = []          # class-level so the server thread can append
    accepting: bool = True    # toggle to simulate a dead endpoint

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if not type(self).accepting:
            self.send_response(500); self.end_headers(); return
        type(self).posts.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.path == "/api/runs":
            self.wfile.write(b'{"id": 1}')
        else:
            self.wfile.write(b'{"ok": true}')

    def log_message(self, *a, **kw): pass


@pytest.fixture
def mock_server():
    _RecordingHandler.posts = []
    _RecordingHandler.accepting = True
    srv = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"http://127.0.0.1:{port}", "handler": _RecordingHandler,
           "srv": srv}
    srv.shutdown()


def _sample(t=None, i=0):
    return Sample(
        t=t or time.time(), gpu_index=i, sm_util=0.5,
        mem_used_bytes=10**9, mem_total_bytes=10**10,
        power_w=300, temp_c=65, sm_clock_mhz=1500, mem_clock_mhz=1500,
        pcie_rx_kbps=1, pcie_tx_kbps=1,
    )


def test_happy_path_posts_batch(mock_server, tmp_path):
    r = Remote(mock_server["url"], flush_hz=5.0,
               buffer_dir=tmp_path / "buf")
    rid = r.start_run("t", "H100", {"k": "v"})
    assert rid == 1
    for i in range(20):
        r.push_sample(_sample(i=i % 2))
    r.push_step({"step": 0, "t_start": 0, "t_end": 1})
    r.end_run()

    # Ingest was hit; buffer stays empty.
    paths = [p for p, _ in mock_server["handler"].posts]
    assert any(p == "/api/runs" for p in paths)
    assert any(p == "/api/runs/1/ingest" for p in paths)
    assert not list((tmp_path / "buf").glob("*.jsonl")) or \
           all(p.stat().st_size == 0 for p in (tmp_path / "buf").glob("*.jsonl"))


def test_outage_buffers_then_drains(mock_server, tmp_path):
    handler = mock_server["handler"]
    r = Remote(mock_server["url"], flush_hz=5.0,
               buffer_dir=tmp_path / "buf")
    r.start_run("t", "H100", {})

    # Push some events, then simulate outage.
    for i in range(5): r.push_sample(_sample(i=0))
    time.sleep(0.4)
    handler.accepting = False
    for i in range(20): r.push_sample(_sample(i=0))
    time.sleep(0.6)

    # Buffer file must exist and have content.
    buf_dir = tmp_path / "buf"
    files = list(buf_dir.glob("*.jsonl"))
    assert len(files) == 1 and files[0].stat().st_size > 0

    # Restore server, wait for drain, verify buffer clears.
    handler.accepting = True
    time.sleep(1.5)
    r.end_run()
    remaining = list(buf_dir.glob("*.jsonl"))
    assert not remaining or all(p.stat().st_size == 0 for p in remaining)


def test_prof_stop_bounded_when_server_never_recovers(mock_server, tmp_path):
    """A permanently-dead server must not hang prof.stop() forever."""
    handler = mock_server["handler"]
    handler.accepting = False  # dead from the start of pushing
    # start_run itself will fail — use a live one first, then kill.
    handler.accepting = True
    r = Remote(mock_server["url"], flush_hz=5.0,
               buffer_dir=tmp_path / "buf")
    r.start_run("t", "H100", {})
    handler.accepting = False

    for i in range(10): r.push_sample(_sample(i=0))
    t0 = time.time()
    r.end_run()
    # Internal cap is 15s + join timeout ≤ 20s. Give slack.
    assert time.time() - t0 < 25.0
