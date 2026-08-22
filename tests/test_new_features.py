"""A1 (W&B) + A2 (regression) + A3 (webhook) + B3 (cost projection)."""
import json
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import gpuprof


# ==== A1 — W&B integration ==============================================

def test_wandb_no_active_run_is_noop():
    """attach_wandb without a wandb.run must not crash — it silently
    no-ops so users can call it unconditionally in code."""
    class _FakeProf:
        def on_step(self, fn): self.step = fn
        def on_end(self, fn): self.end = fn
        _run_id = None
        _local = None
    # Force the "no active run" branch by faking wandb.
    fake = type(sys)("wandb")
    fake.run = None
    sys.modules["wandb"] = fake
    try:
        assert gpuprof.attach_wandb(_FakeProf()) is False
    finally:
        del sys.modules["wandb"]


def test_wandb_step_and_end_callbacks_registered(tmp_path):
    """When wandb.run is present, attach_wandb registers a step
    callback (logs metrics) and an end callback (writes summary)."""
    logged = []
    summary = {}

    class _FakeWandbRun:
        def __init__(self): self.summary = summary

    fake = type(sys)("wandb")
    fake.run = _FakeWandbRun()
    fake.log = lambda payload, step=None, commit=None: logged.append((step, payload))
    sys.modules["wandb"] = fake
    try:
        db = str(tmp_path / "wb.db")
        with gpuprof.profile("wb-test", db_path=db, host_sampling=False,
                              auto=False, summary=False, wandb=True) as prof:
            with prof.step(0) as s:
                with s.phase("forward"):
                    time.sleep(0.001)
                s.record(loss=0.42, tokens=32)
        # step callback → wandb.log called with gpuprof/ metrics
        assert any("gpuprof/step_time_ms" in p for _, p in logged), logged
        assert any(p.get("gpuprof/loss") == 0.42 for _, p in logged), logged
        # end callback → summary populated
        assert any(k.startswith("gpuprof/") for k in summary), summary
    finally:
        del sys.modules["wandb"]


# ==== A2 — Regression detection ========================================

def _seed_run(db, name, step_time_s, rank=None):
    """Create a completed run with N steps of a given duration."""
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO runs(name, started_at, ended_at, gpu_name, "
        "rank, meta_json) VALUES (?, ?, ?, 'MockGPU', ?, '{}')",
        (name, time.time(), time.time() + 1, rank),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    for i in range(20):
        conn.execute(
            "INSERT INTO steps(run_id, step, t_start, t_end) VALUES (?,?,?,?)",
            (run_id, i, i * step_time_s, (i + 1) * step_time_s),
        )
    conn.commit()
    conn.close()
    return run_id


def test_regression_fires_when_slower_than_baseline(tmp_path):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    db = str(tmp_path / "reg.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    # Two baseline runs: 100 ms/step each. New run: 150 ms/step.
    _seed_run(db, "baseline", 0.100)
    _seed_run(db, "baseline", 0.105)
    current = _seed_run(db, "baseline", 0.150)

    r = analyze(db, current)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Regression" in titles, titles
    assert "baseline" in titles


def test_regression_quiet_when_within_noise(tmp_path):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    db = str(tmp_path / "reg.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    # Three runs at ~100 ms — current within 5%.
    _seed_run(db, "baseline", 0.100)
    _seed_run(db, "baseline", 0.100)
    current = _seed_run(db, "baseline", 0.104)

    r = analyze(db, current)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Regression" not in titles, titles


def test_regression_needs_at_least_two_priors(tmp_path):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    db = str(tmp_path / "reg.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    # One prior baseline — not enough for a stable comparison.
    _seed_run(db, "baseline", 0.100)
    current = _seed_run(db, "baseline", 0.200)

    r = analyze(db, current)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Regression" not in titles


# ==== A3 — Webhook alerts ==============================================

class _WebhookRecorder(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        type(self).received.append(json.loads(self.rfile.read(n)) if n else {})
        self.send_response(200); self.end_headers()

    def log_message(self, *a, **kw): pass


@pytest.fixture
def webhook_server():
    _WebhookRecorder.received = []
    srv = HTTPServer(("127.0.0.1", 0), _WebhookRecorder)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"http://127.0.0.1:{port}", "handler": _WebhookRecorder}
    srv.shutdown()


def test_webhook_posted_on_run_end(webhook_server, tmp_path):
    """Generic webhook receives a JSON body with the insight list."""
    db = str(tmp_path / "wh.db")
    with gpuprof.profile("wh-test", db_path=db, host_sampling=False,
                          auto=False, summary=False,
                          webhook=webhook_server["url"]):
        pass
    time.sleep(0.2)  # give the POST a beat
    received = webhook_server["handler"].received
    assert received, "webhook not posted"
    body = received[0]
    assert body["kind"] == "gpuprof.end_of_run"
    assert body["name"] == "wh-test"
    assert "insights" in body


def test_slack_webhook_uses_slack_payload_shape(webhook_server, tmp_path):
    """URLs matching hooks.slack.com produce Slack-native
    `attachments` + `text` instead of the generic schema."""
    # Fake the URL host by making a Slack-shaped POST manually via
    # the alerts module — the recorder doesn't care what host it is.
    from gpuprof.alerts import post_end_of_run_alert
    db = str(tmp_path / "sl.db")
    with gpuprof.profile("sl-test", db_path=db, host_sampling=False,
                          auto=False, summary=False):
        pass
    # Rewrite: post_end_of_run_alert takes url, we test the payload
    # generator directly:
    from gpuprof.alerts import _build_payload
    fake_report = {
        "summary": {"name": "sl-test", "n_steps": 10,
                    "mfu": 0.35, "train_s": 12.0},
        "insights": [
            {"severity": "high", "title": "big issue",
             "recommendation": "do the fix"},
        ],
    }
    slack = _build_payload("https://hooks.slack.com/services/xyz",
                           fake_report)
    generic = _build_payload("https://example.com/webhook", fake_report)
    assert "attachments" in slack and "text" in slack
    assert "kind" in generic and generic["kind"] == "gpuprof.end_of_run"


def test_webhook_failure_never_kills_run(tmp_path):
    """A dead webhook URL must NOT throw out of profile()."""
    db = str(tmp_path / "wh-dead.db")
    # Should complete cleanly — the webhook POST fails silently.
    with gpuprof.profile("wh-dead", db_path=db, host_sampling=False,
                          auto=False, summary=False,
                          webhook="http://127.0.0.1:1"):  # nothing listens
        pass


# ==== B3 — Cost projection =============================================

def test_cost_projection_fires_with_estimate(tmp_path):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    db = str(tmp_path / "cost.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    # 20 steps × 100 ms = 2 s of MockGPU time. At $12.29/hr → ~$0.007.
    rid = _seed_run(db, "cost-test", 0.100)
    r = analyze(db, rid)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "cost" in titles.lower(), titles


def test_cost_projection_uses_env_override(tmp_path, monkeypatch):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    monkeypatch.setenv("GPUPROF_GPU_RATE", "5.00")
    db = str(tmp_path / "cost-env.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()
    rid = _seed_run(db, "cost-env", 3.6)   # 20 × 3.6 s = 72 s wall
    r = analyze(db, rid)
    # Cost = 72s / 3600 × $5 = $0.10 — check the rate override was
    # applied (not the H100 default).
    cost_line = [i for i in r["insights"] if "cost" in i["title"].lower()][0]
    assert cost_line["evidence"]["hourly_rate"] == 5.00


def test_cost_projection_none_when_unknown_gpu(tmp_path):
    from gpuprof.insights import analyze
    from gpuprof.store import apply_schema
    db = str(tmp_path / "cost-unk.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO runs(name, started_at, ended_at, gpu_name, meta_json) "
        "VALUES ('c', ?, ?, 'Very Custom GPU 9000', '{}')",
        (time.time(), time.time() + 1),
    )
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("INSERT INTO steps(run_id, step, t_start, t_end) "
                 "VALUES (?, 0, 0, 1)", (rid,))
    conn.commit(); conn.close()
    r = analyze(db, rid)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "cost" not in titles.lower(), titles
