"""Tests for items 1–12 landed in the finalization pass:

- Version dedup via importlib.metadata            (#7)
- RateLimiter memory cap + reap                   (#2)
- Buffer file sweep at profile() start            (#3)
- selfcheck                                       (#4)
- gc retention CLI                                (#11)
- Static HTML report                              (#12)
- Framework-adapter hint                          (#9)
- Multi-tenant ApiKeySet                          (#10)
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


# ── (#7) version dedup ────────────────────────────────────────────────

def test_version_comes_from_installed_metadata():
    import gpuprof
    assert re.match(r"^\d+\.\d+\.\d+", gpuprof.__version__), gpuprof.__version__
    # If the metadata lookup fails we fall back to "0.0.0+unknown".
    # Test that we're not in the fallback under normal installed conditions.
    assert not gpuprof.__version__.endswith("+unknown"), (
        "installed metadata missing — check pyproject.toml packaging"
    )


# ── (#2) rate-limiter memory bounds ───────────────────────────────────

def test_rate_limiter_evicts_empty_keys():
    """After the sliding window expires, empty deques must be
    removed so the dict doesn't grow one entry per unique IP."""
    from gpuprof.server.auth import RateLimiter
    rl = RateLimiter(max_per_window=5, window_s=0.05)
    for i in range(10):
        rl.allow(f"ip-{i}")
    time.sleep(0.10)  # wait for the window to expire
    rl.allow("trigger")  # any allow() call reaps
    # All prior IPs should have expired and been evicted.
    assert len(rl._hits) == 1
    assert "trigger" in rl._hits


def test_rate_limiter_caps_at_max_keys(monkeypatch):
    from gpuprof.server.auth import RateLimiter
    # Very small cap so the test doesn't need to burn CPU making 10k IPs.
    monkeypatch.setattr(RateLimiter, "_MAX_KEYS", 20)
    rl = RateLimiter(max_per_window=100, window_s=0.05)
    for i in range(200):
        rl.allow(f"ip-{i}")
    time.sleep(0.10)
    rl.allow("post-window")  # trigger reap
    # After the reap, only "post-window" should remain (all others expired).
    assert len(rl._hits) <= 20


# ── (#3) buffer sweep at profile() start ──────────────────────────────

class _CountingHandler(BaseHTTPRequestHandler):
    posts: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
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
    _CountingHandler.posts = []
    srv = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield {"url": f"http://127.0.0.1:{port}"}
    srv.shutdown()


def test_orphaned_buffer_files_sweep_on_start(mock_server, tmp_path):
    """A prior run's buffer file must be auto-drained when a new
    Remote starts — users shouldn't have to remember `gpuprof drain`."""
    from gpuprof.remote import Remote
    buf_dir = tmp_path / "buf"
    buf_dir.mkdir()
    # Simulate an orphaned buffer file from a previous run (run-42).
    (buf_dir / "run-42.jsonl").write_text(
        json.dumps({
            "samples": [{"t": 1.0, "gpu_index": 0, "sm_util": 0.5,
                         "mem_used_bytes": 1, "mem_total_bytes": 2,
                         "power_w": 100, "temp_c": 60,
                         "sm_clock_mhz": 1000, "mem_clock_mhz": 500,
                         "pcie_rx_kbps": 0, "pcie_tx_kbps": 0}],
            "steps": [], "traces": [], "comm_events": [],
            "trace_windows": [], "host_samples": [],
        }) + "\n"
    )
    r = Remote(mock_server["url"], flush_hz=5.0, buffer_dir=buf_dir)
    r.start_run("new-run", "H100", {})
    r.end_run()
    # The orphan should have been drained (→ /api/runs/42/ingest hit)
    # and the file deleted.
    paths = [p for p, _ in _CountingHandler.posts]
    assert "/api/runs/42/ingest" in paths
    assert not (buf_dir / "run-42.jsonl").exists()


# ── (#4) selfcheck ────────────────────────────────────────────────────

def test_selfcheck_runs_and_prints_status():
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "selfcheck"],
        capture_output=True, text=True, timeout=15,
    )
    # In this venv NVML/torch may be missing, so warnings are OK.
    # What must hold: it exits and prints [ok]/[warn]/[fail] rows.
    assert r.returncode in (0, 1), r.stdout + r.stderr
    assert "gpuprof import" in r.stdout
    assert "sqlite" in r.stdout


def test_selfcheck_filters_by_only():
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "selfcheck", "--only", "python"],
        capture_output=True, text=True, timeout=15,
    )
    assert "python" in r.stdout
    # Non-selected checks shouldn't print — nvml e.g. would be absent.
    assert "nvml" not in r.stdout


# ── (#11) gc retention ────────────────────────────────────────────────

def _seed_completed_run(db, name, seconds_ago):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO runs(name, started_at, ended_at, gpu_name, meta_json) "
        "VALUES (?, ?, ?, 'MockGPU', '{}')",
        (name, time.time() - seconds_ago, time.time() - seconds_ago + 10),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("INSERT INTO steps(run_id, step, t_start, t_end) "
                 "VALUES (?, 0, 0, 1)", (run_id,))
    conn.execute("INSERT INTO samples(run_id, t, gpu_index, sm_util) "
                 "VALUES (?, 0, 0, 0.5)", (run_id,))
    conn.commit(); conn.close()
    return run_id


def test_gc_older_than_deletes_stale(tmp_path):
    from gpuprof.store import apply_schema
    db = str(tmp_path / "gc.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    fresh = _seed_completed_run(db, "r-fresh", seconds_ago=10)
    stale = _seed_completed_run(db, "r-stale", seconds_ago=100)

    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "gc", "--db", db,
         "--older-than", "60s"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr

    conn = sqlite3.connect(db)
    ids = {r[0] for r in conn.execute("SELECT id FROM runs").fetchall()}
    n_steps = conn.execute(
        "SELECT COUNT(*) FROM steps WHERE run_id=?", (stale,)
    ).fetchone()[0]
    conn.close()
    assert fresh in ids and stale not in ids
    assert n_steps == 0  # child rows deleted too


def test_gc_keep_last_prunes_per_name(tmp_path):
    from gpuprof.store import apply_schema
    db = str(tmp_path / "gc.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()

    ids = [_seed_completed_run(db, "baseline", seconds_ago=n * 10)
           for n in range(5)]

    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "gc", "--db", db,
         "--keep-last", "2"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr

    conn = sqlite3.connect(db)
    kept = [r[0] for r in conn.execute(
        "SELECT id FROM runs ORDER BY started_at DESC"
    ).fetchall()]
    conn.close()
    # The two most recent runs must survive.
    assert len(kept) == 2
    assert set(kept) == set(ids[:2])   # ids seeded newest-first-index


def test_gc_dry_run_deletes_nothing(tmp_path):
    from gpuprof.store import apply_schema
    db = str(tmp_path / "gc.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()
    _seed_completed_run(db, "a", seconds_ago=100)
    _seed_completed_run(db, "b", seconds_ago=100)

    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "gc", "--db", db,
         "--older-than", "60s", "--dry-run"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr
    assert "dry-run" in r.stdout
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    assert n == 2


# ── (#12) HTML report ─────────────────────────────────────────────────

def test_report_produces_self_contained_html(tmp_path):
    from gpuprof.store import apply_schema
    db = str(tmp_path / "rep.db")
    conn = sqlite3.connect(db); apply_schema(conn); conn.close()
    _seed_completed_run(db, "report-test", seconds_ago=5)

    out = str(tmp_path / "report.html")
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "report", db, "1", "--out", out],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr
    with open(out) as f:
        html = f.read()
    assert "<!doctype html>" in html.lower()
    assert "report-test" in html
    # Must be truly self-contained — no external CSS/JS links.
    assert "<link" not in html and "<script src=" not in html
    # Some content sanity: at least one <svg> for the sparklines,
    # the insights section, and the phase-averages table.
    assert "<svg" in html
    assert "insights" in html.lower()


# ── (#9) framework-adapter hint ───────────────────────────────────────

def test_framework_hint_fires_when_lightning_imported(capfd, monkeypatch):
    """If pytorch_lightning is present in sys.modules AND the user
    is on the auto path, `profile()` prints a stderr hint pointing
    at LightningCallback. (Manual `auto=False` users get no hint —
    they've explicitly opted out of monkey-patching.)"""
    import gpuprof
    monkeypatch.setitem(sys.modules, "pytorch_lightning",
                         type(sys)("pytorch_lightning"))
    monkeypatch.setattr(gpuprof, "_FRAMEWORK_HINTED", False)
    # auto=True fires the hint. torch isn't installed on this venv,
    # so AutoInstrumenter.start() returns False and it no-ops — the
    # hint still prints via `_hint_framework_adapter` first.
    with gpuprof.profile("hint-test", auto=True, summary=False,
                          host_sampling=False):
        pass
    _, err = capfd.readouterr()
    assert "Lightning" in err and "LightningCallback" in err


def test_framework_hint_only_fires_once(capfd, monkeypatch):
    import gpuprof
    monkeypatch.setitem(sys.modules, "pytorch_lightning",
                         type(sys)("pytorch_lightning"))
    monkeypatch.setattr(gpuprof, "_FRAMEWORK_HINTED", False)
    with gpuprof.profile("h1", auto=True, summary=False, host_sampling=False): pass
    with gpuprof.profile("h2", auto=True, summary=False, host_sampling=False): pass
    _, err = capfd.readouterr()
    assert err.count("LightningCallback") == 1


# ── (#10) multi-tenant scoping via ApiKeySet ─────────────────────────

def test_apikeyset_bare_secret_maps_to_default_scope():
    from gpuprof.server.auth import ApiKeySet
    ks = ApiKeySet(["secret-A", "secret-B"])
    ident = ks.resolve("secret-A")
    assert ident == {"user": None, "project": "default"}
    assert ks.resolve("nope") is None


def test_apikeyset_scoped_token_maps_to_user_project():
    from gpuprof.server.auth import ApiKeySet
    ks = ApiKeySet(["alice:teamA:s3cret", "bob:teamB:pw"])
    assert ks.resolve("alice:teamA:s3cret") == {
        "user": "alice", "project": "teamA",
    }
    assert ks.resolve("bob:teamB:pw") == {
        "user": "bob", "project": "teamB",
    }


def test_apikeyset_disabled_returns_default_identity():
    """No keys configured = write-auth off; anon writes land under
    the default scope."""
    from gpuprof.server.auth import ApiKeySet
    ks = ApiKeySet([])
    assert ks.enabled() is False
    assert ks.resolve(None) == {"user": None, "project": "default"}


def test_server_records_run_under_token_scope(tmp_path):
    """End-to-end via TestClient: pushing under `alice:teamA:X` binds
    the run's `owner_user` and `project` columns."""
    from fastapi.testclient import TestClient
    from gpuprof.server.app import create_app
    from gpuprof.server.store import ServerStore
    db = str(tmp_path / "mt.db")
    store = ServerStore(db)
    app = create_app(store=store, api_keys=["alice:teamA:secret"],
                     db_path_for_insights=db)
    with TestClient(app) as c:
        r = c.post("/api/runs", json={"name": "mt-run"},
                   headers={"X-API-Key": "alice:teamA:secret"})
        assert r.status_code == 200
    # Row should reflect the token scope.
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT owner_user, project FROM runs WHERE name='mt-run'"
    ).fetchone()
    conn.close()
    assert row == ("alice", "teamA")
