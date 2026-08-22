"""The server-free flow.

- `gpuprof.profile()` works without any server running.
- The summary block prints to stdout at end of run.
- `gpuprof insights` with no run_id picks the latest run.
- `dashboard=True` spawns an in-process server on a free port.
"""
import os
import re
import socket
import subprocess
import sys
import time

import pytest

import gpuprof


def test_profile_writes_local_db_without_a_server(tmp_path, monkeypatch):
    """Zero-server smoke: profile() → local SQLite → data landed."""
    # Isolate from any GPUPROF_SERVER the parent shell might have set.
    monkeypatch.delenv("GPUPROF_SERVER", raising=False)
    db = str(tmp_path / "solo.db")
    with gpuprof.profile("solo-run", db_path=db,
                          host_sampling=False, summary=False):
        pass
    import sqlite3
    conn = sqlite3.connect(db)
    name = conn.execute("SELECT name FROM runs").fetchone()[0]
    ended = conn.execute("SELECT ended_at FROM runs").fetchone()[0]
    conn.close()
    assert name == "solo-run"
    assert ended is not None       # stop was called, no server was needed


def test_summary_prints_headline_and_insights(tmp_path, capsys):
    with gpuprof.profile("summary-run", db_path=str(tmp_path / "s.db"),
                          host_sampling=False, summary=True):
        pass  # empty loop — still prints "No major bottlenecks detected"
    out = capsys.readouterr().out
    assert "gpuprof · summary-run" in out
    assert "Insights" in out
    assert "gpuprof insights" in out  # points user at the full report


def test_insights_cli_defaults_to_latest_run(tmp_path):
    """`gpuprof insights` with no run_id picks the latest one."""
    db = str(tmp_path / "cli.db")
    # Create three runs so "latest" is a meaningful choice.
    for name in ("a", "b", "c"):
        with gpuprof.profile(name, db_path=db, host_sampling=False,
                              summary=False):
            pass
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "insights", db],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr
    # Latest run is 'c'.
    assert "run 3: 'c'" in r.stdout


def test_insights_cli_no_db_argument_defaults_to_gpuprof_db(tmp_path, monkeypatch):
    """From `gpuprof insights` alone (no args), the CLI should point
    at ./gpuprof.db in the CWD."""
    monkeypatch.chdir(tmp_path)
    # Create ./gpuprof.db by profiling.
    with gpuprof.profile("here", host_sampling=False, summary=False):
        pass
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "insights"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, r.stderr
    assert "run 1: 'here'" in r.stdout


def test_insights_cli_errors_helpfully_on_missing_db(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "gpuprof", "insights",
         str(tmp_path / "does-not-exist.db")],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0
    assert "no DB" in r.stderr


def test_dashboard_true_spawns_local_server(tmp_path, monkeypatch, capsys):
    """dashboard=True prints a http URL and the server actually answers."""
    monkeypatch.delenv("GPUPROF_SERVER", raising=False)
    db = str(tmp_path / "d.db")
    url = None
    with gpuprof.profile("dash", db_path=db, host_sampling=False,
                          summary=False, dashboard=True):
        # Grab the URL from stdout that profile() printed on entry.
        out = capsys.readouterr().out
        m = re.search(r"http://127\.0\.0\.1:\d+", out)
        assert m, out
        url = m.group(0)
        # Hit the server: /api/whoami is a cheap health check that works
        # regardless of auth (returns 'none' when viewer auth is off).
        import urllib.request
        with urllib.request.urlopen(url + "/api/whoami", timeout=2) as resp:
            body = resp.read().decode()
        assert "auth" in body
