"""Top-level ergonomic API: `gpuprof.start` / `gpuprof.profile`."""
import sqlite3
import time

import gpuprof


def test_profile_context_manager_starts_and_stops(tmp_path):
    db = str(tmp_path / "cm.db")
    with gpuprof.profile("cm-test", db_path=db, host_sampling=False):
        pass
    # Run row exists and has ended_at populated (i.e. stop was called).
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT name, ended_at FROM runs WHERE name='cm-test'"
    ).fetchone()
    conn.close()
    assert row and row[1] is not None


def test_profile_stops_on_exception(tmp_path):
    db = str(tmp_path / "err.db")
    try:
        with gpuprof.profile("err-test", db_path=db, host_sampling=False):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # Run should still be marked ended even after an exception.
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT ended_at FROM runs WHERE name='err-test'"
    ).fetchone()
    conn.close()
    assert row and row[0] is not None


def test_start_returns_running_profiler(tmp_path):
    db = str(tmp_path / "s.db")
    prof = gpuprof.start("s-test", db_path=db, host_sampling=False)
    try:
        with prof.step(0) as s:
            with s.phase("forward"):
                time.sleep(0.005)
    finally:
        prof.stop()

    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    assert n == 1


def test_default_run_name_shape():
    from gpuprof import _default_run_name
    name = _default_run_name()
    # <something>-<YYYYMMDD-HHMMSS>
    parts = name.rsplit("-", 2)
    assert len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6


def test_top_level_reexports_integrations():
    # Downstream code should be able to do `from gpuprof import
    # LightningCallback` without touching the .integrations subpath.
    assert hasattr(gpuprof, "LightningCallback")
    assert hasattr(gpuprof, "HFTrainerCallback")
    assert hasattr(gpuprof, "wrap_deepspeed_engine")
