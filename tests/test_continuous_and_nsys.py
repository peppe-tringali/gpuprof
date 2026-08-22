"""Continuous kernel aggregation + nsys integration."""
import json
import sqlite3
import threading
import time

import pytest

from gpuprof.continuous import ContinuousProfiler
from gpuprof.insights import Ctx, rule_kernel_drift
from gpuprof.store import Store, apply_schema


# ---- continuous profiler thread ----------------------------------------

def test_continuous_profiler_emits_windows(tmp_path):
    """The rotating profiler thread must produce time-adjacent
    non-overlapping windows and call the on_window callback for each."""
    try:
        import torch  # noqa
    except ImportError:
        pytest.skip("torch required for the continuous profiler")

    windows = []
    lock = threading.Lock()

    def on_w(w):
        with lock:
            windows.append(w)

    # torch.profiler start/stop carries ~50-200 ms of overhead per
    # rotation depending on the environment, so keep windows large and
    # expectations loose — we're proving the mechanism, not the rate.
    p = ContinuousProfiler(on_window=on_w, window_s=0.4, cuda=False,
                            epoch=time.perf_counter())
    p.start()
    time.sleep(2.0)
    p.stop()

    assert len(windows) >= 1, "no windows emitted"
    # Adjacent windows should not overlap and should have monotone starts.
    for i in range(1, len(windows)):
        assert windows[i]["t_start_rel"] >= windows[i - 1]["t_end_rel"] - 0.05


# ---- trace_windows persist through the store ---------------------------

def test_store_persists_trace_windows(tmp_path):
    db = str(tmp_path / "tw.db")
    s = Store(db)
    s.start_run("t", "H100", "{}")
    s.push_trace_window({
        "t_start_rel": 0.0, "t_end_rel": 1.0,
        "kernels": [{"name": "aten::mm", "self_device_us": 5000.0, "count": 3}],
    })
    s.push_trace_window({
        "t_start_rel": 1.0, "t_end_rel": 2.0,
        "kernels": [{"name": "aten::mm", "self_device_us": 12000.0, "count": 3}],
    })
    s.end_run()

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT t_start_rel, t_end_rel, kernels_json FROM trace_windows "
        "ORDER BY t_start_rel"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0][0] == 0.0 and rows[0][1] == 1.0
    kernels = json.loads(rows[1][2])
    assert kernels[0]["name"] == "aten::mm"


# ---- kernel-drift rule -------------------------------------------------

def _ctx_with_windows(windows):
    return Ctx(
        run_id=1, name="t", gpu_name="H100", meta={}, arch=None,
        n_steps=0, train_s=None, step_s=None, step_times=[],
        phase_avg_s={}, peak_tflops=None,
        trace_windows=windows,
    )


def test_kernel_drift_fires_when_kernel_gets_slower():
    # First half: aten::mm cheap (500 μs/window). Second half: expensive (5000).
    n = 30
    first = [{"t_start_rel": i, "t_end_rel": i + 1,
              "kernels": [{"name": "aten::mm", "self_device_us": 500.0,
                           "self_cpu_us": 0.0}]} for i in range(n // 2)]
    second = [{"t_start_rel": i, "t_end_rel": i + 1,
               "kernels": [{"name": "aten::mm", "self_device_us": 5000.0,
                            "self_cpu_us": 0.0}]}
              for i in range(n // 2, n)]
    r = rule_kernel_drift(_ctx_with_windows(first + second))
    assert r is not None and "aten::mm" in r["title"]


def test_kernel_drift_quiet_when_stable():
    n = 30
    ws = [{"t_start_rel": i, "t_end_rel": i + 1,
           "kernels": [{"name": "aten::mm", "self_device_us": 3000.0,
                        "self_cpu_us": 0.0}]} for i in range(n)]
    assert rule_kernel_drift(_ctx_with_windows(ws)) is None


def test_kernel_drift_needs_enough_windows():
    # Under 20 windows → skip (too little signal).
    ws = [{"t_start_rel": i, "t_end_rel": i + 1,
           "kernels": [{"name": "aten::mm", "self_device_us": 3000.0}]}
          for i in range(10)]
    assert rule_kernel_drift(_ctx_with_windows(ws)) is None


# ---- nsys capture context manager --------------------------------------

def test_nsys_capture_is_safe_without_torch():
    """The context manager must not crash if torch isn't installed —
    it just yields `False`."""
    from gpuprof.nsys import nsys_capture
    with nsys_capture() as active:
        # active is True only under an actual nsys profile run; on a
        # dev box calling cudaProfilerStart it will still return True
        # if torch is available, but doesn't error.
        assert active in (True, False)


# ---- nsys sqlite import ------------------------------------------------

def _fake_nsys_sqlite(path):
    """Build a minimal nsys-export-shaped SQLite DB with two kernels."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, demangledName INTEGER);
        INSERT INTO StringIds(id, value) VALUES
            (1, 'aten::mm'), (2, 'aten::gelu');
    """)
    # Two seconds of "run" — 5 kernels per second, interleaved.
    now = 1_000_000_000_000
    rows = []
    for s in range(2):
        for i in range(5):
            t_start = now + (s * 1_000_000_000) + i * 100_000_000
            rows.append((t_start, t_start + 50_000_000, 1))   # aten::mm 50 ms
            rows.append((t_start + 60_000_000, t_start + 65_000_000, 2))  # gelu 5 ms
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start, end, demangledName) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_nsys_import_populates_trace_windows(tmp_path):
    from gpuprof.nsys import import_nsys_sqlite
    nsys_db = str(tmp_path / "nsys.sqlite")
    gp_db = str(tmp_path / "gp.db")
    _fake_nsys_sqlite(nsys_db)

    # Create a gpuprof run to import into.
    conn = sqlite3.connect(gp_db)
    apply_schema(conn)
    conn.execute(
        "INSERT INTO runs(name, started_at, gpu_name, meta_json) "
        "VALUES ('t', 0, 'H100', '{}')"
    )
    conn.commit()
    run_id = conn.execute("SELECT id FROM runs").fetchone()[0]
    conn.close()

    n = import_nsys_sqlite(nsys_db, gp_db, run_id, bucket_s=1.0)
    assert n >= 1

    conn = sqlite3.connect(gp_db)
    rows = conn.execute(
        "SELECT t_start_rel, t_end_rel, kernels_json FROM trace_windows "
        "WHERE run_id=? ORDER BY t_start_rel", (run_id,),
    ).fetchall()
    conn.close()
    assert rows
    kernels = json.loads(rows[0][2])
    names = [k["name"] for k in kernels]
    assert "aten::mm" in names
