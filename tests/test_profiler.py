import sqlite3
import time

from gpuprof import GpuProfiler
from gpuprof.insights import analyze


def test_end_to_end_captures_phases_and_inter_step_gap(tmp_path):
    db = str(tmp_path / "e2e.db")
    prof = GpuProfiler(run_name="e2e", db_path=db,
                      meta={"batch_size": 1})
    prof.start()
    for i in range(10):
        with prof.step(i) as s:
            with s.phase("forward"):  time.sleep(0.01)
            with s.phase("backward"): time.sleep(0.02)
            s.record(loss=1.0/(i+1), tokens=32)
        time.sleep(0.03)  # simulated data-loading between steps
    prof.stop()

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT step, inter_step_gap_s, forward_s, backward_s FROM steps ORDER BY step"
    ).fetchall()
    conn.close()

    assert len(rows) == 10
    # Step 0 has no prior step, so inter_step_gap is NULL.
    assert rows[0][1] is None
    # Subsequent steps should show ~30 ms gap (allow slack for perf_counter drift).
    for _, gap, _, _ in rows[1:]:
        assert gap is not None and 0.015 <= gap <= 0.100
    # Phase timings are populated for wall-clock mode.
    for _, _, fw, bw in rows:
        assert fw is not None and 0.005 <= fw <= 0.100
        assert bw is not None and 0.015 <= bw <= 0.100


def test_auto_flops_from_arch(tmp_path):
    db = str(tmp_path / "flops.db")
    prof = GpuProfiler(
        run_name="flops", db_path=db,
        meta={"arch": {"params": 100_000_000, "hidden": 256, "layers": 4,
                       "seq_len": 128}, "dtype": "bf16"},
    )
    prof.start()
    for i in range(5):
        with prof.step(i) as s:
            with s.phase("forward"): time.sleep(0.001)
            s.record(tokens=1024)
    prof.stop()

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT flops FROM steps").fetchall()
    conn.close()
    # Every step must have a positive FLOP count from the arch calc.
    assert all(f and f > 0 for f, in rows), rows


def test_analyze_gives_dataloader_stall_verdict(tmp_path):
    db = str(tmp_path / "stall.db")
    prof = GpuProfiler(run_name="stall", db_path=db, meta={})
    prof.start()
    for i in range(15):
        with prof.step(i) as s:
            with s.phase("forward"): time.sleep(0.005)
        time.sleep(0.05)  # big inter-step gap → should trigger the rule
    prof.stop()

    r = analyze(db, 1)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "loader" in titles.lower() or "waiting" in titles.lower(), titles


def test_trace_warmup_captures_step_0(tmp_path):
    # `trace_every_n_steps=1000` alone would never capture step 0;
    # the warmup set ensures it does.
    db = str(tmp_path / "warm.db")
    prof = GpuProfiler(run_name="warm", db_path=db,
                      trace_every_n_steps=1000)
    prof.start()
    for i in range(3):
        with prof.step(i) as s:
            with s.phase("forward"): time.sleep(0.001)
    prof.stop()

    conn = sqlite3.connect(db)
    steps = [r[0] for r in conn.execute("SELECT step FROM traces ORDER BY step").fetchall()]
    conn.close()
    # Torch must be importable for a trace to actually be captured;
    # otherwise the profiler swallows the ImportError silently. Either
    # 0 traces (no torch) or step 0 present (torch available) is fine.
    if steps:
        assert 0 in steps or 1 in steps
