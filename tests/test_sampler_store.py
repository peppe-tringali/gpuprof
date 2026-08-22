import os
import sqlite3
import time

from gpuprof.sampler import Sampler, _detect_gpu_indices, _MockBackend, _NvmlBackend
from gpuprof.store import Store, apply_schema


def test_detect_gpu_indices_respects_env_var(monkeypatch):
    monkeypatch.setenv("GPUPROF_MOCK", "1")
    monkeypatch.setenv("GPUPROF_MOCK_GPUS", "4")
    assert _detect_gpu_indices() == [0, 1, 2, 3]


def test_mock_backend_emits_valid_samples():
    b = _MockBackend(0)
    s = b.sample()
    assert 0.0 <= s.sm_util <= 1.0
    assert s.mem_used_bytes > 0
    assert s.mem_total_bytes > s.mem_used_bytes
    assert s.gpu_index == 0
    assert b.name  # backend carries a device name for the run header


def test_sampler_emits_per_gpu_per_tick():
    collected = []
    s = Sampler(on_sample=collected.append,
                gpu_indices=[0, 1, 2], hz=20.0)
    s.start()
    time.sleep(0.4)
    s.stop()
    # 3 GPUs × ~20 Hz × 0.4s ≈ 24 samples; at least one per GPU.
    indices = {sample.gpu_index for sample in collected}
    assert indices == {0, 1, 2}
    # At least a few per index — allow slack for scheduling jitter.
    per_idx = {i: sum(1 for x in collected if x.gpu_index == i)
               for i in indices}
    assert all(v >= 3 for v in per_idx.values()), per_idx


def test_schema_migrates_old_db(tmp_path):
    path = str(tmp_path / "old.db")
    # Simulate an old DB without the new columns.
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE runs(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                          started_at REAL NOT NULL, ended_at REAL,
                          gpu_name TEXT, meta_json TEXT);
        CREATE TABLE steps(run_id INTEGER NOT NULL, step INTEGER NOT NULL,
                           t_start REAL, t_end REAL,
                           dataloader_wait_s REAL, forward_s REAL,
                           backward_s REAL, optimizer_s REAL,
                           loss REAL, tokens INTEGER, flops REAL);
        CREATE TABLE samples(run_id INTEGER NOT NULL, t REAL);
    """)
    conn.commit()
    apply_schema(conn)
    # New columns must exist now.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
    assert "group_id" in cols and "rank" in cols and "world_size" in cols
    cols = [r[1] for r in conn.execute("PRAGMA table_info(steps)").fetchall()]
    assert "inter_step_gap_s" in cols and "comm_s" in cols
    conn.close()


def test_store_roundtrip_samples_steps_traces(tmp_db):
    from gpuprof.sampler import Sample
    s = Store(tmp_db)
    run_id = s.start_run("t", "H100", "{}", group_id="g", rank=0, world_size=2)
    for i in range(30):
        s.push_sample(Sample(
            t=time.time(), gpu_index=0, sm_util=0.5,
            mem_used_bytes=10**9, mem_total_bytes=80*10**9,
            power_w=300, temp_c=65, sm_clock_mhz=1500, mem_clock_mhz=1500,
            pcie_rx_kbps=1000, pcie_tx_kbps=500,
        ))
        s.push_step({"step": i, "t_start": i*0.1, "t_end": i*0.1+0.09,
                     "inter_step_gap_s": 0.01, "forward_s": 0.03,
                     "backward_s": 0.05, "optimizer_s": 0.005,
                     "comm_s": 0.002, "loss": 1.0/(i+1), "tokens": 128,
                     "flops": 1e12, "dataloader_wait_s": 0.0})
    s.push_trace({"step": 5, "kernels": [{"name": "aten::mm"}]})
    s.end_run()

    conn = sqlite3.connect(tmp_db)
    n_s = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    n_st = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    n_tr = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    grp = conn.execute("SELECT group_id, rank, world_size FROM runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    assert n_s == 30 and n_st == 30 and n_tr == 1
    assert grp == ("g", 0, 2)
