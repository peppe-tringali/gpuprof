"""CPU-bound / I/O-bound / cold-cache dataloader diagnosis.

Unit-tests the three new rules against synthetic contexts, plus an
end-to-end that pushes host samples into the store and runs analyze.
"""
import sqlite3
import time

from gpuprof.insights import (
    Ctx,
    rule_cpu_bound_dataloader,
    rule_io_bound_dataloader,
    rule_cold_cache,
    rule_cache_thrashing,
    rule_prefetch_queue_starved,
    rule_worker_imbalance,
    rule_host_memory_pressure,
    analyze,
)
from gpuprof.store import Store, apply_schema
from gpuprof.host_sampler import HostSample


def _ctx(**over):
    base = dict(
        run_id=1, name="t", gpu_name="H100", meta={}, arch=None,
        n_steps=100, train_s=10.0, step_s=0.1,
        step_times=[0.1] * 100, phase_avg_s={},
        inter_step_gap_s=0.03,      # 30% stall → dataloader-stall active
        tokens_total=None, measured_flops=None, peak_tflops=None,
        mfu=None, mfu_notes=[], avg_sm_util=0.3,
        max_mem=None, mem_total=None, max_temp=None,
        sm_clock_deficit_pct=None,
        pcie_avg_rx_gbps=None, pcie_avg_tx_gbps=None,
        pcie_max_rx_gbps=None,
        traces=[], trace_windows=[],
    )
    base.update(over)
    return Ctx(**base)


# ---- cpu-bound ---------------------------------------------------------

def test_cpu_bound_fires_on_stall_plus_pegged_cpu():
    r = rule_cpu_bound_dataloader(
        _ctx(avg_cpu_pct=92.0, peak_cpu_pct=99.0, n_cpus=16,
             peak_disk_read_bps=10e6)  # disk idle
    )
    assert r is not None
    assert "CPU-bound" in r["title"]


def test_cpu_bound_quiet_without_stall():
    # No stall → skip regardless of CPU
    c = _ctx(inter_step_gap_s=0.001, avg_cpu_pct=99.0)
    assert rule_cpu_bound_dataloader(c) is None


def test_cpu_bound_defers_to_io_when_disk_also_busy():
    # High CPU + high disk read → probably not compute-bound; the I/O
    # rule handles this shape.
    c = _ctx(avg_cpu_pct=90.0, peak_disk_read_bps=700e6)
    assert rule_cpu_bound_dataloader(c) is None


# ---- io-bound ----------------------------------------------------------

def test_io_bound_fires_on_high_read_bandwidth():
    r = rule_io_bound_dataloader(
        _ctx(avg_cpu_pct=40.0, peak_disk_read_bps=800e6,
             peak_disk_iops=1000)
    )
    assert r is not None
    assert "I/O-bound" in r["title"]


def test_io_bound_fires_on_high_iops_small_files():
    # Many small reads with CPU headroom is the classic "random-
    # access on tiny files" shape — WebDataset territory.
    r = rule_io_bound_dataloader(
        _ctx(avg_cpu_pct=50.0, peak_disk_read_bps=50e6,
             peak_disk_iops=8000)
    )
    assert r is not None


def test_io_bound_quiet_when_disk_idle():
    c = _ctx(avg_cpu_pct=95.0, peak_disk_read_bps=1e6, peak_disk_iops=10)
    assert rule_io_bound_dataloader(c) is None


# ---- cold cache --------------------------------------------------------

def test_cold_cache_fires_on_early_disk_and_step_speedup():
    c = _ctx(first_third_disk_read_bps=400e6,
             last_third_disk_read_bps=30e6,       # 13× ratio
             first_third_step_s=0.20,
             last_third_step_s=0.10)              # 2× speedup
    r = rule_cold_cache(c)
    assert r is not None
    assert "Cold-cache" in r["title"]


def test_cold_cache_quiet_when_disk_never_hot():
    c = _ctx(first_third_disk_read_bps=10e6,
             last_third_disk_read_bps=5e6,
             first_third_step_s=0.20,
             last_third_step_s=0.10)
    assert rule_cold_cache(c) is None


def test_cold_cache_quiet_when_step_time_flat():
    # Disk-read dropped but step time didn't improve — probably
    # something else, not cache warmup.
    c = _ctx(first_third_disk_read_bps=400e6,
             last_third_disk_read_bps=30e6,
             first_third_step_s=0.10,
             last_third_step_s=0.10)
    assert rule_cold_cache(c) is None


# ---- cache thrashing ---------------------------------------------------

def test_cache_thrashing_fires_when_cache_full_and_disk_hot():
    c = _ctx(
        peak_mem_cached_bytes=int(50e9),   # 50 GB cache
        mem_total=int(64e9),                # of 64 GB RAM (~78%)
        avg_disk_read_bps=200e6,            # sustained 200 MB/s read
        first_third_disk_read_bps=210e6,
        last_third_disk_read_bps=190e6,     # steady — not cold-cache
    )
    r = rule_cache_thrashing(c)
    assert r is not None and "thrashing" in r["title"].lower()


def test_cache_thrashing_defers_to_cold_cache_on_ramp():
    c = _ctx(
        peak_mem_cached_bytes=int(50e9), mem_total=int(64e9),
        avg_disk_read_bps=200e6,
        first_third_disk_read_bps=400e6,
        last_third_disk_read_bps=20e6,      # dropped off — cold cache
    )
    assert rule_cache_thrashing(c) is None


def test_cache_thrashing_quiet_when_cache_small():
    c = _ctx(peak_mem_cached_bytes=int(1e9), mem_total=int(64e9),
             avg_disk_read_bps=200e6)
    assert rule_cache_thrashing(c) is None


# ---- prefetch queue starved -------------------------------------------

def test_prefetch_queue_starved_fires():
    c = _ctx(prefetch_wait_p50_s=0.005, prefetch_wait_p95_s=0.080,
             frac_steps_with_starve=0.60)
    r = rule_prefetch_queue_starved(c)
    assert r is not None and "prefetch" in r["title"].lower()
    assert r["severity"] == "high"           # >50% starved → high


def test_prefetch_queue_healthy_when_short_wait():
    c = _ctx(prefetch_wait_p50_s=0.001, prefetch_wait_p95_s=0.005,
             frac_steps_with_starve=0.03)
    assert rule_prefetch_queue_starved(c) is None


# ---- worker imbalance -------------------------------------------------

def test_worker_imbalance_fires_when_one_worker_hot():
    # 4 workers, avg 200% total → fair share = 50%. Hottest at 250%.
    c = _ctx(avg_children_cpu_pct=200.0, peak_max_child_cpu_pct=250.0,
             avg_n_children=4.0)
    r = rule_worker_imbalance(c)
    assert r is not None and "imbalance" in r["title"].lower()


def test_worker_imbalance_quiet_when_balanced():
    c = _ctx(avg_children_cpu_pct=400.0, peak_max_child_cpu_pct=110.0,
             avg_n_children=4.0)   # fair share 100, peak barely over
    assert rule_worker_imbalance(c) is None


def test_worker_imbalance_needs_multiple_workers():
    c = _ctx(avg_children_cpu_pct=90.0, peak_max_child_cpu_pct=95.0,
             avg_n_children=1.0)
    assert rule_worker_imbalance(c) is None


# ---- host memory pressure / swap -------------------------------------

def test_host_memory_pressure_fires_on_swap_in():
    c = _ctx(peak_swap_in_bps=5e6, peak_swap_out_bps=1e6,
             min_mem_available_bytes=int(200e6))
    r = rule_host_memory_pressure(c)
    assert r is not None and "swap" in r["title"].lower()


def test_host_memory_pressure_quiet_without_swap():
    c = _ctx(peak_swap_in_bps=0.0, peak_swap_out_bps=0.0,
             min_mem_available_bytes=int(30e9))
    assert rule_host_memory_pressure(c) is None


# ---- end-to-end: host samples flow through the store ------------------

def test_host_samples_persist_and_analyze_can_see_them(tmp_path):
    db = str(tmp_path / "h.db")
    store = Store(db)
    store.start_run("t", "H100", "{}")

    # Fake a "CPU-bound" run: pegged CPU, negligible disk, big
    # inter-step gaps.
    t0 = time.time()
    for i in range(20):
        store.push_step({"step": i, "t_start": i * 0.15,
                         "t_end": i * 0.15 + 0.05,
                         "inter_step_gap_s": 0.10,
                         "forward_s": 0.03, "backward_s": 0.02})
        store.push_host_sample(HostSample(
            t=t0 + i * 0.15,
            cpu_percent=92.0, cpu_max_percent=99.0, n_cpus=8,
            mem_used_bytes=int(30e9), mem_total_bytes=int(64e9),
            mem_cached_bytes=int(5e9), mem_available_bytes=int(30e9),
            swap_in_bps=0.0, swap_out_bps=0.0,
            disk_read_bps=5e6, disk_write_bps=0, disk_iops=10,
            children_cpu_percent=700.0, max_child_cpu_percent=100.0,
            n_children=8,
        ))
    store.end_run()

    # Confirm rows landed.
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM host_samples").fetchone()[0]
    conn.close()
    assert n == 20

    # Analyzer should fire the CPU-bound rule.
    r = analyze(db, 1)
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "CPU-bound" in titles
