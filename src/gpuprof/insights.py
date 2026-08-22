"""Offline analysis over a completed run.

Builds a context of headline metrics + samples + traces, then runs a
list of rules against it. Each rule is a small function that inspects
the context and either returns None (didn't fire) or an insight dict.

Rules live at module level so they're easy to read, add, and test.
Cross-run analysis (rank skew across a group) lives in `analyze_group`.

Runnable::

    python -m gpuprof.insights gpuprof.db 1
    python -m gpuprof.insights gpuprof.db --group my-training-group
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from .flops import (
    TransformerArch, arch_from_meta, peak_tflops,
    transformer_flops, transformer_flops_per_step,
)


# ------------------------------------------------------------------
# Context that rules read from
# ------------------------------------------------------------------

@dataclass
class Ctx:
    run_id: int
    name: str
    gpu_name: str
    meta: dict
    arch: Optional[TransformerArch]
    n_steps: int
    train_s: Optional[float]
    step_s: Optional[float]                  # avg step time in seconds
    step_times: list[float] = field(default_factory=list)
    phase_avg_s: dict = field(default_factory=dict)
    inter_step_gap_s: Optional[float] = None  # avg
    tokens_total: Optional[int] = None
    measured_flops: Optional[float] = None
    peak_tflops: Optional[float] = None
    mfu: Optional[float] = None
    mfu_notes: list = field(default_factory=list)
    avg_sm_util: Optional[float] = None
    max_mem: Optional[int] = None
    mem_total: Optional[int] = None
    max_temp: Optional[float] = None
    sm_clock_deficit_pct: Optional[float] = None  # 1 - avg/max
    pcie_avg_rx_gbps: Optional[float] = None
    pcie_avg_tx_gbps: Optional[float] = None
    pcie_max_rx_gbps: Optional[float] = None
    traces: list = field(default_factory=list)   # list of {step, kernels}
    trace_windows: list = field(default_factory=list)  # continuous-mode aggregates
    # host-side (psutil) aggregates for CPU-bound vs I/O-bound diagnosis:
    avg_cpu_pct: Optional[float] = None       # mean of aggregate cpu%
    peak_cpu_pct: Optional[float] = None      # max sampled aggregate
    peak_cpu_max_core_pct: Optional[float] = None  # hottest core, any time
    peak_disk_read_bps: Optional[float] = None
    avg_disk_read_bps: Optional[float] = None
    peak_disk_iops: Optional[float] = None
    n_cpus: Optional[int] = None
    first_third_disk_read_bps: Optional[float] = None
    last_third_disk_read_bps: Optional[float] = None
    first_third_step_s: Optional[float] = None
    last_third_step_s: Optional[float] = None
    # v0.6 additions — cache-warmth, swap pressure, per-worker imbalance
    avg_mem_cached_bytes: Optional[int] = None
    peak_mem_cached_bytes: Optional[int] = None
    min_mem_available_bytes: Optional[int] = None
    peak_swap_in_bps: Optional[float] = None
    peak_swap_out_bps: Optional[float] = None
    avg_children_cpu_pct: Optional[float] = None
    peak_max_child_cpu_pct: Optional[float] = None
    avg_n_children: Optional[float] = None
    # Prefetch-queue starvation stats derived from per-step dataloader_wait
    prefetch_wait_p50_s: Optional[float] = None
    prefetch_wait_p95_s: Optional[float] = None
    frac_steps_with_starve: Optional[float] = None  # % of steps with wait > threshold
    rank: Optional[int] = None
    world_size: Optional[int] = None
    group_id: Optional[str] = None
    # For cross-run regression detection: the analyzer passes the DB
    # path into the Ctx so rule_regression can query the history.
    db_path: Optional[str] = None


# ------------------------------------------------------------------
# Rules — each returns None or {severity, title, recommendation, evidence}
# ------------------------------------------------------------------

def rule_dataloader_stall(c: Ctx) -> Optional[dict]:
    """Prefer inter-step gap (captures the real wait between step
    contexts) but also fire on explicit dataloader_wait if the user
    wrapped the batch fetch inside a step."""
    gap = c.inter_step_gap_s or 0.0
    wait = c.phase_avg_s.get("dataloader_wait") or 0.0
    total_stall = max(gap, wait)
    if not c.step_s or not total_stall:
        return None
    frac = total_stall / c.step_s
    if frac < 0.15:
        return None
    src = "inter-step gap" if gap >= wait else "dataloader_wait phase"
    sev = "high" if frac > 0.30 else "medium"
    return {
        "severity": sev,
        "title": f"{frac*100:.0f}% of step time waiting on the loader ({src})",
        "recommendation": (
            "GPU is idle waiting on host. Increase DataLoader "
            "num_workers, set pin_memory=True and persistent_workers=True, "
            "and use non_blocking=True on .to(device). Consider "
            "prefetch_factor>=2 and shifting augmentations onto GPU. "
            "If PCIe throughput is also high, the bottleneck is bandwidth, "
            "not workers."
        ),
        "evidence": {"inter_step_gap_s": gap,
                     "dataloader_wait_s": wait,
                     "step_s": c.step_s},
    }


def rule_low_mfu(c: Ctx) -> Optional[dict]:
    if c.mfu is None or c.mfu >= 0.30:
        return None
    dtype = (c.meta.get("dtype") or "").lower() or "unknown-dtype"
    detail = ""
    if c.mfu_notes:
        detail = "  Notes on the estimate: " + "; ".join(c.mfu_notes) + "."
    return {
        "severity": "medium",
        "title": f"MFU {c.mfu*100:.1f}% of {c.peak_tflops:.0f} TFLOPs {dtype} peak",
        "recommendation": (
            "Well below typical 40–60% MFU for transformer training. "
            "Check: batch size / grad accumulation, FlashAttention or SDPA "
            "fused kernel, bf16/fp16 dtype, torch.compile, and avoid "
            "CPU↔GPU sync (`.item()`/`.cpu()`) in the hot path." + detail
        ),
        "evidence": {"mfu": c.mfu, "peak_tflops": c.peak_tflops,
                     "measured_flops_per_step": c.measured_flops},
    }


def rule_memory_pressure(c: Ctx) -> Optional[dict]:
    if not (c.max_mem and c.mem_total): return None
    frac = c.max_mem / c.mem_total
    if frac < 0.90: return None
    return {
        "severity": "high" if frac > 0.97 else "medium",
        "title": f"Peak memory {frac*100:.0f}% of {c.mem_total/1e9:.0f} GB",
        "recommendation": (
            "OOM risk. Reduce batch size, enable gradient checkpointing, "
            "use bf16 activations, offload optimizer state (ZeRO/FSDP), "
            "or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to "
            "reduce fragmentation."
        ),
        "evidence": {"max_mem_bytes": c.max_mem,
                     "mem_total_bytes": c.mem_total},
    }


def rule_kernel_launch_overhead(c: Ctx) -> Optional[dict]:
    if not c.step_s: return None
    phase_sum = sum(v or 0 for v in c.phase_avg_s.values())
    if phase_sum <= 0: return None
    gap = c.step_s - phase_sum
    if gap / c.step_s < 0.10: return None
    return {
        "severity": "low",
        "title": f"{gap/c.step_s*100:.0f}% of step unaccounted "
                 "(kernel-launch / Python overhead)",
        "recommendation": (
            "Step is slower than its measured phases sum. Suggests many "
            "small ops or Python-side overhead. torch.compile / CUDA "
            "graphs commonly recover this. Confirm with a torch.profiler "
            "trace of the same window."
        ),
        "evidence": {"step_s": c.step_s, "phase_sum_s": phase_sum},
    }


def rule_compilation_warmup(c: Ctx) -> Optional[dict]:
    """First step (or two) is >3× the steady-state median → warmup /
    JIT compile / cuBLAS heuristic search."""
    if len(c.step_times) < 8: return None
    early = c.step_times[:2]
    steady = c.step_times[5:15] if len(c.step_times) >= 15 else c.step_times[3:]
    if not steady: return None
    median_steady = statistics.median(steady)
    max_early = max(early)
    if median_steady <= 0 or max_early < 3.0 * median_steady:
        return None
    return {
        "severity": "low",
        "title": (f"First step {max_early*1000:.0f} ms vs steady-state "
                  f"{median_steady*1000:.0f} ms — warmup / JIT compile"),
        "recommendation": (
            "Kernel autotuning, torch.compile, or cuBLAS heuristics likely "
            "ran during the first step. Exclude steps 0–2 from throughput "
            "measurements. If step ~1 is also slow, torch.compile may be "
            "recompiling — check `TORCH_LOGS=recompiles`."
        ),
        "evidence": {"first_two_max_s": max_early,
                     "median_steady_s": median_steady},
    }


def rule_gradient_checkpointing_detected(c: Ctx) -> Optional[dict]:
    """Backward should be ~2× forward without checkpointing; ~3× with.
    Flag when the ratio implies checkpointing but the user didn't
    declare it (or vice versa)."""
    fw = c.phase_avg_s.get("forward") or 0.0
    bw = c.phase_avg_s.get("backward") or 0.0
    if fw <= 0 or bw <= 0: return None
    ratio = bw / fw
    declared = bool(c.arch and c.arch.grad_checkpoint)
    if ratio > 2.7 and not declared:
        return {
            "severity": "low",
            "title": f"backward/forward = {ratio:.1f}× — possible "
                     "gradient checkpointing recompute",
            "recommendation": (
                "Backward is much larger than 2× forward. If you enabled "
                "gradient checkpointing, declare `grad_checkpoint=True` in "
                "the arch so MFU accounts for the extra forward. If you "
                "didn't, investigate: recomputation in autograd, custom "
                "backward, or attention without a fused kernel."
            ),
            "evidence": {"forward_s": fw, "backward_s": bw, "ratio": ratio},
        }
    if declared and ratio < 2.3:
        return {
            "severity": "low",
            "title": f"backward/forward = {ratio:.1f}× but "
                     "grad_checkpoint=True was declared",
            "recommendation": (
                "The declared checkpointing isn't reflected in phase "
                "timing — the checkpoint layer may not be wrapping the "
                "expensive parts. Check `torch.utils.checkpoint.checkpoint` "
                "is called on the transformer blocks, not just the head."
            ),
            "evidence": {"forward_s": fw, "backward_s": bw, "ratio": ratio},
        }
    return None


def rule_sdpa_suboptimal(c: Ctx) -> Optional[dict]:
    """Look for evidence that attention is running as separate bmm+softmax
    rather than a fused kernel (flash/efficient/cudnn)."""
    if not c.traces: return None
    all_names = [k["name"].lower() for t in c.traces for k in t.get("kernels", [])]
    if not all_names: return None
    fused = any("flash" in n or "efficient_attention" in n or "cudnn_attention" in n
                for n in all_names)
    if fused: return None
    # Heuristic: if we see softmax and bmm both in the top kernels, and
    # attention isn't fused, suggest.
    has_softmax = any("softmax" in n for n in all_names)
    has_bmm = any("bmm" in n or "matmul" in n for n in all_names)
    if not (has_softmax and has_bmm): return None
    return {
        "severity": "medium",
        "title": "Attention appears unfused (softmax + bmm in top kernels)",
        "recommendation": (
            "No FlashAttention/efficient-attention/cudnn-attention kernel "
            "in the trace, but softmax + bmm are hot. Use "
            "`torch.nn.functional.scaled_dot_product_attention` (auto-selects "
            "the best backend), or install a Flash Attention build for "
            "your GPU. Common causes: custom attention code, casting to "
            "fp32 inside attention, or requesting `attn_mask` in a shape "
            "that disables the fused kernel."
        ),
        "evidence": {"top_kernels": all_names[:5]},
    }


def rule_thermal_throttling(c: Ctx) -> Optional[dict]:
    if c.max_temp is None or c.sm_clock_deficit_pct is None: return None
    if c.max_temp < 82 or c.sm_clock_deficit_pct < 0.15: return None
    return {
        "severity": "medium",
        "title": (f"Likely thermal throttling — peak {c.max_temp:.0f}°C, "
                  f"SM clock down {c.sm_clock_deficit_pct*100:.0f}%"),
        "recommendation": (
            "SM clocks are meaningfully below the observed max while "
            "temperature is high. Improve airflow, reduce power cap only "
            "if you must, or check for a stuck fan. Sustained throttling "
            "silently caps your throughput."
        ),
        "evidence": {"max_temp_c": c.max_temp,
                     "sm_clock_deficit": c.sm_clock_deficit_pct},
    }


def rule_pcie_saturation(c: Ctx) -> Optional[dict]:
    if c.pcie_max_rx_gbps is None: return None
    # PCIe 4.0 x16 ≈ 31.5 GB/s theoretical; 20+ GB/s sustained is
    # bandwidth-bound territory. PCIe 5.0 x16 ≈ 63 GB/s.
    if c.pcie_max_rx_gbps < 20.0: return None
    return {
        "severity": "medium",
        "title": (f"PCIe RX peaked at {c.pcie_max_rx_gbps:.1f} GB/s "
                  "(bandwidth-bound territory)"),
        "recommendation": (
            "H2D traffic is saturating PCIe. If this coincides with high "
            "dataloader-wait, more workers won't help — batches are too "
            "big to shuttle. Options: fewer/smaller CPU-side augmentations "
            "and do them on GPU; NVLink/NCCL peer copies for multi-GPU "
            "data sharding; keep tensors on GPU across steps."
        ),
        "evidence": {"pcie_max_rx_gbps": c.pcie_max_rx_gbps,
                     "pcie_avg_rx_gbps": c.pcie_avg_rx_gbps},
    }


def rule_small_batch(c: Ctx) -> Optional[dict]:
    """High-ish SM util but low MFU → math kernels launching too small
    to fill the SMs. Batch-size / accumulation typically fixes it."""
    if c.mfu is None or c.avg_sm_util is None: return None
    if c.mfu >= 0.25 or c.avg_sm_util < 0.60: return None
    return {
        "severity": "low",
        "title": (f"Low arithmetic intensity — SM util "
                  f"{c.avg_sm_util*100:.0f}% but MFU only {c.mfu*100:.1f}%"),
        "recommendation": (
            "GPU is busy but not doing much math. Kernels are likely too "
            "small to fill the SMs. Try: larger microbatch, fuse activations "
            "(GELU+matmul), enable torch.compile, or promote dtype to bf16 "
            "so matmul throughput doubles."
        ),
        "evidence": {"mfu": c.mfu, "avg_sm_util": c.avg_sm_util},
    }


def rule_regression(c: Ctx) -> Optional[dict]:
    """Compare this run against the last N runs with the same
    `run_name`. Fires on step-time regression >10% vs the median of
    previous runs. This is the "did my change help?" signal — the
    core reason to run a profiler on repeated experiments.

    Uses the same DB the analyzer is reading from; no server round
    trip. Distributed runs are compared per-rank (same rank + name).
    """
    if not c.db_path or not c.run_id or not c.name or not c.step_s:
        return None
    conn = sqlite3.connect(c.db_path)
    try:
        # Match by name AND rank so DDP runs compare rank-to-rank —
        # otherwise a rank-3 straggler run would pollute the rank-0
        # baseline comparison.
        rank_predicate = "AND rank IS ?" if c.rank is not None else "AND rank IS NULL"
        params = (c.name, c.run_id) + ((c.rank,) if c.rank is not None else ())
        rows = conn.execute(
            f"""SELECT id FROM runs
                WHERE name = ? AND id != ? AND ended_at IS NOT NULL
                {rank_predicate}
                ORDER BY started_at DESC LIMIT 5""",
            params,
        ).fetchall()
        if len(rows) < 2:
            return None  # need at least 2 previous runs for a stable median
        prev_step_avgs = []
        for (rid,) in rows:
            row = conn.execute(
                "SELECT AVG(t_end - t_start) FROM steps WHERE run_id=?",
                (rid,),
            ).fetchone()
            if row and row[0]:
                prev_step_avgs.append(row[0])
    finally:
        conn.close()
    if len(prev_step_avgs) < 2:
        return None
    prev_median = statistics.median(prev_step_avgs)
    if prev_median <= 0:
        return None
    delta = (c.step_s - prev_median) / prev_median
    if delta < 0.10:
        return None  # noise / actually faster
    rank_str = f"rank {c.rank}, " if c.rank is not None else ""
    return {
        "severity": "high" if delta > 0.25 else "medium",
        "title": (f"Regression vs last {len(prev_step_avgs)} runs of "
                  f"'{c.name}' ({rank_str}median): step time "
                  f"{c.step_s*1000:.1f} ms vs {prev_median*1000:.1f} ms "
                  f"({delta*100:.0f}% slower)"),
        "recommendation": (
            "Compare this run's changes against the recent baseline. "
            "The other rules in this report will localize the cause "
            "(new kernel dominance, dataloader shape change, DDP "
            "topology). If the change was intentional (bigger model, "
            "longer context) update your baseline set — this rule "
            "expects apples-to-apples."
        ),
        "evidence": {"current_step_s": c.step_s,
                     "prev_median_step_s": prev_median,
                     "n_previous_runs": len(prev_step_avgs)},
    }


# --- Cost projection (B3) --------------------------------------------

# Reference: AWS on-demand USD/hour per GPU as of 2024 — divided by
# GPUs per instance so `rate × n_gpus × hours` is the whole-run cost.
# Adjust to your fleet's actual price via `GPUPROF_GPU_RATE=8.50`.
_GPU_HOURLY_USD = {
    "H100":     12.29,   # p5.4xlarge / 8
    "H200":     15.00,   # p5e.4xlarge / 8 (approx)
    "A100":      4.10,   # p4d.24xlarge / 8
    "V100":      3.06,   # p3.2xlarge
    "L40":       1.60,
    "L4":        0.71,
    "T4":        0.53,
    "RTX 4090":  0.44,   # RunPod on-demand estimate
    "RTX 3090":  0.34,
    "B200":     30.00,   # placeholder
    "MockGPU":  12.29,   # so demos have a number
}


def _gpu_hourly_rate(gpu_name: Optional[str]) -> Optional[float]:
    import os
    override = os.environ.get("GPUPROF_GPU_RATE")
    if override:
        try: return float(override)
        except ValueError: pass
    if not gpu_name: return None
    for needle, rate in _GPU_HOURLY_USD.items():
        if needle.lower() in gpu_name.lower():
            return rate
    return None


def rule_cost_projection(c: Ctx) -> Optional[dict]:
    """Always-fires info card: what did this run cost, and how much
    was wasted according to MFU?

    Uses AWS on-demand USD/hour as a reference. Override with
    `GPUPROF_GPU_RATE=<usd>` if your fleet's price is different.
    """
    if not c.train_s or not c.gpu_name: return None
    rate = _gpu_hourly_rate(c.gpu_name)
    if rate is None: return None
    n_gpus = len(c.meta.get("gpu_indices") or [0])
    if n_gpus == 0: n_gpus = 1
    hours = c.train_s / 3600.0
    cost = hours * n_gpus * rate

    # Waste model: if MFU is 20%, 80% of the FLOP capacity was
    # unused — attributed to the same wall-clock $. Only meaningful
    # when we have a real MFU number.
    waste_str = ""
    severity = "low"
    if c.mfu is not None and c.mfu > 0:
        waste = cost * (1 - c.mfu)
        waste_str = f"; ~${waste:.2f} of that is idle capacity (MFU {c.mfu*100:.1f}%)"
        # Bump to medium/high on high waste, so long low-MFU runs float
        # up in the insight list.
        if waste > 100.0:  severity = "medium"
        if waste > 1000.0: severity = "high"
    per_gpu_str = f"{n_gpus}× GPU @ ${rate:.2f}/h" if n_gpus > 1 else f"${rate:.2f}/h"
    return {
        "severity": severity,
        "title": (f"Estimated cost: ${cost:.2f} "
                  f"({hours*60:.0f} min · {per_gpu_str}){waste_str}"),
        "recommendation": (
            "Cost estimate uses reference on-demand rates. Override with "
            "`GPUPROF_GPU_RATE=<usd_per_hour>` if your rate differs. "
            "If MFU is low, the other insights above name the "
            "specific fix that would recover the idle capacity."
        ),
        "evidence": {"hours": hours, "n_gpus": n_gpus,
                     "hourly_rate": rate, "total_cost_usd": cost,
                     "mfu": c.mfu},
    }


def rule_kernel_drift(c: Ctx) -> Optional[dict]:
    """Continuous-mode rule: if the kernel mix changed materially
    between the first and second half of the run, point at which
    kernel got worse. Catches "throughput dropped at t=17 min" cases
    that per-step traces miss."""
    if len(c.trace_windows) < 20:
        return None
    n = len(c.trace_windows)
    first_half = c.trace_windows[: n // 2]
    second_half = c.trace_windows[n // 2:]

    def _agg(windows):
        # Aggregate self-device us per kernel across windows.
        totals: dict[str, float] = {}
        for w in windows:
            for k in w.get("kernels", []):
                totals[k["name"]] = totals.get(k["name"], 0.0) + \
                    (k.get("self_device_us") or k.get("self_cpu_us") or 0)
        return totals

    a, b = _agg(first_half), _agg(second_half)
    # Normalize by window count so we compare rates, not totals.
    len_a = max(1, len(first_half))
    len_b = max(1, len(second_half))
    worst_name, worst_delta = None, 0.0
    for name, tb in b.items():
        ta = a.get(name, 0.0)
        rate_a = ta / len_a
        rate_b = tb / len_b
        # Require both a big absolute jump (in per-window microseconds)
        # AND a large multiplicative jump — small kernels going 10×
        # from noise floor aren't interesting.
        delta = rate_b - rate_a
        if rate_a < 100.0: continue        # skip < 0.1 ms/window kernels
        if delta / max(1.0, rate_a) < 0.5: continue
        if delta > worst_delta:
            worst_delta = delta
            worst_name = name
    if worst_name is None:
        return None
    return {
        "severity": "medium",
        "title": (f"Kernel drift: {worst_name} added "
                  f"{worst_delta/1000:.1f} ms/window in the second half"),
        "recommendation": (
            "This kernel's share of device time grew during the run — "
            "throughput slowdown localized to one kernel. Common causes: "
            "memory fragmentation forcing slower allocator paths, "
            "checkpoint save/reload changing cuBLAS heuristics, or a "
            "warmed-up autotune cache being invalidated. `nsys profile "
            "--capture-range=cudaProfilerApi` around a slow window "
            "and gpuprof.nsys_import will confirm the timeline."
        ),
        "evidence": {"kernel": worst_name,
                     "delta_us_per_window": worst_delta},
    }


def _has_dataloader_stall(c: Ctx) -> bool:
    gap = c.inter_step_gap_s or 0.0
    wait = c.phase_avg_s.get("dataloader_wait") or 0.0
    return bool(c.step_s and max(gap, wait) / c.step_s > 0.15)


def rule_prefetch_queue_starved(c: Ctx) -> Optional[dict]:
    """Attributed prefetch wait (from `wrap_dataloader`) is high on
    many steps → the worker pool isn't keeping the queue full.
    Distinct from generic dataloader-stall: this specifically points
    at the prefetch_factor knob."""
    if c.frac_steps_with_starve is None or c.prefetch_wait_p95_s is None:
        return None
    if c.frac_steps_with_starve < 0.20:      # fewer than 20% starved steps
        return None
    if c.prefetch_wait_p95_s < 0.020:        # p95 wait under 20ms — fine
        return None
    return {
        "severity": "high" if c.frac_steps_with_starve > 0.50 else "medium",
        "title": (f"Prefetch queue starved on "
                  f"{c.frac_steps_with_starve*100:.0f}% of steps "
                  f"(p95 wait {c.prefetch_wait_p95_s*1000:.0f} ms)"),
        "recommendation": (
            "Workers can't keep up with training. First knob: raise "
            "`prefetch_factor` on the DataLoader (default 2 — try 4 or 8) "
            "so each worker builds up more slack. If that doesn't fix "
            "it, workers themselves are slow — check the CPU-bound / "
            "I/O-bound rules for direction. `persistent_workers=True` "
            "avoids the per-epoch worker-restart penalty."
        ),
        "evidence": {"frac_starved": c.frac_steps_with_starve,
                     "p50_wait_s": c.prefetch_wait_p50_s,
                     "p95_wait_s": c.prefetch_wait_p95_s},
    }


def rule_cache_thrashing(c: Ctx) -> Optional[dict]:
    """Page cache is at its memory-limited size, yet disk reads keep
    coming — working set doesn't fit in RAM. Distinct from cold-cache
    (which warms up and then quiets down)."""
    if (c.peak_mem_cached_bytes is None
            or c.mem_total is None
            or c.avg_disk_read_bps is None):
        return None
    if c.mem_total <= 0: return None
    cache_frac = c.peak_mem_cached_bytes / c.mem_total
    if cache_frac < 0.40:
        # Cache never got big — not thrashing, just not being used.
        return None
    # Sustained disk reads (not just startup) mean we're missing.
    if c.avg_disk_read_bps < 100e6:
        return None
    # If the cold-cache rule would fire, defer to it — this rule is
    # about the *sustained* pattern, not the ramp-up.
    if (c.first_third_disk_read_bps and c.last_third_disk_read_bps
            and c.first_third_disk_read_bps
                > 3.0 * c.last_third_disk_read_bps):
        return None
    return {
        "severity": "medium",
        "title": (f"Cache thrashing: page cache "
                  f"{c.peak_mem_cached_bytes/1e9:.0f} GB "
                  f"({cache_frac*100:.0f}% of RAM), still reading "
                  f"{c.avg_disk_read_bps/1e6:.0f} MB/s"),
        "recommendation": (
            "Working set is larger than host RAM — every epoch you "
            "re-read data that was evicted. Options: shard the dataset "
            "across nodes, keep only a randomized subset resident, or "
            "add RAM. WebDataset with sequential reads costs less per "
            "miss than random-access on many small files."
        ),
        "evidence": {"peak_mem_cached_bytes": c.peak_mem_cached_bytes,
                     "avg_disk_read_bps": c.avg_disk_read_bps,
                     "mem_total_bytes": c.mem_total},
    }


def rule_worker_imbalance(c: Ctx) -> Optional[dict]:
    """One worker is doing much more work than the average — sharding
    is uneven, or a `worker_init_fn` gave one worker the wrong seed
    stride."""
    if (c.peak_max_child_cpu_pct is None
            or c.avg_children_cpu_pct is None
            or c.avg_n_children is None
            or c.avg_n_children < 2):
        return None
    # Expected fair share if all workers do equal work.
    fair_share = c.avg_children_cpu_pct / c.avg_n_children
    if fair_share <= 0 or c.peak_max_child_cpu_pct < 30:
        return None
    # A single worker peaking at >2.5× the fair share is a red flag.
    if c.peak_max_child_cpu_pct < 2.5 * fair_share:
        return None
    return {
        "severity": "medium",
        "title": (f"Dataloader-worker imbalance — hottest worker at "
                  f"{c.peak_max_child_cpu_pct:.0f}% vs fair share "
                  f"{fair_share:.0f}% across {c.avg_n_children:.0f} workers"),
        "recommendation": (
            "One worker is carrying a disproportionate share. Common "
            "causes: uneven sharding (`Sampler` gives one worker "
            "larger examples), a `worker_init_fn` that doesn't stride "
            "the RNG per worker, or one worker's dataset chunk sitting "
            "on a slower storage tier. Log per-worker sample counts "
            "with `torch.utils.data.get_worker_info()`."
        ),
        "evidence": {"peak_max_child_cpu_pct": c.peak_max_child_cpu_pct,
                     "avg_children_cpu_pct": c.avg_children_cpu_pct,
                     "n_workers": c.avg_n_children},
    }


def rule_host_memory_pressure(c: Ctx) -> Optional[dict]:
    """Host is swapping — pipeline is fighting the OS for RAM. Every
    swap-in eats memory bandwidth."""
    if c.peak_swap_in_bps is None or c.peak_swap_in_bps < 1e6:
        return None                # 1 MB/s of swap-in is a real signal
    return {
        "severity": "high",
        "title": (f"Host memory pressure — swap-in peaked at "
                  f"{c.peak_swap_in_bps/1e6:.0f} MB/s"),
        "recommendation": (
            "The training process is swapping pages back in from disk "
            "— host RAM is oversubscribed and every swap-in burns "
            "memory bandwidth. Reduce prefetch depth, decrease the "
            "worker pool, or upgrade the host. Data pipelines and "
            "PyTorch's pinned-memory allocator can silently pin more "
            "than you think — check `pinned_memory` in "
            "`nvidia-smi` and `RSS` of workers via `ps`."
        ),
        "evidence": {"peak_swap_in_bps": c.peak_swap_in_bps,
                     "peak_swap_out_bps": c.peak_swap_out_bps,
                     "min_mem_available_bytes": c.min_mem_available_bytes},
    }


def rule_cpu_bound_dataloader(c: Ctx) -> Optional[dict]:
    """Dataloader stall AND CPU pegged → workers are compute-bound.
    Adding more workers won't help; the transforms need to move."""
    if not _has_dataloader_stall(c): return None
    if c.avg_cpu_pct is None: return None
    if c.avg_cpu_pct < 80.0: return None
    if c.peak_disk_read_bps and c.peak_disk_read_bps > 500e6:
        # Disk is also busy — mixed cause, defer to the I/O rule.
        return None
    return {
        "severity": "high",
        "title": (f"Dataloader stall + CPU pegged "
                  f"({c.avg_cpu_pct:.0f}% avg across "
                  f"{c.n_cpus or '?'} cores) — workers are CPU-bound"),
        "recommendation": (
            "Adding more workers won't help — every core is already busy. "
            "Move augmentations to GPU (torchvision.transforms.v2 has a "
            "GPU path; `kornia` is designed for it). Simpler transforms, "
            "pre-decoded samples, or reduced JPEG-decode quality can also "
            "help. If your model isn't yet using bf16/fp16, that will "
            "also unstick the CPU by shrinking per-batch bytes."
        ),
        "evidence": {"avg_cpu_pct": c.avg_cpu_pct,
                     "peak_cpu_pct": c.peak_cpu_pct,
                     "peak_disk_read_bps": c.peak_disk_read_bps,
                     "n_cpus": c.n_cpus},
    }


def rule_io_bound_dataloader(c: Ctx) -> Optional[dict]:
    """Dataloader stall AND disk saturated (or CPU well under 80% and
    IOPS high) → I/O-bound. More workers *might* help if CPU has
    headroom, but the storage tier is the bottleneck."""
    if not _has_dataloader_stall(c): return None
    if c.peak_disk_read_bps is None: return None
    # Two paths to fire: high sustained read bandwidth, or high IOPS
    # (random-access on lots of small files) while CPU has headroom.
    high_bw = c.peak_disk_read_bps > 500e6           # 500 MB/s+
    small_files = (c.peak_disk_iops and c.peak_disk_iops > 5000
                   and (c.avg_cpu_pct or 100) < 80)
    if not (high_bw or small_files):
        return None
    return {
        "severity": "high",
        "title": (f"Dataloader stall + disk hot "
                  f"(peak {c.peak_disk_read_bps/1e6:.0f} MB/s"
                  + (f", {c.peak_disk_iops:.0f} IOPS" if c.peak_disk_iops else "")
                  + ") — I/O-bound"),
        "recommendation": (
            "The workers are waiting on storage, not CPU. Options in "
            "rough order of impact: (1) pack samples with WebDataset "
            "or tarballs to sequential-read them, (2) cache the "
            "training set in RAM (`tmpfs`) or a local NVMe tier if "
            "it's on network storage, (3) pre-decode / pre-augment "
            "and store the result, (4) sharded prefetch with a larger "
            "`prefetch_factor` so slow reads overlap compute."
        ),
        "evidence": {"peak_disk_read_bps": c.peak_disk_read_bps,
                     "avg_disk_read_bps": c.avg_disk_read_bps,
                     "peak_disk_iops": c.peak_disk_iops,
                     "avg_cpu_pct": c.avg_cpu_pct},
    }


def rule_cold_cache(c: Ctx) -> Optional[dict]:
    """Early-run disk read rate much higher than steady-state, AND
    step times drop as the cache warms — classic first-epoch pattern
    on network/HDD storage."""
    if (c.first_third_disk_read_bps is None
            or c.last_third_disk_read_bps is None):
        return None
    if c.first_third_disk_read_bps < 50e6:   # need real disk traffic
        return None
    disk_ratio = (c.first_third_disk_read_bps
                  / max(1.0, c.last_third_disk_read_bps))
    if disk_ratio < 3.0:  # only fire on a clear early-vs-steady gap
        return None
    step_speedup = None
    if (c.first_third_step_s and c.last_third_step_s
            and c.last_third_step_s > 0):
        step_speedup = c.first_third_step_s / c.last_third_step_s
    if step_speedup is not None and step_speedup < 1.3:
        return None                # step time didn't actually improve
    return {
        "severity": "medium",
        "title": (f"Cold-cache pattern: disk read {disk_ratio:.1f}× "
                  "higher early in run than at end"),
        "recommendation": (
            "First-epoch behavior looks cache-cold: heavy disk traffic "
            "early, drops off as the OS page cache warms. Either warm "
            "the cache before your throughput measurements or size RAM "
            "to fit the working set. On network storage, pre-stage the "
            "dataset onto local disk."
        ),
        "evidence": {
            "first_third_disk_read_bps": c.first_third_disk_read_bps,
            "last_third_disk_read_bps":  c.last_third_disk_read_bps,
            "first_third_step_s": c.first_third_step_s,
            "last_third_step_s":  c.last_third_step_s,
        },
    }


def rule_comm_dominant(c: Ctx) -> Optional[dict]:
    comm = c.phase_avg_s.get("comm") or 0.0
    if not c.step_s or comm <= 0: return None
    frac = comm / c.step_s
    if frac < 0.20: return None
    return {
        "severity": "high" if frac > 0.40 else "medium",
        "title": f"NCCL comm is {frac*100:.0f}% of step time",
        "recommendation": (
            "Gradient sync dominates the step. Consider: gradient bucketing "
            "size tuning, `gradient_as_bucket_view=True` on DDP, "
            "overlap-more-aggressively via `no_sync()` for micro-batches, "
            "or switch to FSDP with sharded params. Verify NCCL is using "
            "NVLink not PCIe (`NCCL_DEBUG=INFO`)."
        ),
        "evidence": {"comm_s": comm, "step_s": c.step_s},
    }


def rule_high_step_variance(c: Ctx) -> Optional[dict]:
    """A p99/p50 ratio > 2 is a red flag — jitter is expensive and often
    caused by GC pauses, host stalls, or checkpointing spikes."""
    if len(c.step_times) < 20: return None
    sorted_st = sorted(c.step_times[2:])  # skip warmup
    if len(sorted_st) < 10: return None
    p50 = sorted_st[len(sorted_st) // 2]
    p99 = sorted_st[int(len(sorted_st) * 0.99)]
    if p50 <= 0 or p99 / p50 < 2.0: return None
    return {
        "severity": "low",
        "title": f"Step-time p99/p50 = {p99/p50:.1f}× — high tail latency",
        "recommendation": (
            "Some steps are much slower than the median. Common causes: "
            "logging/checkpoint every N steps (batch them or async them), "
            "Python GC pauses (`gc.disable()` during hot loop), or dataloader "
            "workers getting scheduled off. Check `p99` after excluding "
            "known-slow steps."
        ),
        "evidence": {"p50_s": p50, "p99_s": p99},
    }


def rule_first_step_outlier(c: Ctx) -> Optional[dict]:
    """Different from warmup — this fires when the first step is
    massively outsized (>10x) and is likely obscuring downstream
    metrics if not filtered."""
    if len(c.step_times) < 5: return None
    first = c.step_times[0]
    rest = c.step_times[1:]
    med_rest = statistics.median(rest)
    if med_rest <= 0 or first < 10.0 * med_rest:
        return None
    return {
        "severity": "low",
        "title": (f"Step 0 = {first*1000:.0f} ms vs {med_rest*1000:.0f} ms "
                  "median — filter it from averages"),
        "recommendation": (
            "The averages in the summary above already include step 0. Look "
            "at the phase chart's steady-state region for real throughput. "
            "A first-step multiplier of 10× typically comes from cuBLAS "
            "algorithm search, torch.compile guards, or one-time DDP setup."
        ),
        "evidence": {"first_s": first, "median_rest_s": med_rest},
    }


RULES: list[Callable[[Ctx], Optional[dict]]] = [
    rule_dataloader_stall,
    rule_low_mfu,
    rule_memory_pressure,
    rule_kernel_launch_overhead,
    rule_compilation_warmup,
    rule_first_step_outlier,
    rule_high_step_variance,
    rule_gradient_checkpointing_detected,
    rule_sdpa_suboptimal,
    rule_thermal_throttling,
    rule_pcie_saturation,
    rule_small_batch,
    rule_cpu_bound_dataloader,
    rule_io_bound_dataloader,
    rule_cold_cache,
    rule_cache_thrashing,
    rule_prefetch_queue_starved,
    rule_worker_imbalance,
    rule_host_memory_pressure,
    rule_comm_dominant,
    rule_kernel_drift,
    rule_regression,
    rule_cost_projection,
]


# ------------------------------------------------------------------
# Context builder + analyze
# ------------------------------------------------------------------

def _build_ctx(conn: sqlite3.Connection, run_id: int) -> Ctx:
    row = conn.execute(
        "SELECT name, gpu_name, meta_json, started_at, ended_at, "
        "group_id, rank, world_size FROM runs WHERE id=?", (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id} not found")
    name, gpu_name, meta_json, _, _, group_id, rank, world_size = row
    meta = json.loads(meta_json or "{}")
    arch = arch_from_meta(meta)

    steps = conn.execute(
        """SELECT step, t_start, t_end, inter_step_gap_s, dataloader_wait_s,
                  forward_s, backward_s, optimizer_s, comm_s, tokens, flops
           FROM steps WHERE run_id=? ORDER BY step""",
        (run_id,),
    ).fetchall()
    n_steps = len(steps)

    step_times = [r[2] - r[1] for r in steps if r[1] is not None and r[2] is not None]
    train_s = None
    # `is not None` — a first step with t_start == 0.0 is legitimate
    # and must not be treated as "missing timing" (which would leave
    # train_s at None and hide MFU + cost calculations).
    if steps and steps[0][1] is not None and steps[-1][2] is not None:
        train_s = steps[-1][2] - steps[0][1]

    def _avg(col_idx):
        vals = [r[col_idx] for r in steps if r[col_idx] is not None]
        return (sum(vals) / len(vals)) if vals else None

    step_s = (sum(step_times) / len(step_times)) if step_times else None
    inter = _avg(3)
    phases = {
        "dataloader_wait": _avg(4),
        "forward":         _avg(5),
        "backward":        _avg(6),
        "optimizer":       _avg(7),
        "comm":            _avg(8),
    }

    tokens_sum = sum((r[9] or 0) for r in steps) or None
    flops_sum = sum((r[10] or 0) for r in steps) or None

    # Prefer per-step FLOP records (user or arch-computed). If missing
    # and we have an architecture, compute from tokens sum; if missing
    # everything, fall back to 6·P·T.
    measured_flops = flops_sum
    mfu_notes: list[str] = []
    if not measured_flops and tokens_sum:
        if arch:
            fp = transformer_flops_per_step(arch, tokens_sum)
            measured_flops = fp["flops"]
            mfu_notes.extend(fp["notes"])
        elif meta.get("params"):
            measured_flops = transformer_flops(int(meta["params"]), tokens_sum)
            mfu_notes.append("used 6·P·T — declare arch (hidden, layers, "
                             "seq_len, …) in meta for a correct estimate")

    dtype = meta.get("dtype")
    peak = peak_tflops(gpu_name, dtype) if dtype else None
    mfu = None
    if measured_flops and peak and train_s:
        mfu = (measured_flops / train_s / 1e12) / peak

    samples = conn.execute(
        """SELECT sm_util, mem_used_bytes, mem_total_bytes, power_w, temp_c,
                  sm_clock_mhz, pcie_rx_kbps, pcie_tx_kbps
           FROM samples WHERE run_id=?""",
        (run_id,),
    ).fetchall()
    if samples:
        utils = [s[0] for s in samples if s[0] is not None]
        mems = [s[1] for s in samples if s[1] is not None]
        tots = [s[2] for s in samples if s[2] is not None]
        temps = [s[4] for s in samples if s[4] is not None]
        clocks = [s[5] for s in samples if s[5] is not None]
        rx = [s[6] for s in samples if s[6] is not None]
        tx = [s[7] for s in samples if s[7] is not None]
        avg_util = sum(utils) / len(utils) if utils else None
        max_mem = max(mems) if mems else None
        mem_total = max(tots) if tots else None
        max_temp = max(temps) if temps else None
        clock_deficit = None
        if clocks:
            mx, avg = max(clocks), sum(clocks) / len(clocks)
            if mx > 0:
                clock_deficit = 1.0 - (avg / mx)
        # PCIe throughput is reported in KB/s per NVML; convert to GB/s.
        pcie_avg_rx = (sum(rx) / len(rx) / 1e6) if rx else None
        pcie_avg_tx = (sum(tx) / len(tx) / 1e6) if tx else None
        pcie_max_rx = (max(rx) / 1e6) if rx else None
    else:
        avg_util = max_mem = mem_total = max_temp = clock_deficit = None
        pcie_avg_rx = pcie_avg_tx = pcie_max_rx = None

    trace_rows = conn.execute(
        "SELECT step, kernels_json FROM traces WHERE run_id=? ORDER BY step",
        (run_id,),
    ).fetchall()
    traces = [{"step": r[0], "kernels": json.loads(r[1] or "[]")}
              for r in trace_rows]

    # Host-side aggregates for the CPU-bound / I/O-bound / cold-cache
    # / thrashing / imbalance / swap-pressure rules. Push work into
    # SQL — even a many-hours run has small host_samples volume.
    avg_cpu_pct = peak_cpu_pct = peak_cpu_max_core = n_cpus = None
    peak_disk_read = avg_disk_read = peak_iops = None
    first_third_disk = last_third_disk = None
    first_third_step = last_third_step = None
    avg_mem_cached = peak_mem_cached = min_mem_avail = None
    peak_swap_in = peak_swap_out = None
    avg_children_cpu = peak_max_child_cpu = avg_n_children = None
    try:
        row = conn.execute(
            "SELECT AVG(cpu_percent), MAX(cpu_percent), "
            "MAX(cpu_max_percent), MAX(n_cpus), "
            "MAX(disk_read_bps), AVG(disk_read_bps), MAX(disk_iops), "
            "AVG(mem_cached_bytes), MAX(mem_cached_bytes), "
            "MIN(mem_available_bytes), "
            "MAX(swap_in_bps), MAX(swap_out_bps), "
            "AVG(children_cpu_percent), MAX(max_child_cpu_percent), "
            "AVG(n_children) "
            "FROM host_samples WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row and row[0] is not None:
            (avg_cpu_pct, peak_cpu_pct, peak_cpu_max_core, n_cpus,
             peak_disk_read, avg_disk_read, peak_iops,
             avg_mem_cached, peak_mem_cached, min_mem_avail,
             peak_swap_in, peak_swap_out,
             avg_children_cpu, peak_max_child_cpu,
             avg_n_children) = row
    except sqlite3.OperationalError:
        pass                       # old DB without those columns

    # Prefetch-queue starvation stats: wait > 30ms is the "queue was
    # empty" threshold. Track p50, p95, and fraction of steps that
    # crossed the threshold.
    prefetch_p50 = prefetch_p95 = frac_starve = None
    try:
        row = conn.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN dataloader_wait_s > 0.030 THEN 1 ELSE 0 END) "
            "FROM steps WHERE run_id=? AND dataloader_wait_s IS NOT NULL",
            (run_id,),
        ).fetchone()
        if row and row[0]:
            total_with_wait, starved = row
            if total_with_wait >= 10:  # need a few steps to be meaningful
                waits = [r[0] for r in conn.execute(
                    "SELECT dataloader_wait_s FROM steps "
                    "WHERE run_id=? AND dataloader_wait_s IS NOT NULL "
                    "ORDER BY dataloader_wait_s", (run_id,),
                ).fetchall()]
                if waits:
                    prefetch_p50 = waits[len(waits) // 2]
                    prefetch_p95 = waits[min(len(waits) - 1,
                                              int(len(waits) * 0.95))]
                frac_starve = starved / total_with_wait
    except sqlite3.OperationalError:
        pass

    # Cold-cache detection needs "early vs steady" disk read AND step
    # time. Split the run into thirds by wall-clock at ingest time.
    try:
        span = conn.execute(
            "SELECT MIN(t), MAX(t) FROM host_samples WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if span and span[0] is not None and span[1] is not None:
            t0, t1 = span
            third = (t1 - t0) / 3.0 if t1 > t0 else 0
            if third > 0:
                first_third_disk = conn.execute(
                    "SELECT AVG(disk_read_bps) FROM host_samples "
                    "WHERE run_id=? AND t < ?",
                    (run_id, t0 + third),
                ).fetchone()[0]
                last_third_disk = conn.execute(
                    "SELECT AVG(disk_read_bps) FROM host_samples "
                    "WHERE run_id=? AND t >= ?",
                    (run_id, t1 - third),
                ).fetchone()[0]
    except sqlite3.OperationalError:
        pass

    if step_times and len(step_times) >= 6:
        third = len(step_times) // 3
        first_third_step = sum(step_times[:third]) / third
        last_third_step = sum(step_times[-third:]) / third

    try:
        tw_rows = conn.execute(
            "SELECT t_start_rel, t_end_rel, kernels_json FROM trace_windows "
            "WHERE run_id=? ORDER BY t_start_rel",
            (run_id,),
        ).fetchall()
        trace_windows = [
            {"t_start_rel": r[0], "t_end_rel": r[1],
             "kernels": json.loads(r[2] or "[]")}
            for r in tw_rows
        ]
    except sqlite3.OperationalError:
        trace_windows = []  # old DB, no trace_windows table

    return Ctx(
        run_id=run_id, name=name, gpu_name=gpu_name, meta=meta, arch=arch,
        n_steps=n_steps, train_s=train_s, step_s=step_s, step_times=step_times,
        phase_avg_s=phases, inter_step_gap_s=inter,
        tokens_total=int(tokens_sum) if tokens_sum else None,
        measured_flops=measured_flops, peak_tflops=peak, mfu=mfu,
        mfu_notes=mfu_notes,
        avg_sm_util=avg_util, max_mem=max_mem, mem_total=mem_total,
        max_temp=max_temp, sm_clock_deficit_pct=clock_deficit,
        pcie_avg_rx_gbps=pcie_avg_rx, pcie_avg_tx_gbps=pcie_avg_tx,
        pcie_max_rx_gbps=pcie_max_rx, traces=traces,
        trace_windows=trace_windows,
        avg_cpu_pct=avg_cpu_pct, peak_cpu_pct=peak_cpu_pct,
        peak_cpu_max_core_pct=peak_cpu_max_core,
        peak_disk_read_bps=peak_disk_read,
        avg_disk_read_bps=avg_disk_read,
        peak_disk_iops=peak_iops,
        n_cpus=n_cpus,
        first_third_disk_read_bps=first_third_disk,
        last_third_disk_read_bps=last_third_disk,
        first_third_step_s=first_third_step,
        last_third_step_s=last_third_step,
        avg_mem_cached_bytes=avg_mem_cached,
        peak_mem_cached_bytes=peak_mem_cached,
        min_mem_available_bytes=min_mem_avail,
        peak_swap_in_bps=peak_swap_in,
        peak_swap_out_bps=peak_swap_out,
        avg_children_cpu_pct=avg_children_cpu,
        peak_max_child_cpu_pct=peak_max_child_cpu,
        avg_n_children=avg_n_children,
        prefetch_wait_p50_s=prefetch_p50,
        prefetch_wait_p95_s=prefetch_p95,
        frac_steps_with_starve=frac_starve,
        rank=rank, world_size=world_size, group_id=group_id,
        # db_path is filled in by `analyze()` after the Ctx is built —
        # keeping it out of the constructor path lets internal callers
        # (analyze_group's per-rank ctxs) build cheaply.
    )


def analyze(db_path: str, run_id: int) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        c = _build_ctx(conn, run_id)
        c.db_path = db_path  # let rules query cross-run history
    finally:
        conn.close()

    insights = [r(c) for r in RULES]
    insights = [i for i in insights if i]
    if not insights:
        insights.append({
            "severity": "low",
            "title": "No major bottlenecks detected",
            "recommendation": (
                "Phase timings, MFU, memory, and thermals all look "
                "healthy. Consider chasing p99 step time if throughput "
                "matters at the tail."
            ),
        })

    return {
        "summary": {
            "run_id": c.run_id, "name": c.name, "gpu": c.gpu_name,
            "rank": c.rank, "world_size": c.world_size,
            "group_id": c.group_id,
            "n_steps": c.n_steps, "avg_step_s": c.step_s,
            "phase_avg_s": c.phase_avg_s,
            "inter_step_gap_s": c.inter_step_gap_s,
            "train_s": c.train_s, "tokens": c.tokens_total,
            "measured_flops": c.measured_flops, "peak_tflops": c.peak_tflops,
            "mfu": c.mfu, "mfu_notes": c.mfu_notes,
            "avg_sm_util": c.avg_sm_util,
            "max_mem_bytes": c.max_mem, "mem_total_bytes": c.mem_total,
            "max_temp_c": c.max_temp,
            "sm_clock_deficit_pct": c.sm_clock_deficit_pct,
            "pcie_avg_rx_gbps": c.pcie_avg_rx_gbps,
            "pcie_max_rx_gbps": c.pcie_max_rx_gbps,
        },
        "insights": insights,
    }


def _bucket_skew_analysis(conn: sqlite3.Connection, group_id: str) -> list[dict]:
    """For each (step, bucket_id), the collective finishes when the
    *slowest* rank finishes. The per-bucket skew rule looks for a
    specific bucket whose max-min end-time delta across ranks is
    persistently large — meaning the same bucket has the same
    straggler pattern, which is stronger evidence than aggregate step
    skew alone.

    If per-rank clock offsets were estimated (via NCCL ping-pong at
    run start, stored in `runs.rank_offset_s`), we subtract them from
    each rank's t_end_rel to align clocks to rank 0. Precision then
    drops from the ~1 ms `dist.barrier()` floor to ~10-100 μs.
    """
    # First, look up per-run offsets so we can align the clocks.
    offset_rows = conn.execute(
        "SELECT id, COALESCE(rank_offset_s, 0.0) FROM runs WHERE group_id=?",
        (group_id,),
    ).fetchall()
    offset_by_run = {rid: off for rid, off in offset_rows}

    rows = conn.execute(
        "SELECT r.rank, e.run_id, e.step, e.bucket_id, e.t_end_rel "
        "FROM comm_events e JOIN runs r ON r.id = e.run_id "
        "WHERE r.group_id=? AND e.t_end_rel IS NOT NULL",
        (group_id,),
    ).fetchall()
    if not rows:
        return []

    # (step, bucket_id) → {rank: t_end_rel_aligned}
    by_key: dict[tuple[int, int], dict[int, float]] = {}
    for rank, run_id, step, bucket_id, t_end in rows:
        t_end_aligned = t_end - offset_by_run.get(run_id, 0.0)
        by_key.setdefault((step, bucket_id), {})[rank] = t_end_aligned

    per_bucket: dict[int, list[float]] = {}  # bucket_id -> deltas per step
    per_bucket_ranks: dict[int, dict[int, int]] = {}  # bucket -> rank -> late-count
    for (step, bucket_id), rank_to_end in by_key.items():
        if len(rank_to_end) < 2: continue
        ends = list(rank_to_end.values())
        delta = max(ends) - min(ends)
        per_bucket.setdefault(bucket_id, []).append(delta)
        # Attribute lateness to the max-end rank.
        late_rank = max(rank_to_end.items(), key=lambda kv: kv[1])[0]
        per_bucket_ranks.setdefault(bucket_id, {}).setdefault(late_rank, 0)
        per_bucket_ranks[bucket_id][late_rank] += 1

    out = []
    for bucket_id, deltas in per_bucket.items():
        if not deltas: continue
        median = statistics.median(deltas)
        # p95: index into a sorted list at floor(0.95 * len), clamped
        # to the last element. The old `- 1` gave p90 for len=20.
        srt = sorted(deltas)
        p95 = srt[min(len(srt) - 1, int(len(srt) * 0.95))]
        # Attribute to whichever rank was slowest most often.
        rank_counts = per_bucket_ranks.get(bucket_id, {})
        blame_rank, blame_count = (max(rank_counts.items(),
                                       key=lambda kv: kv[1])
                                   if rank_counts else (None, 0))
        out.append({
            "bucket_id": bucket_id, "n_steps": len(deltas),
            "median_delta_s": median, "p95_delta_s": p95,
            "worst_rank": blame_rank, "worst_rank_count": blame_count,
        })
    out.sort(key=lambda b: b["p95_delta_s"], reverse=True)
    return out


def analyze_group(db_path: str, group_id: str) -> dict:
    """Cross-run analysis over all runs in a distributed group.

    Fires rank-skew when the slowest rank's median step is meaningfully
    longer than the fastest, AND if per-bucket comm events are
    available, points at the specific bucket where the straggler shows
    up most.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, rank FROM runs WHERE group_id=? ORDER BY rank",
            (group_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"no runs found for group_id {group_id!r}")
        per_rank = []
        for run_id, rank in rows:
            c = _build_ctx(conn, run_id)
            if c.step_times:
                per_rank.append({
                    "run_id": run_id, "rank": rank,
                    "median_step_s": statistics.median(c.step_times),
                    "mfu": c.mfu, "n_steps": c.n_steps,
                })
        bucket_skew = _bucket_skew_analysis(conn, group_id)
    finally:
        conn.close()

    insights: list[dict] = []
    if len(per_rank) >= 2:
        medians = [r["median_step_s"] for r in per_rank]
        fast, slow = min(medians), max(medians)
        if fast > 0 and slow / fast > 1.10:
            slowest = max(per_rank, key=lambda r: r["median_step_s"])
            fastest = min(per_rank, key=lambda r: r["median_step_s"])
            insights.append({
                "severity": "high" if slow / fast > 1.30 else "medium",
                "title": (f"Rank skew: rank {slowest['rank']} median "
                          f"{slow*1000:.0f} ms vs rank {fastest['rank']} "
                          f"{fast*1000:.0f} ms ({(slow/fast-1)*100:.0f}% slower)"),
                "recommendation": (
                    "Slowest rank drags the collective — every allreduce "
                    "waits for it. Check: uneven data (dataset not sharded "
                    "evenly), heterogeneous GPUs, PCIe topology (one GPU "
                    "on a different NUMA), or a noisy neighbor on that node."
                ),
                "evidence": {"per_rank": per_rank},
            })

    # Per-bucket skew rule: fire on the worst bucket if its p95 delta
    # is > 5 ms AND the same rank is late in ≥ 60% of steps for it.
    for b in bucket_skew[:3]:  # only check the worst three
        if b["p95_delta_s"] < 0.005: continue
        share = (b["worst_rank_count"] / b["n_steps"]) if b["n_steps"] else 0
        if share < 0.6: continue
        insights.append({
            "severity": "medium",
            "title": (f"Bucket {b['bucket_id']}: rank {b['worst_rank']} "
                      f"is late on {share*100:.0f}% of steps "
                      f"(p95 delta {b['p95_delta_s']*1000:.1f} ms)"),
            "recommendation": (
                "One bucket consistently waits for the same rank — a "
                "specific gradient reduction is the bottleneck, not "
                "the whole training loop. Inspect that rank's device "
                "for temperature/clock throttling, NCCL topology "
                "(different NIC / PCIe root), or an extra hook on that "
                "layer. `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL` will "
                "confirm the ring/tree route."
            ),
            "evidence": {"bucket": b},
        })
        break  # one bucket-skew insight is enough

    if not insights:
        insights.append({
            "severity": "low",
            "title": "No rank skew detected",
            "recommendation": (
                "Ranks are within 10% on median step time. Per-bucket "
                "comm events also don't show a specific straggler bucket."
            ),
        })
    return {"summary": {"group_id": group_id,
                        "ranks": per_rank,
                        "bucket_skew": bucket_skew[:10]},
            "insights": insights}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _print_report(r: dict) -> None:
    s = r["summary"]
    if "group_id" in s and "ranks" in s:
        print(f"\n== group {s['group_id']!r} ==")
        for rk in s["ranks"]:
            # Build the MFU segment separately: the whole line would
            # collapse if this were a print-wide ternary (the previous
            # bug ate the rank + median-step fields when MFU was None).
            mfu_str = (f", MFU {rk['mfu']*100:.1f}%"
                       if rk.get("mfu") is not None else "")
            print(f"  rank {rk['rank']}: median step "
                  f"{rk['median_step_s']*1000:.1f} ms" + mfu_str)
    else:
        print(f"\n== run {s['run_id']}: {s['name']!r} on {s['gpu']} ==")
        if s.get("rank") is not None:
            print(f"  rank {s['rank']}/{s.get('world_size')} "
                  f"group {s.get('group_id')!r}")
        train_str = (f"train time: {s['train_s']:.1f}s"
                     if s.get("train_s") else "(no timing)")
        print(f"  steps: {s['n_steps']}  avg step: "
              f"{(s['avg_step_s'] or 0)*1000:.1f} ms  {train_str}")
        p = s["phase_avg_s"]
        print("  phases (avg ms): "
              + "  ".join(f"{k}={(v or 0)*1000:.1f}" for k, v in p.items()))
        if s.get("inter_step_gap_s") is not None:
            print(f"  inter-step gap: "
                  f"{s['inter_step_gap_s']*1000:.1f} ms (real dataloader stall)")
        if s.get("mfu") is not None:
            print(f"  MFU: {s['mfu']*100:.1f}%  "
                  f"(peak {s['peak_tflops']:.0f} TFLOPs)")
            for n in s.get("mfu_notes") or []:
                print(f"    - {n}")
        if s.get("max_mem_bytes"):
            print(f"  peak memory: {s['max_mem_bytes']/1e9:.1f} / "
                  f"{s['mem_total_bytes']/1e9:.0f} GB   "
                  f"avg SM util: {(s.get('avg_sm_util') or 0)*100:.0f}%")
        if s.get("pcie_max_rx_gbps"):
            print(f"  PCIe RX peak: {s['pcie_max_rx_gbps']:.1f} GB/s")

    print("\n== insights ==")
    for it in r["insights"]:
        tag = {"high": "[HIGH]  ", "medium": "[MED]   ",
               "low": "[LOW]   "}[it["severity"]]
        print(f"  {tag}{it['title']}")
        for line in _wrap(it["recommendation"], width=72, indent=" " * 10):
            print(line)
    print()


def _wrap(text: str, width: int, indent: str) -> list[str]:
    words, out, cur = text.split(), [], indent
    for w in words:
        if len(cur) + 1 + len(w) > width and cur.strip():
            out.append(cur.rstrip())
            cur = indent + w
        else:
            cur += (" " if cur != indent else "") + w
    if cur.strip():
        out.append(cur.rstrip())
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        prog="gpuprof insights",
        description=(
            "Show insights for a completed run. With no arguments, "
            "picks the most recent run in ./gpuprof.db."
        ),
    )
    ap.add_argument("db", nargs="?", default="gpuprof.db",
                    help="path to gpuprof SQLite DB (default: ./gpuprof.db)")
    ap.add_argument("run_id", type=int, nargs="?", default=None,
                    help="run id (default: most recent in the DB)")
    ap.add_argument("--group", help="analyze a distributed group by id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.group:
        r = analyze_group(args.db, args.group)
    else:
        # Default the run_id to the latest run in the DB.
        if args.run_id is None:
            args.run_id = _latest_run_id(args.db, ap)
        r = analyze(args.db, args.run_id)

    if args.json:
        print(json.dumps(r, indent=2, default=str))
    else:
        _print_report(r)


def _latest_run_id(db_path: str, ap: argparse.ArgumentParser) -> int:
    if not sqlite3_db_exists(db_path):
        ap.error(f"no DB at {db_path!r} — pass a path or run one training "
                 "loop first")
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT MAX(id) FROM runs").fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        ap.error(f"no runs found in {db_path}")
    return int(row[0])


def sqlite3_db_exists(path: str) -> bool:
    import os
    return os.path.exists(path) and os.path.getsize(path) > 0


if __name__ == "__main__":
    _cli()
