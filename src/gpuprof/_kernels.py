"""Shared torch.profiler event → kernel-dict extraction.

Both `traces.ProfilerTrace.top_kernels` and
`continuous.ContinuousProfiler._extract` need to convert a
`torch.profiler` event list into the same kernel-record shape (name,
count, self/total device us, self/total CPU us). Kept here so the
column set + attribute-name fallbacks (torch 2.5 renamed cuda_time to
device_time) live in one place.
"""
from __future__ import annotations

from typing import Iterable


def _get(event, *names) -> float:
    """Pick the first attribute in `names` that exists and is non-None."""
    for n in names:
        v = getattr(event, n, None)
        if v is not None:
            return float(v)
    return 0.0


def extract_kernels(events: Iterable, top_k: int = 25) -> list[dict]:
    """Turn a torch.profiler `key_averages()` iterable into a
    top-K-by-self-device-time list of dicts.

    Sort key is `self_device_us` with `self_cpu_us` as fallback — the
    "which kernel eats my GPU?" question. On CPU-only runs the CPU
    field ends up doing all the ordering work.
    """
    rows: list[dict] = []
    for e in events:
        rows.append({
            "name": str(e.key),
            "count": int(e.count),
            # torch 2.5+ renamed cuda_time → device_time; support both.
            "self_device_us": _get(e, "self_device_time_total",
                                    "self_cuda_time_total"),
            "device_us":      _get(e, "device_time_total",
                                    "cuda_time_total"),
            "self_cpu_us":    _get(e, "self_cpu_time_total"),
            "cpu_us":         _get(e, "cpu_time_total"),
        })
    rows.sort(
        key=lambda r: r["self_device_us"] or r["self_cpu_us"],
        reverse=True,
    )
    return rows[:top_k]
