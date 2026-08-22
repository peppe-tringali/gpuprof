"""Burst-trace / trace_range mode.

Verifies:
- `trace_range=(a, b)` schedules traces on every step in [a, b).
- `trace_warmup_steps` still trigger regardless of trace_range.
- `sample_hz` is clamped to 100.
"""
import pytest

from gpuprof import GpuProfiler


def test_sample_hz_clamped():
    p = GpuProfiler(run_name="x", db_path=None, sample_hz=10000.0)
    assert p._sample_hz == 100.0
    p2 = GpuProfiler(run_name="x", db_path=None, sample_hz=0.001)
    assert p2._sample_hz == 1.0


def test_should_trace_range_and_warmup():
    p = GpuProfiler(run_name="x", db_path=None,
                    trace_every_n_steps=1000,
                    trace_range=(10, 15),
                    trace_warmup_steps=(0, 1, 5))
    # warmup steps still fire
    assert p._should_trace(0)
    assert p._should_trace(1)
    assert p._should_trace(5)
    # inside range: every step
    assert p._should_trace(10)
    assert p._should_trace(14)
    # boundary is exclusive
    assert not p._should_trace(15)
    # outside both — trace_every=1000 doesn't match steps < 1000
    assert not p._should_trace(20)
    assert p._should_trace(1000)  # trace_every fires
