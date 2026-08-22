"""torch.profiler wrapping — capture a per-kernel trace for one step
every N steps, then extract the top-K kernels by device time.

Traces are expensive: activating torch.profiler for a step adds ~10-30%
overhead to that step. We only turn it on periodically (via
`GpuProfiler(trace_every_n_steps=N)`), and record only aggregated per-op
stats rather than raw event streams — the goal is "which kernels
dominate", not a full chrome-trace.
"""
from __future__ import annotations

from ._kernels import extract_kernels


class ProfilerTrace:
    """Context-managed torch.profiler for one training step.

    Usage:
        with ProfilerTrace(cuda=True) as t:
            # training step runs
            pass
        kernels = t.top_kernels(k=25)
    """

    def __init__(self, cuda: bool = False):
        # Deferred import: don't pull torch in when tracing isn't used.
        import torch.profiler as tp
        activities = [tp.ProfilerActivity.CPU]
        if cuda:
            activities.append(tp.ProfilerActivity.CUDA)
        self._prof = tp.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *args):
        self._prof.__exit__(*args)

    def top_kernels(self, k: int = 25) -> list[dict]:
        try:
            events = self._prof.key_averages()
        except Exception:
            return []
        return extract_kernels(events, top_k=k)
