"""Continuous kernel-aggregate profiling via torch.profiler.

Where `trace_every_n_steps` gives you a per-step snapshot, this gives
you a rolling per-window aggregate: every W seconds, extract the
active profile's `key_averages()`, snapshot the top-K kernels for that
window, and start a fresh profile. Answers "at t=17 min my throughput
dropped — which kernels changed?" without storing a full CUPTI event
stream (which would be gigabytes).

torch.profiler uses CUPTI under the hood, so kernel timings are the
same sub-microsecond precision `nsys` sees. What we don't store are
the individual events — only per-kernel-name aggregates per window.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from ._kernels import extract_kernels


class ContinuousProfiler:
    """Background thread that keeps torch.profiler active on a rolling
    window and emits kernel-aggregate snapshots.

    Overhead: ~2-5% steady CPU when active — opt in via
    `GpuProfiler(continuous_traces_hz=...)`.
    """

    def __init__(
        self,
        on_window: Callable[[dict], None],
        window_s: float = 1.0,
        cuda: bool = False,
        top_k: int = 50,
        epoch: Optional[float] = None,
    ):
        self._on_window = on_window
        self._window_s = max(0.1, float(window_s))
        self._cuda = cuda
        self._top_k = top_k
        self._epoch = epoch or 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gpuprof-continuous",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._window_s * 2 + 1.0)

    def _run(self) -> None:
        # Deferred import so a run without torch still boots.
        try:
            import torch.profiler as tp
        except ImportError:
            return
        activities = [tp.ProfilerActivity.CPU]
        if self._cuda:
            activities.append(tp.ProfilerActivity.CUDA)

        while not self._stop.is_set():
            window_start_rel = time.perf_counter() - self._epoch
            prof = None
            try:
                prof = tp.profile(
                    activities=activities,
                    record_shapes=False,
                    with_stack=False,
                    profile_memory=False,
                )
                prof.__enter__()
                # Rotate every window_s seconds, or exit sooner on stop.
                if self._stop.wait(self._window_s):
                    prof.__exit__(None, None, None)
                    return
                prof.__exit__(None, None, None)
            except Exception:
                # Torch profiler occasionally hits internal state races
                # when start/stop rapidly. Skip this window rather than
                # kill the training thread.
                if prof is not None:
                    try: prof.__exit__(None, None, None)
                    except Exception: pass
                continue

            kernels = self._extract(prof)
            window = {
                "t_start_rel": window_start_rel,
                "t_end_rel": time.perf_counter() - self._epoch,
                "kernels": kernels,
            }
            try: self._on_window(window)
            except Exception: pass

    def _extract(self, prof) -> list[dict]:
        try:
            events = prof.key_averages()
        except Exception:
            return []
        return extract_kernels(events, top_k=self._top_k)
