"""psutil-based host-side sampler.

Answers the "why is my GPU starving?" question by capturing signals
the NVML sampler can't see:

- CPU utilization aggregate + hottest single core
- Aggregate CPU across every dataloader worker (all child processes
  of this rank) plus the hottest single worker
- Host memory: used, available, page-cache size, swap in/out rate
- Disk I/O: read/write bytes-per-sec, combined IOPS

Discriminating rules in `insights` combine these with dataloader-stall
to say *whether* your workers are CPU-bound, I/O-bound, page-cache-
thrashing, memory-pressured, or unevenly loaded.

Overhead: ~0.2% CPU at 1 Hz. Graceful fallback to a no-op when
`psutil` isn't installed — the rest of the tool still works.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class HostSample:
    t: float                    # wall-clock seconds since epoch
    cpu_percent: float          # 0..100 aggregate across all cores
    cpu_max_percent: float      # 0..100 hottest single core
    n_cpus: int                 # number of logical CPUs on this host
    mem_used_bytes: int         # host RAM currently used
    mem_total_bytes: int
    mem_cached_bytes: int       # page cache — grows with warm data
    mem_available_bytes: int    # what the OS says is "usable" right now
    swap_in_bps: float          # bytes/sec swapped IN since last sample
    swap_out_bps: float
    disk_read_bps: float
    disk_write_bps: float
    disk_iops: float            # combined read+write ops/sec
    children_cpu_percent: float # sum of cpu% across child processes
    max_child_cpu_percent: float
    n_children: int             # count of visible child processes


class HostSampler:
    """Background thread that samples host-side counters via psutil.

    Rate is deliberately low (1 Hz default) — dataloader stalls and
    disk-bound behavior are second-scale phenomena; higher polling
    adds overhead without new signal.
    """

    def __init__(
        self,
        on_sample: Callable[[HostSample], None],
        hz: float = 1.0,
    ):
        self._on_sample = on_sample
        self._interval = 1.0 / max(0.1, float(hz))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
        # Rate calculations need consecutive samples of cumulative counters.
        self._prev_disk = None
        self._prev_swap = None
        self._prev_t: Optional[float] = None
        # Child-process cache. psutil.Process.cpu_percent() needs to be
        # primed per process (first call always returns 0), so we hold
        # onto the Process objects rather than re-enumerating them
        # from scratch on every sample.
        self._parent_proc = None
        self._child_procs: dict[int, object] = {}

    @property
    def available(self) -> bool:
        return self._psutil is not None

    def start(self) -> None:
        if self._psutil is None or self._thread is not None:
            return
        # Prime cpu_percent — first call after start always returns 0.
        self._psutil.cpu_percent(interval=None)
        self._psutil.cpu_percent(interval=None, percpu=True)
        self._parent_proc = self._psutil.Process()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gpuprof-host",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2 + 1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._on_sample(self._sample())
            except Exception:
                pass  # never kill training on sampling error
            self._stop.wait(self._interval)

    def _sample(self) -> HostSample:
        p = self._psutil
        t = time.time()

        cpu = p.cpu_percent(interval=None)
        per_core = p.cpu_percent(interval=None, percpu=True)
        cpu_max = max(per_core) if per_core else cpu
        n_cpus = len(per_core) if per_core else 1

        mem = p.virtual_memory()
        # Not every OS reports `cached` — macOS does, Windows doesn't.
        mem_cached = int(getattr(mem, "cached", 0) or 0)
        mem_available = int(getattr(mem, "available", 0) or mem.total - mem.used)

        # Swap: `sin`/`sout` are cumulative BYTES swapped since boot.
        swap_in_bps = swap_out_bps = 0.0
        try:
            swap = p.swap_memory()
        except (RuntimeError, NotImplementedError):
            swap = None
        if swap is not None and self._prev_swap is not None and self._prev_t:
            dt = t - self._prev_t
            if dt > 0:
                swap_in_bps  = max(0.0, (swap.sin  - self._prev_swap.sin) / dt)
                swap_out_bps = max(0.0, (swap.sout - self._prev_swap.sout) / dt)
        self._prev_swap = swap

        # Disk I/O
        try:
            disk = p.disk_io_counters()
        except (RuntimeError, NotImplementedError):
            disk = None
        dr = dw = di = 0.0
        if disk is not None and self._prev_disk is not None and self._prev_t:
            dt = t - self._prev_t
            if dt > 0:
                dr = max(0.0, (disk.read_bytes - self._prev_disk.read_bytes) / dt)
                dw = max(0.0, (disk.write_bytes - self._prev_disk.write_bytes) / dt)
                di = max(0.0, (
                    (disk.read_count + disk.write_count)
                    - (self._prev_disk.read_count + self._prev_disk.write_count)
                ) / dt)
        self._prev_disk = disk
        self._prev_t = t

        # Per-worker CPU: dataloader workers are child processes of
        # this rank's Python. Sum + max across them tells us both
        # aggregate load and worker-level imbalance.
        children_cpu, max_child_cpu, n_children = self._sample_children()

        return HostSample(
            t=t,
            cpu_percent=float(cpu),
            cpu_max_percent=float(cpu_max),
            n_cpus=int(n_cpus),
            mem_used_bytes=int(mem.used),
            mem_total_bytes=int(mem.total),
            mem_cached_bytes=mem_cached,
            mem_available_bytes=mem_available,
            swap_in_bps=float(swap_in_bps),
            swap_out_bps=float(swap_out_bps),
            disk_read_bps=float(dr),
            disk_write_bps=float(dw),
            disk_iops=float(di),
            children_cpu_percent=float(children_cpu),
            max_child_cpu_percent=float(max_child_cpu),
            n_children=int(n_children),
        )

    def _sample_children(self) -> tuple[float, float, int]:
        if self._parent_proc is None:
            return 0.0, 0.0, 0
        p = self._psutil
        try:
            children = self._parent_proc.children(recursive=False)
        except (p.NoSuchProcess, p.AccessDenied):
            return 0.0, 0.0, 0

        # Reconcile the process cache — new children get primed, dead
        # ones evicted. psutil's cpu_percent needs two calls per proc
        # to compute a delta, so re-priming every sample would report
        # zeros for young workers.
        alive_pids = {c.pid for c in children}
        for pid in list(self._child_procs):
            if pid not in alive_pids:
                del self._child_procs[pid]
        for c in children:
            if c.pid not in self._child_procs:
                self._child_procs[c.pid] = c
                try: c.cpu_percent(interval=None)      # prime
                except (p.NoSuchProcess, p.AccessDenied): pass

        percents: list[float] = []
        for c in list(self._child_procs.values()):
            try:
                percents.append(float(c.cpu_percent(interval=None)))
            except (p.NoSuchProcess, p.AccessDenied):
                pass
        if not percents:
            return 0.0, 0.0, 0
        return sum(percents), max(percents), len(percents)
