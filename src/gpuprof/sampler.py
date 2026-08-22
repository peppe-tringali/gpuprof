"""GPU state sampler — multi-GPU capable.

Runs on a background thread and, per tick, samples every configured
GPU and emits one `Sample` per GPU to the callback. Uses NVML when
available; falls back to a synthetic backend so the rest of the stack
is usable on machines without an NVIDIA GPU (laptops, CI).

Env knobs:
- GPUPROF_MOCK=1        → force the mock backend
- GPUPROF_MOCK_GPUS=N   → mock reports N GPUs (default 1)
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Sample:
    t: float               # wall-clock seconds since epoch
    gpu_index: int
    sm_util: float         # 0..1
    mem_used_bytes: int
    mem_total_bytes: int
    power_w: float
    temp_c: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    pcie_rx_kbps: int
    pcie_tx_kbps: int


# NVML init/shutdown is process-global and refcounted. If two backends
# both call `nvmlInit` then one calls `nvmlShutdown`, the counter goes
# to zero and the second backend's handles become invalid. We manage
# the refcount ourselves so multiple concurrent `Sampler` instances
# (nested profilers, test fixtures, DDP ranks) can't poison each other.
_nvml_init_lock = threading.Lock()
_nvml_init_count = 0


def _nvml_init(pynvml) -> None:
    global _nvml_init_count
    with _nvml_init_lock:
        if _nvml_init_count == 0:
            pynvml.nvmlInit()
        _nvml_init_count += 1


def _nvml_shutdown(pynvml) -> None:
    global _nvml_init_count
    with _nvml_init_lock:
        _nvml_init_count -= 1
        if _nvml_init_count <= 0:
            _nvml_init_count = 0
            try: pynvml.nvmlShutdown()
            except Exception: pass


class _NvmlBackend:
    def __init__(self, gpu_index: int):
        import pynvml
        self._pynvml = pynvml
        _nvml_init(pynvml)
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_index = gpu_index
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name

    def sample(self) -> Sample:
        n, h = self._pynvml, self.handle
        util = n.nvmlDeviceGetUtilizationRates(h)
        mem = n.nvmlDeviceGetMemoryInfo(h)
        power = n.nvmlDeviceGetPowerUsage(h) / 1000.0
        temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        sm_clock = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
        mem_clock = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)
        try:
            rx = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_RX_BYTES)
            tx = n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_TX_BYTES)
        except n.NVMLError:
            rx = tx = 0
        return Sample(
            t=time.time(),
            gpu_index=self.gpu_index,
            sm_util=util.gpu / 100.0,
            mem_used_bytes=int(mem.used),
            mem_total_bytes=int(mem.total),
            power_w=float(power),
            temp_c=float(temp),
            sm_clock_mhz=int(sm_clock),
            mem_clock_mhz=int(mem_clock),
            pcie_rx_kbps=int(rx),
            pcie_tx_kbps=int(tx),
        )

    def shutdown(self):
        _nvml_shutdown(self._pynvml)


class _MockBackend:
    """Synthetic samples shaped roughly like a training run — a slow
    wobble plus per-step dips — so charts look plausible during local
    development on a machine without an NVIDIA GPU."""

    def __init__(self, gpu_index: int):
        self.gpu_index = gpu_index
        self.name = "MockGPU"
        # Phase-offset each mocked GPU so multi-GPU charts don't overlap.
        self._t0 = time.time() - gpu_index * 1.7
        self._noise = random.Random(1000 + gpu_index)

    def sample(self) -> Sample:
        dt = time.time() - self._t0
        util = 0.7 + 0.15 * math.sin(dt / 5.0) - 0.2 * abs(math.sin(dt * 2.0))
        util = max(0.0, min(1.0, util + self._noise.uniform(-0.03, 0.03)))
        mem_total = 80 * 1024**3
        mem_used = int(mem_total * (0.55 + 0.05 * math.sin(dt / 30.0)))
        return Sample(
            t=time.time(),
            gpu_index=self.gpu_index,
            sm_util=util,
            mem_used_bytes=mem_used,
            mem_total_bytes=mem_total,
            power_w=350.0 + 80.0 * util + self._noise.uniform(-10, 10),
            temp_c=60.0 + 10.0 * util,
            sm_clock_mhz=1755,
            mem_clock_mhz=1593,
            pcie_rx_kbps=int(1_000_000 * util),
            pcie_tx_kbps=int(500_000 * util),
        )

    def shutdown(self):
        pass


def _detect_gpu_indices() -> list[int]:
    """Enumerate visible GPU indices. Honors CUDA_VISIBLE_DEVICES via NVML."""
    if os.environ.get("GPUPROF_MOCK"):
        return list(range(int(os.environ.get("GPUPROF_MOCK_GPUS", "1"))))
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
        finally:
            pynvml.nvmlShutdown()
        return list(range(n))
    except Exception:
        # NVML unavailable — fall through to mock.
        return list(range(int(os.environ.get("GPUPROF_MOCK_GPUS", "1"))))


def _make_backend(gpu_index: int):
    if os.environ.get("GPUPROF_MOCK"):
        return _MockBackend(gpu_index)
    try:
        return _NvmlBackend(gpu_index)
    except Exception:
        return _MockBackend(gpu_index)


class Sampler:
    """Background thread that samples one or more GPUs at a fixed rate."""

    def __init__(
        self,
        on_sample: Callable[[Sample], None],
        gpu_indices: Optional[list[int]] = None,
        hz: float = 10.0,
    ):
        if gpu_indices is None:
            gpu_indices = _detect_gpu_indices()
        if not gpu_indices:
            gpu_indices = [0]
        self._on_sample = on_sample
        self._backends = [_make_backend(i) for i in gpu_indices]
        self._gpu_indices = list(gpu_indices)
        self._interval = 1.0 / hz
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def gpu_name(self) -> str:
        # Homogeneous nodes are the common case; report the first GPU.
        return self._backends[0].name if self._backends else "unknown"

    @property
    def gpu_indices(self) -> list[int]:
        return list(self._gpu_indices)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gpuprof-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for b in self._backends:
            b.shutdown()

    def _run(self) -> None:
        next_t = time.time()
        while not self._stop.is_set():
            for b in self._backends:
                try:
                    self._on_sample(b.sample())
                except Exception:
                    # Never let a sampling error kill the training process.
                    pass
            next_t += self._interval
            sleep = next_t - time.time()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_t = time.time()  # fell behind; resync
