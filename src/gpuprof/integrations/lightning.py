"""PyTorch Lightning callback.

    from gpuprof.integrations import LightningCallback
    trainer = pl.Trainer(callbacks=[LightningCallback(
        run_name="baseline", cuda=True, trace_every_n_steps=100,
        meta={"arch": {...}, "dtype": "bf16"},
    )])

The callback owns the `GpuProfiler` lifecycle (start on train begin,
stop on train end) and hangs each Lightning training hook onto the
right phase context:

    on_train_batch_start   → prof.step() entered; forward phase begun
    on_before_backward     → forward ended; backward begun
    on_before_optimizer_step → backward ended; optimizer begun
    on_train_batch_end     → optimizer ended; step exited
"""
from __future__ import annotations

from typing import Optional

from ..profiler import GpuProfiler


class GpuProfilerCallback:
    """Lightning `pl.Callback` subclass. We defer the actual base-class
    inheritance to `__init_subclass__`-time via a lazy import so this
    module imports without pytorch_lightning installed."""

    def __init__(self, prof: Optional[GpuProfiler] = None, **prof_kwargs):
        try:
            import pytorch_lightning  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "gpuprof.integrations.LightningCallback needs "
                "pytorch_lightning installed"
            ) from e
        self._prof = prof or GpuProfiler(**prof_kwargs)
        self._step_ctx = None
        self._rec = None
        self._phase_cms: dict = {}
        # A user might feed their own tokens/loss through the recorder.
        self._auto_loss = True

    # -- lifecycle ----------------------------------------------------

    def on_train_start(self, trainer, pl_module):
        self._prof.start()

    def on_train_end(self, trainer, pl_module):
        # Close any still-open phase / step so we don't leak state on
        # unusual exits.
        for name in list(self._phase_cms.keys()):
            self._exit_phase(name)
        if self._step_ctx is not None:
            try: self._step_ctx.__exit__(None, None, None)
            except Exception: pass
            self._step_ctx = None
        self._prof.stop()

    # -- per-batch ----------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_ctx = self._prof.step(batch_idx)
        self._rec = self._step_ctx.__enter__()
        self._enter_phase("forward")

    def on_before_backward(self, trainer, pl_module, loss):
        self._exit_phase("forward")
        self._enter_phase("backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # backward has finished at this point
        self._exit_phase("backward")
        self._enter_phase("optimizer")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._exit_phase("optimizer")
        if self._auto_loss and self._rec is not None:
            loss = _extract_loss(outputs)
            if loss is not None:
                self._rec.record(loss=loss)
        if self._step_ctx is not None:
            self._step_ctx.__exit__(None, None, None)
        self._step_ctx = None
        self._rec = None

    # -- optional: DDP hook ------------------------------------------

    def on_fit_start(self, trainer, pl_module):
        # Lightning's DDP wrapping happens after configure_optimizers;
        # by the time on_fit_start fires the model has been wrapped.
        mod = getattr(pl_module, "trainer", None) and pl_module.trainer.model
        if mod is None:
            mod = pl_module
        # Only try instrument_ddp if the model *looks* like DDP.
        if type(mod).__name__ in ("DistributedDataParallel",):
            try: self._prof.instrument_ddp(mod)
            except Exception: pass

    # -- internals ----------------------------------------------------

    def _enter_phase(self, name: str) -> None:
        if self._rec is None: return
        cm = self._rec.phase(name)
        cm.__enter__()
        self._phase_cms[name] = cm

    def _exit_phase(self, name: str) -> None:
        cm = self._phase_cms.pop(name, None)
        if cm is not None:
            try: cm.__exit__(None, None, None)
            except Exception: pass


def _extract_loss(outputs) -> Optional[float]:
    if outputs is None: return None
    # Lightning may pass a Tensor, a dict with "loss", or a plain float.
    try:
        if hasattr(outputs, "item"): return float(outputs.item())
        if isinstance(outputs, dict) and "loss" in outputs:
            v = outputs["loss"]
            return float(v.item()) if hasattr(v, "item") else float(v)
        if isinstance(outputs, (int, float)): return float(outputs)
    except Exception:
        return None
    return None


# Make the callback a true subclass of pl.Callback lazily. If Lightning
# is installed at import time we can just inherit directly, avoiding
# the metaclass hoops.
try:
    import pytorch_lightning as _pl
    # GpuProfilerCallback FIRST in MRO — pl.Callback ships no-op stubs
    # for every hook and would otherwise win at method resolution.
    _GpuProfilerCallbackImpl = type(
        "GpuProfilerCallback",
        (GpuProfilerCallback, _pl.Callback),
        {},
    )
    GpuProfilerCallback = _GpuProfilerCallbackImpl  # type: ignore
except ImportError:
    pass  # class is still importable; __init__ will raise a clear error
