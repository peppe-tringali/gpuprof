"""DeepSpeed engine adapter.

DeepSpeed's usage pattern is::

    for batch in loader:
        loss = engine(batch)
        engine.backward(loss)
        engine.step()

so we can't inject via a Lightning-style callback. Instead we wrap the
engine object; the profiled version presents the same API and treats
each (forward, backward, step) trio as one profiled step.

    from gpuprof.integrations import wrap_deepspeed_engine

    engine, *_ = deepspeed.initialize(...)
    engine = wrap_deepspeed_engine(engine, run_name="ds-run",
                                    meta={"arch": {...}, "dtype": "bf16"})
    for batch in loader:
        loss = engine(batch)
        engine.backward(loss)
        engine.step()
    engine.close()          # <— required, ends the run
"""
from __future__ import annotations

from typing import Optional

from ..profiler import GpuProfiler


_LOCAL_ATTRS = frozenset({"_engine", "_prof", "_step_idx",
                           "_step_ctx", "_rec"})


class _WrappedEngine:
    def __init__(self, engine, prof: GpuProfiler):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_prof", prof)
        object.__setattr__(self, "_step_idx", 0)
        object.__setattr__(self, "_step_ctx", None)
        object.__setattr__(self, "_rec", None)
        prof.start()

    def __call__(self, *args, **kwargs):
        self._step_ctx = self._prof.step(self._step_idx)
        self._rec = self._step_ctx.__enter__()
        try:
            with self._rec.phase("forward"):
                return self._engine(*args, **kwargs)
        except BaseException:
            # forward crashed — close the step so the pipeline doesn't
            # leak state into the next iteration.
            try: self._step_ctx.__exit__(None, None, None)
            except Exception: pass
            self._step_ctx = self._rec = None
            raise

    def backward(self, loss, *args, **kwargs):
        if self._rec is None:
            return self._engine.backward(loss, *args, **kwargs)
        with self._rec.phase("backward"):
            r = self._engine.backward(loss, *args, **kwargs)
        try: self._rec.record(loss=float(loss.item()))
        except Exception: pass
        return r

    def step(self, *args, **kwargs):
        if self._rec is None:
            return self._engine.step(*args, **kwargs)
        with self._rec.phase("optimizer"):
            r = self._engine.step(*args, **kwargs)
        try: self._step_ctx.__exit__(None, None, None)
        except Exception: pass
        self._step_ctx = self._rec = None
        object.__setattr__(self, "_step_idx", self._step_idx + 1)
        return r

    def close(self):
        if self._step_ctx is not None:
            try: self._step_ctx.__exit__(None, None, None)
            except Exception: pass
        self._prof.stop()

    # Delegate anything else to the underlying engine.
    def __getattr__(self, name):
        return getattr(self._engine, name)

    def __setattr__(self, name, value):
        if name in _LOCAL_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._engine, name, value)


def wrap_deepspeed_engine(
    engine, prof: Optional[GpuProfiler] = None, **prof_kwargs,
):
    """Return a wrapped engine that profiles each forward/backward/step
    trio as one gpuprof step. `engine.close()` (on the wrapper) ends
    the run."""
    prof = prof or GpuProfiler(**prof_kwargs)
    return _WrappedEngine(engine, prof)
