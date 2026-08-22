"""Auto-instrumentation for standard PyTorch training loops.

    import gpuprof
    with gpuprof.profile("my-experiment"):
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

No `prof.step(i)`, no `s.phase("forward")` — the tool patches
`nn.Module.__call__`, `Tensor.backward`, and `Optimizer.step` on entry
to `gpuprof.profile()` and unpatches on exit. Step boundaries come
from `Optimizer.step()`; per-phase timings from the other two hooks.

What auto instrumentation handles well:
- Standard forward → backward → optimizer.step loops
- Gradient accumulation (multiple fwd+bwd per opt.step → one step,
  phases sum)
- Mixed precision via `GradScaler` (still calls the underlying
  optimizer's step)
- Subclassed optimizers that define their own `.step()` (we iterate
  the `Optimizer` subclass tree at patch time)
- DDP models (Module.__call__ still fires on the DDP wrapper)

What it doesn't:
- Inference-only loops (no `optimizer.step()` → step never commits;
  use `auto=False` and instrument explicitly)
- Custom optimizers registered *after* `profile()` started
- `torch.autograd.backward(...)` called directly instead of
  `loss.backward()` (rare — the free function isn't hooked)
- Eval passes *inside* the training loop inflate that step's forward
  time (they don't have their own opt.step to close a step)

Users who want finer control can pass `auto=False` and drop back to
the explicit `prof.step()` / `s.phase()` API.
"""
from __future__ import annotations

import functools
import threading
import time
from typing import Optional


# Sentinel attribute we stamp on our wrappers so nested/repeated
# patches don't wrap the wrapper.
_MARKER = "_gpuprof_wrapped"


class AutoInstrumenter:
    """Monkey-patch the three torch entry points that bound training
    step / phase boundaries. `start()` installs the hooks, `stop()`
    restores them. Only one instance may be active at a time — the
    global lock enforces that.
    """

    _global_lock = threading.Lock()
    _active_instance: "Optional[AutoInstrumenter]" = None

    def __init__(self, prof):
        self._prof = prof
        self._step_idx = 0
        self._step_ctx = None
        self._rec = None
        self._prev_step_end: Optional[float] = None
        # nn.Module.__call__ is re-entered for every submodule; we want
        # the "top-level" forward pass, so we track re-entry depth
        # thread-locally (training normally runs on one thread, but the
        # local is defensive).
        self._depth = threading.local()
        # Live phase context managers keyed by phase name so re-entry
        # (e.g. gradient accumulation) accumulates cleanly.
        self._phase_cm: dict = {}
        # For restore-on-stop.
        self._orig_module_call = None
        self._orig_tensor_backward = None
        self._orig_optimizer_init = None
        self._patched_optimizers: list = []  # (cls, orig_step)

    # -- lifecycle ---------------------------------------------------

    def start(self) -> bool:
        """Install hooks. Returns False if torch isn't installed or
        another AutoInstrumenter is already active."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        with self._global_lock:
            if AutoInstrumenter._active_instance is not None:
                return False
            AutoInstrumenter._active_instance = self
        self._patch_module_call()
        self._patch_tensor_backward()
        self._patch_optimizer_step()
        # Open step 0 lazily — on the first Module.__call__.
        return True

    def stop(self) -> None:
        with self._global_lock:
            if AutoInstrumenter._active_instance is not self:
                return
            AutoInstrumenter._active_instance = None
        # Close any in-flight phase / step so the run doesn't hang
        # with a half-written step.
        for name in list(self._phase_cm):
            self._exit_phase(name)
        if self._step_ctx is not None:
            try: self._step_ctx.__exit__(None, None, None)
            except Exception: pass
            self._step_ctx = None
            self._rec = None
        self._unpatch()

    # -- hooks -------------------------------------------------------

    def _patch_module_call(self) -> None:
        import torch.nn as nn
        orig = nn.Module.__call__
        if getattr(orig, _MARKER, False):
            return
        state = self

        @functools.wraps(orig)
        def wrapped(self_mod, *args, **kwargs):
            # Skip the hook during eval / no_grad blocks. Otherwise a
            # mid-training validation pass opens a step, piles val
            # forward time into the next training step's `forward_s`,
            # and reports a poisoned MFU. This is the classic "eval
            # every N steps" pattern → we must be inert.
            try:
                import torch
                if (not torch.is_grad_enabled()
                        or not self_mod.training):
                    return orig(self_mod, *args, **kwargs)
            except Exception:
                pass
            depth = getattr(state._depth, "d", 0)
            state._depth.d = depth + 1
            top = (depth == 0)
            try:
                if top:
                    state._maybe_open_step()
                    state._enter_phase("forward")
                return orig(self_mod, *args, **kwargs)
            finally:
                if top:
                    state._exit_phase("forward")
                state._depth.d = depth

        setattr(wrapped, _MARKER, True)
        self._orig_module_call = orig
        nn.Module.__call__ = wrapped

    def _patch_tensor_backward(self) -> None:
        import torch
        orig = torch.Tensor.backward
        if getattr(orig, _MARKER, False):
            return
        state = self

        @functools.wraps(orig)
        def wrapped(self_tensor, *args, **kwargs):
            state._maybe_open_step()
            state._enter_phase("backward")
            try:
                return orig(self_tensor, *args, **kwargs)
            finally:
                state._exit_phase("backward")

        setattr(wrapped, _MARKER, True)
        self._orig_tensor_backward = orig
        torch.Tensor.backward = wrapped

    def _patch_optimizer_step(self) -> None:
        """`Optimizer.step` is overridden per subclass (SGD, AdamW, …),
        so walk the subclass tree and wrap each concrete `.step`.

        Also hook `Optimizer.__init__`: HF Trainer and DeepSpeed
        construct their optimizers lazily *inside* `trainer.train()`,
        which is often called from inside our `profile()` block. The
        __init__ hook catches those late arrivals — otherwise we'd
        never see their `.step()` and `_close_step` would never fire,
        leaving the step context perpetually open.
        """
        import torch
        state = self

        def wrap(orig):
            @functools.wraps(orig)
            def hooked(self_opt, *args, **kwargs):
                state._maybe_open_step()
                state._enter_phase("optimizer")
                try:
                    return orig(self_opt, *args, **kwargs)
                finally:
                    state._exit_phase("optimizer")
                    state._close_step()
            setattr(hooked, _MARKER, True)
            return hooked

        def wrap_class(cls) -> None:
            if "step" not in cls.__dict__: return
            orig = cls.step
            if getattr(orig, _MARKER, False): return
            self._patched_optimizers.append((cls, orig))
            cls.step = wrap(orig)

        def walk(cls):
            yield cls
            for sub in cls.__subclasses__():
                yield from walk(sub)

        for cls in walk(torch.optim.Optimizer):
            wrap_class(cls)

        # __init__ hook: wrap the class of any Optimizer subclass
        # instance we haven't seen yet. Adds a few μs to construction.
        orig_init = torch.optim.Optimizer.__init__
        if not getattr(orig_init, _MARKER, False):
            @functools.wraps(orig_init)
            def hooked_init(self_opt, *args, **kwargs):
                orig_init(self_opt, *args, **kwargs)
                # Only wrap if we're still the active instrumenter —
                # otherwise stop() has run and we should be inert.
                if AutoInstrumenter._active_instance is state:
                    wrap_class(type(self_opt))
            setattr(hooked_init, _MARKER, True)
            self._orig_optimizer_init = orig_init
            torch.optim.Optimizer.__init__ = hooked_init
        else:
            self._orig_optimizer_init = None

    def _unpatch(self) -> None:
        """Restore each patched target — but only if OUR wrapper is
        still the outermost one. If another library (Lightning,
        DeepSpeed, HF Accelerate) layered a patch on top of ours
        between start and stop, restoring here would silently kill
        their instrumentation. Safer to leave and let ours no-op."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return
        if self._orig_module_call is not None:
            if getattr(nn.Module.__call__, _MARKER, False):
                nn.Module.__call__ = self._orig_module_call
            self._orig_module_call = None
        if self._orig_tensor_backward is not None:
            if getattr(torch.Tensor.backward, _MARKER, False):
                torch.Tensor.backward = self._orig_tensor_backward
            self._orig_tensor_backward = None
        for cls, orig in self._patched_optimizers:
            try:
                if getattr(cls.step, _MARKER, False):
                    cls.step = orig
            except Exception: pass
        self._patched_optimizers.clear()
        if self._orig_optimizer_init is not None:
            try:
                import torch
                if getattr(torch.optim.Optimizer.__init__, _MARKER, False):
                    torch.optim.Optimizer.__init__ = self._orig_optimizer_init
            except Exception: pass
            self._orig_optimizer_init = None

    # -- step + phase state machine ----------------------------------

    def _maybe_open_step(self) -> None:
        if self._step_ctx is not None:
            return
        # Attribute the wait since the previous step's optimizer.step
        # exit as `dataloader_wait_s`. Give `wrap_dataloader` (the
        # explicit form the user opted into) precedence: only fill in
        # the pending wait if nothing already set it.
        if (self._prev_step_end is not None
                and not self._prof._pending_prefetch_wait_s):
            wait = time.perf_counter() - self._prev_step_end
            self._prof._pending_prefetch_wait_s = max(0.0, wait)
        self._step_ctx = self._prof.step(self._step_idx)
        self._rec = self._step_ctx.__enter__()
        self._step_idx += 1

    def _close_step(self) -> None:
        if self._step_ctx is None:
            return
        try: self._step_ctx.__exit__(None, None, None)
        except Exception: pass
        self._step_ctx = None
        self._rec = None
        self._prev_step_end = time.perf_counter()

    def _enter_phase(self, name: str) -> None:
        if self._rec is None:
            return
        # If a phase of the same name is already open (nested Module
        # calls we couldn't fully depth-track, or unusual patterns),
        # close it before starting the new timer.
        if name in self._phase_cm:
            self._exit_phase(name)
        cm = self._rec.phase(name)
        cm.__enter__()
        self._phase_cm[name] = cm

    def _exit_phase(self, name: str) -> None:
        cm = self._phase_cm.pop(name, None)
        if cm is not None:
            try: cm.__exit__(None, None, None)
            except Exception: pass
