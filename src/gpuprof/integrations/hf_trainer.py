"""HuggingFace Transformers `TrainerCallback` adapter.

    from transformers import TrainingArguments, Trainer
    from gpuprof.integrations import HFTrainerCallback

    trainer = Trainer(
        model=model, args=TrainingArguments(...),
        train_dataset=ds,
        callbacks=[HFTrainerCallback(run_name="hf-baseline",
                                     meta={"arch": {...}, "dtype": "bf16"})],
    )
    trainer.train()

HF Trainer exposes step boundaries but not fine-grained
forward/backward/optimizer separately from the training-step method.
This callback captures:

- step timing (t_start, t_end)
- inter-step gap (auto)
- loss from `on_log`
- tokens_per_step if `TrainingArguments.include_num_input_tokens_seen`
  is on (Trainer computes this) or if the user records it manually

For a finer breakdown, subclass and override `training_step` to add
explicit phase context managers.
"""
from __future__ import annotations

from typing import Optional

from ..profiler import GpuProfiler


class GpuProfilerTrainerCallback:
    def __init__(self, prof: Optional[GpuProfiler] = None, **prof_kwargs):
        try:
            import transformers  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "gpuprof.integrations.HFTrainerCallback needs "
                "transformers installed"
            ) from e
        self._prof = prof or GpuProfiler(**prof_kwargs)
        self._step_ctx = None
        self._rec = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._prof.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self._step_ctx is not None:
            try: self._step_ctx.__exit__(None, None, None)
            except Exception: pass
            self._step_ctx = None
        self._prof.stop()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_ctx = self._prof.step(state.global_step)
        self._rec = self._step_ctx.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_ctx is not None:
            self._step_ctx.__exit__(None, None, None)
        self._step_ctx = None
        self._rec = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._rec is None or not logs:
            return
        loss = logs.get("loss") or logs.get("train_loss")
        if loss is not None:
            try: self._rec.record(loss=float(loss))
            except (TypeError, ValueError): pass


# Late-bind onto `transformers.TrainerCallback` when it's available so
# HF's callback dispatcher recognizes us as a first-class subclass.
try:
    from transformers import TrainerCallback as _HFCb
    # Same MRO trick as the Lightning callback — our methods must
    # shadow the framework's no-op stubs.
    GpuProfilerTrainerCallback = type(
        "GpuProfilerTrainerCallback",
        (GpuProfilerTrainerCallback, _HFCb),
        {},
    )
except ImportError:
    pass
