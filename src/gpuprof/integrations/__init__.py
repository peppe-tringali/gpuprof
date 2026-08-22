"""Framework integrations for gpuprof.

Each integration is an import-on-use adapter — the underlying framework
(Lightning, HF Trainer, DeepSpeed) is imported at construct time so
this package doesn't require any of them to be installed.
"""
from .lightning import GpuProfilerCallback as LightningCallback
from .hf_trainer import GpuProfilerTrainerCallback as HFTrainerCallback
from .deepspeed import wrap_deepspeed_engine

__all__ = [
    "LightningCallback",
    "HFTrainerCallback",
    "wrap_deepspeed_engine",
]
