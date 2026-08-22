"""Minimal end-to-end demo — no torch, no GPU required.

Uses `gpuprof.profile()` (the recommended one-liner form) to exercise
every layer of the stack: NVML/mock sampler, host sampler, step +
phase timing, and the insight rules. Run it against the mock GPU
backend or a real one — set `GPUPROF_MOCK=1 GPUPROF_MOCK_GPUS=4` for
a mocked 4-GPU node.

    GPUPROF_MOCK=1 GPUPROF_MOCK_GPUS=4 \\
    GPUPROF_SERVER=http://localhost:8000 \\
    python examples/toy_train.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gpuprof


def main() -> None:
    with gpuprof.profile("toy-run",
                          meta={"model": "toy-mlp", "batch_size": 32}) as prof:
        print(f"run_id={prof._run_id}  gpu={prof.gpu_name}", flush=True)
        for i in range(60):
            with prof.step(i) as s:
                with s.phase("dataloader_wait"):
                    # Occasional slow batch to exercise the stall rule.
                    time.sleep(random.uniform(0.02, 0.10 if i % 7 else 0.30))
                with s.phase("forward"):   time.sleep(0.05)
                with s.phase("backward"):  time.sleep(0.07)
                with s.phase("optimizer"): time.sleep(0.01)
                s.record(loss=1.0 / (i + 1) + random.random() * 0.1,
                          tokens=32 * 512)
            if i % 10 == 0:
                print(f"  step {i:3d}", flush=True)
    print(f"Done. Inspect with: gpuprof insights ./gpuprof.db 1")


if __name__ == "__main__":
    main()
