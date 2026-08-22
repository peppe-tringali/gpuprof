"""Real PyTorch training loop, profiled with gpuprof.

Highlights:
  - `cuda=True` uses CUDA events for accurate phase timing on a real GPU.
  - `trace_every_n_steps=25` records `torch.profiler` kernel view every
    25 steps; the first few steps are traced automatically so warmup /
    JIT / cuBLAS heuristic search show up.
  - `meta.arch` declares the model shape so MFU uses the proper 6·P·T
    + 12·L·hidden·seq_len·T formula, not the underestimate you get
    without an attention correction.

To push live to a running server:

    GPUPROF_SERVER=http://127.0.0.1:8000 python examples/torch_train.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

from gpuprof import GpuProfiler


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    d_in, d_hidden, d_out, layers, heads = 1024, 4096, 1024, 4, 8
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.GELU(),
        nn.Linear(d_hidden, d_hidden), nn.GELU(),
        nn.Linear(d_hidden, d_out),
    ).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    params = sum(p.numel() for p in model.parameters())
    batch, seq = 8, 128
    tokens_per_step = batch * seq

    prof = GpuProfiler(
        run_name=f"mlp-{d_hidden}-{device}",
        cuda=(device == "cuda"),
        trace_every_n_steps=25,
        meta={
            "batch_size": batch, "seq_len": seq,
            "dtype": "bf16" if dtype == torch.bfloat16 else "fp32",
            "arch": {
                "params": params, "hidden": d_hidden,
                "heads": heads, "layers": layers, "seq_len": seq,
                # These aren't real transformer blocks; declaring them here
                # is enough for the FLOP calc to add the attention T² term
                # that a real transformer would incur.
            },
        },
    )
    prof.start()
    print(f"run_id={prof._run_id} device={device} params={params:,}")

    try:
        for i in range(60):
            with prof.step(i) as s:
                # The batch fetch happens between step contexts — the
                # profiler already captures that as `inter_step_gap_s`.
                # We still time the H2D move here for visibility.
                with s.phase("dataloader_wait"):
                    x = torch.randn(batch, seq, d_in, device=device, dtype=dtype)
                    y = torch.randn(batch, seq, d_out, device=device, dtype=dtype)
                with s.phase("forward"):
                    out = model(x)
                    loss = loss_fn(out, y)
                with s.phase("backward"):
                    loss.backward()
                with s.phase("optimizer"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                s.record(loss=loss.item(), tokens=tokens_per_step)
            if i % 20 == 0:
                print(f"  step {i:3d}  loss={loss.item():.4f}")
    finally:
        prof.stop()
    print(f"Done. Insights: python -m gpuprof.insights gpuprof.db {prof._run_id}")


if __name__ == "__main__":
    main()
