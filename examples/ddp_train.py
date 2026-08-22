"""Minimal DDP training loop with per-rank gpuprof.

Every rank spawns its own GpuProfiler, sharing a `group_id` so the
server groups them and the dashboard's group view shows all ranks
together. `prof.instrument_ddp(model)` registers a comm hook so
NCCL allreduce time is attributed to each step's `comm` phase.

Run:
    GPUPROF_SERVER=http://<server>:8000 GPUPROF_API_KEY=... \\
    torchrun --nproc-per-node=2 examples/ddp_train.py --group my-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpuprof import GpuProfiler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="shared group_id across ranks")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    d_in, d_hidden = 512, 2048
    model = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                          nn.Linear(d_hidden, d_in)).to(device)
    model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    prof = GpuProfiler(
        run_name=f"ddp-rank-{rank}",
        cuda=(device.startswith("cuda")),
        trace_every_n_steps=20,
        meta={"arch": {"params": sum(p.numel() for p in model.parameters()),
                       "hidden": d_hidden, "layers": 2, "seq_len": d_in},
              "dtype": "bf16" if device.startswith("cuda") else "fp32"},
        # rank / world_size are auto-detected from torch.distributed; the
        # group_id is the one thing you actually need to pass.
        group_id=args.group,
    )
    prof.instrument_ddp(model)      # NCCL allreduce → comm phase
    prof.start()
    if rank == 0:
        print(f"run_id={prof._run_id} group={args.group} world_size={world_size}")

    try:
        batch = 16
        for i in range(args.steps):
            with prof.step(i) as s:
                with s.phase("dataloader_wait"):
                    x = torch.randn(batch, d_in, device=device)
                    y = torch.randn(batch, d_in, device=device)
                with s.phase("forward"):
                    loss = loss_fn(model(x), y)
                with s.phase("backward"):
                    loss.backward()
                # `comm` phase is filled by the DDP hook we installed;
                # allreduce fires *during* backward but its time is
                # attributed here for readability.
                with s.phase("optimizer"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                s.record(loss=loss.item(), tokens=batch * d_in)
    finally:
        prof.stop()
        dist.destroy_process_group()
    if rank == 0:
        print(f"Done. Group insights: "
              f"python -m gpuprof.insights gpuprof.db --group {args.group}")


if __name__ == "__main__":
    main()
