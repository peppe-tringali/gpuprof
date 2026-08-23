<div align="center">

# gpuprof

**Live GPU profiler + insight engine for PyTorch training.**

Tells you *why* your GPU is slow, not just that it is.

[![PyPI](https://img.shields.io/pypi/v/gpuprof.svg)](https://pypi.org/project/gpuprof/)
[![CI](https://github.com/peppe-tringali/gpuprof/actions/workflows/test.yml/badge.svg)](https://github.com/peppe-tringali/gpuprof/actions/workflows/test.yml)
[![python](https://img.shields.io/pypi/pyversions/gpuprof.svg)](https://pypi.org/project/gpuprof/)
[![license](https://img.shields.io/pypi/l/gpuprof.svg)](https://github.com/peppe-tringali/gpuprof/blob/main/LICENSE)
[![downloads](https://img.shields.io/pypi/dm/gpuprof.svg)](https://pypi.org/project/gpuprof/)

</div>

---

## Why gpuprof

Every ML observability tool shows you numbers. `nvidia-smi` gives you SM utilization. `torch.profiler` gives you a chrome trace. Weights & Biases logs your loss. When your throughput drops, you're left staring at graphs, guessing.

**gpuprof answers the question in one sentence.**

> **[HIGH]** *Dataloader stall + CPU pegged (95% avg across 16 cores) — workers are CPU-bound. Move augmentations to GPU; `num_workers` won't help.*

> **[MEDIUM]** *Kernel drift: `aten::mm` added 7.2 ms/window in the second half. Look for memory fragmentation or cuBLAS autotune cache invalidation.*

> **[HIGH]** *Bucket 3: rank 1 is late on 100% of steps (p95 delta 20.0 ms). One collective is the bottleneck, not the whole loop.*

23 rules today, each with a specific diagnosis and a specific fix.

---

## 30-second quickstart

```bash
pip install "gpuprof[server,host,nvidia,torch]"
```

For a bare install (mock-GPU mode, useful for CI): `pip install gpuprof`

Add two lines to your training script:

```python
import gpuprof

with gpuprof.profile("my-experiment"):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Run it. **No server, no browser, nothing else.** When the loop exits, a summary block with the top insights prints to stdout — that's the whole product loop working.

```
────────────────────────────────────────────────────────────
gpuprof · my-experiment  (200 steps, avg 121.3 ms)
  Insights (4):
  [HIGH]  34% of step time waiting on the loader (inter-step gap)
  [MED]   MFU 12.4% of 989 TFLOPs bf16 peak
  [LOW]   Estimated cost: $0.14 (2 min · $12.29/h)
  Full report: gpuprof insights gpuprof.db 1
────────────────────────────────────────────────────────────
```

**No wrapping, no `prof.step()`, no `s.phase()`.** Inside the context we patch `nn.Module.__call__`, `Tensor.backward`, and `Optimizer.step`; step boundaries fall out of `optimizer.step()`, per-phase timings from the other two, and dataloader wait is the gap between one step's end and the next model call. On exit everything unpatches — safe even if the loop raises.

To see the report again later:

```bash
gpuprof insights          # latest run in ./gpuprof.db
gpuprof insights gpuprof.db 42   # a specific run
```

## Optional: live browser dashboard

Add `dashboard=True`:

```python
with gpuprof.profile("my-experiment", dashboard=True):
    ...
```

Prints a URL on the first line of output; open it in a browser and watch charts update while the run happens. The dashboard is an in-process server on a free port — no separate `gpuprof serve` command. It stays reachable until your Python process exits.

## Optional: push to a shared team server

Keep a dashboard running for the whole team:

```bash
gpuprof serve --port 8000 --api-key WRITE_KEY --viewer-password ...
```

Then any training run pushed to that server appears in the shared dashboard:

```bash
GPUPROF_SERVER=http://<host>:8000 GPUPROF_API_KEY=WRITE_KEY \
    python train.py
```

Users log in via `--viewer-password`. Postgres backend available via `--storage-url postgres://…`.

## Manual API (opt out of auto-instrumentation)

For unusual training loops (RL, LBFGS, inference-only) or if you don't have torch installed:

```python
with gpuprof.profile("my-experiment", auto=False) as prof:
    for i, batch in enumerate(prof.wrap_dataloader(dataloader)):
        with prof.step(i) as s:
            with s.phase("forward"):   loss = model(batch)
            with s.phase("backward"):  loss.backward()
            with s.phase("optimizer"): opt.step(); opt.zero_grad()
```

Auto instrumentation handles gradient accumulation (multiple fwd+bwd per opt.step accumulate into one step), mixed precision (`GradScaler.step()` still calls the underlying optimizer.step), DDP wrappers, and lazily-constructed optimizers (HF Trainer, DeepSpeed).

## One-line integrations for existing workflows

```python
# Alongside your existing W&B run — MFU + insights land there:
with gpuprof.profile("baseline", wandb=True):
    ...

# Slack alert on completion:
with gpuprof.profile("baseline", webhook="https://hooks.slack.com/services/..."):
    ...
```

Everything is opt-in and can combine. `wandb=True` no-ops without an active `wandb.run`; `webhook=...` never kills training on failure.

### Comparing against your previous runs

Every run written to the same local DB is automatically compared to prior runs of the same `run_name`:

```
[MED]   Regression vs last 3 runs of 'baseline' (median):
        step time 152 ms vs 108 ms (41% slower)
```

This is the "did my change help?" signal — a rule that runs as part of the normal insight pass, no extra command.

### What the run cost

Insights include a cost line by default:

```
[LOW]   Estimated cost: $47.20 (231 min · 8× GPU @ $12.29/h);
        ~$30.68 of that is idle capacity (MFU 35.0%)
```

Override the hourly rate with `GPUPROF_GPU_RATE=<usd_per_hour>` if your fleet's price differs from the reference AWS on-demand list.

```bash
GPUPROF_SERVER=http://localhost:8000 python train.py
```

That's it. Live charts, MFU calc, and insights update in the browser as your run progresses.

---

## Even less code — framework adapters

Already using PyTorch Lightning, HuggingFace Trainer, or DeepSpeed? One line:

```python
from gpuprof import LightningCallback
trainer = pl.Trainer(callbacks=[LightningCallback(run_name="baseline")])
```

```python
from gpuprof import HFTrainerCallback
trainer = Trainer(..., callbacks=[HFTrainerCallback(run_name="baseline")])
```

```python
from gpuprof import wrap_deepspeed_engine
engine = wrap_deepspeed_engine(engine, run_name="baseline")
# use engine(batch) / engine.backward() / engine.step() as normal
engine.close()
```

The callback / wrapper handles start/stop, per-phase timing, dataloader-wait attribution, and all the metrics below — you don't wrap anything by hand.

---

## What it detects — all 21 rules

**Data pipeline (7)** — the "why is my GPU starving?" family:

| Rule | Fires when | Says what to do |
|---|---|---|
| `dataloader_stall` | inter-step gap or explicit wait > 15% of step | Increase `num_workers`, `prefetch_factor`, `pin_memory`, GPU augs |
| `prefetch_queue_starved` | wrap_dataloader wait > 30 ms on ≥20% of steps | Raise `prefetch_factor`; `persistent_workers=True` |
| `cpu_bound_dataloader` | stall + avg CPU > 80%, disk idle | Move augs to GPU (`kornia`, `torchvision.transforms.v2`) |
| `io_bound_dataloader` | stall + disk BW > 500 MB/s or IOPS > 5k with CPU headroom | WebDataset / RAM cache / faster storage tier |
| `cold_cache` | first-third disk read ≥ 3× last-third + step time speeds up | Warm cache before timing; pre-stage to local disk |
| `cache_thrashing` | page cache ≥ 40% RAM + sustained > 100 MB/s disk read | Working set > RAM; shard / more RAM / WebDataset |
| `pcie_saturation` | PCIe RX peak > 20 GB/s | Bandwidth-bound; do augs on GPU; NVLink for multi-GPU |

**Host & scheduling (2):**

| Rule | Fires when | Says what to do |
|---|---|---|
| `worker_imbalance` | hottest worker > 2.5× fair share | Fix `Sampler` sharding / `worker_init_fn` |
| `host_memory_pressure` | swap-in > 1 MB/s | Reduce prefetch / worker pool; the OS is thrashing |

**Compute (6):**

| Rule | Fires when | Says what to do |
|---|---|---|
| `low_mfu` | MFU < 30% of dtype peak | Batch, FlashAttention, bf16, `torch.compile` |
| `small_batch` | SM util > 60% but MFU < 25% | Kernels too small — bigger batch, fuse ops |
| `kernel_launch_overhead` | step > phase sum by > 10% | `torch.compile` / CUDA graphs |
| `gradient_checkpointing_detected` | backward/forward ratio implies (un)declared ckpt | Declare `grad_checkpoint` in arch (or check wrapping) |
| `sdpa_suboptimal` | softmax + bmm hot without a flash kernel | Use `F.scaled_dot_product_attention`; install FlashAttention |
| `kernel_drift` | a kernel's per-window time grew > 50% between halves | Memory fragmentation / cuBLAS cache invalidation |

**Memory & thermals (2):**

| Rule | Fires when | Says what to do |
|---|---|---|
| `memory_pressure` | GPU peak memory > 90% of total | Gradient checkpointing, bf16 activations, ZeRO / FSDP |
| `thermal_throttling` | temp > 82°C + SM clock down > 15% | Improve airflow; check fans |

**Warmup & tail (3):**

| Rule | Fires when | Says what to do |
|---|---|---|
| `compilation_warmup` | first step > 3× steady-state median | Exclude steps 0–2 from throughput measurements |
| `first_step_outlier` | step 0 > 10× median-of-rest | cuBLAS autotune / torch.compile guards — filter it |
| `high_step_variance` | p99/p50 > 2 | Logging/checkpoint spikes; GC pauses; worker scheduling |

**Distributed (2):**

| Rule | Fires when | Says what to do |
|---|---|---|
| `comm_dominant` | NCCL comm > 20% of step | Bucketing, `gradient_as_bucket_view`, `no_sync()`, FSDP |
| `bucket_skew` (group) | one rank late on same bucket ≥ 60% of steps | One collective, one rank — check topology, thermals |

---

## Distributed training — first-class

```bash
torchrun --nproc-per-node=8 train.py
```

That's the whole setup on the training side. `gpuprof` auto-detects `RANK`, `WORLD_SIZE`, and `TORCHELASTIC_RUN_ID` from `torchrun`; every rank pushes to the same server with the same `group_id`. The dashboard shows a **group view** for the whole distributed run, plus:

- **Cross-rank clock alignment.** At `start()`, `dist.barrier()` establishes a shared epoch; then rank 0 does 64 NCCL ping-pongs with each peer to estimate offset to ~10–100 μs. Bucket-skew analysis subtracts offsets before comparing.
- **Per-collective bucket timing** via a DDP comm hook. CUDA events give sub-microsecond precision per rank.
- **Rank-skew rule** across the group. Same rank persistently late on the same bucket → the tool tells you which one.

```bash
gpuprof insights ./gpuprof.db --group my-experiment
# → "Rank skew: rank 3 median 87 ms vs rank 0 62 ms (40% slower)"
# → "Bucket 12: rank 3 is late on 84% of steps (p95 delta 24.7 ms)"
```

---

## Dashboard

Two ways to open it:

- **In-process** — pass `dashboard=True` to `profile()`. Spawns a server on a free port; the URL prints. Dies with your Python process.
- **Standalone** — `gpuprof serve --port 8000` in a separate terminal, then browse `http://localhost:8000`. Persistent; multiple runs / users can share it.

Both serve the same single-page vanilla-JS SPA:

**Live during a run:**
- Per-GPU utilization / memory / power with LTTB downsampling (24 h runs don't freeze the browser)
- Stacked-bar step-time breakdown (dataloader / forward / backward / optimizer / comm)
- Top-kernels table from `torch.profiler` traces
- Insights panel updating live
- Crosshair synced across all charts + hover tooltips + time-window selector

**Cross-run:**
- Compare view (side-by-side headline stats, per-run insight panels)
- Group view for distributed runs (rank overlay + bucket-skew insight)

**Auth:**
- Two independent axes — write API key (`X-API-Key`) + viewer session (HMAC-signed cookie + double-submit CSRF token)
- Login rate limit (10/min per IP)
- Multiple write keys accepted for rotation

---

## Trace modes — from cheap to full-detail

| Mode | Cost | When to use |
|---|---|---|
| `sample_hz=100` (NVML) | ~0.1% CPU | Always on; captures util / memory / power / temp / PCIe |
| `trace_every_n_steps=100` | ~20% overhead on the traced step | Periodic per-kernel snapshots + auto-trace step 0/1/5 for warmup |
| `trace_range=(a,b)` | ~20% overhead on each step in range | Burst-capture a specific window at kernel granularity |
| `continuous_traces_hz=1.0` | ~2–5% steady CPU | Rolling per-second kernel aggregates for the whole run — powers the "kernel drift" rule |
| `nsys profile … + gpuprof nsys-import` | full nsys report per capture window | When you need a chrome-trace timeline |

The `nsys` integration is genuine: `prof.nsys_capture()` brackets a code region with `cudaProfilerStart/Stop`, and `gpuprof nsys-import` parses `nsys export --type sqlite` output and attaches kernel timings to the run in gpuprof's schema.

---

## Storage

- **SQLite** (default) — one file per client, one for the server. Zero setup.
- **Postgres** — `gpuprof serve --storage-url postgres://user:pass@host/db`. Uses `psycopg_pool` on reads. Ready for hundreds of concurrent viewers.
- **On-disk buffer + retry** on the client. If the server drops, batches spool to `~/.gpuprof/buffer/run-<id>.jsonl` and drain automatically when the server returns. If the client exits first, `gpuprof drain --server URL` recovers them.

---

## Architecture

```
┌──────────────────────┐         ┌────────────────────┐         ┌──────────────┐
│  Training process    │         │   gpuprof server   │         │   Browser    │
│                      │         │                    │         │              │
│  gpuprof.profile()   │         │  FastAPI +         │         │  Vanilla-JS  │
│  ├─ NVML sampler     │ ─HTTP─▶ │  SQLite/Postgres   │ ─WS──▶  │  SPA         │
│  ├─ host sampler     │  ~1 Hz  │  writer thread     │  batch  │              │
│  ├─ step + phases    │  batch  │  insights engine   │         │              │
│  ├─ torch.profiler   │         │  auth + rate-limit │         │              │
│  ├─ DDP comm hook    │         │                    │         │              │
│  └─ disk buffer      │  drain  │                    │         │              │
│    (~/.gpuprof/…)    │ ◀─fail─ │                    │         │              │
└──────────────────────┘         └────────────────────┘         └──────────────┘
```

Every component is non-blocking to the training thread:
- Sampler + host sampler run on background daemon threads.
- Store uses a bounded queue + batched writer thread.
- Remote pusher batches every 1 s over HTTP; failed batches spill to disk.
- Server writer thread flushes in transactions; SQLite is in WAL mode so control-plane writes don't block ingest.

---

## Command-line

```bash
gpuprof serve         # dashboard + ingest server
gpuprof insights      # offline insights on a DB
gpuprof drain         # push orphaned buffer files
gpuprof nsys-import   # merge an nsys trace into a run
gpuprof version
```

Each subcommand has its own `--help`.

---

## When gpuprof is a good fit

- ✅ You want continuous, opinionated feedback during training runs.
- ✅ You want to self-host without a cloud MLOps account.
- ✅ You want a tool a small team can `pip install` and be running in a minute.
- ✅ You want distributed-training diagnosis (rank skew, per-bucket comm) without setting up DCGM + Grafana.

## When to use something else

- 🔎 Full CUPTI event timeline / chrome-trace viewer — use `nsys` directly. `gpuprof nsys-import` bridges the two.
- 📊 Deep experiment tracking (hyperparameter sweeps, artifact management) — pair with W&B or MLflow.
- 🏭 Production cluster-wide GPU inventory — DCGM Exporter + Grafana is that job.

---

## Testing

See [`TESTING.md`](TESTING.md) for the full guide. Two quick smokes:

**Just the tool works?**

```bash
pytest tests/                                     # ~140 unit + integration tests
```

**Just the dashboard works?**

```bash
python test_dashboard.py                          # in-process dashboard, mock 4 GPUs
```

Prints a URL; open it in a browser; charts update live for ~30 seconds. Ctrl-C to exit. No NVIDIA hardware required — the sampler has a mock backend for development.

---

## Design notes

- **Non-blocking to training.** Sampler / remote / writer all run on daemon threads with bounded queues; under backpressure events drop rather than block `.step()`.
- **Everything is opt-in.** Continuous traces, CUDA-event comm timing, host sampling, nsys capture — each can be turned off with a kwarg.
- **Graceful degradation.** `pynvml` missing → mock backend. `psutil` missing → host rules silently skip. `torch` missing → traces skip. The tool still runs.
- **One-schema-fits-all.** Client SQLite and server SQLite/Postgres share the same schema shape; the same insights CLI runs against either DB.
- **111 unit + integration tests.** Every rule has synthetic-context coverage plus a round-trip through the store.

---

## Contributing

The codebase is ~5.5k lines of Python + a 900-line vanilla-JS SPA. Layout:

```
src/gpuprof/
  ├─ profiler.py         core: sampler + step context + DDP hook
  ├─ sampler.py          NVML / mock GPU backend
  ├─ host_sampler.py     psutil host CPU / disk / mem
  ├─ continuous.py       rolling torch.profiler kernel aggregates
  ├─ traces.py           per-step torch.profiler snapshots
  ├─ nsys.py             nsys capture + import
  ├─ flops.py            TransformerArch + MFU math (attn T², MoE, LoRA)
  ├─ insights.py         21 rules + group analyzer + CLI
  ├─ remote.py           HTTP pusher + disk-buffer recovery
  ├─ store.py            client SQLite writer
  ├─ drain.py            offline buffer drain CLI
  ├─ integrations/       Lightning / HF Trainer / DeepSpeed adapters
  └─ server/
      ├─ app.py          FastAPI routes + WS fan-out
      ├─ auth.py         session cookie + CSRF + rate limit
      ├─ store.py        server SQLite backend
      ├─ pg_store.py     Postgres backend (psycopg_pool)
      └─ static/         single-file SPA
```

Adding a new insight rule is one function + one test. See `insights.py` — every rule is a small pure function that returns a dict or `None`.

---

## License

MIT.
