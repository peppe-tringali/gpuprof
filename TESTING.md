# Testing gpuprof

Six ways to test, from fastest to most realistic. Each is self-contained
— pick the one that matches what you're checking.

## Prerequisites

Two Python venvs (already set up in the repo):

- `.venv/` — Python 3.14, no torch. Fast; used for unit tests, auth,
  server, mock backend, offline drain.
- `.venv-torch/` — Python 3.13 with torch + Lightning + Transformers.
  Used for the continuous profiler, framework integration tests, and
  the real PyTorch example.

If you're setting up fresh:

```bash
# base venv
python3.14 -m venv .venv
.venv/bin/pip install -e ".[server,postgres,dev]"   # dev extra pulls in psutil

# torch venv (needs Python 3.13 — torch has no 3.14 wheels yet)
/usr/local/bin/python3.13 -m venv .venv-torch
.venv-torch/bin/pip install -e ".[server,postgres,dev]" torch pytorch-lightning transformers
```

Optional extras beyond the defaults:

- `pip install -e ".[host]"` — installs `psutil` for the host sampler
  (CPU / disk / mem) that powers the CPU-bound vs I/O-bound rules.
  Already included in `[dev]`, so if you installed dev deps you have
  it.

---

## 1. Unit + integration tests (~30 s, no GPU needed)

Every code change should keep these green.

```bash
.venv/bin/python -m pytest tests/                       # 84 tests
.venv/bin/python -m pytest tests/ -v                    # verbose
.venv/bin/python -m pytest tests/test_insights.py -v    # just the rules
.venv/bin/python -m pytest tests/ -k drain -v           # only drain tests
.venv/bin/python -m pytest tests/ --tb=long             # full tracebacks
```

For torch-specific paths (continuous profiler, Lightning callback,
DeepSpeed adapter):

```bash
.venv-torch/bin/python -m pytest tests/                 # same 84, different skips
```

Both venvs should print `84 passed, N skipped`. The skipped tests are
the ones that only make sense in the *other* venv (e.g. "raises when
Lightning is not installed" only runs on the venv without Lightning).

## 2. Local end-to-end with the mock backend (~1 min, no GPU)

Boot the server + dashboard, push a mocked training run into it, look
at the browser.

**Terminal A — server:**

```bash
.venv/bin/gpuprof serve --port 8000
```

Open <http://127.0.0.1:8000> in a browser.

**Terminal B — mocked 4-GPU training run:**

```bash
GPUPROF_MOCK=1 GPUPROF_MOCK_GPUS=4 \
GPUPROF_SERVER=http://127.0.0.1:8000 \
.venv/bin/python examples/toy_train.py
```

You should see:

- A run appear in the run list.
- Run-detail view with 4 utilization lines (one per mocked GPU).
- Stacked-bar phase breakdown updating live.
- One or two insights firing on the dataloader stall the toy loop
  simulates.

## 3. Real PyTorch on a Mac / CPU box (~2 min)

Same shape but with a real MLP going through actual autograd,
`torch.profiler` traces every 25 steps, and MFU calculated from the
declared architecture.

```bash
# Server
.venv/bin/gpuprof serve --port 8000

# Client (torch venv)
GPUPROF_SERVER=http://127.0.0.1:8000 \
.venv-torch/bin/python examples/torch_train.py
```

Expectations:

- MFU is low on CPU (that's correct — the low-MFU rule fires).
- The kernels table populates from real `aten::mm` / `aten::addmm` calls.
- Insights include compilation warmup if step 0 or 1 was much slower
  than steady state.

## 4. Auth + rate limit + optional Postgres (~2 min)

Exercise the hardened auth surface.

```bash
# Server with all the flags
GPUPROF_SECRET=stable-secret \
.venv/bin/gpuprof serve --port 8000 \
    --api-key WRITE_KEY --viewer-password viewpass
# Optionally: --storage-url postgres://user:pass@host:5432/db
```

In another terminal:

```bash
# 401 without a session cookie
curl -s -o /dev/null -w "no cookie:   HTTP %{http_code}\n" \
    http://127.0.0.1:8000/api/runs                                    # → 401

# Login flow
curl -s -c /tmp/c -o /dev/null -w "bad password: HTTP %{http_code}\n" \
    -X POST -H "Content-Type: application/json" \
    -d '{"password":"wrong"}' http://127.0.0.1:8000/api/login         # → 401

curl -s -c /tmp/c -o /dev/null -w "good login:   HTTP %{http_code}\n" \
    -X POST -H "Content-Type: application/json" \
    -d '{"password":"viewpass"}' http://127.0.0.1:8000/api/login      # → 200

curl -s -b /tmp/c -o /dev/null -w "with cookie:  HTTP %{http_code}\n" \
    http://127.0.0.1:8000/api/runs                                    # → 200

# CSRF: /api/logout requires X-XSRF-TOKEN
XSRF=$(grep XSRF-TOKEN /tmp/c | awk '{print $NF}')
curl -s -b /tmp/c -o /dev/null -w "no CSRF:     HTTP %{http_code}\n" \
    -X POST http://127.0.0.1:8000/api/logout                          # → 403

curl -s -b /tmp/c -o /dev/null -w "with CSRF:   HTTP %{http_code}\n" \
    -X POST -H "X-XSRF-TOKEN: $XSRF" \
    http://127.0.0.1:8000/api/logout                                  # → 200

# Write side: ingest requires X-API-Key
curl -s -o /dev/null -w "no api-key:  HTTP %{http_code}\n" \
    -X POST -H "Content-Type: application/json" \
    -d '{"name":"t"}' http://127.0.0.1:8000/api/runs                  # → 403

# Rate limit — 15 bad logins in a row should hit 429 partway through
for i in $(seq 1 15); do
    curl -s -o /dev/null -w "%{http_code} " \
        -X POST -H "Content-Type: application/json" \
        -d '{"password":"wrong"}' http://127.0.0.1:8000/api/login
done; echo
# Expect: 401 401 ... 429 429 429 429 429
```

## 5. Offline recovery flow (~3 min)

Verifies the disk-buffer + drain path — kill the server mid-run,
restart it, confirm the client backfills.

**Short outage (recovers in-band):**

```bash
# Terminal A
.venv/bin/gpuprof serve --port 8000 --db recovery.db

# Terminal B — client pushes for ~10 seconds
GPUPROF_SERVER=http://127.0.0.1:8000 GPUPROF_MOCK=1 \
.venv/bin/python examples/toy_train.py
```

During those ~10 s, kill the server (Ctrl-C in Terminal A), wait 3
seconds, restart with the same `--db`. The client's `prof.stop()` has
a 15 s drain window and will backfill automatically.

**Long outage (client exited before server came back):**

```bash
# Kill the server, let the client finish, THEN restart the server.
# Files land in ~/.gpuprof/buffer/. Recover them offline:
.venv/bin/gpuprof drain --server http://127.0.0.1:8000
# → drained N events, files removed on success
```

## 6. Distributed (needs multiple CUDA GPUs or a mocked multi-rank setup)

On a machine with multiple CUDA GPUs:

```bash
GPUPROF_SERVER=http://<server>:8000 GPUPROF_API_KEY=... \
torchrun --nproc-per-node=2 examples/ddp_train.py --group experiment-1

# Cross-rank bucket-skew analysis:
.venv/bin/gpuprof insights gpuprof_server.db --group experiment-1
```

Without real GPUs, you can simulate two ranks with different pacing to
exercise the group insight rule:

```bash
.venv/bin/gpuprof serve --port 8000 &

# Rank 0 (fast)
GPUPROF_SERVER=http://127.0.0.1:8000 GPUPROF_MOCK=1 python -c "
from gpuprof import GpuProfiler
import time
p = GpuProfiler(run_name='r0', db_path=None, rank=0, world_size=2,
                group_id='sim', meta={})
p.start()
for i in range(30):
    with p.step(i) as s:
        with s.phase('forward'):  time.sleep(0.02)
        with s.phase('backward'): time.sleep(0.04)
p.stop()"

# Rank 1 (40% slower — should trigger rank-skew rule)
GPUPROF_SERVER=http://127.0.0.1:8000 GPUPROF_MOCK=1 python -c "
from gpuprof import GpuProfiler
import time
p = GpuProfiler(run_name='r1', db_path=None, rank=1, world_size=2,
                group_id='sim', meta={})
p.start()
for i in range(30):
    with p.step(i) as s:
        with s.phase('forward'):  time.sleep(0.028)
        with s.phase('backward'): time.sleep(0.056)
p.stop()"

# Cross-rank analysis
curl -s http://127.0.0.1:8000/api/groups/sim/insights | python3 -m json.tool
```

## 6b. Diagnosing WHY the GPU is starving

If a training run's GPU sits idle and you want the tool to distinguish
between "CPU-bound augmentations", "I/O-bound disk", and "cold cache",
the profiler now collects host-side samples via psutil (CPU per core,
memory pressure, disk read/write throughput, IOPS). Three rules fire
on top of the existing dataloader-stall detection:

- **`rule_cpu_bound_dataloader`** — stall + CPU pegged, disk idle →
  "Move augmentations to GPU; more workers won't help."
- **`rule_io_bound_dataloader`** — stall + high disk bandwidth or
  high IOPS with CPU headroom → "WebDataset / tarball packing / RAM
  cache; the storage tier is the bottleneck."
- **`rule_cold_cache`** — early-run disk-read rate much higher than
  steady state AND step time speeds up → "First-epoch cache warmup;
  pre-stage or size RAM."

Simulate a CPU-bound run (no GPU required):

```bash
GPUPROF_MOCK=1 .venv/bin/python -c "
import time, sys
sys.path.insert(0, 'src')
from gpuprof import GpuProfiler
from gpuprof.host_sampler import HostSample

# Disable psutil sampling so the test is deterministic; feed synthetic
# host samples that look like a CPU-bound pipeline.
prof = GpuProfiler(run_name='cpu-bound-demo', host_sampling=False, meta={})
prof.start()
t0 = time.time()
for i in range(15):
    prof._on_host_sample(HostSample(
        t=t0+i, cpu_percent=95.0, cpu_max_percent=100.0, n_cpus=16,
        mem_used_bytes=int(40e9), mem_total_bytes=int(64e9),
        disk_read_bps=2e6, disk_write_bps=0, disk_iops=8,
    ))
for i in range(30):
    with prof.step(i) as s:
        with s.phase('forward'):  time.sleep(0.005)
        with s.phase('backward'): time.sleep(0.010)
    time.sleep(0.08)                # simulated dataloader stall
prof.stop()
"
.venv/bin/gpuprof insights gpuprof.db 1
```

Expected verdict:

```
[HIGH]  Dataloader stall + CPU pegged (95% avg across 16 cores)
        — workers are CPU-bound
        Adding more workers won't help ... move augmentations to
        GPU (torchvision.transforms.v2 GPU path, kornia) ...
```

Swap `cpu_percent` for low and `disk_read_bps` for high (e.g. 800e6)
to see the I/O-bound rule fire instead.

Four more discriminating rules live alongside the CPU/IO ones and
require the same `host_sampling` path:

- **`rule_prefetch_queue_starved`** — fires when many steps show long
  waits from `wrap_dataloader`. Points at `prefetch_factor`.
- **`rule_cache_thrashing`** — page cache is at its memory-limited
  size but disk reads keep coming (working set > RAM, distinct from
  cold-cache warmup).
- **`rule_worker_imbalance`** — one child process is hot, siblings
  aren't (uneven sharding / `worker_init_fn`).
- **`rule_host_memory_pressure`** — swap-in > 0 (pipeline fighting
  OS for RAM).

To exercise the prefetch-queue-starved rule directly:

```python
from gpuprof import GpuProfiler

class SlowLoader:
    def __iter__(self):
        import time
        for _ in range(100):
            time.sleep(0.100)         # 100 ms per batch → queue starved
            yield {"x": 1}

prof = GpuProfiler(run_name="starved", host_sampling=False, meta={})
prof.start()
for i, batch in enumerate(prof.wrap_dataloader(SlowLoader())):
    with prof.step(i) as s:
        with s.phase("forward"):
            import time; time.sleep(0.005)
    if i >= 30: break
prof.stop()
# → "Prefetch queue starved on 100% of steps (p95 wait 100 ms)"
```

## 7. Offline insights on a completed run

```bash
.venv/bin/gpuprof insights gpuprof.db 1               # human-readable
.venv/bin/gpuprof insights gpuprof.db 1 --json        # machine-readable
.venv/bin/gpuprof insights gpuprof.db --group my-run  # cross-rank
```

---

## Debugging — where to look when something is off

| Symptom | Look at |
|---|---|
| Server auth failures | `python -m gpuprof.server` stderr — logs bad-password IPs |
| No data on the dashboard | `sqlite3 gpuprof_server.db 'SELECT COUNT(*) FROM samples;'` |
| Client side data missing | `sqlite3 gpuprof.db '.tables'`, then `SELECT * FROM runs;` |
| Client couldn't reach server | `ls ~/.gpuprof/buffer/` — files here mean the client buffered but couldn't drain; run `gpuprof.drain` |
| WS not updating | Browser DevTools → Network → WS → `/watch/<id>` — batch messages should arrive every ~1 s |
| Server exits on SIGTERM but data missing | Should not happen (lifespan flushes the writer thread); if it does, check the writer log line "gpuprof-server-writer" |
| SQLite "database is locked" | `sqlite3 gpuprof_server.db 'PRAGMA journal_mode;'` — should print `wal`; if not, the DB was created before v0.4 |

## Good first test to run right now

Simplest smoke, no GPU needed, exercises ~90% of the code:

```bash
.venv/bin/python -m pytest tests/                    # confirm all 84 pass
.venv/bin/gpuprof serve --port 8000 &     # server
sleep 2
GPUPROF_MOCK=1 GPUPROF_MOCK_GPUS=4 GPUPROF_SERVER=http://127.0.0.1:8000 \
    .venv/bin/python examples/toy_train.py
# then open http://127.0.0.1:8000 in a browser
```

You should see a mocked 4-GPU run with live-updating charts, phase
breakdown, and one or two insights fired. That's the whole product
loop working end to end.
