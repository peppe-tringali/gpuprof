# Changelog

All notable changes to gpuprof are documented here. This project
follows [SemVer](https://semver.org/); until the 1.0 release
minor-version bumps may contain schema migrations that require an
existing `gpuprof.db` to be rebuilt (they're always additive — old
DBs auto-migrate).

## [Unreleased]

### Added
- `LICENSE` (MIT).
- GitHub Actions CI workflow (`.github/workflows/test.yml`) running
  the mock-GPU test matrix on Linux + macOS across Python 3.11–3.13,
  plus a torch-enabled job on CPU.
- `CONTRIBUTING.md` with the "how to add an insight rule" recipe.
- `gpuprof selfcheck` — probes the environment (NVML, psutil, torch,
  distributed) so users can verify a real-hardware install in one
  command.
- `gpuprof gc` — retention CLI; delete runs older than N days, or
  keep only the last N per name.
- Auto-detected framework hint on `profile()`: if Lightning / HF /
  DeepSpeed are already imported in a training loop, a warning
  points the user at the dedicated adapter.
- Static HTML report export (`gpuprof report DB RUN --out=r.html`) —
  self-contained single-file report for offline sharing / PR
  attachments.
- Multi-tenant server auth: per-user API tokens with project scoping,
  layered on top of the existing single-key mode (fully backward
  compatible).

### Fixed
- Rate limiter memory leak — `Auth.RateLimiter` reaps empty deques
  and caps the tracked-keys dict so a long-running server can't
  accumulate one entry per unique IP forever.
- On-disk buffer files from prior runs are auto-swept at
  `profile()` start; users no longer need to remember
  `gpuprof drain` for orphaned buffers.
- Version is now sourced from `importlib.metadata` in a single
  place; no more drift between `pyproject.toml` and `__init__.py`.

## [0.6.0] — 2026-08-22

Big-jump release.

### Added — user-visible
- One-line integration: `with gpuprof.profile("name"):`. Auto-patches
  `nn.Module.__call__` / `Tensor.backward` / `Optimizer.step` for
  zero-code phase capture.
- End-of-run summary printed to stdout (no server required).
- In-process live dashboard via `dashboard=True`.
- W&B integration (`wandb=True`) — MFU + insights land in
  `wandb.run.summary`.
- Webhook/Slack alerts (`webhook="https://…"`).
- Regression detection rule — compares against prior runs of same
  `run_name`.
- Cost projection rule with per-GPU pricing table, overridable via
  `GPUPROF_GPU_RATE`.
- Host sampler (`psutil`) — CPU, memory, disk I/O, page cache, swap,
  per-worker CPU. Powers 7 new discriminating rules.
- Continuous kernel-aggregate profiling via rotating
  `torch.profiler`.
- `nsys` integration — capture context manager + SQLite import.
- Cross-rank clock alignment via NCCL ping-pong (opt-in).
- Distributed group insights: rank-skew + per-bucket comm timing.
- Postgres backend with connection pool.
- Framework adapters: Lightning, HF Trainer, DeepSpeed.
- Server auth: HMAC-signed session cookie + CSRF double-submit +
  login rate limit + multi-key rotation.
- Dashboard: LTTB downsampling, hover tooltips, crosshair sync,
  time-window selector, compare-runs view, group/rank overlay.

### Added — insight rules
- 23 rules total (up from 4 in the first pass). Grouped by concern:
  data pipeline (8), compute (6), memory + thermals (2), warmup (3),
  distributed (2), regression (1), cost (1).

### Fixed
- NVML `nvmlShutdown` refcount so nested samplers don't invalidate
  each other's handles.
- `nn.Module.__call__` hook skips eval/no_grad blocks so validation
  passes don't pollute the training step's `forward_s`.
- Monkey-patches restore only if our marker still owns the entry,
  never clobbering another library's patch that layered on top.
- SQLite writer-thread connection is now thread-local; no
  cross-thread close.
- `_build_ctx` no longer treats `t_start == 0.0` as "missing", which
  silently disabled MFU + cost calculations on the first run of a
  perf_counter epoch.

## [0.1.0] — 2026-08-21

Initial: NVML sampler, per-step + per-phase timing, SQLite writer,
4 insight rules.
