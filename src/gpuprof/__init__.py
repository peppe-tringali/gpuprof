"""gpuprof — a GPU profiler + insight engine for PyTorch training.

The one-line integration is:

    import gpuprof
    with gpuprof.profile("my-experiment"):
        # standard PyTorch loop — no wrapping needed
        for batch in loader:
            loss = model(batch); loss.backward()
            optimizer.step(); optimizer.zero_grad()

You do NOT need to run a server for this to work. The tool writes to
`./gpuprof.db` locally; insights print to stdout at the end of the
run; and `gpuprof insights` (no args) shows the report again later.

If you want a live browser dashboard:

    with gpuprof.profile("my-experiment", dashboard=True):
        ...

spawns an in-process server on a free port and prints the URL.
"""
from __future__ import annotations

import os
import sys
import time as _time
from contextlib import contextmanager
from typing import Optional

from .profiler import GpuProfiler
from .integrations import (
    LightningCallback,
    HFTrainerCallback,
    wrap_deepspeed_engine,
)
from .integrations.wandb import attach_wandb

# Single source of truth for the version — pulled from installed
# package metadata so pyproject.toml is authoritative. Editable
# installs report the version from the installed .dist-info; a bare
# `python -c "import gpuprof"` without a build works because setuptools
# writes a stub .egg-info at install time.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("gpuprof")
except Exception:  # pragma: no cover — fallback for uninstalled trees
    __version__ = "0.0.0+unknown"

__all__ = [
    "GpuProfiler",
    "start", "profile",
    "LightningCallback", "HFTrainerCallback", "wrap_deepspeed_engine",
    "attach_wandb",
    "__version__",
]


def _default_run_name() -> str:
    """Sensible auto-name: script basename + short timestamp."""
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "run"
    if script.endswith(".py"):
        script = script[:-3]
    if not script:
        script = "run"
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    return f"{script}-{stamp}"


def start(run_name: Optional[str] = None, **kwargs) -> GpuProfiler:
    """Create and start a profiler with sensible defaults.

    - `run_name` defaults to `<script>-<YYYYMMDD-HHMMSS>`.
    - Server URL / API key are picked up from `GPUPROF_SERVER` /
      `GPUPROF_API_KEY` if not passed.
    - Rank / world_size / group_id are auto-detected from torchrun
      env vars (`RANK`, `WORLD_SIZE`, `TORCHELASTIC_RUN_ID`).

    All other kwargs pass through to `GpuProfiler`.
    """
    prof = GpuProfiler(run_name or _default_run_name(), **kwargs)
    prof.start()
    return prof


@contextmanager
def profile(
    run_name: Optional[str] = None,
    *,
    auto: bool = True,
    summary: bool = True,
    dashboard: bool = False,
    wandb: bool = False,
    webhook: Optional[str] = None,
    **kwargs,
):
    """Zero-config profiler context manager.

        import gpuprof
        with gpuprof.profile("my-experiment"):
            # standard PyTorch training loop
            ...

    Behavior:
    - `auto=True` (default) monkey-patches `nn.Module.__call__`,
      `Tensor.backward`, and `Optimizer.step` so step + phase timings
      are captured with no code changes. Unpatched on exit.
    - `summary=True` (default) prints a headline + top-5 insights to
      stdout when the context exits. Answers "how did that run go?"
      without a separate command.
    - `dashboard=True` spawns an in-process browser dashboard on a
      free loopback port and points the profiler at it — WebSocket
      updates flow to the browser exactly like a remote setup, and
      you never had to run `gpuprof serve`.

    No server is required for the standard workflow — data lands in
    `./gpuprof.db` and `gpuprof insights` (no args) reads it back.
    """
    # Where the data ends up — the summary printer and any downstream
    # `gpuprof insights` call read from this path.
    db_path = kwargs.get("db_path", "gpuprof.db")

    dashboard_url = None
    if dashboard:
        try:
            from ._local_dashboard import start_local_dashboard
            # Point the in-process server at a shadow file so double-
            # writing from the client's local Store doesn't duplicate
            # rows. The dashboard reads the shadow; insights and the
            # end-of-run summary read from the persistent `db_path`
            # the local Store owns. If the server later crashes
            # mid-run, the client still has the local DB.
            shadow = str(db_path) + ".dashboard"
            dashboard_url = start_local_dashboard(shadow)
        except Exception as e:                              # pragma: no cover
            print(f"[gpuprof] dashboard start failed ({e}); "
                  "continuing without live dashboard.",
                  file=sys.stderr, flush=True)
        if dashboard_url is not None:
            print(f"[gpuprof] live dashboard: {dashboard_url}",
                  flush=True)
            kwargs["server_url"] = dashboard_url
            # KEY: keep the local Store enabled (do NOT set db_path=None).
            # The whole point is that a dashboard failure mid-run must
            # not destroy the training run's data.
        else:
            print(f"[gpuprof] dashboard failed to bind; "
                  "continuing with local file only.",
                  file=sys.stderr, flush=True)

    prof = start(run_name, **kwargs)

    # If the user is inside a framework loop, the dedicated adapter
    # gives cleaner step boundaries than the generic auto-instr.
    # Warn once so they know they're leaving signal on the table.
    _hint_framework_adapter(auto)

    if wandb:
        attach_wandb(prof)

    instr = None
    if auto:
        from .auto import AutoInstrumenter
        instr = AutoInstrumenter(prof)
        if not instr.start():
            instr = None                    # torch missing → explicit-only

    try:
        yield prof
    finally:
        if instr is not None:
            instr.stop()
        prof.stop()
        if summary and prof._run_id is not None:
            _print_run_summary(db_path, prof._run_id)
        if webhook and prof._run_id is not None:
            from .alerts import post_end_of_run_alert
            post_end_of_run_alert(webhook, db_path, prof._run_id)
        if dashboard_url is not None:
            print(f"[gpuprof] dashboard still at {dashboard_url} "
                  "until this process exits.", flush=True)


_FRAMEWORK_HINTED = False


def _hint_framework_adapter(auto: bool) -> None:
    """If the user's process has already imported Lightning / HF /
    DeepSpeed, they're probably inside a framework loop where the
    dedicated adapter would place step boundaries more precisely
    than the generic Module.__call__ patch. Warn once per process."""
    global _FRAMEWORK_HINTED
    if _FRAMEWORK_HINTED or not auto:
        return
    detected = []
    if "pytorch_lightning" in sys.modules or "lightning" in sys.modules:
        detected.append(("Lightning", "LightningCallback"))
    if "transformers" in sys.modules:
        detected.append(("HuggingFace Transformers", "HFTrainerCallback"))
    if "deepspeed" in sys.modules:
        detected.append(("DeepSpeed", "wrap_deepspeed_engine"))
    if not detected:
        return
    _FRAMEWORK_HINTED = True
    for fw, adapter in detected:
        print(f"[gpuprof] {fw} is imported — for cleaner phase "
              f"attribution consider `gpuprof.{adapter}` instead "
              "of the bare `profile()` call.",
              file=sys.stderr, flush=True)


def _print_run_summary(db_path: str, run_id: int) -> None:
    """One-glance summary printed to stdout when a run ends."""
    try:
        from .insights import analyze
        r = analyze(db_path, run_id)
    except Exception:
        return  # never let summary printing kill a run
    s = r["summary"]
    lines = ["", "─" * 60,
             f"gpuprof · {s['name']}  ({s['n_steps']} steps"]
    if s.get("avg_step_s"):
        lines[-1] += f", avg {s['avg_step_s']*1000:.1f} ms)"
    else:
        lines[-1] += ")"
    if s.get("mfu") is not None:
        lines.append(
            f"  MFU: {s['mfu']*100:.1f}% of {s['peak_tflops']:.0f} TFLOPs"
        )
    if s.get("inter_step_gap_s") is not None:
        lines.append(
            f"  inter-step gap: {s['inter_step_gap_s']*1000:.1f} ms"
        )
    lines.append(f"  Insights ({len(r['insights'])}):")
    for it in r["insights"][:5]:
        tag = {"high":   "  [HIGH]  ",
               "medium": "  [MED]   ",
               "low":    "  [LOW]   "}[it["severity"]]
        lines.append(f"{tag}{it['title']}")
    lines.append(f"  Full report: gpuprof insights {db_path} {run_id}")
    lines.append("─" * 60)
    print("\n".join(lines), flush=True)
