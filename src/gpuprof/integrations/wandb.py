"""W&B integration.

    import gpuprof
    with gpuprof.profile("baseline", wandb=True):
        # your standard PyTorch training loop
        ...

Or explicit:

    prof = gpuprof.start("baseline")
    gpuprof.attach_wandb(prof)
    ...
    prof.stop()

What lands in W&B:

- **Per-step metrics** — `gpuprof/step_time_ms`, `gpuprof/forward_ms`,
  `gpuprof/backward_ms`, `gpuprof/optimizer_ms`,
  `gpuprof/dataloader_wait_ms`, `gpuprof/loss`. Logged with the same
  `step` value your existing `wandb.log` uses.
- **Run summary** — `gpuprof/mfu`, `gpuprof/avg_step_ms`,
  `gpuprof/avg_sm_util`, `gpuprof/peak_mem_gb`. Fixed after the run.
- **Insights** — a text block in `wandb.run.summary["gpuprof/insights"]`
  formatted as Markdown so the W&B UI renders it nicely.

Requires `wandb`. If not installed or no active run, `attach_wandb`
silently no-ops.
"""
from __future__ import annotations

from typing import Optional


def attach_wandb(prof, *, prefix: str = "gpuprof") -> bool:
    """Attach the profiler to the active `wandb.run`. Returns True if
    successful, False otherwise (wandb missing or no active run)."""
    try:
        import wandb
    except ImportError:
        return False
    if wandb.run is None:
        return False

    p = prefix

    def on_step(step_dict: dict) -> None:
        def _ms(k): return (step_dict.get(k) or 0.0) * 1000.0
        payload = {
            f"{p}/step_time_ms": (step_dict["t_end"] - step_dict["t_start"]) * 1000.0,
            f"{p}/inter_step_gap_ms": _ms("inter_step_gap_s"),
            f"{p}/dataloader_wait_ms": _ms("dataloader_wait_s"),
            f"{p}/forward_ms":   _ms("forward_s"),
            f"{p}/backward_ms":  _ms("backward_s"),
            f"{p}/optimizer_ms": _ms("optimizer_s"),
            f"{p}/comm_ms":      _ms("comm_s"),
        }
        if step_dict.get("loss") is not None:
            payload[f"{p}/loss"] = step_dict["loss"]
        if step_dict.get("flops") and step_dict.get("t_end") and step_dict.get("t_start"):
            dt = step_dict["t_end"] - step_dict["t_start"]
            if dt > 0:
                payload[f"{p}/tflops_per_step"] = step_dict["flops"] / dt / 1e12
        try:
            wandb.log(payload, step=step_dict["step"], commit=False)
        except Exception:
            pass  # best-effort — never let logging break training

    def on_end() -> None:
        _post_summary_to_wandb(prof, prefix=p)

    prof.on_step(on_step)
    prof.on_end(on_end)
    return True


def _post_summary_to_wandb(prof, *, prefix: str) -> None:
    """After the run ends, load the insights report and publish it to
    W&B as summary metrics + a Markdown text block."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    if prof._run_id is None or prof._local is None:
        return  # nowhere to read insights from
    db_path = prof._local._path
    try:
        from ..insights import analyze
        r = analyze(db_path, prof._run_id)
    except Exception:
        return

    s = r.get("summary", {})
    p = prefix
    numeric = {}
    for k, sk in [("mfu", "mfu"), ("avg_step_s", "avg_step_ms"),
                   ("avg_sm_util", "avg_sm_util"),
                   ("inter_step_gap_s", "avg_dataloader_wait_ms")]:
        v = s.get(k)
        if v is None: continue
        if sk.endswith("_ms"): v *= 1000.0
        numeric[f"{p}/{sk}"] = v
    if s.get("max_mem_bytes"):
        numeric[f"{p}/peak_mem_gb"] = s["max_mem_bytes"] / 1e9
    if s.get("peak_tflops"):
        numeric[f"{p}/peak_tflops"] = s["peak_tflops"]
    try:
        for k, v in numeric.items():
            wandb.run.summary[k] = v
    except Exception:
        pass

    md = _insights_markdown(r)
    try:
        wandb.run.summary[f"{p}/insights"] = md
    except Exception:
        pass


def _insights_markdown(r: dict) -> str:
    """Format the insight list as Markdown so W&B renders it nicely
    in the run overview."""
    lines = []
    s = r.get("summary", {})
    lines.append(f"### {s.get('name', 'run')} · gpuprof insights")
    if s.get("mfu") is not None:
        lines.append(f"**MFU:** {s['mfu']*100:.1f}% "
                     f"(peak {s.get('peak_tflops', 0):.0f} TFLOPs)")
    if s.get("avg_step_s"):
        lines.append(f"**Avg step:** {s['avg_step_s']*1000:.1f} ms")
    lines.append("")
    if not r.get("insights"):
        return "\n".join(lines)
    for it in r["insights"]:
        badge = {"high": "🔴 **HIGH**",
                 "medium": "🟡 **MED**",
                 "low": "🟢 LOW"}.get(it["severity"], "•")
        lines.append(f"- {badge} — {it['title']}")
        rec = it.get("recommendation")
        if rec:
            lines.append(f"  <br>_{rec}_")
    return "\n".join(lines)
