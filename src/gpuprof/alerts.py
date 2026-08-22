"""End-of-run webhook alerts (Slack / generic).

    gpuprof.profile("run", webhook="https://hooks.slack.com/services/...")

Fires one POST when the run ends, containing the run's headline
metrics and the top insights. Best-effort: a webhook failure never
kills training. Auto-detects the payload shape from the URL — Slack
webhooks get Slack-native `attachments`; anything else gets a plain
JSON body.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


def post_end_of_run_alert(
    webhook_url: str,
    db_path: str,
    run_id: int,
    *,
    timeout: float = 5.0,
) -> bool:
    """Post the end-of-run report to `webhook_url`. Returns True on
    HTTP 2xx; False otherwise (network, non-2xx, missing DB, etc.).
    Never raises — the training-run summary printer stays intact
    even if the alert fails."""
    try:
        from .insights import analyze
        r = analyze(db_path, run_id)
    except Exception:
        return False
    payload = _build_payload(webhook_url, r)
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def _is_slack(url: str) -> bool:
    return "hooks.slack.com" in url


def _build_payload(url: str, report: dict) -> dict:
    s = report.get("summary", {})
    insights = report.get("insights", [])
    high = [i for i in insights if i.get("severity") == "high"]
    med  = [i for i in insights if i.get("severity") == "medium"]

    if _is_slack(url):
        return _slack_payload(s, insights, high, med)
    # Generic webhook — a stable JSON schema that other tools can
    # ingest (Discord, PagerDuty via a transform, homerolled bots…).
    return {
        "kind": "gpuprof.end_of_run",
        "run_id": s.get("run_id"),
        "name": s.get("name"),
        "gpu": s.get("gpu"),
        "rank": s.get("rank"),
        "world_size": s.get("world_size"),
        "n_steps": s.get("n_steps"),
        "avg_step_ms": (s.get("avg_step_s") or 0) * 1000,
        "mfu": s.get("mfu"),
        "train_s": s.get("train_s"),
        "insights": [
            {"severity": i["severity"], "title": i["title"]}
            for i in insights
        ],
    }


def _slack_payload(s: dict, insights: list, high: list, med: list) -> dict:
    """Slack-native payload with color-coded attachments."""
    mfu_str = (f" · MFU {s['mfu']*100:.1f}%"
               if s.get("mfu") is not None else "")
    step_str = (f" · {s['n_steps']} steps"
                if s.get("n_steps") else "")
    when = (f" in {s['train_s']:.0f}s"
            if s.get("train_s") else "")
    headline = (f"*gpuprof · {s.get('name', 'run')}* finished"
                f"{step_str}{mfu_str}{when}")

    # One attachment per insight, colored by severity.
    color = {"high": "danger", "medium": "warning", "low": "good"}
    attachments = []
    for it in insights[:8]:                 # cap so Slack doesn't truncate
        rec = it.get("recommendation") or ""
        attachments.append({
            "color": color.get(it.get("severity"), "good"),
            "title": it["title"],
            "text": rec[:800] if rec else "",
            "footer": f"severity: {it.get('severity')}",
        })
    text_parts = [headline]
    if high:
        text_parts.append(f":rotating_light: *{len(high)} high-severity* "
                          + ("issues" if len(high) > 1 else "issue"))
    return {
        "text": "\n".join(text_parts),
        "attachments": attachments,
    }
