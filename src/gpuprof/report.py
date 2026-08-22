"""`gpuprof report DB RUN --out=report.html` — self-contained HTML.

Renders a single file with the run's headline, per-phase timings,
insight verdicts, and inline SVG sparklines for the sampled series.
No external CSS or JS — the whole file is one `<html>` document you
can attach to a PR, email, or drop into a shared drive.

Not the same as the live dashboard: this is a *snapshot* meant for
sharing after the run. If you want interactivity, use `gpuprof
serve` + browser.
"""
from __future__ import annotations

import argparse
import html
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Optional


# ---------- Small helpers -------------------------------------------

def _fmt_ms(v):    return "—" if v is None else f"{v*1000:.1f} ms"
def _fmt_pct(v):   return "—" if v is None else f"{v*100:.1f}%"
def _fmt_gb(v):    return "—" if v is None else f"{v/1e9:.1f} GB"
def _fmt_usd(v):   return "—" if v is None else f"${v:,.2f}"


def _load_run(conn, run_id: int) -> dict:
    row = conn.execute(
        "SELECT id, name, gpu_name, started_at, ended_at, "
        "group_id, rank, world_size, meta_json FROM runs WHERE id=?",
        (run_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"run_id {run_id} not found in DB")
    return {
        "id": row[0], "name": row[1], "gpu": row[2],
        "started_at": row[3], "ended_at": row[4],
        "group_id": row[5], "rank": row[6], "world_size": row[7],
        "meta": json.loads(row[8] or "{}"),
    }


def _load_step_series(conn, run_id: int) -> dict:
    rows = conn.execute(
        "SELECT step, t_start, t_end, forward_s, backward_s, "
        "optimizer_s, dataloader_wait_s, comm_s, loss "
        "FROM steps WHERE run_id=? ORDER BY step",
        (run_id,),
    ).fetchall()
    if not rows:
        return {"step_ms": [], "fw": [], "bw": [], "op": [], "dl": [], "cm": [], "loss": []}
    return {
        "step_ms": [(r[2] - r[1]) * 1000 for r in rows],
        "fw":  [(r[3] or 0) * 1000 for r in rows],
        "bw":  [(r[4] or 0) * 1000 for r in rows],
        "op":  [(r[5] or 0) * 1000 for r in rows],
        "dl":  [(r[6] or 0) * 1000 for r in rows],
        "cm":  [(r[7] or 0) * 1000 for r in rows],
        "loss": [r[8] for r in rows if r[8] is not None],
    }


def _load_sample_series(conn, run_id: int) -> dict:
    """Aggregate samples across GPUs so the report shows one line."""
    rows = conn.execute(
        "SELECT sm_util, mem_used_bytes, power_w "
        "FROM samples WHERE run_id=? ORDER BY t",
        (run_id,),
    ).fetchall()
    return {
        "util":  [r[0] for r in rows if r[0] is not None],
        "mem":   [r[1] for r in rows if r[1] is not None],
        "power": [r[2] for r in rows if r[2] is not None],
    }


# ---------- SVG sparkline -------------------------------------------

def _sparkline(values, width: int = 480, height: int = 60,
                stroke: str = "#4f8bf9",
                fill: Optional[str] = "rgba(79,139,249,0.15)",
                y_min: Optional[float] = None,
                y_max: Optional[float] = None) -> str:
    """Return inline SVG for a small line chart. No external deps."""
    if not values:
        return f'<svg width="{width}" height="{height}"><text x="8" y="30" fill="#8590a6" font-size="11">no data</text></svg>'
    vs = list(values)
    mn = y_min if y_min is not None else min(vs)
    mx = y_max if y_max is not None else max(vs)
    if mx == mn: mx = mn + 1
    n = len(vs)
    def _x(i): return (i / max(1, n - 1)) * (width - 8) + 4
    def _y(v): return height - 4 - ((v - mn) / (mx - mn)) * (height - 8)
    pts = " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(vs))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}">']
    if fill:
        area = pts + f" {_x(n-1):.1f},{height-4:.1f} {_x(0):.1f},{height-4:.1f}"
        parts.append(f'<polygon points="{area}" fill="{fill}" />')
    parts.append(f'<polyline points="{pts}" fill="none" '
                  f'stroke="{stroke}" stroke-width="1.5" />')
    # Baseline min/max labels.
    parts.append(f'<text x="4" y="12" fill="#8590a6" font-size="10">{mx:.2f}</text>')
    parts.append(f'<text x="4" y="{height-4}" fill="#8590a6" font-size="10">{mn:.2f}</text>')
    parts.append('</svg>')
    return "".join(parts)


# ---------- Report HTML ---------------------------------------------

_CSS = """
:root { color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1d24; --muted: #6b7280; --border: #e5e7eb;
  --card: #f9fafb; --accent: #4f8bf9; --ok: #52c41a; --warn: #f5a623; --err: #ff4d4f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1017; --fg: #e6e8ee; --muted: #8590a6;
          --border: #232a34; --card: #161b22; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
       font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }
.wrap { max-width: 900px; margin: 0 auto; }
header { border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 20px; }
h1 { margin: 0 0 4px 0; font-size: 24px; }
.subtitle { color: var(--muted); font-size: 13px; }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 16px 18px; margin-bottom: 16px; }
.card h2 { margin: 0 0 10px 0; font-size: 11px; font-weight: 600;
           color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
         gap: 20px; margin-bottom: 4px; }
.stat .n { font-size: 22px; font-weight: 600; }
.stat .u { font-size: 12px; color: var(--muted); margin-left: 3px; }
.stat .l { font-size: 11px; color: var(--muted); text-transform: uppercase;
           letter-spacing: 0.05em; margin-top: 4px; }
.insight { padding: 12px 14px; border-radius: 6px; border-left: 3px solid;
           margin-bottom: 8px; background: rgba(120,120,120,0.06); }
.insight.high { border-color: var(--err); }
.insight.medium { border-color: var(--warn); }
.insight.low { border-color: var(--ok); }
.insight .title { font-weight: 600; margin-bottom: 4px; font-size: 14px; }
.insight .rec { color: var(--muted); font-size: 13px; line-height: 1.5; }
.footer { color: var(--muted); font-size: 11px; margin-top: 32px; text-align: center; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 12px; border-bottom: 1px solid var(--border);
         font-size: 13px; text-align: left; }
th { color: var(--muted); font-weight: 500; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.05em; }
"""


def _render_html(run: dict, insights_report: dict, steps: dict,
                  samples: dict) -> str:
    s = insights_report.get("summary", {})
    e = html.escape
    def esc(x): return e(str(x)) if x is not None else "—"

    # Stat tiles
    def _stat(label, value, unit=""):
        return (f'<div class="stat"><div><span class="n">{value}</span>'
                f'<span class="u">{unit}</span></div>'
                f'<div class="l">{label}</div></div>')

    stats_html = "".join([
        _stat("steps", s.get("n_steps", 0)),
        _stat("avg step", f"{(s.get('avg_step_s') or 0)*1000:.1f}", "ms"),
        _stat("MFU", (f"{s['mfu']*100:.1f}"
                     if s.get('mfu') is not None else "—"), "%"),
        _stat("avg SM util", (f"{(s.get('avg_sm_util') or 0)*100:.0f}"), "%"),
        _stat("peak mem", (f"{(s.get('max_mem_bytes') or 0)/1e9:.1f}"
                           if s.get('max_mem_bytes') else "—"), "GB"),
    ])

    # Insights
    insight_cards = []
    for it in insights_report.get("insights", []):
        sev = it.get("severity", "low")
        insight_cards.append(
            f'<div class="insight {sev}">'
            f'<div class="title">[{sev.upper()}] {e(it["title"])}</div>'
            f'<div class="rec">{e(it.get("recommendation", ""))}</div>'
            "</div>"
        )
    insights_html = ("\n".join(insight_cards)
                     if insight_cards
                     else '<div class="rec">(no insights)</div>')

    # Sparklines
    def _series_block(title, values, unit, stroke="#4f8bf9"):
        svg = _sparkline(values, stroke=stroke)
        n_pts = f" · {len(values)} pts" if values else ""
        return (f'<div class="card"><h2>{title}{n_pts}</h2>{svg}'
                f'<div class="subtitle">unit: {unit}</div></div>')

    when = (esc(run.get("started_at") and __import__("time").strftime(
        "%Y-%m-%d %H:%M UTC",
        __import__("time").gmtime(run["started_at"]))) or "—")

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>gpuprof · {e(run['name'])}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <header>
    <h1>{e(run['name'])}</h1>
    <div class="subtitle">run {run['id']} · {e(run.get('gpu'))} · started {when}</div>
  </header>

  <div class="card">
    <h2>headline</h2>
    <div class="stats">{stats_html}</div>
  </div>

  <div class="card">
    <h2>insights ({len(insights_report.get('insights', []))})</h2>
    {insights_html}
  </div>

  {_series_block("step time", steps.get('step_ms') or [], "ms")}
  {_series_block("SM utilization", [v*100 for v in (samples.get('util') or [])], "%")}
  {_series_block("memory used", [v/1e9 for v in (samples.get('mem') or [])], "GB", stroke="#b57edc")}
  {_series_block("power", samples.get('power') or [], "W", stroke="#f5a623")}
  {_series_block("loss", steps.get('loss') or [], "", stroke="#52c41a")}

  <div class="card">
    <h2>phase averages</h2>
    <table>
      <thead><tr><th>phase</th><th>avg</th></tr></thead>
      <tbody>{
      "".join(
          f"<tr><td>{k}</td><td>{_fmt_ms((s.get('phase_avg_s') or {}).get(k))}</td></tr>"
          for k in ("dataloader_wait", "forward", "backward", "optimizer", "comm")
      )
      }</tbody>
    </table>
  </div>

  <div class="footer">
    Generated by gpuprof · self-contained · no external dependencies
  </div>
</div></body></html>"""


# ---------- CLI -----------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        prog="gpuprof report",
        description=(
            "Render a self-contained HTML report for a completed run. "
            "One file, no external CSS/JS — safe to attach to a PR "
            "or share via email."
        ),
    )
    ap.add_argument("db", help="gpuprof SQLite DB (e.g. ./gpuprof.db)")
    ap.add_argument("run_id", type=int, help="run id in the DB")
    ap.add_argument("--out", default=None,
                    help="output HTML path (default: report-<name>.html)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        ap.error(f"no DB at {args.db!r}")

    conn = sqlite3.connect(args.db)
    try:
        run = _load_run(conn, args.run_id)
        steps = _load_step_series(conn, args.run_id)
        samples = _load_sample_series(conn, args.run_id)
    finally:
        conn.close()

    from .insights import analyze
    r = analyze(args.db, args.run_id)

    out = args.out or f"report-{run['name']}-{run['id']}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(_render_html(run, r, steps, samples))
    print(f"wrote {out}")


if __name__ == "__main__":
    _cli()
