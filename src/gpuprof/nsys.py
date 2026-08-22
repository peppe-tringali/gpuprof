"""nsys (NVIDIA Nsight Systems) integration.

Two pieces:

1. **Capture-range control** — `prof.nsys_capture()` context manager
   that brackets a code region with `cudaProfilerStart`/`Stop`. The
   user launches with
   `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`
   and gets an nsys report scoped to exactly the code they wrapped.
   Useful for capturing "just the problematic 5 steps" at full nsys
   fidelity without a multi-GB whole-run trace.

2. **Report import** — `python -m gpuprof.nsys_import <sqlite> --run-id N`
   parses an nsys SQLite export and inserts per-kernel timings into
   the run's `trace_windows` table so they appear next to the rest of
   the run's data.

    To produce the input:
        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
            -o run.nsys-rep python train.py
        nsys export --type sqlite --output run.sqlite run.nsys-rep
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager


@contextmanager
def nsys_capture():
    """Bracket the enclosed code with cudaProfilerStart/Stop.

    A no-op if torch isn't installed. Silent if the runtime call
    fails (e.g. running under `nsys profile` without the capture
    range flag).
    """
    try:
        import torch
        torch.cuda.cudart().cudaProfilerStart()
        active = True
    except Exception:
        active = False
    try:
        yield active
    finally:
        if active:
            try:
                import torch
                torch.cuda.cudart().cudaProfilerStop()
            except Exception:
                pass


# ------------------------------------------------------------------
# nsys SQLite import
# ------------------------------------------------------------------

def import_nsys_sqlite(nsys_db: str, gpuprof_db: str, run_id: int,
                       bucket_s: float = 1.0) -> int:
    """Parse an nsys SQLite export and add per-window kernel aggregates
    to a gpuprof run's `trace_windows` table.

    Args:
        nsys_db: path to `nsys export --type sqlite` output.
        gpuprof_db: gpuprof SQLite DB path (client-side or server-side).
        run_id: run to attach the imported traces to.
        bucket_s: aggregation window in seconds (default 1.0).

    Returns number of windows inserted.
    """
    src = sqlite3.connect(nsys_db)
    try:
        # nsys export schema: CUPTI_ACTIVITY_KIND_KERNEL has columns
        # start, end (nanoseconds since profile start), demangledName
        # (kernel name), globalPid, deviceId.
        try:
            rows = src.execute(
                "SELECT k.start, k.end, s.value AS name "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id = k.demangledName "
                "ORDER BY k.start"
            ).fetchall()
        except sqlite3.OperationalError:
            # Older nsys export schema uses `shortName` instead.
            rows = src.execute(
                "SELECT k.start, k.end, s.value AS name "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id = k.shortName "
                "ORDER BY k.start"
            ).fetchall()
    finally:
        src.close()

    if not rows:
        return 0

    # Aggregate by (window, kernel_name).
    t0_ns = rows[0][0]
    bucket_ns = int(bucket_s * 1e9)
    windows: dict[int, dict[str, dict]] = {}
    for start_ns, end_ns, name in rows:
        wkey = (start_ns - t0_ns) // bucket_ns
        w = windows.setdefault(wkey, {})
        agg = w.setdefault(name, {"name": name, "count": 0,
                                    "self_device_us": 0.0})
        agg["count"] += 1
        agg["self_device_us"] += (end_ns - start_ns) / 1000.0

    dst = sqlite3.connect(gpuprof_db)
    try:
        # Ensure schema exists.
        from .store import apply_schema
        apply_schema(dst)
        cur = dst.cursor()
        inserted = 0
        for wkey in sorted(windows):
            kernels = list(windows[wkey].values())
            kernels.sort(key=lambda k: k["self_device_us"], reverse=True)
            kernels = kernels[:50]
            t_start_rel = (wkey * bucket_ns) / 1e9
            t_end_rel = ((wkey + 1) * bucket_ns) / 1e9
            cur.execute(
                "INSERT INTO trace_windows(run_id, t_start_rel, t_end_rel, "
                "kernels_json) VALUES (?,?,?,?)",
                (run_id, t_start_rel, t_end_rel, json.dumps(kernels)),
            )
            inserted += 1
        dst.commit()
    finally:
        dst.close()
    return inserted


def _cli() -> None:
    ap = argparse.ArgumentParser(prog="python -m gpuprof.nsys_import")
    ap.add_argument("nsys_sqlite", help="path to `nsys export --type sqlite` output")
    ap.add_argument("--gpuprof-db", required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--bucket-s", type=float, default=1.0,
                    help="aggregation window in seconds")
    args = ap.parse_args()
    n = import_nsys_sqlite(args.nsys_sqlite, args.gpuprof_db,
                            args.run_id, bucket_s=args.bucket_s)
    print(f"imported {n} trace windows into run {args.run_id}")


if __name__ == "__main__":
    _cli()
