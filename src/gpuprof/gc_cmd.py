"""`gpuprof gc` — retention CLI.

Two modes, either or both:

    gpuprof gc --older-than 30d    # delete runs older than 30 days
    gpuprof gc --keep-last 20      # keep only the 20 most recent
                                    # per run-name
    gpuprof gc --dry-run           # show what would go, delete nothing

Reads the same `./gpuprof.db` by default; `--db` for a different one.
Deletes are done in a single transaction so a crash mid-gc leaves
the DB consistent.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path


_DURATION_RE = re.compile(r"^(\d+)\s*([smhdw])$")

_UNIT_S = {
    "s": 1, "m": 60, "h": 3600,
    "d": 86_400, "w": 7 * 86_400,
}


def _parse_duration(s: str) -> float:
    m = _DURATION_RE.match(s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(
            f"bad duration {s!r} — try `30d`, `2w`, `12h`"
        )
    return int(m.group(1)) * _UNIT_S[m.group(2)]


def _run_ids_older_than(conn, cutoff: float) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM runs WHERE started_at < ? AND ended_at IS NOT NULL",
        (cutoff,),
    ).fetchall()
    return [r[0] for r in rows]


def _run_ids_beyond_keep_last(conn, keep: int) -> list[int]:
    """For each run-name, return the ids past the `keep` most recent."""
    rows = conn.execute(
        """
        SELECT id, name, started_at
        FROM runs
        WHERE ended_at IS NOT NULL
        ORDER BY name, started_at DESC
        """
    ).fetchall()
    by_name: dict[str, list[int]] = {}
    for rid, name, _ in rows:
        by_name.setdefault(name, []).append(rid)
    stale = []
    for name, ids in by_name.items():
        stale.extend(ids[keep:])
    return stale


def _delete_runs(conn, ids: list[int]) -> None:
    """Delete a set of runs + all their child rows in one transaction."""
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    with conn:
        for table in ("samples", "steps", "traces", "comm_events",
                      "trace_windows", "host_samples"):
            try:
                conn.execute(
                    f"DELETE FROM {table} WHERE run_id IN ({marks})", ids,
                )
            except sqlite3.OperationalError:
                # Older DBs may not have every table — that's OK.
                pass
        conn.execute(f"DELETE FROM runs WHERE id IN ({marks})", ids)


def _summarize(conn, ids: list[int]) -> str:
    if not ids:
        return "(no runs)"
    marks = ",".join("?" * len(ids))
    row = conn.execute(
        f"""SELECT COUNT(*), MIN(started_at), MAX(started_at)
            FROM runs WHERE id IN ({marks})""", ids,
    ).fetchone()
    n, mn, mx = row
    span = ""
    if mn and mx:
        # Show wall-clock dates so a user can eyeball what's going.
        span = (f" (from {time.strftime('%Y-%m-%d', time.localtime(mn))} "
                f"to {time.strftime('%Y-%m-%d', time.localtime(mx))})")
    return f"{n} run(s){span}"


def _cli() -> None:
    ap = argparse.ArgumentParser(
        prog="gpuprof gc",
        description="Delete old runs from a gpuprof SQLite DB.",
    )
    ap.add_argument("--db", default="gpuprof.db")
    ap.add_argument("--older-than", type=_parse_duration, metavar="DURATION",
                    help="delete completed runs whose start is older than "
                    "this (e.g. `30d`, `2w`, `12h`)")
    ap.add_argument("--keep-last", type=int, metavar="N",
                    help="for each run-name, keep only the N most recent")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would go, delete nothing")
    ap.add_argument("--vacuum", action="store_true",
                    help="run VACUUM after delete to reclaim disk")
    args = ap.parse_args()

    if args.older_than is None and args.keep_last is None:
        ap.error("pass at least one of --older-than or --keep-last")

    if not Path(args.db).exists():
        ap.error(f"no DB at {args.db!r}")

    conn = sqlite3.connect(args.db)
    try:
        stale: set[int] = set()
        if args.older_than is not None:
            cutoff = time.time() - args.older_than
            older = _run_ids_older_than(conn, cutoff)
            print(f"--older-than: {_summarize(conn, older)}")
            stale.update(older)
        if args.keep_last is not None:
            beyond = _run_ids_beyond_keep_last(conn, args.keep_last)
            print(f"--keep-last {args.keep_last}: "
                  f"{_summarize(conn, beyond)}")
            stale.update(beyond)

        if not stale:
            print("nothing to delete.")
            return

        ids = sorted(stale)
        if args.dry_run:
            print(f"\n[dry-run] would delete {len(ids)} run(s): "
                  f"ids={ids[:10]}{'…' if len(ids) > 10 else ''}")
            return

        _delete_runs(conn, ids)
        print(f"deleted {len(ids)} run(s) + all their child rows.")
        if args.vacuum:
            conn.execute("VACUUM")
            print("VACUUM done.")
    finally:
        conn.close()


if __name__ == "__main__":
    _cli()
