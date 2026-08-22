"""python -m gpuprof.drain — push orphaned per-run buffer files to a server.

Used to recover data when a server outage outlasted the training
process, so `prof.stop()` couldn't drain its on-disk buffer before the
script exited. Files that drain completely are deleted; the rest stay
on disk for a subsequent attempt.

    python -m gpuprof.drain --server http://127.0.0.1:8000 \
        --api-key SECRET [--dir ~/.gpuprof/buffer]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ._batch import batch_size, empty_batch, merge_into


DRAIN_BATCH_CAP = 5000  # events per POST — mirror remote.py
_RUN_FILENAME_RE = re.compile(r"^run-(\d+)$")


def _post(server: str, path: str, body: dict,
          api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def _extract_run_id(path: Path) -> int | None:
    m = _RUN_FILENAME_RE.match(path.stem)
    return int(m.group(1)) if m else None


def _collect_batch(lines: list[str]) -> tuple[dict, int]:
    """Read up to DRAIN_BATCH_CAP events from `lines` into one merged batch.
    Returns (batch, consumed_line_count)."""
    combined = empty_batch()
    consumed = 0
    for line in lines:
        consumed += 1
        s = line.strip()
        if not s:
            continue
        try:
            merge_into(combined, json.loads(s))
        except Exception:
            continue  # skip corrupt line, move on
        if batch_size(combined) >= DRAIN_BATCH_CAP:
            break
    return combined, consumed


def drain_file(path: Path, server: str, api_key: str | None) -> tuple[bool, int]:
    """Return (fully_drained, events_pushed)."""
    run_id = _extract_run_id(path)
    if run_id is None:
        return (False, 0)

    lines = path.read_text().splitlines()
    if not lines:
        try: path.unlink()
        except OSError: pass
        return (True, 0)

    events_pushed = 0
    while lines:
        batch, consumed = _collect_batch(lines)
        try:
            _post(server, f"/api/runs/{run_id}/ingest", batch, api_key)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            # Partial success possible from earlier iterations; keep
            # the remaining lines on disk for a later attempt.
            path.write_text("\n".join(lines) + "\n")
            print(f"  {path.name}: post failed ({e}); "
                  f"{events_pushed} events pushed",
                  file=sys.stderr)
            return (False, events_pushed)
        events_pushed += batch_size(batch)
        lines = lines[consumed:]

    try: path.unlink()
    except OSError: pass
    return (True, events_pushed)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m gpuprof.drain")
    ap.add_argument("--server", required=True, help="server URL")
    ap.add_argument("--api-key", default=os.environ.get("GPUPROF_API_KEY"))
    ap.add_argument("--dir", default=os.environ.get("GPUPROF_BUFFER_DIR") or
                    str(Path.home() / ".gpuprof" / "buffer"))
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"buffer dir {root} does not exist — nothing to drain")
        return

    files = sorted(root.glob("run-*.jsonl"))
    if not files:
        print(f"no buffer files under {root}")
        return

    total = 0
    ok = 0
    for f in files:
        drained, n = drain_file(f, args.server, args.api_key)
        total += n
        if drained:
            ok += 1
            print(f"  {f.name}: drained {n} events")
        else:
            print(f"  {f.name}: partial ({n} events pushed, file kept)")
    print(f"done — {ok}/{len(files)} files fully drained, {total} events pushed")


if __name__ == "__main__":
    main()
