"""HTTP pusher: buffers samples/steps/traces and POSTs them to the
server in batches.

Two levels of buffering:
  1. In-memory queue drained by a background thread — non-blocking to
     training under normal conditions.
  2. On-disk JSONL buffer per run — if the server is unreachable, failed
     batches append to `~/.gpuprof/buffer/run-<id>.jsonl` and are
     drained the next time a POST succeeds. Bounded by BUFFER_MAX_BYTES;
     when the cap is hit, the oldest half is dropped.

Uses stdlib only (urllib), so it adds no dependency on the client.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from ._batch import BATCH_KINDS, batch_size, empty_batch, merge_into
from .sampler import Sample


def _collect_from_lines(lines: list[str]) -> tuple[dict, int]:
    """Parse up to DRAIN_BATCH_CAP events from `lines` into one merged
    batch. Returns (batch, consumed_line_count) — matching drain.py."""
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
            continue
        if batch_size(combined) >= DRAIN_BATCH_CAP:
            break
    return combined, consumed


BUFFER_MAX_BYTES = 50 * 1024 * 1024      # cap disk buffer at 50 MB/run
DRAIN_BATCH_CAP = 5000                    # max events per drain POST


class Remote:
    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        flush_hz: float = 1.0,
        buffer_dir: Optional[Path] = None,
    ):
        self._url = server_url.rstrip("/")
        self._api_key = api_key
        self._q: queue.Queue = queue.Queue(maxsize=100_000)
        self._stop = threading.Event()
        self._interval = 1.0 / flush_hz
        self._run_id: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._buffer_dir = Path(
            buffer_dir or os.environ.get("GPUPROF_BUFFER_DIR")
            or (Path.home() / ".gpuprof" / "buffer")
        )
        self._buffer_path: Optional[Path] = None

    @property
    def run_id(self) -> Optional[int]:
        return self._run_id

    def start_run(self, name: str, gpu_name: str, meta: dict,
                  group_id: Optional[str] = None,
                  rank: Optional[int] = None,
                  world_size: Optional[int] = None) -> int:
        body = {
            "name": name, "gpu_name": gpu_name, "meta": meta,
            "group_id": group_id, "rank": rank, "world_size": world_size,
        }
        data = self._request("POST", "/api/runs", body)
        self._run_id = int(data["id"])
        self._buffer_path = self._buffer_dir / f"run-{self._run_id}.jsonl"
        # Sweep orphaned buffer files from prior runs. If any exist,
        # try to drain them into the current server so users don't
        # need to remember `gpuprof drain` after a network flake.
        try:
            self._sweep_orphaned_buffers()
        except Exception:
            pass  # never let cleanup break the new run
        self._thread = threading.Thread(
            target=self._pump, daemon=True, name="gpuprof-remote",
        )
        self._thread.start()
        return self._run_id

    def _sweep_orphaned_buffers(self) -> None:
        """Best-effort drain of `run-*.jsonl` files left behind by
        prior runs. Only touched here (not during the training loop)
        so it can't add latency to the hot path."""
        if not self._buffer_dir.exists():
            return
        for path in sorted(self._buffer_dir.glob("run-*.jsonl")):
            # Skip our own file — the pump owns it now.
            if self._buffer_path is not None and path == self._buffer_path:
                continue
            try:
                # Route through the shared drain code so this is
                # exactly consistent with `gpuprof drain`.
                from .drain import drain_file
                drain_file(path, self._url, self._api_key)
            except Exception:
                # Leave the file for the next attempt / manual drain.
                continue

    def end_run(self) -> None:
        self._stop.set()
        # Give the pump time to drain both the in-memory queue and any
        # on-disk buffer if the server is available; the pump itself
        # imposes an internal cap so this can't hang forever.
        if self._thread is not None:
            self._thread.join(timeout=20.0)
        try:
            self._request("POST", f"/api/runs/{self._run_id}/end", {})
        except Exception:
            pass

    def push_sample(self, s: Sample) -> None:
        try: self._q.put_nowait(("sample", _sample_to_dict(s)))
        except queue.Full: pass

    def push_step(self, step: dict) -> None:
        try: self._q.put_nowait(("step", step))
        except queue.Full: pass

    def push_trace(self, trace: dict) -> None:
        try: self._q.put_nowait(("trace", trace))
        except queue.Full: pass

    def push_comm_event(self, ev: dict) -> None:
        try: self._q.put_nowait(("comm", ev))
        except queue.Full: pass

    def push_trace_window(self, w: dict) -> None:
        try: self._q.put_nowait(("tw", w))
        except queue.Full: pass

    def push_host_sample(self, h) -> None:
        try: self._q.put_nowait(("host", _host_sample_to_dict(h)))
        except queue.Full: pass

    def set_rank_offset(self, offset_s: float) -> None:
        """Best-effort POST to record the run's clock offset. Fires
        once at start; the server persists it on the run row."""
        try:
            self._request(
                "POST",
                f"/api/runs/{self._run_id}/rank_offset",
                {"offset_s": float(offset_s)},
            )
        except Exception:
            pass

    # -----------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        req = urllib.request.Request(
            self._url + path,
            data=json.dumps(body).encode(),
            headers=headers, method=method,
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else {}

    def _pump(self) -> None:
        # Keep looping while there is work to do. On shutdown we also
        # keep trying while the disk buffer has content — so a brief
        # server outage that overlaps prof.stop() still recovers — but
        # cap the wait so a permanently-dead server can't hang forever.
        SHUTDOWN_DRAIN_CAP_S = 15.0
        shutdown_deadline: Optional[float] = None

        while True:
            stop = self._stop.is_set()
            if stop and shutdown_deadline is None:
                shutdown_deadline = time.time() + SHUTDOWN_DRAIN_CAP_S
            done = (self._q.empty() and not self._buffer_nonempty())
            if stop and done:
                return
            if stop and shutdown_deadline and time.time() > shutdown_deadline:
                return

            cur = self._collect_current_batch()

            # If we have prior failed batches on disk, try them first.
            if self._buffer_nonempty():
                drained = self._try_drain_buffer()
                if drained:
                    if _nonempty(cur):
                        try:
                            self._request(
                                "POST",
                                f"/api/runs/{self._run_id}/ingest", cur,
                            )
                        except Exception:
                            self._buffer_append(cur)
                else:
                    if _nonempty(cur):
                        self._buffer_append(cur)
                    # Server unreachable — brief backoff so we don't spin
                    # on the shutdown path when the queue is drained.
                    if stop and self._q.empty():
                        self._stop.wait(0.5)
            else:
                if _nonempty(cur):
                    try:
                        self._request(
                            "POST",
                            f"/api/runs/{self._run_id}/ingest", cur,
                        )
                    except (urllib.error.URLError, TimeoutError,
                            ConnectionError, OSError):
                        self._buffer_append(cur)

    def _collect_current_batch(self) -> dict:
        # Bucketed batch shape mirrors the wire format one-for-one so
        # POST bodies and disk-buffer lines have identical keys.
        samples: list = []
        steps: list = []
        traces: list = []
        comm: list = []
        windows: list = []
        host: list = []
        deadline = time.time() + self._interval
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                kind, payload = self._q.get(timeout=remaining)
            except queue.Empty:
                break
            if kind == "sample": samples.append(payload)
            elif kind == "step": steps.append(payload)
            elif kind == "trace": traces.append(payload)
            elif kind == "comm": comm.append(payload)
            elif kind == "tw": windows.append(payload)
            elif kind == "host": host.append(payload)
            if (len(samples) + len(steps) + len(traces)
                    + len(comm) + len(windows) + len(host)) >= 500:
                break
        return {"samples": samples, "steps": steps,
                "traces": traces, "comm_events": comm,
                "trace_windows": windows, "host_samples": host}

    # ---- disk buffer -------------------------------------------------

    # Trim is amortized O(1): only run when the file has grown 25%
    # past the cap. Rewriting on every append was O(N²) under load.
    _BUFFER_TRIM_HIGH_WATERMARK = int(BUFFER_MAX_BYTES * 1.25)

    def _buffer_nonempty(self) -> bool:
        p = self._buffer_path
        try:
            return p is not None and p.exists() and p.stat().st_size > 0
        except OSError:
            return False

    def _buffer_append(self, batch: dict) -> None:
        if self._buffer_path is None:
            return
        try:
            self._buffer_dir.mkdir(parents=True, exist_ok=True)
            with open(self._buffer_path, "a") as f:
                f.write(json.dumps(batch) + "\n")
            # Amortized: check the (cheap) file size but only rewrite
            # when we're well past the cap.
            if self._buffer_path.stat().st_size > self._BUFFER_TRIM_HIGH_WATERMARK:
                self._buffer_trim()
        except OSError:
            pass  # last resort: drop

    def _buffer_trim(self) -> None:
        """Drop the oldest half of the buffer. Uses a tmp file + rename
        so we don't sit with a partially-written buffer if we crash."""
        p = self._buffer_path
        try:
            if p is None or not p.exists() or p.stat().st_size <= BUFFER_MAX_BYTES:
                return
            with open(p, "r") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2:]
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w") as f:
                f.writelines(keep)
            os.replace(tmp, p)  # atomic
        except OSError:
            pass

    def _try_drain_buffer(self) -> bool:
        """Drain up to DRAIN_BATCH_CAP events from the on-disk buffer.
        Returns True if the buffer is now empty."""
        p = self._buffer_path
        if p is None:
            return True
        try:
            with open(p, "r") as f:
                lines = f.readlines()
        except OSError:
            return False
        if not lines:
            return True

        combined, consumed = _collect_from_lines(lines)

        try:
            self._request(
                "POST", f"/api/runs/{self._run_id}/ingest", combined,
            )
        except Exception:
            return False

        remaining = lines[consumed:]
        try:
            if remaining:
                # tmp + rename for the same crash-safety reason as trim.
                tmp = p.with_suffix(p.suffix + ".tmp")
                with open(tmp, "w") as f:
                    f.writelines(remaining)
                os.replace(tmp, p)
                return False
            else:
                p.unlink(missing_ok=True)
                return True
        except OSError:
            return not remaining


def _sample_to_dict(s: Sample) -> dict:
    return {
        "t": s.t, "gpu_index": s.gpu_index, "sm_util": s.sm_util,
        "mem_used_bytes": s.mem_used_bytes, "mem_total_bytes": s.mem_total_bytes,
        "power_w": s.power_w, "temp_c": s.temp_c,
        "sm_clock_mhz": s.sm_clock_mhz, "mem_clock_mhz": s.mem_clock_mhz,
        "pcie_rx_kbps": s.pcie_rx_kbps, "pcie_tx_kbps": s.pcie_tx_kbps,
    }


def _host_sample_to_dict(h) -> dict:
    return {
        "t": h.t, "cpu_percent": h.cpu_percent,
        "cpu_max_percent": h.cpu_max_percent, "n_cpus": h.n_cpus,
        "mem_used_bytes": h.mem_used_bytes, "mem_total_bytes": h.mem_total_bytes,
        "mem_cached_bytes": h.mem_cached_bytes,
        "mem_available_bytes": h.mem_available_bytes,
        "swap_in_bps": h.swap_in_bps, "swap_out_bps": h.swap_out_bps,
        "disk_read_bps": h.disk_read_bps, "disk_write_bps": h.disk_write_bps,
        "disk_iops": h.disk_iops,
        "children_cpu_percent": h.children_cpu_percent,
        "max_child_cpu_percent": h.max_child_cpu_percent,
        "n_children": h.n_children,
    }


def _nonempty(batch: dict) -> bool:
    """Wire-format batch has any events at all."""
    return any(batch.get(k) for k in BATCH_KINDS)
