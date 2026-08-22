"""SQLite server store. Same interface as `PostgresServerStore`.

Both flavors present the same duck-typed API so the FastAPI app is
storage-agnostic. The queue-drain writer thread lives in
`gpuprof._writer` and is shared between the two backends.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional

from .._writer import BatchedWriter, BatchesDict
from ..store import (
    apply_schema,
    _insert_samples, _insert_steps, _insert_traces,
    _insert_comm, _insert_windows, _insert_host,
)
from ..sampler import Sample
from ..host_sampler import HostSample


_KIND_SAMPLE = "sample"
_KIND_STEP   = "step"
_KIND_TRACE  = "trace"
_KIND_COMM   = "comm"
_KIND_WINDOW = "tw"
_KIND_HOST   = "host"
_KINDS = (_KIND_SAMPLE, _KIND_STEP, _KIND_TRACE, _KIND_COMM,
          _KIND_WINDOW, _KIND_HOST)


_RUN_COLS = (
    "id, name, gpu_name, started_at, ended_at, "
    "group_id, rank, world_size, meta_json, "
    "COALESCE(project, 'default'), owner_user"
)


def _row_to_run(r) -> dict:
    """Shape a `runs` row tuple into the dict shape the API returns.

    Ordered to match `_RUN_COLS`. If you add a column to that SELECT
    list, add the field here too."""
    return {
        "id": r[0], "name": r[1], "gpu": r[2],
        "started_at": r[3], "ended_at": r[4],
        "group_id": r[5], "rank": r[6], "world_size": r[7],
        "meta": json.loads(r[8] or "{}"),
        "project": r[9] if len(r) > 9 else "default",
        "owner_user": r[10] if len(r) > 10 else None,
    }


class ServerStore:
    def __init__(self, path: str):
        self._path = path
        conn = sqlite3.connect(path)
        try:
            apply_schema(conn)
        finally:
            conn.close()
        # Thread-local — see the client Store for the same reason.
        self._tls = threading.local()
        self._writer = BatchedWriter(
            kinds=_KINDS,
            flush_batch=self._flush,
            batch_size=500,
            thread_name="gpuprof-server-writer",
        )
        self._writer.start()

    def close(self) -> None:
        # Same reasoning as the client Store: don't close the writer
        # thread's SQLite connection from *this* thread — it was
        # opened on the writer, and SQLite raises ProgrammingError on
        # cross-thread close.
        self._writer.close(timeout=15.0)

    # -- control-plane writes -----------------------------------------

    def create_run(self, name: str, gpu_name: str, meta_json: str,
                   group_id: Optional[str] = None,
                   rank: Optional[int] = None,
                   world_size: Optional[int] = None,
                   owner_user: Optional[str] = None,
                   project: str = "default") -> int:
        conn = sqlite3.connect(self._path)
        try:
            cur = conn.execute(
                "INSERT INTO runs(name, started_at, gpu_name, group_id, "
                "rank, world_size, meta_json, owner_user, project) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (name, time.time(), gpu_name, group_id, rank,
                 world_size, meta_json, owner_user, project),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def end_run(self, run_id: int) -> None:
        conn = sqlite3.connect(self._path)
        try:
            conn.execute(
                "UPDATE runs SET ended_at=? WHERE id=?",
                (time.time(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_rank_offset(self, run_id: int, offset_s: float) -> None:
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("UPDATE runs SET rank_offset_s=? WHERE id=?",
                         (float(offset_s), run_id))
            conn.commit()
        finally:
            conn.close()

    # -- ingest (producer threads) ------------------------------------

    def push_sample(self, run_id: int, d: dict) -> None:
        # Server receives samples as dicts over HTTP; wrap them back
        # into the `Sample` shape the shared insert helper expects.
        self._writer.enqueue(_KIND_SAMPLE, (run_id, _dict_to_sample(d)))

    def push_step(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_STEP, (run_id, d))

    def push_trace(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_TRACE, (run_id, d))

    def push_comm_event(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_COMM, (run_id, d))

    def push_trace_window(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_WINDOW, (run_id, d))

    def push_host_sample(self, run_id: int, d: dict) -> None:
        # HTTP wire form is a dict; wrap it back into HostSample for
        # the shared insert helper.
        self._writer.enqueue(_KIND_HOST, (run_id, _dict_to_host_sample(d)))

    # -- reads --------------------------------------------------------

    def list_runs(self) -> list[dict]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                f"SELECT {_RUN_COLS} "
                "FROM runs ORDER BY started_at DESC LIMIT 200"
            ).fetchall()
            return [_row_to_run(r) for r in rows]
        finally:
            conn.close()

    def list_runs_in_group(self, group_id: str) -> list[dict]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                "SELECT id, name, gpu_name, started_at, ended_at, "
                "group_id, rank, world_size, meta_json FROM runs "
                "WHERE group_id=? ORDER BY rank ASC",
                (group_id,),
            ).fetchall()
            return [_row_to_run(r) for r in rows]
        finally:
            conn.close()

    def get_run(self, run_id: int) -> Optional[dict]:
        conn = sqlite3.connect(self._path)
        try:
            r = conn.execute(
                f"SELECT {_RUN_COLS} FROM runs WHERE id=?", (run_id,),
            ).fetchone()
            return _row_to_run(r) if r else None
        finally:
            conn.close()

    def list_traces(self, run_id: int, limit: int = 20) -> list[dict]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                "SELECT step, captured_at, kernels_json FROM traces "
                "WHERE run_id=? ORDER BY step DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
            return [
                {"step": r[0], "captured_at": r[1],
                 "kernels": json.loads(r[2] or "[]")}
                for r in rows
            ]
        finally:
            conn.close()

    def list_trace_windows(self, run_id: int, limit: int = 500) -> list[dict]:
        conn = sqlite3.connect(self._path)
        try:
            rows = conn.execute(
                "SELECT t_start_rel, t_end_rel, kernels_json FROM trace_windows "
                "WHERE run_id=? ORDER BY t_start_rel DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
            return [
                {"t_start_rel": r[0], "t_end_rel": r[1],
                 "kernels": json.loads(r[2] or "[]")}
                for r in reversed(rows)
            ]
        finally:
            conn.close()

    def snapshot(self, run_id: int, max_samples: int = 300,
                 max_steps: int = 500) -> dict:
        conn = sqlite3.connect(self._path)
        try:
            samples = [
                {"t": r[0], "gpu_index": r[1], "sm_util": r[2],
                 "mem_used_bytes": r[3], "mem_total_bytes": r[4],
                 "power_w": r[5], "temp_c": r[6],
                 "sm_clock_mhz": r[7], "mem_clock_mhz": r[8],
                 "pcie_rx_kbps": r[9], "pcie_tx_kbps": r[10]}
                for r in reversed(conn.execute(
                    """SELECT t, gpu_index, sm_util, mem_used_bytes,
                              mem_total_bytes, power_w, temp_c,
                              sm_clock_mhz, mem_clock_mhz,
                              pcie_rx_kbps, pcie_tx_kbps
                       FROM samples WHERE run_id=?
                       ORDER BY t DESC LIMIT ?""",
                    (run_id, max_samples),
                ).fetchall())
            ]
            steps = [
                {"step": r[0], "t_start": r[1], "t_end": r[2],
                 "inter_step_gap_s": r[3], "dataloader_wait_s": r[4],
                 "forward_s": r[5], "backward_s": r[6],
                 "optimizer_s": r[7], "comm_s": r[8],
                 "loss": r[9], "tokens": r[10], "flops": r[11]}
                for r in reversed(conn.execute(
                    """SELECT step, t_start, t_end, inter_step_gap_s,
                              dataloader_wait_s, forward_s, backward_s,
                              optimizer_s, comm_s, loss, tokens, flops
                       FROM steps WHERE run_id=? ORDER BY step DESC LIMIT ?""",
                    (run_id, max_steps),
                ).fetchall())
            ]
            trace_row = conn.execute(
                "SELECT step, captured_at, kernels_json FROM traces "
                "WHERE run_id=? ORDER BY step DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            latest_trace = None
            if trace_row:
                latest_trace = {
                    "step": trace_row[0], "captured_at": trace_row[1],
                    "kernels": json.loads(trace_row[2] or "[]"),
                }
        finally:
            conn.close()
        run = self.get_run(run_id) or {}
        return {
            "samples": samples, "steps": steps,
            "latest_trace": latest_trace,
            "meta": run.get("meta", {}), "gpu_name": run.get("gpu"),
            "name": run.get("name"),
        }

    # -- writer callback ----------------------------------------------

    def _flush(self, batches: BatchesDict) -> None:
        """Called by the writer thread. Batches are tuples of
        (run_id, payload); group by run_id then delegate to the shared
        insert helpers."""
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            self._tls.conn = conn
        # Group per-kind by run_id so each SQL executemany carries a
        # single run's rows (matches the helper signature).
        for kind in _KINDS:
            grouped: dict[int, list] = {}
            for rid, payload in batches[kind]:
                grouped.setdefault(rid, []).append(payload)
            for rid, rows in grouped.items():
                _INSERT_FN[kind](conn, rid, rows)
        conn.commit()


_INSERT_FN = {
    _KIND_SAMPLE: _insert_samples,
    _KIND_STEP:   _insert_steps,
    _KIND_TRACE:  _insert_traces,
    _KIND_COMM:   _insert_comm,
    _KIND_WINDOW: _insert_windows,
    _KIND_HOST:   _insert_host,
}


def _dict_to_sample(d: dict) -> Sample:
    """HTTP-wire dict → Sample dataclass for the shared insert helper."""
    return Sample(
        t=d["t"], gpu_index=d["gpu_index"], sm_util=d["sm_util"],
        mem_used_bytes=d["mem_used_bytes"],
        mem_total_bytes=d["mem_total_bytes"],
        power_w=d["power_w"], temp_c=d["temp_c"],
        sm_clock_mhz=d["sm_clock_mhz"], mem_clock_mhz=d["mem_clock_mhz"],
        pcie_rx_kbps=d["pcie_rx_kbps"], pcie_tx_kbps=d["pcie_tx_kbps"],
    )


def _dict_to_host_sample(d: dict) -> HostSample:
    return HostSample(
        t=d["t"],
        cpu_percent=d.get("cpu_percent", 0.0),
        cpu_max_percent=d.get("cpu_max_percent", 0.0),
        n_cpus=d.get("n_cpus", 0),
        mem_used_bytes=d.get("mem_used_bytes", 0),
        mem_total_bytes=d.get("mem_total_bytes", 0),
        mem_cached_bytes=d.get("mem_cached_bytes", 0),
        mem_available_bytes=d.get("mem_available_bytes", 0),
        swap_in_bps=d.get("swap_in_bps", 0.0),
        swap_out_bps=d.get("swap_out_bps", 0.0),
        disk_read_bps=d.get("disk_read_bps", 0.0),
        disk_write_bps=d.get("disk_write_bps", 0.0),
        disk_iops=d.get("disk_iops", 0.0),
        children_cpu_percent=d.get("children_cpu_percent", 0.0),
        max_child_cpu_percent=d.get("max_child_cpu_percent", 0.0),
        n_children=d.get("n_children", 0),
    )
