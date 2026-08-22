"""PostgreSQL server store. Same interface as `ServerStore`.

Thread model:
- The writer thread (BatchedWriter from `.._writer`) owns a persistent
  autocommit-off connection so it can batch inserts inside
  transactions without contention.
- Read paths borrow a connection from a `psycopg_pool.ConnectionPool`
  shared by FastAPI worker threads.

Requires `psycopg[binary]>=3` and `psycopg-pool>=3.1` (declared in the
`postgres` extra).
"""
from __future__ import annotations

import json
import time
from typing import Optional

from .._writer import BatchedWriter, BatchesDict
from .store import _row_to_run  # shape helper is store-agnostic


PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    started_at DOUBLE PRECISION NOT NULL,
    ended_at DOUBLE PRECISION,
    gpu_name TEXT,
    group_id TEXT,
    rank INTEGER,
    world_size INTEGER,
    meta_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_group ON runs(group_id);
CREATE TABLE IF NOT EXISTS samples (
    run_id BIGINT NOT NULL,
    t DOUBLE PRECISION NOT NULL,
    gpu_index INTEGER,
    sm_util DOUBLE PRECISION,
    mem_used_bytes BIGINT,
    mem_total_bytes BIGINT,
    power_w DOUBLE PRECISION,
    temp_c DOUBLE PRECISION,
    sm_clock_mhz INTEGER,
    mem_clock_mhz INTEGER,
    pcie_rx_kbps BIGINT,
    pcie_tx_kbps BIGINT
);
CREATE INDEX IF NOT EXISTS ix_samples_run_t ON samples(run_id, t);
CREATE TABLE IF NOT EXISTS steps (
    run_id BIGINT NOT NULL,
    step INTEGER NOT NULL,
    t_start DOUBLE PRECISION,
    t_end DOUBLE PRECISION,
    inter_step_gap_s DOUBLE PRECISION,
    dataloader_wait_s DOUBLE PRECISION,
    forward_s DOUBLE PRECISION,
    backward_s DOUBLE PRECISION,
    optimizer_s DOUBLE PRECISION,
    comm_s DOUBLE PRECISION,
    loss DOUBLE PRECISION,
    tokens BIGINT,
    flops DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_steps_run ON steps(run_id, step);
CREATE TABLE IF NOT EXISTS traces (
    run_id BIGINT NOT NULL,
    step INTEGER NOT NULL,
    captured_at DOUBLE PRECISION,
    kernels_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_traces_run ON traces(run_id, step);
CREATE TABLE IF NOT EXISTS comm_events (
    run_id BIGINT NOT NULL,
    step INTEGER NOT NULL,
    bucket_id INTEGER NOT NULL,
    t_start_rel DOUBLE PRECISION,
    t_end_rel DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_comm_events_run ON comm_events(run_id, step, bucket_id);
CREATE TABLE IF NOT EXISTS trace_windows (
    run_id BIGINT NOT NULL,
    t_start_rel DOUBLE PRECISION NOT NULL,
    t_end_rel DOUBLE PRECISION,
    kernels_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_trace_windows_run ON trace_windows(run_id, t_start_rel);
CREATE TABLE IF NOT EXISTS host_samples (
    run_id BIGINT NOT NULL,
    t DOUBLE PRECISION NOT NULL,
    cpu_percent DOUBLE PRECISION,
    cpu_max_percent DOUBLE PRECISION,
    n_cpus INTEGER,
    mem_used_bytes BIGINT,
    mem_total_bytes BIGINT,
    disk_read_bps DOUBLE PRECISION,
    disk_write_bps DOUBLE PRECISION,
    disk_iops DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_host_samples_run_t ON host_samples(run_id, t);
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS mem_cached_bytes BIGINT;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS mem_available_bytes BIGINT;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS swap_in_bps DOUBLE PRECISION;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS swap_out_bps DOUBLE PRECISION;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS children_cpu_percent DOUBLE PRECISION;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS max_child_cpu_percent DOUBLE PRECISION;
ALTER TABLE host_samples ADD COLUMN IF NOT EXISTS n_children INTEGER;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS rank_offset_s DOUBLE PRECISION;
"""

_KIND_SAMPLE = "sample"
_KIND_STEP   = "step"
_KIND_TRACE  = "trace"
_KIND_COMM   = "comm"
_KIND_WINDOW = "tw"
_KIND_HOST   = "host"
_KINDS = (_KIND_SAMPLE, _KIND_STEP, _KIND_TRACE, _KIND_COMM,
          _KIND_WINDOW, _KIND_HOST)


# ------------------------------------------------------------------
# Insert helpers (Postgres dialect).
# ------------------------------------------------------------------

def _pg_insert_samples(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO samples(run_id, t, gpu_index, sm_util, "
        "mem_used_bytes, mem_total_bytes, power_w, temp_c, "
        "sm_clock_mhz, mem_clock_mhz, pcie_rx_kbps, pcie_tx_kbps) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        [(rid, d["t"], d["gpu_index"], d["sm_util"],
          d["mem_used_bytes"], d["mem_total_bytes"],
          d["power_w"], d["temp_c"],
          d["sm_clock_mhz"], d["mem_clock_mhz"],
          d["pcie_rx_kbps"], d["pcie_tx_kbps"]) for rid, d in rows],
    )


def _pg_insert_steps(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO steps(run_id, step, t_start, t_end, "
        "inter_step_gap_s, dataloader_wait_s, forward_s, "
        "backward_s, optimizer_s, comm_s, loss, tokens, flops) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        [(rid, d["step"], d["t_start"], d["t_end"],
          d.get("inter_step_gap_s"), d.get("dataloader_wait_s"),
          d.get("forward_s"), d.get("backward_s"),
          d.get("optimizer_s"), d.get("comm_s"),
          d.get("loss"), d.get("tokens"), d.get("flops"))
         for rid, d in rows],
    )


def _pg_insert_traces(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO traces(run_id, step, captured_at, kernels_json) "
        "VALUES (%s,%s,%s,%s)",
        [(rid, d["step"],
          d.get("captured_at") or time.time(),
          json.dumps(d.get("kernels", []))) for rid, d in rows],
    )


def _pg_insert_comm(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO comm_events(run_id, step, bucket_id, "
        "t_start_rel, t_end_rel) VALUES (%s,%s,%s,%s,%s)",
        [(rid, d["step"], d["bucket_id"],
          d.get("t_start_rel"), d.get("t_end_rel")) for rid, d in rows],
    )


def _pg_insert_windows(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO trace_windows(run_id, t_start_rel, t_end_rel, "
        "kernels_json) VALUES (%s,%s,%s,%s)",
        [(rid, d["t_start_rel"], d.get("t_end_rel"),
          json.dumps(d.get("kernels", []))) for rid, d in rows],
    )


def _pg_insert_host(cur, rows):
    if not rows: return
    cur.executemany(
        "INSERT INTO host_samples(run_id, t, cpu_percent, cpu_max_percent, "
        "n_cpus, mem_used_bytes, mem_total_bytes, mem_cached_bytes, "
        "mem_available_bytes, swap_in_bps, swap_out_bps, "
        "disk_read_bps, disk_write_bps, disk_iops, "
        "children_cpu_percent, max_child_cpu_percent, n_children) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        [(rid, d["t"], d.get("cpu_percent"), d.get("cpu_max_percent"),
          d.get("n_cpus"),
          d.get("mem_used_bytes"), d.get("mem_total_bytes"),
          d.get("mem_cached_bytes"), d.get("mem_available_bytes"),
          d.get("swap_in_bps"), d.get("swap_out_bps"),
          d.get("disk_read_bps"), d.get("disk_write_bps"),
          d.get("disk_iops"),
          d.get("children_cpu_percent"), d.get("max_child_cpu_percent"),
          d.get("n_children")) for rid, d in rows],
    )


_INSERT_FN = {
    _KIND_SAMPLE: _pg_insert_samples,
    _KIND_STEP:   _pg_insert_steps,
    _KIND_TRACE:  _pg_insert_traces,
    _KIND_COMM:   _pg_insert_comm,
    _KIND_WINDOW: _pg_insert_windows,
    _KIND_HOST:   _pg_insert_host,
}


# ------------------------------------------------------------------
# Store
# ------------------------------------------------------------------

class PostgresServerStore:
    def __init__(self, dsn: str, pool_min: int = 2, pool_max: int = 10):
        # Deferred imports so a server without psycopg still boots for
        # the SQLite path.
        import psycopg
        from psycopg_pool import ConnectionPool
        self._dsn = dsn
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(PG_SCHEMA)
        self._pool = ConnectionPool(
            dsn, min_size=pool_min, max_size=pool_max, open=True,
        )
        self._writer_conn = None
        self._writer = BatchedWriter(
            kinds=_KINDS,
            flush_batch=self._flush,
            batch_size=500,
            thread_name="gpuprof-pg-writer",
        )
        self._writer.start()

    def close(self) -> None:
        self._writer.close(timeout=10.0)
        if self._writer_conn is not None:
            try: self._writer_conn.close()
            except Exception: pass
            self._writer_conn = None
        try: self._pool.close()
        except Exception: pass

    # -- control-plane writes -----------------------------------------

    def create_run(self, name: str, gpu_name: str, meta_json: str,
                   group_id: Optional[str] = None,
                   rank: Optional[int] = None,
                   world_size: Optional[int] = None) -> int:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs(name, started_at, gpu_name, "
                "group_id, rank, world_size, meta_json) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (name, time.time(), gpu_name, group_id, rank,
                 world_size, meta_json),
            )
            return cur.fetchone()[0]

    def end_run(self, run_id: int) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE runs SET ended_at=%s WHERE id=%s",
                        (time.time(), run_id))

    def set_rank_offset(self, run_id: int, offset_s: float) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE runs SET rank_offset_s=%s WHERE id=%s",
                        (float(offset_s), run_id))

    # -- ingest -------------------------------------------------------

    def push_sample(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_SAMPLE, (run_id, d))

    def push_step(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_STEP, (run_id, d))

    def push_trace(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_TRACE, (run_id, d))

    def push_comm_event(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_COMM, (run_id, d))

    def push_trace_window(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_WINDOW, (run_id, d))

    def push_host_sample(self, run_id: int, d: dict) -> None:
        self._writer.enqueue(_KIND_HOST, (run_id, d))

    # -- reads --------------------------------------------------------

    def list_runs(self) -> list[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, gpu_name, started_at, ended_at, "
                "group_id, rank, world_size, meta_json "
                "FROM runs ORDER BY started_at DESC LIMIT 200",
            )
            return [_row_to_run(r) for r in cur.fetchall()]

    def list_runs_in_group(self, group_id: str) -> list[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, gpu_name, started_at, ended_at, "
                "group_id, rank, world_size, meta_json "
                "FROM runs WHERE group_id=%s ORDER BY rank NULLS LAST",
                (group_id,),
            )
            return [_row_to_run(r) for r in cur.fetchall()]

    def get_run(self, run_id: int) -> Optional[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, gpu_name, started_at, ended_at, "
                "group_id, rank, world_size, meta_json FROM runs WHERE id=%s",
                (run_id,),
            )
            r = cur.fetchone()
            return _row_to_run(r) if r else None

    def list_traces(self, run_id: int, limit: int = 20) -> list[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT step, captured_at, kernels_json FROM traces "
                "WHERE run_id=%s ORDER BY step DESC LIMIT %s",
                (run_id, limit),
            )
            return [
                {"step": r[0], "captured_at": r[1],
                 "kernels": json.loads(r[2] or "[]")}
                for r in cur.fetchall()
            ]

    def list_trace_windows(self, run_id: int, limit: int = 500) -> list[dict]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT t_start_rel, t_end_rel, kernels_json FROM trace_windows "
                "WHERE run_id=%s ORDER BY t_start_rel DESC LIMIT %s",
                (run_id, limit),
            )
            return [
                {"t_start_rel": r[0], "t_end_rel": r[1],
                 "kernels": json.loads(r[2] or "[]")}
                for r in reversed(cur.fetchall())
            ]

    def snapshot(self, run_id: int, max_samples: int = 300,
                 max_steps: int = 500) -> dict:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT t, gpu_index, sm_util, mem_used_bytes, mem_total_bytes, "
                "power_w, temp_c, sm_clock_mhz, mem_clock_mhz, "
                "pcie_rx_kbps, pcie_tx_kbps FROM samples WHERE run_id=%s "
                "ORDER BY t DESC LIMIT %s",
                (run_id, max_samples),
            )
            samples = [
                {"t": r[0], "gpu_index": r[1], "sm_util": r[2],
                 "mem_used_bytes": r[3], "mem_total_bytes": r[4],
                 "power_w": r[5], "temp_c": r[6],
                 "sm_clock_mhz": r[7], "mem_clock_mhz": r[8],
                 "pcie_rx_kbps": r[9], "pcie_tx_kbps": r[10]}
                for r in reversed(cur.fetchall())
            ]
            cur.execute(
                "SELECT step, t_start, t_end, inter_step_gap_s, "
                "dataloader_wait_s, forward_s, backward_s, optimizer_s, "
                "comm_s, loss, tokens, flops FROM steps WHERE run_id=%s "
                "ORDER BY step DESC LIMIT %s",
                (run_id, max_steps),
            )
            steps = [
                {"step": r[0], "t_start": r[1], "t_end": r[2],
                 "inter_step_gap_s": r[3], "dataloader_wait_s": r[4],
                 "forward_s": r[5], "backward_s": r[6],
                 "optimizer_s": r[7], "comm_s": r[8],
                 "loss": r[9], "tokens": r[10], "flops": r[11]}
                for r in reversed(cur.fetchall())
            ]
            cur.execute(
                "SELECT step, captured_at, kernels_json FROM traces "
                "WHERE run_id=%s ORDER BY step DESC LIMIT 1",
                (run_id,),
            )
            row = cur.fetchone()
            latest_trace = (
                {"step": row[0], "captured_at": row[1],
                 "kernels": json.loads(row[2] or "[]")} if row else None
            )
        run = self.get_run(run_id) or {}
        return {
            "samples": samples, "steps": steps,
            "latest_trace": latest_trace,
            "meta": run.get("meta", {}), "gpu_name": run.get("gpu"),
            "name": run.get("name"),
        }

    # -- writer callback ----------------------------------------------

    def _flush(self, batches: BatchesDict) -> None:
        import psycopg
        if self._writer_conn is None:
            self._writer_conn = psycopg.connect(self._dsn)
            self._writer_conn.autocommit = False
        conn = self._writer_conn
        with conn.cursor() as cur:
            for kind in _KINDS:
                _INSERT_FN[kind](cur, batches[kind])
        conn.commit()
