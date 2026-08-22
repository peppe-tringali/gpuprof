"""Client-side SQLite writer for runs, samples, steps, traces, and
per-bucket comm events. The queue-drain machinery lives in `_writer`
so both client and server stores share the same batching logic.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from ._writer import BatchedWriter, BatchesDict
from .sampler import Sample


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    gpu_name TEXT,
    group_id TEXT,
    rank INTEGER,
    world_size INTEGER,
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    run_id INTEGER NOT NULL,
    t REAL NOT NULL,
    gpu_index INTEGER,
    sm_util REAL,
    mem_used_bytes INTEGER,
    mem_total_bytes INTEGER,
    power_w REAL,
    temp_c REAL,
    sm_clock_mhz INTEGER,
    mem_clock_mhz INTEGER,
    pcie_rx_kbps INTEGER,
    pcie_tx_kbps INTEGER
);
CREATE INDEX IF NOT EXISTS ix_samples_run_t ON samples(run_id, t);
CREATE TABLE IF NOT EXISTS steps (
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    t_start REAL,
    t_end REAL,
    inter_step_gap_s REAL,
    dataloader_wait_s REAL,
    forward_s REAL,
    backward_s REAL,
    optimizer_s REAL,
    comm_s REAL,
    loss REAL,
    tokens INTEGER,
    flops REAL
);
CREATE INDEX IF NOT EXISTS ix_steps_run ON steps(run_id, step);
CREATE TABLE IF NOT EXISTS traces (
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    captured_at REAL,
    kernels_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_traces_run ON traces(run_id, step);
CREATE TABLE IF NOT EXISTS comm_events (
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    bucket_id INTEGER NOT NULL,
    t_start_rel REAL,
    t_end_rel REAL
);
CREATE INDEX IF NOT EXISTS ix_comm_events_run ON comm_events(run_id, step, bucket_id);
CREATE TABLE IF NOT EXISTS trace_windows (
    run_id INTEGER NOT NULL,
    t_start_rel REAL NOT NULL,
    t_end_rel REAL,
    kernels_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_trace_windows_run ON trace_windows(run_id, t_start_rel);
CREATE TABLE IF NOT EXISTS host_samples (
    run_id INTEGER NOT NULL,
    t REAL NOT NULL,
    cpu_percent REAL,
    cpu_max_percent REAL,
    n_cpus INTEGER,
    mem_used_bytes INTEGER,
    mem_total_bytes INTEGER,
    mem_cached_bytes INTEGER,
    mem_available_bytes INTEGER,
    swap_in_bps REAL,
    swap_out_bps REAL,
    disk_read_bps REAL,
    disk_write_bps REAL,
    disk_iops REAL,
    children_cpu_percent REAL,
    max_child_cpu_percent REAL,
    n_children INTEGER
);
CREATE INDEX IF NOT EXISTS ix_host_samples_run_t ON host_samples(run_id, t);
"""

# Idempotent migrations applied on top of SCHEMA. ALTER re-runs return
# a "duplicate column" error; apply_schema swallows that specific case
# and re-raises everything else.
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN group_id TEXT",
    "ALTER TABLE runs ADD COLUMN rank INTEGER",
    "ALTER TABLE runs ADD COLUMN world_size INTEGER",
    "ALTER TABLE steps ADD COLUMN inter_step_gap_s REAL",
    "ALTER TABLE steps ADD COLUMN comm_s REAL",
    "CREATE INDEX IF NOT EXISTS ix_runs_group ON runs(group_id)",
    """CREATE TABLE IF NOT EXISTS comm_events (
        run_id INTEGER NOT NULL, step INTEGER NOT NULL,
        bucket_id INTEGER NOT NULL,
        t_start_rel REAL, t_end_rel REAL)""",
    "CREATE INDEX IF NOT EXISTS ix_comm_events_run ON comm_events(run_id, step, bucket_id)",
    """CREATE TABLE IF NOT EXISTS trace_windows (
        run_id INTEGER NOT NULL,
        t_start_rel REAL NOT NULL,
        t_end_rel REAL, kernels_json TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_trace_windows_run ON trace_windows(run_id, t_start_rel)",
    "ALTER TABLE runs ADD COLUMN rank_offset_s REAL",
    # v0.7 — server-side multi-tenant scoping. Any pre-existing run
    # gets project="default" and user=NULL (i.e. no scoping applied).
    "ALTER TABLE runs ADD COLUMN project TEXT DEFAULT 'default'",
    "ALTER TABLE runs ADD COLUMN owner_user TEXT",
    "CREATE INDEX IF NOT EXISTS ix_runs_project ON runs(project)",
    "CREATE INDEX IF NOT EXISTS ix_runs_owner ON runs(owner_user)",
    # host_samples added in v0.5 for CPU-bound vs I/O-bound diagnosis.
    """CREATE TABLE IF NOT EXISTS host_samples (
        run_id INTEGER NOT NULL, t REAL NOT NULL,
        cpu_percent REAL, cpu_max_percent REAL, n_cpus INTEGER,
        mem_used_bytes INTEGER, mem_total_bytes INTEGER,
        disk_read_bps REAL, disk_write_bps REAL, disk_iops REAL)""",
    "CREATE INDEX IF NOT EXISTS ix_host_samples_run_t ON host_samples(run_id, t)",
    # v0.6: fields for cache-thrashing / swap-pressure / per-worker rules.
    "ALTER TABLE host_samples ADD COLUMN mem_cached_bytes INTEGER",
    "ALTER TABLE host_samples ADD COLUMN mem_available_bytes INTEGER",
    "ALTER TABLE host_samples ADD COLUMN swap_in_bps REAL",
    "ALTER TABLE host_samples ADD COLUMN swap_out_bps REAL",
    "ALTER TABLE host_samples ADD COLUMN children_cpu_percent REAL",
    "ALTER TABLE host_samples ADD COLUMN max_child_cpu_percent REAL",
    "ALTER TABLE host_samples ADD COLUMN n_children INTEGER",
]

# Kinds enqueued via the BatchedWriter — one list per SQL insert.
_KIND_SAMPLE = "sample"
_KIND_STEP   = "step"
_KIND_TRACE  = "trace"
_KIND_COMM   = "comm"
_KIND_WINDOW = "tw"
_KIND_HOST   = "host"
_KINDS = (_KIND_SAMPLE, _KIND_STEP, _KIND_TRACE, _KIND_COMM,
          _KIND_WINDOW, _KIND_HOST)


def apply_schema(conn) -> None:
    """Create schema + apply pending migrations, idempotently."""
    # WAL mode lets the drain-thread's persistent connection and short-
    # lived control-plane connections (start_run, end_run,
    # set_rank_offset) coexist without "database is locked" errors on
    # busy runs. Set once; SQLite persists the mode with the DB file.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # in-memory / non-file DBs don't support WAL
    conn.executescript(SCHEMA)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            # "duplicate column name" is the intended no-op for a
            # re-run migration; anything else must surface.
            msg = str(e).lower()
            if "duplicate column" not in msg and "already exists" not in msg:
                raise
    conn.commit()


# ------------------------------------------------------------------
# Insert helpers — small, single-purpose so they're easy to test.
# ------------------------------------------------------------------

def _insert_samples(conn, run_id: int, samples: list[Sample]) -> None:
    if not samples: return
    conn.executemany(
        """INSERT INTO samples(run_id, t, gpu_index, sm_util,
               mem_used_bytes, mem_total_bytes, power_w, temp_c,
               sm_clock_mhz, mem_clock_mhz, pcie_rx_kbps, pcie_tx_kbps)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, s.t, s.gpu_index, s.sm_util,
          s.mem_used_bytes, s.mem_total_bytes, s.power_w, s.temp_c,
          s.sm_clock_mhz, s.mem_clock_mhz,
          s.pcie_rx_kbps, s.pcie_tx_kbps) for s in samples],
    )


def _insert_steps(conn, run_id: int, steps: list[dict]) -> None:
    if not steps: return
    conn.executemany(
        """INSERT INTO steps(run_id, step, t_start, t_end,
               inter_step_gap_s, dataloader_wait_s, forward_s,
               backward_s, optimizer_s, comm_s, loss, tokens, flops)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, d["step"], d["t_start"], d["t_end"],
          d.get("inter_step_gap_s"), d.get("dataloader_wait_s"),
          d.get("forward_s"), d.get("backward_s"),
          d.get("optimizer_s"), d.get("comm_s"),
          d.get("loss"), d.get("tokens"), d.get("flops"))
         for d in steps],
    )


def _insert_traces(conn, run_id: int, traces: list[dict]) -> None:
    if not traces: return
    conn.executemany(
        "INSERT INTO traces(run_id, step, captured_at, kernels_json) "
        "VALUES (?,?,?,?)",
        [(run_id, t["step"],
          t.get("captured_at") or time.time(),
          json.dumps(t.get("kernels", []))) for t in traces],
    )


def _insert_comm(conn, run_id: int, events: list[dict]) -> None:
    if not events: return
    conn.executemany(
        "INSERT INTO comm_events(run_id, step, bucket_id, "
        "t_start_rel, t_end_rel) VALUES (?,?,?,?,?)",
        [(run_id, e["step"], e["bucket_id"],
          e.get("t_start_rel"), e.get("t_end_rel")) for e in events],
    )


def _insert_windows(conn, run_id: int, windows: list[dict]) -> None:
    if not windows: return
    conn.executemany(
        "INSERT INTO trace_windows(run_id, t_start_rel, t_end_rel, "
        "kernels_json) VALUES (?,?,?,?)",
        [(run_id, w["t_start_rel"], w.get("t_end_rel"),
          json.dumps(w.get("kernels", []))) for w in windows],
    )


def _insert_host(conn, run_id: int, samples: list) -> None:
    if not samples: return
    conn.executemany(
        """INSERT INTO host_samples(run_id, t, cpu_percent, cpu_max_percent,
               n_cpus, mem_used_bytes, mem_total_bytes,
               mem_cached_bytes, mem_available_bytes,
               swap_in_bps, swap_out_bps,
               disk_read_bps, disk_write_bps, disk_iops,
               children_cpu_percent, max_child_cpu_percent, n_children)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, h.t, h.cpu_percent, h.cpu_max_percent, h.n_cpus,
          h.mem_used_bytes, h.mem_total_bytes,
          h.mem_cached_bytes, h.mem_available_bytes,
          h.swap_in_bps, h.swap_out_bps,
          h.disk_read_bps, h.disk_write_bps, h.disk_iops,
          h.children_cpu_percent, h.max_child_cpu_percent, h.n_children)
         for h in samples],
    )


# ------------------------------------------------------------------
# Store — one run's worth of writes.
# ------------------------------------------------------------------

class Store:
    def __init__(self, path: str | Path):
        self._path = str(path)
        self._run_id: Optional[int] = None
        # Thread-local so the writer thread's SQLite connection is
        # created and closed on the same thread — SQLite's default
        # `check_same_thread=True` otherwise raises `ProgrammingError`.
        self._tls = threading.local()
        self._ensure_schema()
        self._writer = BatchedWriter(
            kinds=_KINDS,
            flush_batch=self._flush,
            thread_name="gpuprof-writer",
        )

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(self._path)
        try:
            apply_schema(conn)
        finally:
            conn.close()

    def start_run(self, name: str, gpu_name: str, meta_json: str = "{}",
                  group_id: Optional[str] = None,
                  rank: Optional[int] = None,
                  world_size: Optional[int] = None) -> int:
        conn = sqlite3.connect(self._path)
        try:
            cur = conn.execute(
                "INSERT INTO runs(name, started_at, gpu_name, group_id, "
                "rank, world_size, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, time.time(), gpu_name, group_id, rank, world_size,
                 meta_json),
            )
            conn.commit()
            self._run_id = cur.lastrowid
        finally:
            conn.close()
        self._writer.start()
        return self._run_id

    def end_run(self) -> None:
        # 15 s is generous — writer's own idle-flush is 250 ms and the
        # queue drains fast under WAL. Old 5 s timeout could truncate
        # under a big backlog on shutdown.
        self._writer.close(timeout=15.0)
        # NOTE: don't touch the writer thread's SQLite connection here.
        # It was created on the writer thread; SQLite forbids cross-
        # thread close (ProgrammingError). The thread exits on its own
        # after `_writer.close()` joins, and Python GC finalizes the
        # connection.
        conn = sqlite3.connect(self._path)
        try:
            conn.execute(
                "UPDATE runs SET ended_at=? WHERE id=?",
                (time.time(), self._run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def set_rank_offset(self, offset_s: float) -> None:
        """Persist the software-estimated cross-rank clock offset."""
        if self._run_id is None: return
        conn = sqlite3.connect(self._path)
        try:
            conn.execute(
                "UPDATE runs SET rank_offset_s=? WHERE id=?",
                (float(offset_s), self._run_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -- push (from producer threads) ---------------------------------

    def push_sample(self, s: Sample) -> None:
        self._writer.enqueue(_KIND_SAMPLE, s)

    def push_step(self, step: dict) -> None:
        self._writer.enqueue(_KIND_STEP, step)

    def push_trace(self, trace: dict) -> None:
        self._writer.enqueue(_KIND_TRACE, trace)

    def push_comm_event(self, ev: dict) -> None:
        """ev = {step, bucket_id, t_start_rel, t_end_rel}"""
        self._writer.enqueue(_KIND_COMM, ev)

    def push_trace_window(self, w: dict) -> None:
        """w = {t_start_rel, t_end_rel, kernels: [...]}"""
        self._writer.enqueue(_KIND_WINDOW, w)

    def push_host_sample(self, h) -> None:
        """h = HostSample dataclass"""
        self._writer.enqueue(_KIND_HOST, h)

    # -- flush (writer thread) ----------------------------------------

    def _flush(self, batches: BatchesDict) -> None:
        """Called by the writer thread. Owns a per-thread connection so
        WAL commits amortize across batches, without cross-thread
        SQLite handoff (which raises ProgrammingError)."""
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            self._tls.conn = conn
        rid = self._run_id
        _insert_samples(conn, rid, batches[_KIND_SAMPLE])
        _insert_steps(conn,   rid, batches[_KIND_STEP])
        _insert_traces(conn,  rid, batches[_KIND_TRACE])
        _insert_comm(conn,    rid, batches[_KIND_COMM])
        _insert_windows(conn, rid, batches[_KIND_WINDOW])
        _insert_host(conn,    rid, batches[_KIND_HOST])
        conn.commit()
