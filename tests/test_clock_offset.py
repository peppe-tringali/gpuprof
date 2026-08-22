"""Cross-rank clock-offset estimation + application in bucket skew."""
import sqlite3

from gpuprof.insights import analyze_group
from gpuprof.store import apply_schema


def _seed(db, group_id, per_rank_events, offsets):
    """per_rank_events: {rank: [(step, bucket_id, t_end_rel), ...]}
    offsets:            {rank: rank_offset_s}
    """
    conn = sqlite3.connect(db)
    try:
        apply_schema(conn)
        for rank, events in per_rank_events.items():
            cur = conn.execute(
                "INSERT INTO runs(name, started_at, gpu_name, group_id, "
                "rank, world_size, meta_json, rank_offset_s) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"r{rank}", 0.0, "H100", group_id, rank,
                 len(per_rank_events), "{}", offsets.get(rank, 0.0)),
            )
            run_id = cur.lastrowid
            for step_i in range(20):
                conn.execute(
                    "INSERT INTO steps(run_id, step, t_start, t_end) "
                    "VALUES (?,?,?,?)",
                    (run_id, step_i, step_i * 0.1, step_i * 0.1 + 0.05),
                )
            for step, bucket_id, t_end in events:
                conn.execute(
                    "INSERT INTO comm_events(run_id, step, bucket_id, "
                    "t_start_rel, t_end_rel) VALUES (?,?,?,?,?)",
                    (run_id, step, bucket_id, t_end - 0.001, t_end),
                )
        conn.commit()
    finally:
        conn.close()


def test_offset_removes_spurious_skew(tmp_path):
    """Two ranks whose clocks are 10 ms apart (perfectly correlated
    offset) should show NO skew once the estimated offset is applied.
    Without offset application, the raw numbers would falsely fire."""
    db = str(tmp_path / "off.db")
    # Rank 0 and rank 1 do the SAME work at the SAME "true" time, but
    # rank 1's clock reads 10 ms later. Offset field encodes that.
    r0, r1 = [], []
    for step in range(20):
        for bucket_id in range(4):
            base = 0.010 * bucket_id + 0.001
            r0.append((step, bucket_id, base))
            r1.append((step, bucket_id, base + 0.010))
    _seed(db, "gO", {0: r0, 1: r1}, {0: 0.0, 1: 0.010})

    r = analyze_group(db, "gO")
    titles = " | ".join(i["title"] for i in r["insights"])
    # After offset application, no bucket skew should fire.
    assert "Bucket" not in titles


def test_offset_preserves_real_skew(tmp_path):
    """Rank 1 has a 10ms clock offset AND is genuinely 20ms slow on
    bucket 3 — offset removes the false 10ms, the real 20ms remains
    and fires."""
    db = str(tmp_path / "real.db")
    r0, r1 = [], []
    for step in range(20):
        for bucket_id in range(4):
            base = 0.010 * bucket_id + 0.001
            r0.append((step, bucket_id, base))
            extra = 0.020 if bucket_id == 3 else 0.0
            r1.append((step, bucket_id, base + 0.010 + extra))
    _seed(db, "gR", {0: r0, 1: r1}, {0: 0.0, 1: 0.010})

    r = analyze_group(db, "gR")
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Bucket 3" in titles and "rank 1" in titles
