"""Cross-rank per-bucket skew analysis."""
import sqlite3

from gpuprof.insights import analyze_group
from gpuprof.store import Store, apply_schema


def _seed(db, group_id, per_rank_events):
    """per_rank_events: {rank: [(step, bucket_id, t_end_rel), ...]}
    Creates one run per rank with a small comm_events table populated."""
    conn = sqlite3.connect(db)
    try:
        apply_schema(conn)
        for rank, events in per_rank_events.items():
            cur = conn.execute(
                "INSERT INTO runs(name, started_at, gpu_name, group_id, "
                "rank, world_size, meta_json) VALUES (?,?,?,?,?,?,?)",
                (f"r{rank}", 0.0, "H100", group_id, rank,
                 len(per_rank_events), "{}"),
            )
            run_id = cur.lastrowid
            # Add step rows so analyze_group's step_times are populated.
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


def test_bucket_skew_flags_the_late_bucket(tmp_path):
    db = str(tmp_path / "s.db")
    # Bucket 3 is 20 ms slower on rank 1 for every step. Bucket 0 is
    # tight. Analysis should point at bucket 3, rank 1.
    r0_events, r1_events = [], []
    for step in range(20):
        for bucket_id in range(4):
            base = 0.010 * bucket_id
            r0_events.append((step, bucket_id, base + 0.001))
            late = 0.020 if bucket_id == 3 else 0.0
            r1_events.append((step, bucket_id, base + 0.001 + late))
    _seed(db, "g", {0: r0_events, 1: r1_events})

    r = analyze_group(db, "g")
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Bucket 3" in titles and "rank 1" in titles
    bskew = r["summary"]["bucket_skew"]
    assert bskew[0]["bucket_id"] == 3
    assert bskew[0]["worst_rank"] == 1


def test_bucket_skew_quiet_when_ranks_are_tight(tmp_path):
    db = str(tmp_path / "s.db")
    events = [[], []]
    for step in range(20):
        for bucket_id in range(4):
            for rank in (0, 1):
                events[rank].append((step, bucket_id, 0.010 * bucket_id + 0.001))
    _seed(db, "g", {0: events[0], 1: events[1]})
    r = analyze_group(db, "g")
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Bucket" not in titles
