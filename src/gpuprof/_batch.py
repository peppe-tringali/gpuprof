"""One source of truth for the ingest batch schema.

Every place that ships events between client and server used to
hardcode the field list (`samples`, `steps`, `traces`, …). When a new
event kind was added, half a dozen sites needed updating — and one of
them (`drain.py`) fell out of sync and silently dropped `comm_events`
and `trace_windows`. Now every batch producer / consumer refers to
`BATCH_KINDS` and building an empty batch is `empty_batch()`.
"""
from __future__ import annotations


# Ordered so the JSON payload has a stable shape for tests / debugging.
BATCH_KINDS: tuple[str, ...] = (
    "samples",
    "steps",
    "traces",
    "comm_events",
    "trace_windows",
    "host_samples",
)


def empty_batch() -> dict:
    return {k: [] for k in BATCH_KINDS}


def batch_size(batch: dict) -> int:
    return sum(len(batch.get(k, ())) for k in BATCH_KINDS)


def batch_nonempty(batch: dict) -> bool:
    return any(batch.get(k) for k in BATCH_KINDS)


def merge_into(dst: dict, src: dict) -> None:
    """Append every kind's items from `src` into `dst` in place."""
    for k in BATCH_KINDS:
        dst.setdefault(k, []).extend(src.get(k, ()))
