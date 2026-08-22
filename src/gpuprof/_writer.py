"""BatchedWriter — the queue-drained background writer used by every
storage backend.

Before this module, the same "daemon thread pulls from a bounded
Queue, buckets events by `kind`, calls `_flush` when a batch fills
up or the queue idles" logic was written three times: client Store,
server SqliteServerStore, server PostgresServerStore. Each of the
three grew slightly different bugs (batch-clear ordering, exit
condition drift, missing final flush). Now the orchestration lives
here and each backend just supplies a `flush_batch` callable.

Under backpressure enqueue() drops rather than blocks — the producer
(training loop, HTTP handler) must never wait on I/O.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable


BatchesDict = dict[str, list]
FlushFn = Callable[[BatchesDict], None]


class BatchedWriter:
    """Queue-drained batch writer running on a daemon thread.

    The writer batches by event kind. On each flush the callback
    receives a dict mapping each kind to the list of payloads collected
    for it (in arrival order). The dict is reused across flushes —
    callbacks must not retain references to the lists.
    """

    def __init__(
        self,
        kinds: tuple[str, ...],
        flush_batch: FlushFn,
        *,
        batch_size: int = 200,
        queue_size: int = 100_000,
        idle_timeout_s: float = 0.25,
        thread_name: str = "gpuprof-writer",
    ):
        self._kinds = kinds
        self._flush_batch = flush_batch
        self._batch_size = batch_size
        self._idle_s = idle_timeout_s
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=thread_name,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def close(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=timeout)

    def enqueue(self, kind: str, payload) -> None:
        """Push one event. Drops silently under backpressure."""
        try:
            self._q.put_nowait((kind, payload))
        except queue.Full:
            pass

    # -- internals ----------------------------------------------------

    def _run(self) -> None:
        batches: BatchesDict = {k: [] for k in self._kinds}
        # Loop until both stop is signaled AND the queue is fully drained
        # — a caller-initiated close() should still commit any events
        # that landed after the stop flag was set.
        while not (self._stop.is_set() and self._q.empty()):
            try:
                kind, payload = self._q.get(timeout=self._idle_s)
            except queue.Empty:
                self._maybe_flush(batches)
                continue
            if kind in batches:
                batches[kind].append(payload)
            if sum(len(b) for b in batches.values()) >= self._batch_size:
                self._maybe_flush(batches)
        # Final flush on the way out so nothing sits half-written.
        self._maybe_flush(batches)

    def _maybe_flush(self, batches: BatchesDict) -> None:
        if not any(batches.values()):
            return
        try:
            self._flush_batch(batches)
        finally:
            # Always clear so a caller exception doesn't repeat the batch.
            for k in batches:
                batches[k].clear()
