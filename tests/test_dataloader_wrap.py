"""`GpuProfiler.wrap_dataloader` attributes real prefetch-queue wait."""
import sqlite3
import time

from gpuprof import GpuProfiler


class _FakeLoader:
    """Stand-in for a torch DataLoader. Returns batches with a
    controllable per-item delay so we can simulate a slow / empty
    prefetch queue deterministically."""

    def __init__(self, delays: list[float]):
        self._delays = delays
        self.batch_size = 32          # attribute passthrough test

    def __iter__(self):
        for d in self._delays:
            time.sleep(d)
            yield {"batch": True}

    def __len__(self):
        return len(self._delays)


def test_wrap_dataloader_captures_wait_into_step(tmp_path):
    """A 50ms per-batch delay must land in dataloader_wait_s (± jitter)."""
    db = str(tmp_path / "pf.db")
    prof = GpuProfiler(run_name="pf", db_path=db,
                       host_sampling=False, meta={})
    loader = _FakeLoader([0.05] * 6)
    prof.start()
    for i, batch in enumerate(prof.wrap_dataloader(loader)):
        with prof.step(i) as s:
            with s.phase("forward"):  time.sleep(0.005)
    prof.stop()

    conn = sqlite3.connect(db)
    waits = [r[0] for r in conn.execute(
        "SELECT dataloader_wait_s FROM steps ORDER BY step").fetchall()]
    conn.close()

    # Every batch fetch cost 50 ms — the wrapper attributes each wait
    # to the step that follows, so we expect ~50 ms on every step.
    assert len(waits) == 6
    for w in waits:
        assert w is not None and 0.030 <= w <= 0.150, w


def test_wrap_dataloader_preserves_len_and_attrs():
    """The wrapper is a drop-in replacement — len() and passthrough
    attributes must work so torchvision/HF training loops don't care."""
    prof = GpuProfiler(run_name="pf", db_path=None,
                       host_sampling=False, meta={})
    loader = _FakeLoader([0.001] * 3)
    wrapped = prof.wrap_dataloader(loader)
    assert len(wrapped) == 3
    assert wrapped.batch_size == 32          # __getattr__ delegation
