"""Framework-integration adapters.

Only import surface + basic wiring is verified here — we don't require
Lightning / HF / DeepSpeed to be installed to run the tests. When they
*are* installed, additional behavioral tests run.
"""
import sqlite3
import time

import pytest

from gpuprof import GpuProfiler


# ---- Lightning ---------------------------------------------------------

def test_lightning_import_without_pl_raises_clearly():
    """If pytorch_lightning isn't installed, the class still imports;
    only __init__ raises the friendly error."""
    from gpuprof.integrations.lightning import GpuProfilerCallback  # noqa
    try:
        import pytorch_lightning  # noqa
        pytest.skip("pl is installed; behavior differs")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pytorch_lightning"):
        GpuProfilerCallback(run_name="x", db_path=None)


def test_lightning_callback_lifecycle_simulated(tmp_path):
    """Manually drive the Lightning hook sequence — proves the phase
    handoff works without needing an actual Trainer.fit()."""
    try:
        import pytorch_lightning  # noqa
    except ImportError:
        pytest.skip("pytorch_lightning not installed")
    from gpuprof.integrations import LightningCallback
    cb = LightningCallback(run_name="lt", db_path=str(tmp_path / "l.db"),
                           meta={})
    cb.on_train_start(None, None)
    for i in range(3):
        cb.on_train_batch_start(None, None, batch=None, batch_idx=i)
        time.sleep(0.005)
        cb.on_before_backward(None, None, loss=None)
        time.sleep(0.005)
        cb.on_before_optimizer_step(None, None, optimizer=None)
        time.sleep(0.002)
        cb.on_train_batch_end(None, None, outputs={"loss": 0.5}, batch=None, batch_idx=i)
    cb.on_train_end(None, None)

    conn = sqlite3.connect(str(tmp_path / "l.db"))
    rows = conn.execute("SELECT step, forward_s, backward_s, optimizer_s, loss FROM steps").fetchall()
    conn.close()
    assert len(rows) == 3
    # Upper bounds generous for shared CI — see DeepSpeed test below
    # for the same rationale.
    for step, fw, bw, op, loss in rows:
        assert fw and 0.003 <= fw <= 0.300, fw
        assert bw and 0.003 <= bw <= 0.300, bw
        assert op and 0.001 <= op <= 0.300, op
        assert loss == 0.5


# ---- HuggingFace Trainer -----------------------------------------------

def test_hf_import_without_transformers_raises_clearly():
    from gpuprof.integrations.hf_trainer import GpuProfilerTrainerCallback
    try:
        import transformers  # noqa
        pytest.skip("transformers is installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="transformers"):
        GpuProfilerTrainerCallback(run_name="x", db_path=None)


# ---- DeepSpeed wrapper -------------------------------------------------

class _FakeDsEngine:
    """Behaves enough like a DeepSpeedEngine to test the wrapper."""

    def __init__(self):
        self.forward_calls = 0
        self.backward_calls = 0
        self.step_calls = 0

    def __call__(self, batch):
        time.sleep(0.005)
        self.forward_calls += 1
        class L:
            def item(inner): return 0.42
        return L()

    def backward(self, loss):
        time.sleep(0.003)
        self.backward_calls += 1

    def step(self):
        time.sleep(0.002)
        self.step_calls += 1


def test_deepspeed_wrapper_captures_all_three_phases(tmp_path):
    from gpuprof.integrations import wrap_deepspeed_engine
    ds = _FakeDsEngine()
    engine = wrap_deepspeed_engine(ds, run_name="ds",
                                   db_path=str(tmp_path / "ds.db"))
    for _ in range(4):
        loss = engine(batch=None)
        engine.backward(loss)
        engine.step()
    engine.close()

    conn = sqlite3.connect(str(tmp_path / "ds.db"))
    rows = conn.execute("SELECT step, forward_s, backward_s, optimizer_s, loss FROM steps ORDER BY step").fetchall()
    conn.close()
    assert len(rows) == 4
    # Upper bounds widened for shared CI runners — nominal sleeps
    # are 5 / 3 / 2 ms; on macOS-latest runners they can measure up
    # to ~55 ms. Lower bounds still catch "phase not measured".
    for step, fw, bw, op, loss in rows:
        assert fw and 0.003 <= fw <= 0.300, fw
        assert bw and 0.001 <= bw <= 0.300, bw
        assert op and 0.0005 <= op <= 0.300, op
        assert loss == 0.42
    assert ds.forward_calls == 4 and ds.backward_calls == 4 and ds.step_calls == 4


# ---- Postgres store (import-only) --------------------------------------

def test_postgres_store_imports():
    """Doesn't require a Postgres server — just that the module loads
    and the schema constant is well-formed."""
    from gpuprof.server.pg_store import PostgresServerStore, PG_SCHEMA
    assert "CREATE TABLE" in PG_SCHEMA
    # PostgresServerStore.__init__ will try to connect; importing is enough here.
    assert callable(PostgresServerStore)
