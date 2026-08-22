"""Regression tests for the adversarial-review fixes.

Each test corresponds to one issue from the audit; the docstring
names the finding so a future regression is easy to trace back.
"""
import sqlite3
import time

import pytest

import gpuprof
from gpuprof.sampler import _NvmlBackend, _MockBackend, _nvml_init_count


# ── C1 ─ dashboard failure must not destroy the run ──────────────────

def test_c1_dashboard_keeps_local_store_enabled(tmp_path, monkeypatch, capsys):
    """dashboard=True must NOT set db_path=None. If uvicorn crashes
    mid-run we still need a local DB to fall back on."""
    monkeypatch.delenv("GPUPROF_SERVER", raising=False)
    db = str(tmp_path / "c1.db")
    with gpuprof.profile("c1", db_path=db, host_sampling=False,
                          summary=False, dashboard=True):
        pass
    # The local DB must have received the run row even though
    # dashboard=True was set — nothing should have been redirected
    # exclusively through the in-process server.
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT name FROM runs").fetchone()
    conn.close()
    assert row and row[0] == "c1"


def test_c1_dashboard_start_returns_none_when_port_never_listens(monkeypatch):
    """_pick_free_port could race — the dashboard helper must return
    None (not a bogus URL) if uvicorn never binds."""
    from gpuprof import _local_dashboard as ld
    monkeypatch.setattr(ld, "_port_is_listening", lambda p: False)
    # Poll happens 40× 0.05 s = 2 s. Speed it up so tests stay fast.
    orig_sleep = time.sleep
    monkeypatch.setattr(ld.time, "sleep", lambda s: orig_sleep(0.001))
    url = ld.start_local_dashboard("/tmp/never-binds.db")
    assert url is None


# ── C2 ─ NVML refcount must survive multiple init/shutdown ───────────

def test_c2_nvml_refcount(monkeypatch):
    """Two backends creating and destroying NVML handles must not
    poison each other. Before the fix, the second backend's shutdown
    left the first backend's handles invalid."""
    # Use the mock backend so we don't need real NVML. The refcount
    # code lives in _NvmlBackend; simulate the calls directly.
    import gpuprof.sampler as sm

    class _FakePynvml:
        inits = 0
        shutdowns = 0
        NVMLError = Exception
        NVMLError_Uninitialized = Exception

        @classmethod
        def nvmlInit(cls): cls.inits += 1

        @classmethod
        def nvmlShutdown(cls): cls.shutdowns += 1

    p = _FakePynvml
    # Simulate three concurrent backends init+shutdown.
    for _ in range(3): sm._nvml_init(p)
    assert p.inits == 1                       # init only once
    for _ in range(2): sm._nvml_shutdown(p)
    assert p.shutdowns == 0                    # not yet — refcount still 1
    sm._nvml_shutdown(p)
    assert p.shutdowns == 1                    # now


# ── C5 ─ clock-offset pings default OFF ──────────────────────────────

def test_c5_estimate_clock_offset_default_false():
    """A single stalled/crashed peer must not freeze `profile()` on
    entry. Default off; users opt in when they need μs precision."""
    from gpuprof.profiler import GpuProfiler
    prof = GpuProfiler(run_name="c5", db_path=None, host_sampling=False)
    assert prof._estimate_offset is False


# ── H5 ─ wrap_dataloader precedence over auto's inter-step estimate ─

def test_h5_wrap_dataloader_precedence(tmp_path):
    """When both `wrap_dataloader` and auto-instrumentation could
    attribute the wait, `wrap_dataloader` wins — it's the explicit
    signal from the user."""
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x): return self.lin(x).sum()

    class Loader:
        def __iter__(self):
            for _ in range(4):
                time.sleep(0.05)       # 50 ms per batch — the "real" wait
                yield torch.randn(2, 4)

        def __len__(self): return 4

    db = str(tmp_path / "h5.db")
    with gpuprof.profile("h5", db_path=db, host_sampling=False,
                          summary=False) as prof:
        model = Model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for batch in prof.wrap_dataloader(Loader()):
            loss = model(batch)
            loss.backward()
            opt.step()

    conn = sqlite3.connect(db)
    waits = [r[0] for r in conn.execute(
        "SELECT dataloader_wait_s FROM steps ORDER BY step"
    ).fetchall()]
    conn.close()
    # All four steps see the ~50 ms Loader delay.
    for w in waits:
        assert w is not None and 0.030 <= w <= 0.150, waits


# ── H6 ─ Optimizer subclasses constructed after profile.start ────────

def test_h6_lazy_optimizer_construction_still_wrapped(tmp_path):
    """HF Trainer / DeepSpeed construct optimizers inside train().
    Our __init__ hook must catch them."""
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x): return self.lin(x).sum()

    db = str(tmp_path / "h6.db")
    with gpuprof.profile("h6", db_path=db, host_sampling=False,
                          summary=False):
        model = Model()
        # Construct the optimizer AFTER profile() has entered.
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(3):
            loss = model(torch.randn(2, 4))
            loss.backward()
            opt.step()

    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
    conn.close()
    # Three optimizer.step calls → three committed steps.
    assert n == 3


# ── H1 ─ eval / no_grad blocks don't poison the next training step ──

def test_h1_eval_pass_skipped_by_auto_hook(tmp_path):
    """A mid-training validation pass with `model.eval()` +
    `torch.no_grad()` must NOT get counted into the surrounding
    training step's `forward_s`."""
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            time.sleep(0.010)          # noticeable per-forward time
            return self.lin(x).sum()

    db = str(tmp_path / "h1.db")
    with gpuprof.profile("h1", db_path=db, host_sampling=False,
                          summary=False):
        model = Model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        # Training step:
        loss = model(torch.randn(2, 4))
        loss.backward()
        # Simulate a big mid-training validation loop.
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(torch.randn(2, 4))
        model.train()
        # Same optimizer.step for the ONE training forward above.
        opt.step()

    conn = sqlite3.connect(db)
    fw = conn.execute("SELECT forward_s FROM steps").fetchone()[0]
    conn.close()
    # ONE training forward at ~10 ms — not 6 (1 train + 5 val).
    assert fw is not None
    assert 0.005 <= fw <= 0.030, fw
