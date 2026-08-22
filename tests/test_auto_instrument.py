"""Auto-instrumentation: `with gpuprof.profile("x"):` captures step,
forward, backward, and optimizer time without any explicit
`prof.step()` / `s.phase()` calls."""
import sqlite3
import time

import pytest

import gpuprof

torch = pytest.importorskip("torch")


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        # Add a sleep so the auto-tracked forward phase has real time
        # (kernel launch on CPU is otherwise sub-ms).
        time.sleep(0.005)
        return self.linear(x).sum()


def test_auto_captures_all_phases_no_manual_wrapping(tmp_path):
    """Zero-config form — no prof.step, no s.phase. All fields
    populate."""
    db = str(tmp_path / "auto.db")
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with gpuprof.profile("auto-test",
                          db_path=db, host_sampling=False):
        for _ in range(5):
            x = torch.randn(4, 8)
            loss = model(x)          # ← forward auto-tracked
            loss.backward()          # ← backward auto-tracked
            opt.step()               # ← step boundary + optimizer phase
            opt.zero_grad()

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT step, forward_s, backward_s, optimizer_s "
        "FROM steps ORDER BY step"
    ).fetchall()
    conn.close()

    assert len(rows) == 5, rows
    for step, fw, bw, op in rows:
        assert fw is not None and fw > 0.001, f"forward not captured @ {step}"
        assert bw is not None and bw > 0.0,   f"backward not captured @ {step}"
        assert op is not None and op > 0.0,   f"optimizer not captured @ {step}"


def test_auto_captures_dataloader_wait_between_steps(tmp_path):
    """The gap between one opt.step() and the next model() call is
    attributed to `dataloader_wait_s` on the new step."""
    db = str(tmp_path / "dl.db")
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with gpuprof.profile("dl-test",
                          db_path=db, host_sampling=False):
        for _ in range(4):
            x = torch.randn(4, 8)
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
            time.sleep(0.05)         # simulated loader wait between steps

    conn = sqlite3.connect(db)
    waits = [r[0] for r in conn.execute(
        "SELECT dataloader_wait_s FROM steps ORDER BY step"
    ).fetchall()]
    conn.close()
    # Step 0 has no prior step → no wait. Steps 1..3 should see ~50 ms.
    assert waits[0] is None or waits[0] < 0.020, waits
    for w in waits[1:]:
        assert w is not None and 0.030 <= w <= 0.150, waits


def test_gradient_accumulation_accumulates_into_one_step(tmp_path):
    """Multiple fwd+bwd per opt.step should register as ONE gpuprof
    step whose forward/backward times are the sum of the micro-batches."""
    db = str(tmp_path / "acc.db")
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with gpuprof.profile("accum-test",
                          db_path=db, host_sampling=False):
        for _ in range(3):
            for _ in range(4):       # 4 micro-batches per opt.step
                loss = model(torch.randn(4, 8))
                loss.backward()
            opt.step()
            opt.zero_grad()

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT forward_s FROM steps ORDER BY step"
    ).fetchall()
    conn.close()

    assert len(rows) == 3
    # 4 micro-batches × ~5 ms sleep = ~20 ms accumulated forward.
    for (fw,) in rows:
        assert fw is not None and fw > 0.015, fw


def test_auto_unpatches_cleanly(tmp_path):
    """After `profile()` exits, torch's monkey-patched methods must be
    restored — no residual instrumentation on later training."""
    import torch.nn as nn
    orig_call = nn.Module.__call__
    orig_backward = torch.Tensor.backward
    orig_sgd_step = torch.optim.SGD.step

    with gpuprof.profile("unpatch",
                          db_path=str(tmp_path / "u.db"),
                          host_sampling=False):
        # Inside the block, they should be different.
        assert nn.Module.__call__ is not orig_call
        assert torch.Tensor.backward is not orig_backward
        assert torch.optim.SGD.step is not orig_sgd_step

    # Outside, they must be restored.
    assert nn.Module.__call__ is orig_call
    assert torch.Tensor.backward is orig_backward
    assert torch.optim.SGD.step is orig_sgd_step


def test_auto_unpatches_on_exception(tmp_path):
    """A training-loop exception must still unpatch on its way out."""
    import torch.nn as nn
    orig_call = nn.Module.__call__
    try:
        with gpuprof.profile("boom",
                              db_path=str(tmp_path / "b.db"),
                              host_sampling=False):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert nn.Module.__call__ is orig_call


def test_auto_false_leaves_torch_unpatched(tmp_path):
    """`auto=False` opt-out: no patching happens."""
    import torch.nn as nn
    orig_call = nn.Module.__call__
    with gpuprof.profile("no-auto",
                          db_path=str(tmp_path / "n.db"),
                          host_sampling=False, auto=False):
        assert nn.Module.__call__ is orig_call
    assert nn.Module.__call__ is orig_call
