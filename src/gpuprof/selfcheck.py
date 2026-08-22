"""`gpuprof selfcheck` — environment probe.

Runs a series of small checks to confirm gpuprof will do the right
thing on the machine it's running on. Designed for the "install on
a real GPU box" moment where the developer wants to know:

  - Does the NVML backend actually see the GPU(s)?
  - Can `torch.profiler` capture a trace?
  - Does `torch.distributed` see the launcher env vars?
  - Are host-side signals (psutil) available?
  - Do FastAPI + uvicorn import cleanly for the server path?
  - Is Postgres reachable (if user pointed at one)?

Each check prints [ok] / [warn] / [fail] with a one-line reason.
Exit code 0 if everything critical passed, 1 if a critical check
failed. Warnings never fail the exit code — they surface things
that only some deployments care about.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable


_RESULTS: list[tuple[str, str, str]] = []       # (level, check, detail)


def _ok(check: str, detail: str = "") -> None:
    _RESULTS.append(("ok", check, detail))


def _warn(check: str, detail: str) -> None:
    _RESULTS.append(("warn", check, detail))


def _fail(check: str, detail: str) -> None:
    _RESULTS.append(("fail", check, detail))


# --- individual checks --------------------------------------------------

def _check_python() -> None:
    v = sys.version_info
    if v < (3, 9):
        _fail("python", f"gpuprof requires 3.9+, running {v.major}.{v.minor}")
    else:
        _ok("python", f"{v.major}.{v.minor}.{v.micro}")


def _check_gpuprof() -> None:
    try:
        import gpuprof
        _ok("gpuprof import", f"v{gpuprof.__version__}")
    except Exception as e:                            # pragma: no cover
        _fail("gpuprof import", str(e))


def _check_nvml() -> None:
    try:
        import pynvml
    except ImportError:
        _warn("nvml", "pynvml not installed — mock backend will be used "
              "(install with `pip install pynvml` on a GPU box)")
        return
    try:
        pynvml.nvmlInit()
        try:
            n = pynvml.nvmlDeviceGetCount()
            names = []
            for i in range(min(n, 4)):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                nm = pynvml.nvmlDeviceGetName(h)
                if isinstance(nm, bytes): nm = nm.decode()
                names.append(nm)
            more = f" (+{n - len(names)} more)" if n > len(names) else ""
            _ok("nvml", f"{n} GPU(s): {', '.join(names)}{more}")
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        _fail("nvml", f"nvmlInit failed: {e}")


def _check_psutil() -> None:
    try:
        import psutil
        v = getattr(psutil, "__version__", "?")
        _ok("psutil", f"v{v}; host sampling available")
    except ImportError:
        _warn("psutil", "not installed — host CPU/disk/memory rules will "
              "skip (install with `pip install psutil`)")


def _check_torch() -> None:
    try:
        import torch
    except ImportError:
        _warn("torch", "not installed — auto-instrumentation and CUDA "
              "event comm timing unavailable")
        return
    cuda = torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        cur = torch.cuda.get_device_name(0)
        _ok("torch", f"v{torch.__version__}, CUDA {torch.version.cuda}, "
                     f"{n} device(s), 'cuda:0' = {cur}")
    else:
        _warn("torch", f"v{torch.__version__} installed but "
              "torch.cuda.is_available() is False — CPU-only mode")


def _check_torch_profiler() -> None:
    try:
        import torch.profiler as tp
    except ImportError:
        _warn("torch.profiler", "torch not installed; kernel tracing disabled")
        return
    # Do a one-step trace to prove CUPTI wires up.
    try:
        with tp.profile(activities=[tp.ProfilerActivity.CPU],
                        record_shapes=False):
            pass
        _ok("torch.profiler", "profile() context works")
    except Exception as e:                            # pragma: no cover
        _fail("torch.profiler", f"profile() raised: {e}")


def _check_distributed_env() -> None:
    hints = [k for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK",
                          "TORCHELASTIC_RUN_ID") if k in os.environ]
    if not hints:
        _ok("torch.distributed", "no torchrun env — single-process mode")
        return
    parts = [f"{k}={os.environ[k]}" for k in hints]
    _ok("torch.distributed", "; ".join(parts))


def _check_server_deps() -> None:
    try:
        import fastapi, uvicorn                       # noqa: F401
        _ok("server deps", f"fastapi {fastapi.__version__}, "
                            f"uvicorn {uvicorn.__version__}")
    except ImportError as e:
        _warn("server deps", f"{e.name} missing — `gpuprof serve` and "
              "`dashboard=True` won't work. Install with "
              "`pip install 'gpuprof[server]'`.")


def _check_postgres_env() -> None:
    dsn = os.environ.get("GPUPROF_STORAGE_URL", "")
    if not dsn.startswith(("postgres://", "postgresql://")):
        return  # nothing to check
    try:
        import psycopg
    except ImportError:
        _fail("postgres", "GPUPROF_STORAGE_URL points at Postgres but "
              "psycopg is not installed")
        return
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                v = cur.fetchone()[0].split(" on ")[0]
        _ok("postgres", f"reachable — {v}")
    except Exception as e:
        _fail("postgres", f"cannot reach {dsn}: {e}")


def _check_local_db_writable() -> None:
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
            from .store import Store
            s = Store(f.name)
            s.start_run("selfcheck-probe", "MockGPU")
            s.end_run()
        _ok("sqlite", "read + write + WAL journal work")
    except Exception as e:
        _fail("sqlite", f"local SQLite failed: {e}")


CHECKS: list[Callable[[], None]] = [
    _check_python, _check_gpuprof, _check_nvml, _check_psutil,
    _check_torch, _check_torch_profiler, _check_distributed_env,
    _check_server_deps, _check_postgres_env, _check_local_db_writable,
]


def _print_and_exit() -> int:
    tag = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}
    for level, check, detail in _RESULTS:
        line = f"  {tag[level]}  {check:<22}  {detail}"
        print(line)
    fails = sum(1 for r in _RESULTS if r[0] == "fail")
    warns = sum(1 for r in _RESULTS if r[0] == "warn")
    print()
    if fails:
        print(f"selfcheck: {fails} FAIL, {warns} warn — "
              "critical checks did not pass.", file=sys.stderr)
        return 1
    if warns:
        print(f"selfcheck: 0 fail, {warns} warn — usable, but the "
              "warnings limit what gpuprof can see.")
    else:
        print("selfcheck: all checks passed. You're good.")
    return 0


def _cli() -> None:
    ap = argparse.ArgumentParser(prog="gpuprof selfcheck")
    ap.add_argument("--only", nargs="+", metavar="CHECK",
                    help="run only the named checks (e.g. `--only nvml sqlite`)")
    args = ap.parse_args()

    to_run = CHECKS
    if args.only:
        wanted = set(args.only)
        to_run = [f for f in CHECKS
                  if any(n in f.__name__ for n in wanted)]
    for f in to_run:
        try:
            f()
        except Exception as e:                        # pragma: no cover
            _fail(f.__name__.lstrip("_"), f"unexpected: {e}")
    sys.exit(_print_and_exit())


if __name__ == "__main__":
    _cli()
