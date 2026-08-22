"""Unified CLI: `gpuprof <command>`.

Dispatches to the same code paths as `python -m gpuprof.server` etc.,
but exposed under one command so the pip-installed `gpuprof` script
covers everything.
"""
from __future__ import annotations

import sys


USAGE = """\
gpuprof — GPU profiler + insight engine for PyTorch training

Usage: gpuprof <command> [args]

Commands:
  serve         Launch the dashboard server
                (`gpuprof serve --port 8000 --api-key SECRET`)
  insights      Run the offline insights CLI on a completed run
                (`gpuprof insights ./gpuprof.db 1`)
  drain         Push orphaned buffer files to a running server
                (`gpuprof drain --server http://host:8000`)
  nsys-import   Import an `nsys export --type sqlite` file into a run
                (`gpuprof nsys-import trace.sqlite --gpuprof-db db --run-id 5`)
  selfcheck     Probe the environment (NVML, torch, psutil, server deps)
                (`gpuprof selfcheck`)
  gc            Delete old runs from a gpuprof SQLite DB
                (`gpuprof gc --older-than 30d --keep-last 20`)
  report        Export a self-contained HTML report for a run
                (`gpuprof report ./gpuprof.db 1 --out=report.html`)
  version       Print installed gpuprof version

Run `gpuprof <command> --help` for command-specific usage.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    # Shift argv so downstream argparse sees `<cmd>` as argv[0].
    sys.argv = [f"gpuprof {cmd}"] + sys.argv[2:]

    if cmd == "serve":
        from .server.__main__ import main as _m
        _m()
    elif cmd == "insights":
        from .insights import _cli
        _cli()
    elif cmd == "drain":
        from .drain import main as _m
        _m()
    elif cmd == "nsys-import":
        from .nsys import _cli
        _cli()
    elif cmd == "selfcheck":
        from .selfcheck import _cli
        _cli()
    elif cmd == "gc":
        from .gc_cmd import _cli
        _cli()
    elif cmd == "report":
        from .report import _cli
        _cli()
    elif cmd == "version":
        from . import __version__
        print(__version__)
    else:
        print(f"unknown command: {cmd!r}\n\n{USAGE}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
