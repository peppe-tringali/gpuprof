"""python -m gpuprof.server — launch the FastAPI app under uvicorn.

Picks the storage backend from --storage-url (or --db for the legacy
SQLite shortcut). Postgres requires `pip install gpuprof[postgres]`.
"""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from .app import create_app


def _build_store(storage_url: str, sqlite_db: str):
    """Return (store, sqlite_path_for_insights).

    Postgres callers get sqlite_path_for_insights=None — the insights
    module currently only knows SQLite; that route 501s in Postgres mode.
    """
    if storage_url.startswith("postgres://") or storage_url.startswith("postgresql://"):
        try:
            from .pg_store import PostgresServerStore
        except ImportError as e:
            print(f"Postgres backend requires psycopg: {e}", file=sys.stderr)
            print("  pip install 'gpuprof[postgres]'", file=sys.stderr)
            sys.exit(2)
        return PostgresServerStore(storage_url), None
    if storage_url.startswith("sqlite://"):
        path = storage_url[len("sqlite://"):]
    else:
        path = sqlite_db
    from .store import ServerStore
    return ServerStore(path), path


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m gpuprof.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="gpuprof_server.db",
                    help="legacy SQLite shortcut; equivalent to --storage-url sqlite://PATH")
    ap.add_argument("--storage-url", default=None,
                    help="sqlite://PATH or postgres://user:pass@host:port/db")
    ap.add_argument("--api-key",
                    default=os.environ.get("GPUPROF_API_KEY"),
                    help="one or more comma-separated write API keys; "
                    "if unset, writes are open")
    ap.add_argument("--viewer-password",
                    default=os.environ.get("GPUPROF_VIEWER_PASSWORD"),
                    help="enable session-cookie viewer auth")
    ap.add_argument("--secure-cookies", action="store_true",
                    help="set the Secure flag on cookies (use behind TLS)")
    args = ap.parse_args()

    storage_url = args.storage_url or f"sqlite://{args.db}"
    store, sqlite_path = _build_store(storage_url, args.db)

    api_keys = None
    if args.api_key:
        api_keys = [k.strip() for k in args.api_key.split(",") if k.strip()]

    app = create_app(
        store=store,
        api_keys=api_keys,
        viewer_password=args.viewer_password,
        secure_cookies=args.secure_cookies,
        db_path_for_insights=sqlite_path,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
