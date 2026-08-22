"""In-process local dashboard for `gpuprof.profile(dashboard=True)`.

Spawns a uvicorn server in a daemon thread on a free loopback port,
pointed at the client's local SQLite. Client pushes ingest to the
in-process server so WebSocket updates flow to the browser exactly
like a remote setup — no `gpuprof serve` command needed for the
"just show me a live dashboard" case.

The thread is a daemon: when the training process exits, the
dashboard exits with it. Users who want the dashboard to outlive the
training script should run `gpuprof serve` in a separate terminal
instead.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Optional


def start_local_dashboard(db_path: str) -> Optional[str]:
    """Start an in-process dashboard pointed at `db_path`. Returns the
    URL, or None if uvicorn/fastapi aren't installed.
    """
    try:
        import uvicorn
    except ImportError:
        return None

    from .server.app import create_app
    from .server.store import ServerStore

    port = _pick_free_port()
    store = ServerStore(db_path)
    app = create_app(
        store=store,
        db_path_for_insights=db_path,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",       # quiet — user just wants the URL
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, daemon=True, name="gpuprof-dashboard",
    )
    thread.start()
    # Poll until the port is genuinely listening. If we return before
    # uvicorn has bound the socket, `profile()` will disable the local
    # Store thinking we have a working server, and the very first
    # ingest POST will fail — losing the whole run. Return None if
    # the server never comes up so the caller can fall back to
    # local-only mode.
    for _ in range(40):
        time.sleep(0.05)
        if _port_is_listening(port):
            return f"http://127.0.0.1:{port}"
    return None


def _pick_free_port() -> int:
    """Ask the OS for a free ephemeral port on loopback. Closing the
    socket releases the port; there's a tiny race before uvicorn
    binds it, but on a workstation loopback that's not a problem."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _port_is_listening(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.05)
        return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False
    finally:
        s.close()
