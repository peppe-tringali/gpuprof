"""FastAPI server: HTTP ingest, WebSocket fan-out, static SPA.

Auth primitives (session cookie + CSRF + rate limiter) live in
`.auth`. Storage is a duck-typed store instance so either backend can
drive the same routes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI, HTTPException, Request, Response,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from .auth import (
    Auth, RateLimiter,
    SESSION_COOKIE as _SESSION_COOKIE,
    XSRF_COOKIE as _XSRF_COOKIE,
    SESSION_TTL_S as _SESSION_TTL_S,
)

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("gpuprof.server")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

def create_app(
    store,
    api_keys: Optional[list[str]] = None,
    viewer_password: Optional[str] = None,
    secure_cookies: bool = False,
    db_path_for_insights: Optional[str] = None,
) -> FastAPI:
    """Build the app.

    - `store`: SqliteServerStore or PostgresServerStore instance.
    - `api_keys`: list of accepted X-API-Key values. Empty disables write auth.
    - `viewer_password`: enables session-cookie viewer auth if set.
    - `secure_cookies`: set the Secure flag on cookies. Turn on when TLS.
    - `db_path_for_insights`: for the SQLite backend, path used by the
      offline insights module. When None, insights routes 501.
    """
    api_keys = set(api_keys or [])
    auth = Auth(viewer_password)
    login_limiter = RateLimiter(max_per_window=10, window_s=60.0)

    @asynccontextmanager
    async def lifespan(app):
        yield
        # Flush the writer thread on shutdown so a SIGTERM during
        # heavy ingest doesn't lose queued rows.
        try: store.close()
        except Exception: log.exception("store.close failed")

    app = FastAPI(title="gpuprof", lifespan=lifespan)

    subs: dict[int, set[WebSocket]] = {}
    subs_lock = asyncio.Lock()
    latest: dict[int, dict] = {}
    MAX_SAMPLES, MAX_STEPS, MAX_TRACES = 600, 1000, 20

    def _check_key(request: Request) -> None:
        if not api_keys:
            return
        got = request.headers.get("X-API-Key")
        if not got or got not in api_keys:
            raise HTTPException(403, "bad or missing api key")

    async def _broadcast(run_id: int, msg: dict) -> None:
        """Send `msg` to every subscriber. Uses asyncio.gather so a
        slow subscriber can't block the others; dead ones are pruned."""
        conns = list(subs.get(run_id, ()))
        if not conns:
            return
        results = await asyncio.gather(
            *(ws.send_json(msg) for ws in conns),
            return_exceptions=True,
        )
        dead = [ws for ws, r in zip(conns, results)
                if isinstance(r, BaseException)]
        if dead:
            async with subs_lock:
                s = subs.get(run_id)
                if s is not None:
                    for ws in dead: s.discard(ws)

    def _cookie_kwargs() -> dict:
        return dict(
            httponly=True, samesite="strict",
            secure=secure_cookies, max_age=_SESSION_TTL_S,
        )

    # -- auth routes --------------------------------------------------

    @app.get("/api/whoami")
    def whoami(request: Request):
        if not auth.enabled():
            return {"auth": "none"}
        cookie = request.cookies.get(_SESSION_COOKIE)
        if cookie and auth.verify_cookie(cookie):
            return {"auth": "ok"}
        return {"auth": "required"}

    @app.post("/api/login")
    async def login(request: Request, response: Response):
        if not auth.enabled():
            raise HTTPException(400, "viewer auth is not enabled")
        client = (request.client.host if request.client else "unknown")
        if not login_limiter.allow(client):
            raise HTTPException(429, "too many login attempts, try again later")
        body = await request.json()
        if body.get("password") != auth.viewer_password:
            log.warning("bad login attempt from %s", client)
            raise HTTPException(401, "bad password")
        response.set_cookie(_SESSION_COOKIE, auth.make_cookie(), **_cookie_kwargs())
        # CSRF token is *not* HttpOnly — the SPA needs to read it in JS
        # and echo it back in the X-XSRF-TOKEN header.
        xsrf = secrets.token_urlsafe(32)
        response.set_cookie(
            _XSRF_COOKIE, xsrf,
            httponly=False, samesite="strict",
            secure=secure_cookies, max_age=_SESSION_TTL_S,
        )
        return {"ok": True}

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        auth.require_csrf(request)
        response.delete_cookie(_SESSION_COOKIE)
        response.delete_cookie(_XSRF_COOKIE)
        return {"ok": True}

    # -- ingest -------------------------------------------------------

    @app.post("/api/runs")
    async def create_run(request: Request):
        _check_key(request)
        body = await request.json()
        run_id = store.create_run(
            name=body.get("name", "unnamed"),
            gpu_name=body.get("gpu_name", ""),
            meta_json=json.dumps(body.get("meta", {})),
            group_id=body.get("group_id"),
            rank=body.get("rank"),
            world_size=body.get("world_size"),
        )
        latest[run_id] = {
            "name": body.get("name", "unnamed"),
            "gpu_name": body.get("gpu_name", ""),
            "meta": body.get("meta", {}),
            "group_id": body.get("group_id"),
            "rank": body.get("rank"),
            "world_size": body.get("world_size"),
            "started_at": time.time(),
            "ended_at": None,
            "samples": [], "steps": [], "traces": [],
        }
        await _broadcast(0, {
            "kind": "run_start", "run_id": run_id,
            "name": body.get("name"), "gpu": body.get("gpu_name"),
            "group_id": body.get("group_id"), "rank": body.get("rank"),
        })
        return {"id": run_id}

    @app.post("/api/runs/{run_id}/ingest")
    async def ingest(run_id: int, request: Request):
        _check_key(request)
        body = await request.json()
        samples = body.get("samples", [])
        steps = body.get("steps", [])
        traces = body.get("traces", [])
        comm_events = body.get("comm_events", [])
        trace_windows = body.get("trace_windows", [])
        host_samples = body.get("host_samples", [])

        for s in samples: store.push_sample(run_id, s)
        for s in steps: store.push_step(run_id, s)
        for t in traces: store.push_trace(run_id, t)
        for e in comm_events:
            if hasattr(store, "push_comm_event"):
                store.push_comm_event(run_id, e)
        for w in trace_windows:
            if hasattr(store, "push_trace_window"):
                store.push_trace_window(run_id, w)
        for h in host_samples:
            if hasattr(store, "push_host_sample"):
                store.push_host_sample(run_id, h)

        info = latest.setdefault(run_id, {
            "samples": [], "steps": [], "traces": [], "meta": {},
            "gpu_name": "", "name": "unnamed",
            "started_at": time.time(), "ended_at": None,
        })
        info["samples"].extend(samples)
        info["steps"].extend(steps)
        info["traces"].extend(traces)
        if len(info["samples"]) > MAX_SAMPLES:
            del info["samples"][:len(info["samples"]) - MAX_SAMPLES]
        if len(info["steps"]) > MAX_STEPS:
            del info["steps"][:len(info["steps"]) - MAX_STEPS]
        if len(info["traces"]) > MAX_TRACES:
            del info["traces"][:len(info["traces"]) - MAX_TRACES]

        # Broadcast the whole ingest as one WS message — the previous
        # per-event fanout meant a batch of 500 samples fired 500
        # awaited sends. Now the browser gets one message per POST and
        # demuxes it into its own state.
        await _broadcast(run_id, {
            "kind": "batch",
            "samples": samples, "steps": steps, "traces": traces,
            "trace_windows": trace_windows,
        })
        return {"ok": True, "n_samples": len(samples),
                "n_steps": len(steps), "n_traces": len(traces),
                "n_comm_events": len(comm_events),
                "n_trace_windows": len(trace_windows),
                "n_host_samples": len(host_samples)}

    @app.post("/api/runs/{run_id}/rank_offset")
    async def set_rank_offset(run_id: int, request: Request):
        _check_key(request)
        body = await request.json()
        if hasattr(store, "set_rank_offset"):
            store.set_rank_offset(run_id, float(body.get("offset_s", 0.0)))
        return {"ok": True}

    @app.get("/api/runs/{run_id}/trace_windows")
    def get_trace_windows(run_id: int, request: Request, limit: int = 500):
        auth.require(request)
        if hasattr(store, "list_trace_windows"):
            return store.list_trace_windows(run_id, limit=limit)
        return []

    @app.post("/api/runs/{run_id}/end")
    async def end_run(run_id: int, request: Request):
        _check_key(request)
        store.end_run(run_id)
        # `latest` is a per-run rolling ring of samples/steps for
        # browsers connecting mid-run. Once the run ends we can drop
        # it — future WS connects fall back to `store.snapshot` which
        # reads the finalized data from the DB. Not popping here was
        # a memory leak: every completed run stayed resident forever.
        latest.pop(run_id, None)
        await _broadcast(run_id, {"kind": "end"})
        await _broadcast(0, {"kind": "run_end", "run_id": run_id})
        return {"ok": True}

    # -- read ---------------------------------------------------------

    @app.get("/api/runs")
    def list_runs(request: Request):
        auth.require(request)
        return store.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int, request: Request):
        auth.require(request)
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, "not found")
        return run

    @app.get("/api/runs/{run_id}/insights")
    def get_insights(run_id: int, request: Request):
        auth.require(request)
        if not db_path_for_insights:
            raise HTTPException(501, "insights disabled (SQLite path not configured)")
        from ..insights import analyze
        try:
            return analyze(db_path_for_insights, run_id)
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/runs/{run_id}/traces")
    def get_traces(run_id: int, request: Request):
        auth.require(request)
        return store.list_traces(run_id, limit=MAX_TRACES)

    @app.get("/api/groups")
    def list_groups(request: Request):
        auth.require(request)
        runs = store.list_runs()
        by_group: dict = {}
        for r in runs:
            gid = r.get("group_id")
            if not gid: continue
            g = by_group.setdefault(gid, {"group_id": gid, "runs": []})
            g["runs"].append(r)
        return list(by_group.values())

    @app.get("/api/groups/{group_id}")
    def get_group(group_id: str, request: Request):
        auth.require(request)
        runs = store.list_runs_in_group(group_id)
        if not runs:
            raise HTTPException(404, "group not found")
        return {"group_id": group_id, "runs": runs}

    @app.get("/api/groups/{group_id}/insights")
    def group_insights(group_id: str, request: Request):
        auth.require(request)
        if not db_path_for_insights:
            raise HTTPException(501, "insights disabled")
        from ..insights import analyze_group
        try:
            return analyze_group(db_path_for_insights, group_id)
        except ValueError as e:
            raise HTTPException(404, str(e))

    # -- WebSocket ----------------------------------------------------

    @app.websocket("/watch/{run_id}")
    async def watch(ws: WebSocket, run_id: int):
        if not auth.require_ws(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        snap = latest.get(run_id)
        if snap is None:
            snap = store.snapshot(run_id, max_samples=MAX_SAMPLES,
                                  max_steps=MAX_STEPS)
        await ws.send_json({
            "kind": "snapshot",
            "samples": snap.get("samples", []),
            "steps": snap.get("steps", []),
            "traces": snap.get("traces", []),
            "latest_trace": snap.get("latest_trace"),
            "meta": snap.get("meta", {}),
            "gpu_name": snap.get("gpu_name") or "",
            "name": snap.get("name") or "",
        })
        async with subs_lock:
            subs.setdefault(run_id, set()).add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            async with subs_lock:
                subs.get(run_id, set()).discard(ws)

    # -- static -------------------------------------------------------

    @app.get("/")
    def index():
        return FileResponse(str(STATIC / "index.html"))

    return app
