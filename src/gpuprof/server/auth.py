"""Server-side auth primitives.

Two independent axes:
- **Write auth** (`X-API-Key` header): a set of accepted keys; empty
  set disables write auth (dev only).
- **Viewer auth** (session cookie): HMAC-SHA256-signed cookie with a
  double-submit CSRF token for state-changing session-authed routes.

Also lives here: the sliding-window `RateLimiter` used on the login
route. Extracted from `app.py` so the route factory can focus on
routing, not on cryptographic details.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections import deque
from typing import Optional

from fastapi import HTTPException, Request, WebSocket


SESSION_COOKIE = "gpuprof_session"
XSRF_COOKIE = "XSRF-TOKEN"
SESSION_TTL_S = 30 * 24 * 3600


log = logging.getLogger("gpuprof.server.auth")


class Auth:
    """HMAC-signed session cookies. Optional; if `viewer_password` is
    None the read side is open."""

    def __init__(self, viewer_password: Optional[str]):
        self.viewer_password = viewer_password
        env_secret = os.environ.get("GPUPROF_SECRET", "")
        # Random per-process default. Set GPUPROF_SECRET explicitly for
        # cookies to survive a server restart.
        self._secret = env_secret.encode() if env_secret else secrets.token_bytes(32)

    def enabled(self) -> bool:
        return bool(self.viewer_password)

    def make_cookie(self) -> str:
        payload = {"exp": int(time.time()) + SESSION_TTL_S}
        body = _b64u(json.dumps(payload).encode())
        sig = _b64u(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())
        return f"{body}.{sig}"

    def verify_cookie(self, raw: str) -> bool:
        try:
            body, sig = raw.split(".", 1)
            expected = hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
            got = _b64u_decode(sig)
            if not hmac.compare_digest(expected, got):
                return False
            payload = json.loads(_b64u_decode(body))
            return payload.get("exp", 0) > time.time()
        except Exception:
            return False

    def require(self, request: Request) -> None:
        """Raise 401 if viewer auth is on and the caller isn't logged in."""
        if not self.enabled():
            return
        cookie = request.cookies.get(SESSION_COOKIE)
        if not cookie or not self.verify_cookie(cookie):
            raise HTTPException(401, "auth required")

    def require_csrf(self, request: Request) -> None:
        """Double-submit CSRF token check for session-authed state changes."""
        if not self.enabled():
            return
        cookie_tok = request.cookies.get(XSRF_COOKIE)
        header_tok = request.headers.get("X-XSRF-TOKEN")
        if not cookie_tok or not header_tok or not hmac.compare_digest(
            cookie_tok, header_tok,
        ):
            raise HTTPException(403, "csrf token missing or mismatched")

    def require_ws(self, ws: WebSocket) -> bool:
        if not self.enabled():
            return True
        cookie = ws.cookies.get(SESSION_COOKIE)
        return bool(cookie and self.verify_cookie(cookie))


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


class RateLimiter:
    """Per-key sliding window. Thread-safe. Small in-memory cost:
    O(N_active_keys × window_hits)."""

    def __init__(self, max_per_window: int, window_s: float):
        self._max = max_per_window
        self._win = window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] < now - self._win:
                q.popleft()
            if len(q) >= self._max:
                return False
            q.append(now)
            return True
