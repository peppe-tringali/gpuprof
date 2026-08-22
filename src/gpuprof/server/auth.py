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


class ApiKeySet:
    """Multi-tenant write-auth token registry.

    Two token shapes are accepted, both via `X-API-Key`:

    - **Bare secret** — an opaque token that just proves "you may
      write". Runs pushed under it are recorded with `owner_user=None`
      and `project="default"`. This is the backward-compatible form.
    - **Scoped token** — `user:project:secret`. Runs land with
      `owner_user=user` and `project=project`. Different users on
      the same server thus share the same DB but each other's runs
      can be filtered by project or owner in the dashboard.

    Provide keys via `--api-key` (repeatable), a comma-separated
    string, or `GPUPROF_API_KEY`. Constant-time comparison prevents
    a timing side channel on the secret.
    """

    def __init__(self, keys: Optional[list[str]]):
        # Store the raw strings; parse on lookup. Small N.
        self._keys = [k.strip() for k in (keys or []) if k.strip()]

    def enabled(self) -> bool:
        return bool(self._keys)

    def resolve(self, presented: Optional[str]) -> Optional[dict]:
        """Return `{"user": str|None, "project": str}` if `presented`
        matches any configured key, else None. When write-auth is
        disabled, always returns the "anonymous default" identity."""
        if not self._keys:
            return {"user": None, "project": "default"}
        if not presented:
            return None
        for k in self._keys:
            # Constant-time; avoids leaking length via short-circuit.
            if hmac.compare_digest(presented, k):
                return _parse_scope(k)
        return None


def _parse_scope(token: str) -> dict:
    """`user:project:secret` → {user, project}. Bare secret → default."""
    parts = token.split(":", 2)
    if len(parts) == 3 and parts[0] and parts[1]:
        return {"user": parts[0], "project": parts[1]}
    return {"user": None, "project": "default"}


class RateLimiter:
    """Per-key sliding window. Thread-safe.

    Two safety nets against unbounded memory growth on long-running
    servers that see many distinct IPs:

    - Empty deques are evicted on the read path (so a burst of new
      IPs doesn't leave a lingering entry per IP after the window).
    - A hard cap on the number of live keys; when reached we evict
      the oldest-touched keys en masse. A saturated bucket a
      malicious client tries to plant is thus bounded in RAM.
    """

    _MAX_KEYS = 10_000

    def __init__(self, max_per_window: int, window_s: float):
        self._max = max_per_window
        self._win = window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            # Amortized reap on every allow — the dict is bounded to
            # ~10k keys and each reap is O(N). At 100 logins/sec that's
            # ~1M dict lookups/sec worst case — trivial CPU. Prevents
            # any IP that has ever hit the endpoint from leaving a
            # persistent entry once its window expires.
            self._reap_locked(now)
            q = self._hits.setdefault(key, deque())
            while q and q[0] < now - self._win:
                q.popleft()
            allowed = len(q) < self._max
            if allowed:
                q.append(now)
            if not q:
                self._hits.pop(key, None)
            return allowed

    def _reap_locked(self, now: float) -> None:
        """Sweep the whole table for expired keys."""
        cutoff = now - self._win
        expired = [k for k, q in self._hits.items()
                   if not q or q[-1] < cutoff]
        for k in expired:
            self._hits.pop(k, None)
        # Hard cap defense: if a burst punched past the sweep, drop
        # the oldest by inspection.
        if len(self._hits) > self._MAX_KEYS:
            excess = len(self._hits) - self._MAX_KEYS
            # deque[-1] is the most recent hit; sort ascending by it
            # and drop the tail.
            oldest = sorted(self._hits.items(),
                            key=lambda kv: kv[1][-1] if kv[1] else 0)[:excess]
            for k, _ in oldest:
                self._hits.pop(k, None)
