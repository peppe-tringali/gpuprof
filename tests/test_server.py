"""Server tests via FastAPI TestClient.

Covers the two auth axes (write api-key, viewer session), CSRF, login
rate limiting, ingest, and group / insights routes.
"""
import time

import pytest
from fastapi.testclient import TestClient

from gpuprof.server.app import create_app
from gpuprof.server.store import ServerStore


@pytest.fixture
def app_no_auth(tmp_path):
    store = ServerStore(str(tmp_path / "srv.db"))
    app = create_app(store=store, api_keys=None, viewer_password=None,
                     db_path_for_insights=str(tmp_path / "srv.db"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def app_full_auth(tmp_path, monkeypatch):
    # Deterministic secret so cookies verify.
    monkeypatch.setenv("GPUPROF_SECRET", "test-secret")
    store = ServerStore(str(tmp_path / "srv.db"))
    app = create_app(store=store, api_keys=["WKEY"],
                     viewer_password="vpass",
                     db_path_for_insights=str(tmp_path / "srv.db"))
    with TestClient(app) as c:
        yield c


# ---- unauthed / write path --------------------------------------------

def test_whoami_no_auth(app_no_auth):
    r = app_no_auth.get("/api/whoami").json()
    assert r == {"auth": "none"}


def test_ingest_open_without_api_key(app_no_auth):
    r = app_no_auth.post("/api/runs", json={"name": "t", "gpu_name": "H100"})
    assert r.status_code == 200
    run_id = r.json()["id"]
    r2 = app_no_auth.post(f"/api/runs/{run_id}/ingest", json={
        "samples": [], "steps": [], "traces": [],
    })
    assert r2.status_code == 200


# ---- write auth --------------------------------------------------------

def test_ingest_requires_api_key_when_configured(app_full_auth):
    r = app_full_auth.post("/api/runs", json={"name": "x"})
    assert r.status_code == 403


def test_ingest_accepts_correct_key(app_full_auth):
    r = app_full_auth.post("/api/runs", json={"name": "x"},
                            headers={"X-API-Key": "WKEY"})
    assert r.status_code == 200


def test_ingest_multiple_keys(tmp_path):
    store = ServerStore(str(tmp_path / "srv.db"))
    app = create_app(store=store, api_keys=["k1", "k2"], viewer_password=None)
    with TestClient(app) as c:
        for k in ("k1", "k2"):
            r = c.post("/api/runs", json={"name": "x"},
                       headers={"X-API-Key": k})
            assert r.status_code == 200
        r = c.post("/api/runs", json={"name": "x"},
                   headers={"X-API-Key": "wrong"})
        assert r.status_code == 403


# ---- viewer auth -------------------------------------------------------

def test_reads_gated_when_viewer_pw_set(app_full_auth):
    assert app_full_auth.get("/api/runs").status_code == 401


def test_login_flow_and_reads_open(app_full_auth):
    r = app_full_auth.post("/api/login", json={"password": "wrong"})
    assert r.status_code == 401
    r = app_full_auth.post("/api/login", json={"password": "vpass"})
    assert r.status_code == 200
    # Cookies now present in the test client jar.
    assert app_full_auth.cookies.get("gpuprof_session")
    assert app_full_auth.cookies.get("XSRF-TOKEN")
    assert app_full_auth.get("/api/runs").status_code == 200


def test_logout_requires_csrf_token(app_full_auth):
    app_full_auth.post("/api/login", json={"password": "vpass"})
    # No CSRF header → rejected
    assert app_full_auth.post("/api/logout").status_code == 403
    xsrf = app_full_auth.cookies.get("XSRF-TOKEN")
    assert app_full_auth.post("/api/logout",
                              headers={"X-XSRF-TOKEN": xsrf}).status_code == 200


def test_login_rate_limit(app_full_auth):
    hits = 0
    for _ in range(15):
        r = app_full_auth.post("/api/login", json={"password": "wrong"})
        if r.status_code == 429:
            hits += 1
    # First 10 should be 401; anything past that is 429. Assert at least
    # one 429 fired.
    assert hits > 0


# ---- groups / distributed runs ----------------------------------------

def test_group_endpoint(app_no_auth):
    for rank in range(2):
        r = app_no_auth.post("/api/runs", json={
            "name": f"rank-{rank}", "gpu_name": "H100",
            "group_id": "trainX", "rank": rank, "world_size": 2,
        })
        assert r.status_code == 200
    r = app_no_auth.get("/api/groups")
    assert r.status_code == 200
    groups = r.json()
    assert len(groups) == 1 and groups[0]["group_id"] == "trainX"
    assert len(groups[0]["runs"]) == 2

    detail = app_no_auth.get("/api/groups/trainX").json()
    ranks = sorted(r["rank"] for r in detail["runs"])
    assert ranks == [0, 1]


def test_group_insights_route_no_skew(app_no_auth):
    # Two ranks with similar step timing → no skew.
    for rank in range(2):
        rid = app_no_auth.post("/api/runs", json={
            "name": f"r{rank}", "gpu_name": "H100",
            "group_id": "gX", "rank": rank, "world_size": 2,
        }).json()["id"]
        # Push a handful of similar-timing steps.
        steps = [{"step": i, "t_start": i*0.1, "t_end": i*0.1 + 0.09}
                 for i in range(20)]
        app_no_auth.post(f"/api/runs/{rid}/ingest",
                         json={"samples": [], "steps": steps, "traces": []})
        app_no_auth.post(f"/api/runs/{rid}/end")
    # Give the writer thread a beat.
    time.sleep(0.4)
    r = app_no_auth.get("/api/groups/gX/insights").json()
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "No rank skew" in titles


def test_group_insights_route_detects_skew(app_no_auth):
    for rank, factor in [(0, 1.0), (1, 1.6)]:  # rank 1 is 60% slower
        rid = app_no_auth.post("/api/runs", json={
            "name": f"r{rank}", "gpu_name": "H100",
            "group_id": "gS", "rank": rank, "world_size": 2,
        }).json()["id"]
        steps = [{"step": i, "t_start": i*0.1,
                  "t_end": i*0.1 + 0.09 * factor} for i in range(30)]
        app_no_auth.post(f"/api/runs/{rid}/ingest",
                         json={"samples": [], "steps": steps, "traces": []})
        app_no_auth.post(f"/api/runs/{rid}/end")
    time.sleep(0.4)
    r = app_no_auth.get("/api/groups/gS/insights").json()
    titles = " | ".join(i["title"] for i in r["insights"])
    assert "Rank skew" in titles


# ---- WebSocket auth ---------------------------------------------------

def test_ws_denied_without_cookie(app_full_auth):
    with pytest.raises(Exception):
        with app_full_auth.websocket_connect("/watch/1"):
            pass


def test_ws_ok_after_login(app_full_auth):
    app_full_auth.post("/api/login", json={"password": "vpass"})
    # Create a run first so watch has something to snapshot.
    rid = app_full_auth.post("/api/runs", json={"name": "t"},
                              headers={"X-API-Key": "WKEY"}).json()["id"]
    with app_full_auth.websocket_connect(f"/watch/{rid}") as ws:
        snap = ws.receive_json()
        assert snap["kind"] == "snapshot"
        assert "samples" in snap and "steps" in snap
