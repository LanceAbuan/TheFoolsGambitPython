"""Production hardening (FS-9): TTL reaper, static mount of web/dist, POST rate limit."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, reaper
from server.main import create_app

REAP_SCAN = 0.05  # test scan interval (real: 30s)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "hardening.db")
    monkeypatch.setattr(reaper, "SCAN_INTERVAL_SECONDS", REAP_SCAN)
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def test_reaper_auto_resigns_idle_game(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """A game idle past FSG_GAME_TTL_MIN is finished + persisted + broadcast."""
    monkeypatch.setattr(config, "GAME_TTL_MIN", 0.01)  # minutes -> 0.6s TTL
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    # Backdate past the TTL (real games age by wall clock; tests skip the wait).
    session = engine.get_session(gid)
    session.last_activity = time.time() - 60.0

    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = client.get(f"/api/play/{gid}").json()
        if snap["status"] == "finished":
            break
        time.sleep(REAP_SCAN)
    snap = client.get(f"/api/play/{gid}").json()
    assert snap["status"] == "finished"
    assert snap["winner"] == "black"  # the human (White) abandoned -> loss
    games = client.get("/api/games", params={"kind": "play"}).json()["games"]
    row = next(g for g in games if g["id"] == gid)
    assert row["status"] == "finished"  # persisted
    assert engine.get_session(gid).reason == "abandoned"


def test_reaper_leaves_active_games_alone(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "GAME_TTL_MIN", 30.0)  # default TTL: nothing is expired
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    assert reaper.expired_sessions() == []
    time.sleep(REAP_SCAN * 4)  # a few reaper ticks
    assert client.get(f"/api/play/{gid}").json()["status"] == "active"


def test_expired_sessions_filters_by_status_and_ttl(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """Only active sessions past the TTL are reaper candidates."""
    monkeypatch.setattr(config, "GAME_TTL_MIN", 30.0)
    gid = client.post("/api/play/new", json={"color": "white", "opponent": "fool"}).json()["game_id"]
    session = engine.get_session(gid)
    assert reaper.expired_sessions() == []  # fresh game is well inside the TTL
    session.last_activity = time.time() - 10_000.0
    assert [s.id for s in reaper.expired_sessions()] == [gid]
    session.status = "finished"
    assert reaper.expired_sessions() == []  # finished games are never re-touched


def test_static_mount_serves_built_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>FOOL-INDEX</body></html>", encoding="utf-8")
    (dist / "app.js").write_text("console.log('fool')", encoding="utf-8")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "static.db")
    monkeypatch.setattr(config, "WEB_DIST", dist)
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "FOOL-INDEX" in index.text
        assert client.get("/app.js").status_code == 200
        assert client.get("/api/health").json() == {"ok": True}  # API still wins
        # SPA fallback: client-side routes resolve to index.html on hard navigation
        deep = client.get("/play")
        assert deep.status_code == 200
        assert "FOOL-INDEX" in deep.text
        # unknown /api paths 404 as JSON, never masquerade as the SPA
        assert client.get("/api/definitely-not-here").status_code == 404


def test_no_static_mount_when_dist_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "nostatic.db")
    monkeypatch.setattr(config, "WEB_DIST", tmp_path / "missing-dist")
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as client:
        assert client.get("/api/health").json() == {"ok": True}
        assert client.get("/").status_code == 404  # nothing mounted at /


def test_post_rate_limit_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ratelimit.db")
    monkeypatch.setattr(config, "WEB_DIST", tmp_path / "missing-dist")
    monkeypatch.setattr(config, "RATE_LIMIT_CAPACITY", 2)
    monkeypatch.setattr(config, "RATE_LIMIT_REFILL", 0.0001)  # effectively no refill
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as client:
        assert client.post("/api/play/new", json={"color": "white"}).status_code == 200
        assert client.post("/api/play/new", json={"color": "white"}).status_code == 200
        limited = client.post("/api/play/new", json={"color": "white"})
        assert limited.status_code == 429
        assert limited.json()["detail"] == "rate limit exceeded"
        for _ in range(5):  # GETs are never limited
            assert client.get("/api/health").status_code == 200


def test_rate_limit_refills_over_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ratelimit2.db")
    monkeypatch.setattr(config, "WEB_DIST", tmp_path / "missing-dist")
    monkeypatch.setattr(config, "RATE_LIMIT_CAPACITY", 1)
    monkeypatch.setattr(config, "RATE_LIMIT_REFILL", 1000.0)  # full refill ~instantly
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as client:
        for _ in range(5):  # the bucket keeps refilling to capacity between posts
            r = client.post("/api/play/new", json={"color": "white"})
            assert r.status_code == 200
            time.sleep(0.01)
