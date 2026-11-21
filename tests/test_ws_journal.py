"""WS training journal (FS-6/FS-6b): backlog on subscribe, 2s live tail, live/game_end
frame forwarding, stale-replay guard, and the ~10 frames/sec live-frame cap."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, telemetry
from server import journal as journal_mod
from server.main import create_app

INTERVAL = 0.25  # test journal interval (real: 2s); live budget = int(10 * 0.25) = 2/flush
SEED_LINE = '{"message": "history"}'


def _write_journal(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """App with a journal seeded pre-boot (so the tail starts at its EOF)."""
    journal_path = tmp_path / "train_log.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(SEED_LINE + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "TRAIN_LOG_PATH", journal_path)
    monkeypatch.setattr(telemetry, "JOURNAL_INTERVAL_SECONDS", INTERVAL)
    db.reset_for_tests()
    engine.reset_for_tests()
    with TestClient(create_app()) as c:
        yield c


def _recv_until(ws: Any, predicate: Any, timeout: float = 3.0) -> dict[str, Any]:
    """Read frames (reader thread) until one satisfies `predicate`; raises on timeout."""
    received: dict[str, Any] = {}

    def _pump() -> None:
        while True:
            frame = ws.receive_json()
            received["last"] = frame
            if predicate(frame):
                return

    watcher = threading.Thread(target=_pump, daemon=True)
    watcher.start()
    watcher.join(timeout)
    assert not watcher.is_alive(), "timed out waiting for a matching frame"
    return received["last"]


def test_parse_line_maps_to_log_payload():
    """Journal lines map onto the pinned {level, message} log payload."""
    assert journal_mod.parse_line('{"level": "warn", "message": "gpu hot"}') == {
        "level": "warn",
        "message": "gpu hot",
    }
    # structured event without a message field -> compact JSON passthrough
    parsed = journal_mod.parse_line('{"event": "ckpt", "tag": "a"}')
    assert parsed is not None
    assert parsed["level"] == "info"
    assert "ckpt" in parsed["message"]
    # non-JSON text passes through verbatim
    assert journal_mod.parse_line("plain text line") == {"level": "info", "message": "plain text line"}
    assert journal_mod.parse_line("   ") is None
    assert journal_mod.parse_line(b'{"level": "error", "message": "boom"}')["level"] == "error"


def test_backlog_caps_at_50_oldest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "train_log.jsonl"
    _write_journal(path, [json.dumps({"message": f"line-{i}"}) for i in range(60)])
    backlog = journal_mod.backlog(path=path)
    assert len(backlog) == 50
    assert backlog[0]["message"] == "line-10"  # oldest of the last 50
    assert backlog[-1]["message"] == "line-59"
    assert journal_mod.backlog(path=tmp_path / "missing.jsonl") == []


def test_subscribe_sends_backlog_after_metrics(client: TestClient, tmp_path: Path):
    path = tmp_path / "train_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:  # appended pre-subscribe -> in backlog
        handle.write('{"level": "info", "message": "iter 1 done"}\n')
        handle.write("garbage line kept as info\n")
        handle.write('{"level": "warn", "message": "iter 2 slow"}\n')
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        first = ws.receive_json()
        assert first["type"] == "metrics"
        frames = [ws.receive_json() for _ in range(4)]  # seed + 3 appended, oldest first
        assert [f["type"] for f in frames] == ["log", "log", "log", "log"]
        assert [f["payload"]["message"] for f in frames] == [
            "history",
            "iter 1 done",
            "garbage line kept as info",
            "iter 2 slow",
        ]
        assert frames[2]["payload"]["level"] == "info"
        assert frames[3]["payload"] == {"level": "warn", "message": "iter 2 slow"}
        seqs = [first["seq"], *(f["seq"] for f in frames)]
        assert seqs == sorted(seqs)  # one monotonic stream: snapshot + backlog share it


def test_journal_pushes_new_lines_while_subscribed(client: TestClient, tmp_path: Path):
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        assert ws.receive_json()["type"] == "metrics"  # snapshot
        assert ws.receive_json()["payload"]["message"] == "history"  # backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"level": "info", "message": "live iter 3"}) + "\n")

        push = _recv_until(ws, lambda f: f.get("payload", {}).get("message") == "live iter 3")
        assert push["type"] == "log"
        assert push["payload"] == {"level": "info", "message": "live iter 3"}


def test_journal_idles_without_subscribers(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, str]] = []
    original = telemetry.hub.send

    async def _spy(topic: str, mtype: str, payload: dict[str, Any]) -> None:
        calls.append((topic, mtype))
        await original(topic, mtype, payload)

    monkeypatch.setattr(telemetry.hub, "send", _spy)
    with (tmp_path / "train_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"message": "unwatched"}\n')
    time.sleep(INTERVAL * 4)  # several ticks with zero train subscribers
    assert calls == []  # idle guard: no file-driven broadcasts


# ---------------------------------------------------------------------------
# FS-6b: live/game_end frame routing, stale-replay guard, rate cap.
# ---------------------------------------------------------------------------

def _live_line(game: int, ply: int, ts: float, fen: str | None = None, run_id: str = "run-1") -> str:
    return json.dumps(
        {
            "kind": "live",
            "game": game,
            "ply": ply,
            "move": f"{ply:04d}",
            "fen": fen or f"fen-g{game}-p{ply}",
            "run_id": run_id,
            "ts": ts,
        }
    )


def _collector(ws: Any) -> queue.Queue[dict[str, Any]]:
    """Background reader pumping every incoming frame into a queue."""
    out: queue.Queue[dict[str, Any]] = queue.Queue()

    def _pump() -> None:
        while True:
            try:
                out.put(ws.receive_json())
            except Exception:  # noqa: BLE001 — socket closed at context exit
                return

    threading.Thread(target=_pump, daemon=True).start()
    return out


def _drain(source: queue.Queue[dict[str, Any]], quiet: float = 0.6, first: float = 5.0) -> list[dict[str, Any]]:
    """Collect frames until ``quiet`` seconds pass with nothing new."""
    frames = [source.get(timeout=first)]
    while True:
        try:
            frames.append(source.get(timeout=quiet))
        except queue.Empty:
            return frames


def _subscribe(client: TestClient) -> tuple[Any, queue.Queue[dict[str, Any]]]:
    """Open a train-topic WS, consume metrics + backlog, return (ws, collector)."""
    ws = client.websocket_connect("/ws")
    ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
    out = _collector(ws)
    first = out.get(timeout=5)
    assert first["type"] == "metrics"  # subscribe-time snapshot
    return ws, out


def test_classify_routes_live_game_end_and_log():
    """Routing: live/game_end get dedicated frames; everything else stays a log frame."""
    routed = journal_mod.classify_line(
        '{"kind": "live", "game": 1, "ply": 5, "move": "e2e4", "fen": "F", "run_id": "r", "ts": 100.5}'
    )
    assert routed == (
        "live",
        {"game": 1, "ply": 5, "move": "e2e4", "fen": "F", "run_id": "r", "ts": 100.5},
    )
    routed = journal_mod.classify_line('{"kind": "game_end", "game": 2, "result": "1-0", "plies": 84, "ts": 7.0}')
    assert routed == ("game_end", {"game": 2, "result": "1-0", "plies": 84, "ts": 7.0})
    # alias-tolerant: other field names + string numerics still route
    routed = journal_mod.classify_line(
        '{"kind": "Live", "game_id": "3", "move_number": "7", "uci": "d2d4", "position": "P", "run": "r2", "timestamp": 9}'
    )
    assert routed == (
        "live",
        {"game": 3, "ply": 7, "move": "d2d4", "fen": "P", "run_id": "r2", "ts": 9.0},
    )
    # live line without a routable game degrades to a log frame (never dropped)
    fallback = journal_mod.classify_line('{"kind": "live", "ply": 3, "fen": "X"}')
    assert fallback is not None
    assert fallback[0] == "log"
    assert "live" in fallback[1]["message"]
    # unknown kind -> log frame (compact JSON), nothing crashes
    unknown = journal_mod.classify_line('{"kind": "iteration", "loss": 0.5}')
    assert unknown is not None
    assert unknown[0] == "log"
    assert "iteration" in unknown[1]["message"]
    assert journal_mod.classify_line("   ") is None


def test_throttle_coalesces_per_game_and_caps():
    """Newest fen wins per game; oldest live frames drop when over budget; ends pass."""
    events: list[tuple[str, dict[str, Any]]] = [("log", {"level": "info", "message": "iter"})]
    for game in (1, 2, 3):
        for ply in (1, 2):
            events.append(
                ("live", {"game": game, "ply": ply, "move": "m", "fen": "f", "run_id": "r",
                          "ts": float(ply * 10 + game)})
            )
    events.append(("game_end", {"game": 1, "result": "1-0", "plies": 40, "ts": 99.0}))
    out = telemetry.throttle_frames(events, interval=0.2)  # budget = int(10 * 0.2) = 2
    assert [(t, p.get("game"), p.get("ply")) for t, p in out] == [
        ("log", None, None),
        ("live", 2, 2),   # coalesced newest per game, ts-ascending, oldest dropped
        ("live", 3, 2),
        ("game_end", 1, None),  # ends always last and never dropped
    ]


def test_throttle_never_drops_game_ends():
    """Even an absurd game_end burst passes untouched (terminal events are exempt)."""
    events = [
        ("game_end", {"game": g, "result": "1-0", "plies": 10, "ts": float(g)}) for g in range(30)
    ]
    out = telemetry.throttle_frames(events, interval=0.05)  # budget = max(1, 0) = 1
    assert len(out) == 30


def test_live_frames_forwarded_with_schema(client: TestClient, tmp_path: Path):
    """Appended live lines reach the client as typed frames with the documented payload."""
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["payload"]["message"] == "history"  # backlog seed

        batch = (
            _live_line(1, 5, 100.5, fen="FEN-A")
            + "\n"
            + _live_line(2, 3, 100.6, fen="FEN-B")
            + "\n"
            + json.dumps({"kind": "iteration", "loss": 0.25, "ts": 100.7})
            + "\n"
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(batch)

        frames = _drain(out)
        by_type: dict[str, list[dict[str, Any]]] = {}
        for frame in frames:
            assert frame["topic"] == "train"
            by_type.setdefault(frame["type"], []).append(frame["payload"])
        assert by_type["log"] == [
            {"level": "info", "message": json.dumps({"kind": "iteration", "loss": 0.25, "ts": 100.7}, sort_keys=True)}
        ]
        assert by_type["live"] == [  # exact documented schema, ts-ascending
            {"game": 1, "ply": 5, "move": "0005", "fen": "FEN-A", "run_id": "run-1", "ts": 100.5},
            {"game": 2, "ply": 3, "move": "0003", "fen": "FEN-B", "run_id": "run-1", "ts": 100.6},
        ]
        assert "game_end" not in by_type


def test_game_end_frame_forwarded(client: TestClient, tmp_path: Path):
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["type"] == "log"  # backlog seed

        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"kind": "game_end", "game": 7, "result": "0-1", "plies": 84, "ts": 200.25}) + "\n"
            )

        frames = _drain(out)
        ends = [f for f in frames if f["type"] == "game_end"]
        assert len(ends) == 1
        assert ends[0]["topic"] == "train"
        assert ends[0]["payload"] == {"game": 7, "result": "0-1", "plies": 84, "ts": 200.25}


def test_backlog_excludes_live_and_game_end(client: TestClient, tmp_path: Path):
    """Position frames never replay on subscribe; live flows from subscription time on."""
    path = tmp_path / "train_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:  # pre-subscribe: stale positions
        handle.write(_live_line(1, 1, 1.0, fen="OLD") + "\n")
        handle.write(json.dumps({"kind": "game_end", "game": 1, "result": "1-0", "plies": 30, "ts": 2.0}) + "\n")
        handle.write('{"message": "iter 9 done"}\n')
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        backlog_frames = [out.get(timeout=5), out.get(timeout=5)]
        assert [f["type"] for f in backlog_frames] == ["log", "log"]  # live/game_end excluded
        assert [f["payload"]["message"] for f in backlog_frames] == ["history", "iter 9 done"]

        # ...but live frames flow from subscription time onward
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_live_line(1, 2, 3.0, fen="NEW") + "\n")
        frame = out.get(timeout=5)
        assert frame["type"] == "live"
        assert frame["payload"]["fen"] == "NEW"
        assert not any(f["type"] == "live" and f["payload"]["fen"] == "OLD" for f in backlog_frames)


def test_live_lines_appended_before_subscribe_never_replay(client: TestClient, tmp_path: Path):
    """The seek-to-now guard: idle-accumulated live lines are skipped at subscribe."""
    path = tmp_path / "train_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:  # nobody subscribed yet
        for ply in range(1, 6):
            handle.write(_live_line(1, ply, float(ply), fen=f"STALE-{ply}") + "\n")
    time.sleep(INTERVAL * 2)  # let idle ticks pass (they must not queue anything)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["payload"]["message"] == "history"  # only the seed backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write(_live_line(1, 6, 6.0, fen="FRESH") + "\n")
        frames = _drain(out)
        lives = [f for f in frames if f["type"] == "live"]
        assert len(lives) == 1
        assert lives[0]["payload"]["fen"] == "FRESH"  # no STALE-* positions arrived


def test_unknown_kind_line_does_not_crash_tail(client: TestClient, tmp_path: Path):
    """Mystery kinds flow as log frames; the tail keeps working afterwards."""
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["type"] == "log"  # seed backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "mystery-event", "x": 1}\n')
            handle.write("not even json\n")
            handle.write(_live_line(4, 2, 10.0) + "\n")  # tail still routes live after oddities

        frames = _drain(out)
        logs = [f["payload"]["message"] for f in frames if f["type"] == "log"]
        assert any("mystery-event" in message for message in logs)
        assert any("not even json" in message for message in logs)
        lives = [f for f in frames if f["type"] == "live"]
        assert [f["payload"] for f in lives] == [
            {"game": 4, "ply": 2, "move": "0002", "fen": "fen-g4-p2", "run_id": "run-1", "ts": 10.0}
        ]


def test_single_game_burst_coalesces_to_newest_fen(client: TestClient, tmp_path: Path):
    """12 plies of one game within one tick -> one live frame: the newest fen wins."""
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["type"] == "log"  # seed backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(_live_line(1, ply, float(ply)) for ply in range(1, 13)) + "\n")

        frames = _drain(out)
        lives = [f["payload"] for f in frames if f["type"] == "live"]
        # Coalesced per flush; a mid-write read split can yield a second flush.
        assert 1 <= len(lives) <= 2
        assert [p["ply"] for p in lives] == sorted(p["ply"] for p in lives)
        assert lives[-1]["ply"] == 12  # newest fen wins: intermediate plies dropped (correct)
        assert lives[-1]["fen"] == "fen-g1-p12"


def test_multi_game_burst_capped_to_budget(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A burst over the per-flush budget keeps only the most recent position(s)."""
    # Interval 0.05 -> budget = max(1, int(10 * 0.05)) = 1 live frame per flush.
    monkeypatch.setattr(telemetry, "JOURNAL_INTERVAL_SECONDS", 0.05)
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["type"] == "log"  # seed backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(_live_line(game, 1, float(game)) for game in range(1, 13)) + "\n")

        frames = _drain(out, quiet=0.8)  # > 10 flush intervals: the cap is the steady state
        lives = [f["payload"] for f in frames if f["type"] == "live"]
        # Budget 1 per flush; a mid-write read split can yield a second flush.
        assert 1 <= len(lives) <= 2
        games = [p["game"] for p in lives]
        assert games == sorted(games)  # each flush keeps its max-ts game
        assert games[-1] == 12  # most recent position wins
        assert lives[-1]["ts"] == 12.0
