"""WS training journal (FS-6/FS-6b): backlog on subscribe, 2s live tail, live/game_end
frame forwarding, stale-replay guard, and the ~10 frames/sec live-frame cap.
Audit F-3: O(window) backlog reads and rotation-tolerant tailing."""
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
        # Additive `replayed: true` marks subscribe-replayed backlog lines.
        assert frames[3]["payload"] == {
            "level": "warn", "message": "iter 2 slow", "replayed": True,
        }
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


# ---------------------------------------------------------------------------
# Audit F-3: O(window) backlog reads + rotation-tolerant tail.
# ---------------------------------------------------------------------------

def test_backlog_window_is_last_50_raw_lines_then_filter(tmp_path: Path):
    """Pinned semantics: the window is the last 50 RAW lines (position frames
    consume window slots, then drop from the output) — not the last 50 logs."""
    path = tmp_path / "train_log.jsonl"
    lines = [json.dumps({"message": f"log-{i}"}) for i in range(10)]
    lines += [_live_line(1, ply, float(ply)) for ply in range(1, 41)]  # 40 live lines
    lines.append(json.dumps({"message": "tail-log"}))
    _write_journal(path, lines)
    backlog = journal_mod.backlog(path=path)
    assert [f["message"] for f in backlog] == [f"log-{i}" for i in range(1, 10)] + ["tail-log"]


def test_tail_start_offsets_are_exact(tmp_path: Path):
    """_tail_start bounds the last-N-lines byte span for newline-terminated and
    unterminated files alike; 0 when the file holds fewer than N lines."""
    path = tmp_path / "train_log.jsonl"
    lines = [f"line-{i:04d}".encode() for i in range(300)]  # 9 chars each
    path.write_bytes(b"".join(line + b"\n" for line in lines))  # LF-terminated: 10 B/line
    current = path.stat().st_size
    assert current == 300 * 10
    # 51 newlines counted from EOF on a terminated file bounds the last 50 lines
    # (the trailing newline consumes one count): start of line 250 (0-based).
    assert journal_mod._tail_start(path, 51, current) == 250 * 10
    # more newlines requested than exist -> the whole file
    assert journal_mod._tail_start(path, 301, current) == 0

    path.write_bytes(b"\n".join(lines))  # no trailing newline: all 51 lines bound
    assert journal_mod._tail_start(path, 51, path.stat().st_size) == 249 * 10


def test_tail_start_scans_backward_across_long_lines(tmp_path: Path):
    """Lines larger than TAIL_CHUNK: the backward walk crosses many blocks and
    still yields exact last-50 semantics (a fixed byte window would truncate)."""
    path = tmp_path / "train_log.jsonl"
    big = "x" * (journal_mod.TAIL_CHUNK + 256)
    _write_journal(path, [json.dumps({"message": f"big-{i:02d}", "pad": big}) for i in range(60)])
    backlog = journal_mod.backlog(path=path)
    assert len(backlog) == 50
    assert backlog[0]["message"] == "big-10"  # oldest of the last 50
    assert backlog[-1]["message"] == "big-59"


def test_backlog_never_reads_whole_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The F-3 regression gate: a 50k-line journal serves the exact last-50
    backlog with Path.read_text poisoned — only the O(window) seek path may
    answer a subscribe (the defect was read_text().splitlines() per subscribe)."""
    path = tmp_path / "train_log.jsonl"
    _write_journal(path, [json.dumps({"message": f"line-{i}"}) for i in range(50_000)])

    def _poison(_self: Path, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("backlog() performed a full-file read_text")

    monkeypatch.setattr(Path, "read_text", _poison)
    backlog = journal_mod.backlog(path=path)
    assert len(backlog) == 50
    assert backlog[0]["message"] == "line-49950"
    assert backlog[-1]["message"] == "line-49999"


def test_subscribe_backlog_on_huge_journal(client: TestClient, tmp_path: Path):
    """Functional O(1) tail: subscribing with 50k lines on disk still streams
    exactly the last-50-window backlog, oldest first, and nothing older."""
    path = tmp_path / "train_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for i in range(50_000):
            handle.write(json.dumps({"message": f"bulk-{i}"}) + "\n")
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        first = ws.receive_json()
        assert first["type"] == "metrics"
        frames = [ws.receive_json() for _ in range(50)]
        assert all(f["type"] == "log" for f in frames)
        assert frames[0]["payload"]["message"] == "bulk-49950"
        assert frames[-1]["payload"]["message"] == "bulk-49999"


def test_seek_to_now_reconciles_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """After a replace-with-larger-file rotation, seek_to_now anchors at the NEW
    file's EOF (never a stale offset into its middle) and the tail resumes clean."""
    path = tmp_path / "train_log.jsonl"
    monkeypatch.setattr(config, "TRAIN_LOG_PATH", path)
    _write_journal(path, [json.dumps({"message": f"old-{i:02d}"}) for i in range(20)])
    journal_mod.reset_cursor()
    journal_mod.delta_from_cursor()  # cursor mechanics are pure file logic (no hub needed)
    assert journal_mod._cursor == path.stat().st_size

    path.unlink()
    _write_journal(path, [json.dumps({"message": f"new-{i:03d}"}) for i in range(80)])  # larger file
    journal_mod.seek_to_now()
    assert journal_mod._cursor == path.stat().st_size  # re-anchored, not max()'d into the middle
    assert journal_mod.delta_from_cursor() == []  # nothing stale to push


def test_truncated_journal_restarts_tail_mid_subscribe(client: TestClient, tmp_path: Path):
    """Truncate-in-place rotation: the tail resets to the top of the (now
    shorter) file and fresh lines flow; no crash, no stale-cursor stall."""
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["payload"]["message"] == "history"  # seed backlog

        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"message": "pre-rotate"}\n')
        pre = [f["payload"].get("message") for f in _drain(out) if f["type"] == "log"]
        assert "pre-rotate" in pre  # cursor is now past the pre-rotate content

        path.write_text('{"message": "fresh-0"}\n{"message": "fresh-1"}\n', encoding="utf-8")
        fresh = [f["payload"].get("message") for f in _drain(out) if f["type"] == "log"]
        assert "fresh-0" in fresh and "fresh-1" in fresh

        # the tail keeps flowing after the rotation event
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"message": "post-rotate"}\n')
        post = [f["payload"].get("message") for f in _drain(out) if f["type"] == "log"]
        assert "post-rotate" in post


def test_replaced_journal_larger_than_cursor_restarts_at_top(client: TestClient, tmp_path: Path):
    """Delete-and-recreate rotation where the NEW file already outgrew the old
    cursor: only a file-identity check sees it — a size-only tail would seek
    mid-file and stream a garbage slice, never the new file's head."""
    path = tmp_path / "train_log.jsonl"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"seq": 1, "type": "subscribe", "topic": "train"})
        out = _collector(ws)
        assert out.get(timeout=5)["type"] == "metrics"
        assert out.get(timeout=5)["payload"]["message"] == "history"

        with path.open("a", encoding="utf-8") as handle:  # advance the cursor ~450 bytes in
            for i in range(20):
                handle.write(json.dumps({"message": f"pre-{i:02d}"}) + "\n")
        pre = [f["payload"].get("message") for f in _drain(out) if f["type"] == "log"]
        assert "pre-19" in pre  # cursor is now past the pre-rotation content

        path.unlink()  # rotate: replace with a file LARGER than the old cursor
        _write_journal(path, [json.dumps({"message": f"rotated-{i:03d}"}) for i in range(60)])

        rotated = [f["payload"].get("message") for f in _drain(out, quiet=0.8) if f["type"] == "log"]
        assert "rotated-000" in rotated  # restarted at the new file's TOP
        assert "rotated-059" in rotated
