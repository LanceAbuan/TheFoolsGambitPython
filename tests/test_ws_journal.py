"""WS training journal (FS-6): backlog on subscribe + 2s live tail, idle when unwatched."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server import config, db, engine, telemetry
from server import journal as journal_mod
from server.main import create_app

INTERVAL = 0.1  # test journal interval (real: 2s)
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
