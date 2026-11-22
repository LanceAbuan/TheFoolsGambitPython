"""Hub fan-out isolation (WS deaf-window defect, WP-slice): per-subscriber queues.

Regression tests for the reproduced production defect: the landed hub awaited
``ws.send_json`` per subscriber inside ``hub.send``, so ONE consumer that
stopped reading (frozen tab / slow monitor — anything that still answers
protocol pings, which uvicorn's ping-timeout never cleans up) suspended the
whole fan-out task: every train subscriber starved until the wedged socket
died. The harness measured an 11 fps → 1.8 fps degradation followed by total
starvation on the landed code; the per-subscriber-queue hub keeps everyone
else at full rate and drops the wedged consumer instead.

These tests exercise :class:`server.ws.Hub` mechanics directly (no sockets):
a stuck ``send_json`` stands in for transport backpressure.
"""
from __future__ import annotations

import asyncio

import pytest

from server import ws as ws_mod
from server.ws import Hub


class _StuckSocket:
    """Consumer that never drains: its send never completes (zero window)."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.close_calls: list[int] = []

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)  # first frame may land; then the transport "fills"
        await asyncio.Event().wait()  # never set: the wedged-send stand-in

    async def close(self, code: int = 1000) -> None:
        self.close_calls.append(code)


class _WedgedCloseSocket(_StuckSocket):
    """Overflowed consumer whose transport also wedges on the close handshake."""

    async def close(self, code: int = 1000) -> None:
        self.close_calls.append(code)
        await asyncio.Event().wait()  # never set: the wedged-close stand-in


class _EagerSocket:
    """Healthy consumer: send completes immediately."""

    def __init__(self) -> None:
        self.received: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.received.append(message)


class _ExplodingSocket:
    """Consumer whose transport is dead: every send raises (disconnect)."""

    async def send_json(self, message: dict) -> None:
        raise ConnectionResetError("transport closed")


async def _cleanup(hub: Hub) -> None:
    for topic in list(hub._subs):
        for sub in list(hub._subs[topic].values()):
            await hub.unsubscribe(topic, sub.ws)


def test_wedged_subscriber_does_not_block_fanout():
    """THE defect: one wedged consumer must not delay frames to the others."""

    async def scenario() -> None:
        hub = Hub()
        stuck, eager = _StuckSocket(), _EagerSocket()
        await hub.subscribe("train", stuck)
        await hub.subscribe("train", eager)
        try:
            # Landed code: this await suspended inside the wedged socket's
            # send until that connection died — minutes of topic-wide silence.
            await asyncio.wait_for(hub.send("train", "log", {"message": "one"}), timeout=2.0)
            await asyncio.wait_for(hub.send("train", "log", {"message": "two"}), timeout=2.0)
            await asyncio.sleep(0.05)  # let the eager pump deliver
            assert [m["payload"]["message"] for m in eager.received] == ["one", "two"]
            assert hub.subscriber_count("train") == 2  # wedged one is still *registered*
        finally:
            await _cleanup(hub)

    asyncio.run(scenario())


def test_queue_overflow_drops_only_the_slow_subscriber(monkeypatch: pytest.MonkeyPatch):
    """A consumer behind by more than SUBSCRIBER_QUEUE_LIMIT frames is dropped,
    its socket is closed with 1013 ("Try Again Later") so the client's onclose
    reconnect fires, and the rest of the topic keeps flowing at full rate."""

    async def scenario() -> None:
        hub = Hub()
        stuck, eager = _StuckSocket(), _EagerSocket()
        await hub.subscribe("train", stuck)
        await hub.subscribe("train", eager)
        try:
            for i in range(ws_mod.SUBSCRIBER_QUEUE_LIMIT + 10):
                await hub.send("train", "log", {"message": f"f{i}"})
            await asyncio.sleep(0.1)  # _drop task + eager pump need a few ticks
            assert hub.subscriber_count("train") == 1, "wedged subscriber not dropped"
            # The drop closes the transport: without this the endpoint would
            # keep reading on a permanently deaf socket (no client watchdog).
            assert stuck.close_calls == [ws_mod.OVERFLOW_CLOSE_CODE], "dropped socket left open"
            assert ws_mod.OVERFLOW_CLOSE_CODE == 1013
            # The healthy subscriber never missed a frame.
            assert len(eager.received) == ws_mod.SUBSCRIBER_QUEUE_LIMIT + 10
            # And the topic still flows afterwards.
            await hub.send("train", "log", {"message": "after"})
            await asyncio.sleep(0.05)
            assert eager.received[-1]["payload"]["message"] == "after"
        finally:
            await _cleanup(hub)

    asyncio.run(scenario())


def test_overflow_drop_close_timeout_does_not_leak_task(monkeypatch: pytest.MonkeyPatch):
    """A wedged close handshake is bounded: the drop task finishes (no leak)
    and the subscription is still removed."""

    async def scenario() -> None:
        monkeypatch.setattr(ws_mod, "OVERFLOW_CLOSE_TIMEOUT_SECONDS", 0.05)
        hub = Hub()
        stuck, eager = _WedgedCloseSocket(), _EagerSocket()
        await hub.subscribe("train", stuck)
        await hub.subscribe("train", eager)
        try:
            for i in range(ws_mod.SUBSCRIBER_QUEUE_LIMIT + 10):
                await hub.send("train", "log", {"message": f"f{i}"})
            await asyncio.sleep(0.2)  # 0.05s close timeout + task teardown
            assert hub.subscriber_count("train") == 1
            assert stuck.close_calls == [ws_mod.OVERFLOW_CLOSE_CODE]
            # The timed-out close must not leave its task behind.
            assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("hub-drop-train")]
            assert len(eager.received) == ws_mod.SUBSCRIBER_QUEUE_LIMIT + 10
        finally:
            await _cleanup(hub)

    asyncio.run(scenario())


def test_seq_monotonic_across_snapshots_and_fanout():
    """Subscribe-time snapshots (send_to) and fan-outs share one monotonic
    per-topic counter, and per-connection order is preserved."""

    async def scenario() -> None:
        hub = Hub()
        eager = _EagerSocket()
        await hub.subscribe("train", eager)
        try:
            await hub.send_to(eager, "train", "metrics", {"elo": 1.0})
            await hub.send("train", "log", {"message": "live"})
            await hub.send_to(eager, "train", "log", {"message": "backlog", "replayed": True})
            await asyncio.sleep(0.05)
            types = [m["type"] for m in eager.received]
            assert types == ["metrics", "log", "log"]
            seqs = [m["seq"] for m in eager.received]
            assert seqs == sorted(seqs) and len(set(seqs)) == 3
            # send_to to an unregistered socket is a no-op, not an error.
            await hub.send_to(_EagerSocket(), "train", "metrics", {"elo": 2.0})
        finally:
            await _cleanup(hub)

    asyncio.run(scenario())


def test_send_error_drops_subscriber_without_breaking_fanout():
    """A dying transport (disconnect mid-flight) is unsubscribed by its pump;
    hub.send itself never raises."""

    async def scenario() -> None:
        hub = Hub()
        dead, eager = _ExplodingSocket(), _EagerSocket()
        await hub.subscribe("train", dead)
        await hub.subscribe("train", eager)
        try:
            await hub.send("train", "log", {"message": "boom"})
            await asyncio.sleep(0.1)  # pump error path + unsubscribe
            assert hub.subscriber_count("train") == 1, "dead subscriber not dropped"
            assert [m["payload"]["message"] for m in eager.received] == ["boom"]
        finally:
            await _cleanup(hub)

    asyncio.run(scenario())


def test_unsubscribe_cancels_the_pump():
    """Disconnecting leaves no pump tasks behind (no leaked producers)."""

    async def scenario() -> None:
        hub = Hub()
        eager = _EagerSocket()
        await hub.subscribe("train", eager)
        sub = next(iter(hub._subs["train"].values()))
        task = sub.task
        assert task is not None
        await hub.unsubscribe("train", eager)
        await asyncio.sleep(0.05)
        assert task.cancelled() or task.done()
        assert hub.subscriber_count("train") == 0

    asyncio.run(scenario())


def test_idle_topic_fanout_is_cheap():
    """No subscribers → no message build; the seq still allocates (landed
    semantics: the counter is topic-global, so monotonicity holds across
    empty stretches). The telemetry idle path skips hub.send entirely."""

    async def scenario() -> None:
        hub = Hub()
        await hub.send("train", "metrics", {"elo": 1.0})
        assert hub.subscriber_count("train") == 0
        assert hub._seq["train"] == 1  # allocated, just unconsumed
        eager = _EagerSocket()
        await hub.subscribe("train", eager)
        await hub.send("train", "metrics", {"elo": 2.0})
        await asyncio.sleep(0.05)
        assert eager.received[0]["seq"] == 2  # stream stays monotonic

    asyncio.run(scenario())
