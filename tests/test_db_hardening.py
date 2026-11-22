"""DB connection hardening: busy_timeout + synchronous=NORMAL (WAL pairing).

The server's pooled connection is guarded by an in-process lock, but other
processes (training worker, arena jobs) open the same SQLite file. Without a
busy timeout, a write landing while another process holds the write lock fails
immediately with "database is locked"; with busy_timeout=5000 it waits its
turn instead.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from server import config, db


@pytest.fixture()
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dedicated temp FSG_DB with the pooled connection dropped first."""
    path = tmp_path / "hardened.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    db.reset_for_tests()
    return path


def test_connect_sets_busy_timeout_and_synchronous_normal(db_path: Path) -> None:
    """The hardened pragmas are present on the pooled connection (and WAL stays on)."""
    conn = db.connect()
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # 1 == NORMAL
    assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


def test_busy_timeout_waits_out_a_concurrent_writer(db_path: Path) -> None:
    """A second connection holding the write lock must not error our writes.

    The holder grabs the write lock (BEGIN IMMEDIATE) and releases it ~0.3s
    later, from its own thread. The pooled connection's write must block for
    that window and then succeed — proving the busy timeout is doing the
    waiting that a default connection would spend raising "database is locked".
    """
    db.connect()  # pooled connection, hardened pragmas

    other: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)  # a second "process"
    other.execute("BEGIN IMMEDIATE")
    other.execute(
        "INSERT INTO games (id, kind, created_at, updated_at, orientation, status)"
        " VALUES ('other', 'play', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'white', 'active')"
    )

    def _release() -> None:
        time.sleep(0.3)
        other.commit()  # finishing the transaction releases the write lock
        other.close()

    worker = threading.Thread(target=_release, name="lock-holder")
    worker.start()

    start = time.monotonic()
    db.tx(
        lambda conn: conn.execute(
            "INSERT INTO games (id, kind, created_at, updated_at, orientation, status)"
            " VALUES ('ours', 'play', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'black', 'active')"
        )
    )
    waited = time.monotonic() - start
    worker.join(2.0)

    assert waited >= 0.25  # actually waited for the lock instead of failing fast
    rows: list[dict[str, Any]] = [
        dict(r) for r in db.tx(lambda conn: conn.execute("SELECT id FROM games ORDER BY id").fetchall())
    ]
    assert [row["id"] for row in rows] == ["other", "ours"]  # both writes landed
