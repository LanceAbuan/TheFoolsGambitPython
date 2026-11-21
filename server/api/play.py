"""Play API: create games, move, undo, resign — §14.0 pinned contract."""
from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .. import db, engine
from ..engine import Session
from ..ws import hub, lobby_state

router = APIRouter()

Strength = Literal["wandering", "curious", "enlightened"]


class NewGameIn(BaseModel):
    color: Literal["white", "black", "random"] = "white"
    opponent: Literal["fool", "stockfish", "nn"] = "fool"
    strength: Strength = "curious"


class MoveIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_sq: str = Field(alias="from", pattern=r"^[a-h][1-8]$")
    to_sq: str = Field(alias="to", pattern=r"^[a-h][1-8]$")
    promotion: Literal["q", "r", "b", "n"] | None = None


def _load(gid: str) -> Session:
    try:
        return engine.get_session(gid)
    except engine.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


async def _persist(session: Session) -> None:
    await asyncio.to_thread(db.persist_session, session)


@router.post("/play/new")
async def new_game(body: NewGameIn) -> dict:
    try:
        session = engine.create_game(body.color, body.opponent, body.strength)
        await engine.opening_reply(session)
    except engine.GameError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    await _persist(session)
    snap = session.snapshot()
    await hub.send(f"game:{session.id}", "state", snap)
    await hub.send("lobby", "lobby", {"active_games": lobby_state()})
    return {"game_id": session.id, "fen": snap["fen"], "orientation": session.orientation}


@router.get("/play/{gid}")
async def game_state(gid: str) -> dict:
    return _load(gid).snapshot()


@router.post("/play/{gid}/move")
async def move(gid: str, body: MoveIn) -> dict:
    session = _load(gid)
    try:
        await engine.player_move(session, body.from_sq, body.to_sq, body.promotion)
    except engine.GameError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    await _persist(session)
    snap = session.snapshot()
    await hub.send(
        f"game:{gid}",
        "position",
        {
            "fen": snap["fen"],
            "last_move": snap["last_move"],
            "move_number": (len(snap["moves"]) + 1) // 2,
        },
    )
    if session.status == "finished":
        await hub.send(f"game:{gid}", "state", snap)
    return snap


@router.post("/play/{gid}/undo")
async def undo(gid: str) -> dict:
    session = _load(gid)
    try:
        await engine.undo(session)
    except engine.GameError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    await _persist(session)
    snap = session.snapshot()
    await hub.send(f"game:{gid}", "state", snap)
    return snap


@router.post("/play/{gid}/resign")
async def resign(gid: str) -> dict:
    session = _load(gid)
    try:
        await engine.resign(session)
    except engine.GameError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
    await _persist(session)
    snap = session.snapshot()
    await hub.send(f"game:{gid}", "state", snap)
    await hub.send("lobby", "lobby", {"active_games": lobby_state()})
    return snap
