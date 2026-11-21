"""Games archive: listing + PGN export — §14.0 pinned contract."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .. import db, engine

router = APIRouter()


@router.get("/games")
def list_games(kind: str | None = None, limit: int = 50) -> dict:
    return {"games": db.list_games(kind, max(1, min(limit, 200)))}


@router.get("/games/{gid}/pgn", response_class=PlainTextResponse)
def game_pgn(gid: str) -> str:
    try:
        return engine.get_session(gid).pgn
    except engine.NotFound:
        pass
    replay = db.moves_for(gid)
    if replay is None:
        raise HTTPException(status_code=404, detail=f"game {gid} not found")
    sans, orientation = replay
    return engine.build_pgn(sans, orientation)
