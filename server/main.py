"""FastAPI application factory."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db, engine, telemetry
from .api import games, play, train
from .config import ALLOWED_ORIGINS
from .ws import router as ws_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Run the WS telemetry feed; stop it (and every UCI process) on shutdown."""
    telemetry.start()
    try:
        yield
    finally:
        await telemetry.stop()
        stopped = await engine.shutdown_engines()
        if stopped:
            log.info("stopped %d stockfish process(es) on shutdown", stopped)


def create_app() -> FastAPI:
    app = FastAPI(title="The Fool's Gambit", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    db.connect()
    app.include_router(play.router, prefix="/api")
    app.include_router(games.router, prefix="/api")
    app.include_router(train.router, prefix="/api")
    app.include_router(ws_router)

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app


app = create_app()
