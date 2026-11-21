"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config, db, engine, model_server, reaper, telemetry
from .api import games, play, train
from .config import ALLOWED_ORIGINS
from .ratelimit import TokenBucketRateLimit
from .ws import router as ws_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Backfill the arena ledger, run the background feeds + TTL reaper; clean up on shutdown."""
    backfilled = await asyncio.to_thread(db.backfill_arena)
    if backfilled:
        log.info("backfilled %d arena ledger row(s) at startup", backfilled)
    telemetry.start()
    reaper.start()
    warmup = asyncio.create_task(model_server.warmup_async(), name="champion-warmup")
    try:
        yield
    finally:
        warmup.cancel()
        with suppress(asyncio.CancelledError):
            await warmup
        await telemetry.stop()
        await reaper.stop()
        stopped = await engine.shutdown_engines()
        if stopped:
            log.info("stopped %d stockfish process(es) on shutdown", stopped)
        model_server.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="The Fool's Gambit", version="2.0.0", lifespan=lifespan)
    # Rate limit sits inside CORS so 429s carry CORS headers for the dev origin.
    app.add_middleware(TokenBucketRateLimit)
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

    # FS-9: single-origin serving of the built web client when it exists (CORS
    # above stays for the Vite dev server). Mounted last so /api + /ws win.
    if config.WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(config.WEB_DIST), html=True), name="web")
        log.info("serving web client from %s", config.WEB_DIST)

    return app


app = create_app()
