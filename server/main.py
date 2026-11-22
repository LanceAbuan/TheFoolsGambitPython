"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import backfill, config, db, engine, model_server, reaper, telemetry
from .api import analyze, games, play, train
from .config import ALLOWED_ORIGINS
from .ratelimit import TokenBucketRateLimit
from .ws import router as ws_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Migrate + backfill the DB, rehydrate active game sessions, run the
    background feeds + periodic ledger backfill + TTL reaper; clean up on
    shutdown."""
    await asyncio.to_thread(db.migrate)
    backfilled = await asyncio.to_thread(db.backfill_arena)
    if backfilled:
        log.info("backfilled %d arena ledger row(s) at startup", backfilled)
    # Keep the mirror current while up: the trainer appends ledger rows
    # continuously and a startup-only import left SQLite (and anything reading
    # it) stale until the next restart. Idempotent tick (see server.backfill).
    backfill.start()
    # Restart resilience: rebuild in-progress games from SQLite before serving,
    # so GET/WS/moves on a pre-restart game and the lobby's active list survive
    # deploys. Runs before the reaper starts so restored sessions are visible
    # to its first scan.
    await asyncio.to_thread(engine.rehydrate_active)
    telemetry.start()
    reaper.start()
    warmup = asyncio.create_task(model_server.warmup_async(), name="champion-warmup")
    try:
        yield
    finally:
        warmup.cancel()
        with suppress(asyncio.CancelledError):
            await warmup
        await backfill.stop()
        await telemetry.stop()
        await reaper.stop()
        stopped = await engine.shutdown_engines()
        if stopped:
            log.info("stopped %d stockfish process(es) on shutdown", stopped)
        if await analyze.shutdown():
            log.info("stopped analysis stockfish on shutdown")
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
    app.include_router(analyze.router, prefix="/api")
    app.include_router(ws_router)

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    # FS-9: single-origin serving of the built web client when it exists (CORS
    # above stays for the Vite dev server). Mounted last so /api + /ws win.
    if config.WEB_DIST.is_dir():
        if (config.WEB_DIST / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=str(config.WEB_DIST / "assets")), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str) -> FileResponse:
            """Serve real files (favicon etc.); everything else is the SPA's index.html
            so client-side routes like /play, /watch resolve on hard navigation.
            Unknown API/WS paths 404 instead of masquerading as the SPA."""
            if full_path == "api" or full_path.startswith(("api/", "ws", "ws/")):
                raise HTTPException(status_code=404, detail="Not Found")
            candidate = (config.WEB_DIST / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(config.WEB_DIST.resolve()):
                return FileResponse(candidate)
            return FileResponse(config.WEB_DIST / "index.html")

        log.info("serving web client from %s", config.WEB_DIST)

    return app


app = create_app()
