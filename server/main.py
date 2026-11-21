"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .api import games, play, train
from .config import ALLOWED_ORIGINS
from .ws import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="The Fool's Gambit", version="2.0.0")
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
