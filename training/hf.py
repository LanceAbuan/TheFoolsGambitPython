"""Hugging Face Hub publishing, per regime (FS-5, plan §5.2 provenance).

Each regime owns its own repo (`RegimeConfig.hf_repo`, e.g.
``LanceAbuan/gambit-scratch``) so every public model carries its regime
lineage. Auth comes from ``HF_TOKEN``; with no token the upload is a graceful
no-op (logged, returns False) so local runs never block on the Hub.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "checkpoint.pt"
METADATA_FILENAME = "metadata.json"


def resolve_hf_token() -> str:
    """``HF_TOKEN`` env value ('' when unset — callers treat that as no-op)."""
    return os.environ.get("HF_TOKEN", "").strip()


def upload_checkpoint(
    repo_id: str,
    checkpoint_path: Path | str,
    metadata: dict[str, Any] | None = None,
    token: str | None = None,
) -> bool:
    """Upload one checkpoint (+ optional metadata.json) to `repo_id`.

    Returns True on success, False on no-token or any Hub failure — training
    must never crash because a push failed.
    """
    resolved_token = token if token is not None else resolve_hf_token()
    if not resolved_token:
        log.info("[HF] HF_TOKEN unset — skipping upload to %s (local-only run)", repo_id)
        return False
    try:
        from huggingface_hub import HfApi, upload_file  # injected/mocked in tests
    except ImportError:
        log.warning("[HF] huggingface_hub not installed — skipping upload to %s", repo_id)
        return False

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        log.warning("[HF] checkpoint %s missing — skipping upload to %s", checkpoint_path, repo_id)
        return False
    try:
        api = HfApi(token=resolved_token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception:
            log.debug("[HF] create_repo(%s) failed or exists", repo_id, exc_info=True)
        upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=CHECKPOINT_FILENAME,
            repo_id=repo_id,
            token=resolved_token,
        )
        if metadata is not None:
            import json
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                meta_path = Path(tmp) / METADATA_FILENAME
                meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
                upload_file(
                    path_or_fileobj=str(meta_path),
                    path_in_repo=METADATA_FILENAME,
                    repo_id=repo_id,
                    token=resolved_token,
                )
        log.info("[HF] pushed %s -> %s", checkpoint_path.name, repo_id)
        return True
    except Exception as exc:  # noqa: BLE001 - HF push must never crash training
        log.error("[HF] push to %s failed: %s", repo_id, exc)
        return False
