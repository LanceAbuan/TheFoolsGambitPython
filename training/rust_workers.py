"""WP-H H4: the Rust self-play worker backend - spool IPC + weight hot-reload.

Topology (the Lc0/KataGo shape: native engine + Python training harness):

    scripts/run_rust_workers.py          pipeline.run_training(selfplay_backend="rust_workers")
      spawns N x scripts/rust_worker.py    drains <run-root>/spool/inbox/*.jsonl
        loop:                              convert -> TrainingBuffer.add (unchanged)
          watch models/latest.json         per iteration: export model -> ONNX
          play ONE selfplay_game                            -> models/latest.json
          write spool JSONL (atomic)

The worker processes are independent Rust-backed interpreters: every per-move
step (rules, encoder v2, MCTS, ONNX inference) runs inside
``gambit_core.selfplay_game`` with the GIL released - Python appears only once
per finished game, to serialize one record to disk.

IPC contract (simplest-robust per WP-H H4): a spool directory, ONE JSONL file
per finished game, written atomically (temp file + ``os.replace``) so the
trainer never reads a partial game. The trainer claims files by renaming them
into ``inbox/processed/`` after a full read+validate; malformed records go to
``inbox/rejected/`` (never crash training; never lose bytes).

Record schema ``gambit.selfplay.v1`` (versioned - bump the id on ANY
breaking change; the reader refuses unknown ids):

    schema            "gambit.selfplay.v1"
    encoding          "f32.le.b64" - board/policy tensors are base64 of raw
                      little-endian float32 bytes (bit-exact round-trip)
    weight_version    the model version the game played against (trainer-
                      exported pointer version); EVERY game is tagged
    worker_id/seed/opening_seed/sims/max_plies/temperature/
    temperature_plies worker + game-reproduction metadata
    opening_fen/result/termination/ply_count/moves/ucis/fens
                      the H3 selfplay_game fields, verbatim
    examples[]        {board_tensor_b64, policy_b64, value} - the exact
                      pipeline.py buffer.add record payload

Weight hot-reload: the trainer exports the current net to
``<run-root>/onnx/model-<weight_version>.onnx`` and publishes
``<run-root>/onnx/latest.json`` (path + sha256 + version) atomically AFTER the
model file is in place. Workers watch the POINTER file (mtime+size), then
verify the model file's hash once against the pointer before adopting it.
Reload decisions happen BETWEEN games only - a worker resolves the pointer
once per loop iteration and hands ONE fixed path to ``selfplay_game``, which
loads its own OrtModel per call, so a game can never mix weights.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # heavy torch/chess imports stay lazy (worker CLI imports fast)
    from .selfplay import SelfPlayResult

log = logging.getLogger(__name__)

# --- schema constants -------------------------------------------------------
SPOOL_SCHEMA_ID = "gambit.selfplay.v2"  # WP2: examples carry policy_masked
SPOOL_SCHEMA_IDS = {"gambit.selfplay.v1", "gambit.selfplay.v2"}
SPOOL_ENCODING = "f32.le.b64"

# pipeline.py buffer.add record payload: (104, 8, 8) f32 planes + (4672,) f32
# policy + scalar value in {-1, 0, 1} (the H3/encoder-v2 contract).
BOARD_PLANES = 104
POLICY_MOVES = 4672

INBOX_DIRNAME = "inbox"
PROCESSED_DIRNAME = "processed"
REJECTED_DIRNAME = "rejected"

POINTER_FILENAME = "latest.json"


class RecordError(ValueError):
    """A spool record violates the gambit.selfplay.v1 schema."""


class ModelPointerError(RuntimeError):
    """The published model pointer is missing/inconsistent with its file."""


# --- tensor codecs ----------------------------------------------------------


def _b64_encode_f32(arr: np.ndarray) -> str:
    """float32 ndarray -> base64 of its raw little-endian bytes."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def _b64_decode_f32(payload: str, shape: tuple[int, ...], what: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:  # binascii.Error covers bad base64
        raise RecordError(f"{what}: invalid base64 ({exc})") from exc
    expected = int(np.prod(shape)) * 4
    if len(raw) != expected:
        raise RecordError(f"{what}: decoded {len(raw)} bytes, expected {expected} for {shape}")
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=True)


# --- record encode / decode / validate --------------------------------------


def game_record_from_rust(
    game: dict[str, Any],
    *,
    weight_version: int,
    worker_id: str,
    seed: int,
    opening_seed: int | None,
    sims: int,
    max_plies: int,
    temperature: float,
    temperature_plies: int,
) -> dict[str, Any]:
    """``gambit_core.selfplay_game`` output dict -> one spool record (schema v1).

    The game dict's examples carry numpy arrays (board_tensor (104, 8, 8) f32,
    policy (4672,) f32, value float); tensors are base64-encoded bit-exactly.
    """
    return {
        "schema": SPOOL_SCHEMA_ID,
        "encoding": SPOOL_ENCODING,
        "weight_version": int(weight_version),
        "worker_id": str(worker_id),
        "seed": int(seed),
        "opening_seed": None if opening_seed is None else int(opening_seed),
        "sims": int(sims),
        "max_plies": int(max_plies),
        "temperature": float(temperature),
        "temperature_plies": int(temperature_plies),
        "opening_fen": game["opening_fen"],
        "result": game["result"],
        "termination": game["termination"],
        "ply_count": int(game["ply_count"]),
        "moves": list(game["moves"]),
        "ucis": list(game["ucis"]),
        "fens": list(game["fens"]),
        "finished_at": time.time(),
        "examples": [
            {
                "board_tensor_b64": _b64_encode_f32(np.asarray(ex["board_tensor"])),
                "policy_b64": _b64_encode_f32(np.asarray(ex["policy"])),
                "value": float(ex["value"]),
                "policy_masked": bool(ex.get("policy_masked", False)),
                "root_value": float(ex.get("root_value", 0.0)),
            }
            for ex in game["examples"]
        ],
    }

def validate_game_record(record: dict[str, Any]) -> None:
    """Raise RecordError unless ``record`` is a well-formed schema-v1 game.

    Structural + range validation only (shapes, dtypes, value domain,
    list-length agreement): values MAY legitimately differ between backends
    (H6(a) compares schema, not trajectories).
    """
    if not isinstance(record, dict):
        raise RecordError("record must be a JSON object")
    if record.get("schema") not in SPOOL_SCHEMA_IDS:
        raise RecordError(
            f"unsupported schema {record.get('schema')!r} "
            f"(expected one of {sorted(SPOOL_SCHEMA_IDS)})"
        )
    if record.get("encoding") != SPOOL_ENCODING:
        raise RecordError(f"unsupported encoding {record.get('encoding')!r}")
    for key in ("weight_version", "seed", "sims", "max_plies", "ply_count"):
        if not isinstance(record.get(key), int):
            raise RecordError(f"{key} must be an integer, got {record.get(key)!r}")
    for key in ("worker_id", "opening_fen", "result", "termination"):
        if not isinstance(record.get(key), str) or not record.get(key):
            raise RecordError(f"{key} must be a non-empty string")
    if record["result"] not in ("1-0", "0-1", "1/2-1/2"):
        raise RecordError(f"result {record['result']!r} is not a legal game result")
    if record["termination"] not in ("game-over", "max-plies"):
        raise RecordError(f"termination {record['termination']!r} is not recognized")
    for key in ("moves", "ucis", "fens"):
        if not isinstance(record.get(key), list) or len(record[key]) != record["ply_count"]:
            raise RecordError(f"{key} must be a list of ply_count={record['ply_count']} strings")
    examples = record.get("examples")
    if not isinstance(examples, list) or len(examples) != record["ply_count"]:
        raise RecordError(f"examples must be a list of ply_count={record['ply_count']} records")
    if record["ply_count"] == 0:
        raise RecordError("game recorded zero plies")
    for i, ex in enumerate(examples):
        required = {"board_tensor_b64", "policy_b64", "value"}
        allowed = required | {"policy_masked", "root_value"}
        if not isinstance(ex, dict) or not required <= set(ex) or not set(ex) <= allowed:
            raise RecordError(
                f"examples[{i}] keys must be board_tensor_b64/policy_b64/value"
                " (+optional policy_masked)"
            )
        if "policy_masked" in ex and not isinstance(ex["policy_masked"], bool):
            raise RecordError(f"examples[{i}].policy_masked must be a bool")
        if "root_value" in ex:
            rv = ex["root_value"]
            if not isinstance(rv, (int, float)) or not (-1.0 <= float(rv) <= 1.0):
                raise RecordError(f"examples[{i}].root_value must be in [-1, 1]")
        value = ex["value"]
        # WP3 soft-Z: value targets are capped mixtures of the game result and
        # the root eval, so any finite float in [-1, 1] is valid - the old
        # hard {-1, 0, 1} check quarantined every soft-Z game (silent data loss).
        if not isinstance(value, (int, float)) or not (-1.0 <= float(value) <= 1.0):
            raise RecordError(f"examples[{i}].value must be in [-1, 1], got {value!r}")
        planes = _b64_decode_f32(
            ex["board_tensor_b64"], (BOARD_PLANES, 8, 8), f"examples[{i}].board_tensor"
        )
        policy = _b64_decode_f32(ex["policy_b64"], (POLICY_MOVES,), f"examples[{i}].policy")
        if not np.isfinite(planes).all():
            raise RecordError(f"examples[{i}].board_tensor has non-finite values")
        if not np.isfinite(policy).all() or float(policy.min()) < 0.0:
            raise RecordError(f"examples[{i}].policy must be finite and non-negative")
        total = float(policy.sum())
        if total > 0.0 and abs(total - 1.0) > 1e-3:
            raise RecordError(
                f"examples[{i}].policy sums to {total}, expected ~1 (or all zeros)"
            )


def read_game_record(path: Path) -> dict[str, Any]:
    """Read + validate one spool JSONL file; tensors decoded to numpy."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordError(f"invalid JSON in {path}: {exc}") from exc
    validate_game_record(record)
    for i, ex in enumerate(record["examples"]):
        ex["board_tensor"] = _b64_decode_f32(
            ex.pop("board_tensor_b64"), (BOARD_PLANES, 8, 8), f"examples[{i}].board_tensor"
        )
        ex["policy"] = _b64_decode_f32(
            ex.pop("policy_b64"), (POLICY_MOVES,), f"examples[{i}].policy"
        )
    return record


# --- spool directory I/O -----------------------------------------------------


def write_game_record(spool_dir: Path | str, record: dict[str, Any]) -> Path:
    """Write ONE game record as ``<spool>/inbox/<name>.jsonl`` - atomically.

    Temp file in the SAME directory (rename within one volume is atomic on
    Windows and POSIX), then ``os.replace``. Workers never truncate an inbox
    file in place, so the trainer can never observe a partial game.
    """
    inbox = Path(spool_dir) / INBOX_DIRNAME
    inbox.mkdir(parents=True, exist_ok=True)
    name = (
        f"game-{record.get('worker_id', 'w')}-{record.get('weight_version', 0)}"
        f"-{uuid.uuid4().hex}.jsonl"
    )
    final = inbox / name
    tmp = inbox / f".tmp-{uuid.uuid4().hex}"
    tmp.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp, final)
    return final


def record_to_buffer_examples(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Spool record -> the exact ``TrainingBuffer.add`` payload shape.

    Same keys the python self-play path feeds the buffer (pipeline.py):
    ``{"board_tensor": (104,8,8) f32, "policy": (4672,) f32, "value": float}``.
    """
    return [
        {"board_tensor": ex["board_tensor"], "policy": ex["policy"], "value": ex["value"]}
        for ex in record["examples"]
    ]


def record_to_selfplay_result(record: dict[str, Any], tb: Any = None) -> "SelfPlayResult":
    """Spool record -> a ``SelfPlayResult`` the pipeline consumes unchanged.

    The PGN is rendered from the recorded SANs (python-chess importer, with
    the FEN header when the game starts from a sampled opening).
    """
    import chess

    from .selfplay import SelfPlayResult, TrainingExample

    game = chess.pgn.Game()
    if record["opening_fen"] != chess.STARTING_FEN:
        game.headers["FEN"] = record["opening_fen"]
        game.headers["SetUp"] = "1"
    replay = chess.Board(record["opening_fen"])
    node = game
    tb_values: list[float | None] = []
    use_tb = tb is not None and getattr(tb, "enabled", False)
    for san in record["moves"]:
        move = replay.parse_san(san)
        node = node.add_variation(move)
        tb_values.append(tb.value_for(replay) if use_tb else None)
        replay.push(move)
    game.headers["Result"] = record["result"]
    pgn = str(
        game.accept(chess.pgn.StringExporter(headers=None, variations=False, comments=False))
    )
    return SelfPlayResult(
        # TrainingExample objects - the exact thing the python self-play
        # path returns, so pipeline.py's buffer.add loop is untouched.
        examples=[
            TrainingExample(
                planes=ex["board_tensor"],
                policy=ex["policy"],
                value=(
                    tb_values[i]
                    if i < len(tb_values) and tb_values[i] is not None
                    else ex["value"]
                ),
                policy_mask=0.0 if ex.get("policy_masked") else 1.0,
            )
            for i, ex in enumerate(record["examples"])
        ],
        moves=list(record["moves"]),
        pgn=pgn,
        result=record["result"],
        ply_count=int(record["ply_count"]),
    )


def _game_weight_version(path: Path) -> int | None:
    """Weight version parsed from ``game-<worker>-<wv>-<hash>.jsonl``."""
    parts = path.stem.split("-")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return None


def _is_below_version(path: Path, min_weight_version: int | None) -> bool:
    if min_weight_version is None:
        return False
    version = _game_weight_version(path)
    return version is not None and version < min_weight_version


def purge_stale_games(
    spool_dir: Path | str,
    min_weight_version: int,
    high_water: int = 512,
    batch: int = 256,
) -> int:
    """Delete UNCLAIMED games below ``min_weight_version``, only under pressure.

    Staleness does not need deletion to be safe any more - ``collect_from_spool``
    never reads a game below the horizon - so deletion is purely a disk
    high-water mechanism: nothing happens until the inbox exceeds ``high_water``
    files, then up to ``batch`` stale games are dropped. This keeps a fast
    trainer from shredding live fleet output (in-flight games carry the
    previous model's version tag).
    """
    inbox = Path(spool_dir) / INBOX_DIRNAME
    if not inbox.is_dir():
        return 0
    files = [p for p in inbox.glob("game-*.jsonl") if p.is_file()]
    if len(files) <= high_water:
        return 0
    purged = 0
    for path in files:
        if purged >= batch:
            break
        if _is_below_version(path, min_weight_version):
            with contextlib.suppress(OSError):
                path.unlink()
                purged += 1
    if purged:
        log.info(
            "[RUST-SPOOL] purged %d stale games (weight_version < %d)",
            purged,
            min_weight_version,
        )
    return purged


def prune_processed_games(
    spool_dir: Path | str,
    keep: int = 20_000,
    keep_hours: float = 48.0,
) -> int:
    """Bound the disk footprint of ``inbox/processed`` (retention).

    Claimed games are kept only for replay/debugging, but at GPU-speed self-play
    (~500 games/hr x ~5 MB) an unpruned archive grows ~60 GB/day. Keep the newest
    ``keep`` files OR anything younger than ``keep_hours``, whichever is more
    generous, and delete the rest. Runs are idempotent and cheap (a single
    directory scan) so the drain can call it every iteration.
    """
    processed = Path(spool_dir) / INBOX_DIRNAME / PROCESSED_DIRNAME
    if not processed.is_dir():
        return 0
    files = [(p, p.stat().st_mtime, p.stat().st_size) for p in processed.glob("game-*.jsonl")]
    if len(files) <= keep:
        return 0
    cutoff = time.time() - keep_hours * 3600.0
    # newest-first: everything beyond the ``keep`` window is a candidate, but
    # recent files are protected regardless of count.
    files.sort(key=lambda item: item[1], reverse=True)
    removed = 0
    freed = 0
    for path, mtime, size in files[keep:]:
        if mtime >= cutoff:
            break
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
            freed += size
    if removed:
        log.info(
            "[RUST-SPOOL] pruned %d processed games (%.1f GB), kept %d",
            removed,
            freed / 1e9,
            keep,
        )
    return removed


def collect_from_spool(
    spool_dir: Path | str,
    want: int,
    timeout_s: float = 600.0,
    poll_s: float = 1.0,
    min_weight_version: int | None = None,
) -> list[dict[str, Any]]:
    """Drain up to ``want`` finished games from the spool, NEWEST FIRST.

    Newest-first keeps training data fresh when workers outproduce the trainer
    (the old mtime-ascending order trained on the oldest backlog). Games whose
    weight version is below ``min_weight_version`` are skipped entirely (never
    read, never claimed) - they are too many policy versions stale to train on.
    Each claimed file is read + validated fully, THEN moved into
    ``inbox/processed/``; invalid records are quarantined into ``inbox/rejected/``.
    """
    inbox = Path(spool_dir) / INBOX_DIRNAME
    processed = inbox / PROCESSED_DIRNAME
    rejected = inbox / REJECTED_DIRNAME
    for d in (inbox, processed, rejected):
        d.mkdir(parents=True, exist_ok=True)

    games: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while len(games) < want:
        candidates = sorted(
            (
                p
                for p in inbox.glob("game-*.jsonl")
                if p.is_file() and not _is_below_version(p, min_weight_version)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            if time.monotonic() >= deadline:
                log.warning(
                    "[RUST-SPOOL] timeout after %.0fs: %d/%d games arrived",
                    timeout_s,
                    len(games),
                    want,
                )
                break
            time.sleep(poll_s)
            continue
        for path in candidates:
            if len(games) >= want:
                break
            try:
                record = read_game_record(path)
            except RecordError as exc:
                log.error("[RUST-SPOOL] rejecting %s: %s", path.name, exc)
                with contextlib.suppress(OSError):
                    os.replace(path, rejected / path.name)
                continue
            games.append(record)
            os.replace(path, processed / path.name)
    return games

# --- model pointer store (weight versioning + hot-reload) --------------------


@dataclass(frozen=True)
class ModelPointer:
    """One published worker model: file path + hash + trainer version."""

    weight_version: int
    path: Path
    sha256: str
    exported_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_version": self.weight_version,
            # File NAME only: an absolute path broke the fleet the moment the
            # repo moved drives (every worker reported "names a missing model
            # file"). Readers resolve it against the models dir.
            "path": self.path.name,
            "sha256": self.sha256,
            "exported_at": self.exported_at,
        }


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{uuid.uuid4().hex}"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_model_pointer(
    models_dir: Path | str, onnx_path: Path | str, weight_version: int
) -> ModelPointer:
    """Publish ``latest.json`` for an ALREADY-exported model file (atomic)."""
    models_dir = Path(models_dir)
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise ModelPointerError(f"cannot publish pointer: {onnx_path} is missing")
    pointer = ModelPointer(
        weight_version=int(weight_version),
        path=onnx_path.resolve(),
        sha256=_sha256_file(onnx_path),
        exported_at=time.time(),
    )
    _atomic_write_json(models_dir / POINTER_FILENAME, pointer.to_dict())
    return pointer


def read_model_pointer(models_dir: Path | str) -> ModelPointer | None:
    """Read ``latest.json``; None when no model has been published yet."""
    path = Path(models_dir) / POINTER_FILENAME
    if not path.is_file():
        return None
    try:
        # utf-8-sig: PowerShell / editors happily write a BOM, and a BOM turned
        # every worker into "pointer unusable, waiting for the trainer" (a
        # whole fleet idle on one invisible byte).
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        model_path = Path(data["path"])
        if not model_path.is_absolute():
            # Relative entries (the current format) resolve against the models
            # dir, so a repo/drive move cannot orphan the fleet.
            model_path = Path(models_dir) / model_path
        return ModelPointer(
            weight_version=int(data["weight_version"]),
            path=model_path,
            sha256=str(data["sha256"]),
            exported_at=float(data.get("exported_at", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelPointerError(f"corrupt model pointer {path}: {exc}") from exc


def export_model_to_onnx(
    model: Any,
    out_dir: Path | str,
    weight_version: int,
) -> ModelPointer:
    """Export a GambitNet to ``<out_dir>/model-<weight_version>.onnx`` (atomic).

    Writes the model to a temp file, ``os.replace``s it into place (a worker
    reading mid-export sees either the old file or the new one, never a torn
    one), then publishes the pointer. Same opset/IO contract as H1's
    ``scripts/export_onnx.py`` (opset 17, dynamic batch, input/policy/value).
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"model-{int(weight_version)}.onnx"
    tmp = out_dir / f".tmp-model-{int(weight_version)}-{uuid.uuid4().hex}.onnx"
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, int(model.config.in_channels), 8, 8)
        torch.onnx.export(
            model,
            dummy,
            str(tmp),
            input_names=["input"],
            output_names=["policy", "value"],
            dynamic_axes={
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        os.replace(tmp, final)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        if was_training:
            model.train()
    return write_model_pointer(out_dir, final, weight_version)


class ModelWatcher:
    """Worker-side hot-reload watcher: pointer mtime+size, model hash check.

    ``current()`` is called BETWEEN games only. It re-reads ``latest.json``
    when its (mtime, size) fingerprint changes, then verifies the model
    file's sha256 against the pointer BEFORE adopting the new version - a
    truncated or mismatched file is rejected loudly instead of poisoning a
    game. The adopted pointer is cached until the pointer file changes.
    """

    def __init__(self, models_dir: Path | str) -> None:
        self.models_dir = Path(models_dir)
        self._cached: ModelPointer | None = None
        self._fingerprint: tuple[float, int] | None = None

    def current(self) -> ModelPointer | None:
        pointer_path = self.models_dir / POINTER_FILENAME
        if not pointer_path.is_file():
            return None
        stat = pointer_path.stat()
        fingerprint = (stat.st_mtime, stat.st_size)
        if self._fingerprint is not None and fingerprint == self._fingerprint:
            return self._cached
        pointer = read_model_pointer(self.models_dir)
        if pointer is None:
            return None
        if not pointer.path.is_file():
            raise ModelPointerError(
                f"pointer v{pointer.weight_version} names a missing model file: {pointer.path}"
            )
        actual = _sha256_file(pointer.path)
        if actual != pointer.sha256:
            raise ModelPointerError(
                f"pointer v{pointer.weight_version} hash mismatch on {pointer.path.name}: "
                f"pointer says {pointer.sha256[:12]}..., file is {actual[:12]}..."
            )
        self._fingerprint = fingerprint
        self._cached = pointer
        return pointer