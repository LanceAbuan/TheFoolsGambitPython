"""WP-H H6 gates: the rust_workers backend (spool IPC + hot-reload + wiring).

(a) example-parity - the spool record a rust game produces decodes to the
    EXACT pipeline.py buffer.add schema the python path feeds the buffer
    (board_tensor (104,8,8) f32, policy (4672,) f32 non-negative summing to
    ~1-or-0, value in {-1,0,1}); legal-move/SAN/FEN replay is checked against
    python-chess on BOTH sides. Values MAY differ between backends (different
    rng streams) - schema is the contract.
(b) perf - scripts/bench_selfplay_backends.py (measured, not asserted here).
(c) soak - scripts/run_rust_workers.py --duration-seconds 86400 (documented
    in docs/OPTIMIZATION_PLAN.md WP-H; orchestrator-run).

Rust-backed tests skip cleanly when the wheel lacks ``selfplay_game``, the
model, or the ort dylib (same guards as tests/test_gambit_core_selfplay.py).
"""

import io
import json
import os
import pathlib
import sys
import time

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gambit_core  # noqa: E402

from training.pipeline import NIGHT_V2_PROFILE, TrainingProfile  # noqa: E402
from training.rust_workers import (  # noqa: E402
    ModelPointerError,
    ModelWatcher,
    RecordError,
    collect_from_spool,
    export_model_to_onnx,
    game_record_from_rust,
    read_game_record,
    read_model_pointer,
    record_to_buffer_examples,
    record_to_selfplay_result,
    write_game_record,
    write_model_pointer,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gambitnet-v2.onnx"


def _fake_game_dict(ply_count: int = 2, *, weight_version: int = 5, policy_hot: int = 77):
    """A gambit_core.selfplay_game-shaped dict with deterministic content."""
    rng = np.random.default_rng(weight_version)
    examples = []
    for i in range(ply_count):
        planes = (rng.standard_normal((104, 8, 8)) * 0.1).astype(np.float32)
        policy = np.zeros(4672, dtype=np.float32)
        policy[policy_hot] = 0.7
        policy[(policy_hot + 1) % 4672] = 0.3
        examples.append({"board_tensor": planes, "policy": policy, "value": 0.0})
    return {
        "examples": examples,
        "moves": ["e4", "e5"][:ply_count],
        "ucis": ["e2e4", "e7e5"][:ply_count],
        "fens": [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ][:ply_count],
        "opening_fen": chess.STARTING_FEN,
        "result": "1/2-1/2",
        "termination": "max-plies",
        "weight_version": weight_version,
        "ply_count": ply_count,
    }


def _fake_record(**kwargs):
    return game_record_from_rust(
        _fake_game_dict(**kwargs),
        weight_version=kwargs.get("weight_version", 5),
        worker_id="w0",
        seed=11,
        opening_seed=22,
        sims=8,
        max_plies=24,
        temperature=1.0,
        temperature_plies=10,
    )


# --- H5: profile flag --------------------------------------------------------


def test_profile_default_backend_is_python():
    """H5: the flag exists and defaults to python. (NIGHT_V2_PROFILE was
    flipped to rust_workers by the 2026-11-24 user-approved flip, PR #152 -
    the generic default stays python.)"""
    profile = TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model)
    assert profile.selfplay_backend == "python"
    assert profile.spool_dir is None
    assert NIGHT_V2_PROFILE.selfplay_backend == "rust_workers"


def test_profile_rejects_unknown_backend():
    with pytest.raises(ValueError, match="selfplay_backend"):
        TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model, selfplay_backend="rust")
    with pytest.raises(ValueError, match="selfplay_backend"):
        TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model, selfplay_backend="rust_workers ")
    accepted = TrainingProfile(name="t", model=NIGHT_V2_PROFILE.model, selfplay_backend="rust_workers")
    assert accepted.selfplay_backend == "rust_workers"


# --- spool record schema (versioned) -----------------------------------------


def test_record_roundtrip_is_bit_exact(tmp_path):
    record = _fake_record()
    path = write_game_record(tmp_path, record)
    assert path.parent.name == "inbox" and path.suffix == ".jsonl"
    decoded = read_game_record(path)
    assert decoded["schema"] == "gambit.selfplay.v2"
    assert decoded["weight_version"] == 5
    assert decoded["worker_id"] == "w0"
    assert decoded["seed"] == 11 and decoded["opening_seed"] == 22
    original = _fake_game_dict()
    for ex_decoded, ex_orig in zip(decoded["examples"], original["examples"]):
        assert ex_decoded["board_tensor"].dtype == np.float32
        assert ex_decoded["board_tensor"].shape == (104, 8, 8)
        assert np.array_equal(ex_decoded["board_tensor"], ex_orig["board_tensor"])
        assert np.array_equal(ex_decoded["policy"], ex_orig["policy"])
        assert ex_decoded["value"] == ex_orig["value"]
    assert decoded["moves"] == original["moves"]
    assert decoded["ucis"] == original["ucis"]
    assert decoded["fens"] == original["fens"]


def test_record_validation_rejects_bad_payloads(tmp_path):
    good = _fake_record()
    # wrong schema id
    bad = dict(good, schema="gambit.selfplay.v99")
    with pytest.raises(RecordError, match="unsupported schema"):
        read_game_record(_write_raw(tmp_path, bad))
    # board tensor truncated
    bad = json.loads(json.dumps(good))
    bad["examples"][0]["board_tensor_b64"] = bad["examples"][0]["board_tensor_b64"][:-8]
    with pytest.raises(RecordError, match="expected"):
        read_game_record(_write_raw(tmp_path, bad))
    # soft-Z value targets are legal: any finite float in [-1, 1]
    soft = json.loads(json.dumps(good))
    soft["examples"][0]["value"] = 0.5
    assert read_game_record(_write_raw(tmp_path, soft))["examples"][0]["value"] == 0.5
    # ...but out-of-range / non-finite still refuses
    bad = json.loads(json.dumps(good))
    bad["examples"][0]["value"] = 1.5
    with pytest.raises(RecordError, match="value"):
        read_game_record(_write_raw(tmp_path, bad))
    bad = json.loads(json.dumps(good))
    bad["examples"][0]["value"] = float("nan")
    with pytest.raises(RecordError, match="value"):
        read_game_record(_write_raw(tmp_path, bad))
    # ply_count/list disagreement
    bad = json.loads(json.dumps(good))
    bad["ucis"] = bad["ucis"][:-1]
    with pytest.raises(RecordError, match="ucis"):
        read_game_record(_write_raw(tmp_path, bad))
    # unnormalized policy
    bad = json.loads(json.dumps(good))
    bad["examples"][0]["policy_b64"] = __import__("base64").b64encode(
        (np.ones(4672, dtype="<f4") * 0.5).tobytes()
    ).decode()
    with pytest.raises(RecordError, match="sums to"):
        read_game_record(_write_raw(tmp_path, bad))
    # torn json (a non-atomic writer would produce this)
    torn = tmp_path / "inbox" / "game-torn.jsonl"
    torn.parent.mkdir(exist_ok=True)
    torn.write_text(json.dumps(good)[:-40], encoding="utf-8")
    with pytest.raises(RecordError, match="invalid JSON"):
        read_game_record(torn)


def _write_raw(tmp_path, payload) -> pathlib.Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir(exist_ok=True)
    path = inbox / "game-bad.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- trainer-side spool drain (claim / quarantine / timeout) ------------------


def test_collect_prefers_newest_and_skips_stale(tmp_path):
    """Freshness: newest games claimed first; games below the staleness horizon
    are skipped unread (not deleted - deletion is a disk-pressure mechanism)."""
    write_game_record(tmp_path, _fake_record(weight_version=2))  # stale
    time.sleep(0.02)
    write_game_record(tmp_path, _fake_record(weight_version=9))  # fresh, second
    time.sleep(0.02)
    write_game_record(tmp_path, _fake_record(weight_version=10))  # newest

    games = collect_from_spool(
        tmp_path, want=1, timeout_s=2.0, poll_s=0.05, min_weight_version=8
    )
    assert [g["weight_version"] for g in games] == [10], "newest claimed first"
    # nothing was destroyed; the version-9 game waits for the next drain
    remaining = sorted(p.name for p in (tmp_path / "inbox").glob("game-*.jsonl"))
    versions_left = [int(n.split("-")[2]) for n in remaining]
    assert 2 in versions_left, "stale game is skipped, not silently deleted"
    assert 9 in versions_left


def test_purge_stale_games_only_under_pressure(tmp_path):
    """Disk high-water: stale files survive a normal inbox, and only stale
    files are dropped once the inbox exceeds the water mark."""
    from training.rust_workers import purge_stale_games

    for version in (1, 2, 3):
        write_game_record(tmp_path, _fake_record(weight_version=version))

    assert purge_stale_games(tmp_path, min_weight_version=10, high_water=100) == 0
    assert len(list((tmp_path / "inbox").glob("game-*.jsonl"))) == 3

    assert purge_stale_games(tmp_path, min_weight_version=3, high_water=0) == 2
    left = sorted(int(p.stem.split("-")[2]) for p in (tmp_path / "inbox").glob("game-*.jsonl"))
    assert left == [3], "only below-horizon games removed"


def test_collect_claims_and_quarantines(tmp_path):
    write_game_record(tmp_path, _fake_record(weight_version=1))
    time.sleep(0.02)  # distinct mtimes -> deterministic drain order
    write_game_record(tmp_path, _fake_record(weight_version=2))
    (tmp_path / "inbox" / "game-bad.jsonl").write_text("{not json", encoding="utf-8")

    games = collect_from_spool(tmp_path, want=2, timeout_s=2.0, poll_s=0.05)
    assert len(games) == 2
    assert {g["weight_version"] for g in games} == {1, 2}
    # the two good games are claimed; the malformed file is left for the
    # next drain pass (quarantine is lazy - the drain stops at `want`)
    assert not list((tmp_path / "inbox").glob("game-w0-*.jsonl")), "good games claimed"
    processed = list((tmp_path / "inbox" / "processed").glob("game-*.jsonl"))
    assert len(processed) == 2
    # a second drain quarantines the bad file and returns nothing
    assert collect_from_spool(tmp_path, want=1, timeout_s=0.2, poll_s=0.05) == []
    rejected = list((tmp_path / "inbox" / "rejected").glob("game-*.jsonl"))
    assert len(rejected) == 1


def test_collect_timeout_returns_shortfall(tmp_path):
    write_game_record(tmp_path, _fake_record())
    games = collect_from_spool(tmp_path, want=4, timeout_s=0.3, poll_s=0.05)
    assert len(games) == 1, "returns what arrived before the deadline"


def test_buffer_examples_match_python_schema(tmp_path):
    """The drained payload is the pipeline buffer.add contract, key-for-key."""
    record = read_game_record(write_game_record(tmp_path, _fake_record()))
    examples = record_to_buffer_examples(record)
    assert all(set(ex) == {"board_tensor", "policy", "value"} for ex in examples)
    assert all(ex["board_tensor"].shape == (104, 8, 8) for ex in examples)
    assert all(ex["policy"].shape == (4672,) for ex in examples)


def test_record_to_selfplay_result_pgn_replays(tmp_path):
    record = read_game_record(write_game_record(tmp_path, _fake_record()))
    result = record_to_selfplay_result(record)
    assert result.result == "1/2-1/2"
    assert result.ply_count == 2
    assert result.moves == ["e4", "e5"]
    # the synthesized PGN replays the recorded SANs from the opening FEN
    game = chess.pgn.read_game(io.StringIO(result.pgn))
    assert game is not None
    replayed = list(game.mainline_moves())
    assert len(replayed) == 2
    board = chess.Board()
    for san, move in zip(result.moves, replayed):
        assert board.parse_san(san) == move
        board.push(move)

# --- model pointer store + hot-reload watcher --------------------------------


def test_pointer_store_roundtrip(tmp_path):
    model_file = tmp_path / "model-1.onnx"
    model_file.write_bytes(b"stub-onnx-bytes")
    pointer = write_model_pointer(tmp_path, model_file, weight_version=1)
    assert pointer.weight_version == 1
    assert pointer.path == model_file.resolve()
    assert len(pointer.sha256) == 64
    loaded = read_model_pointer(tmp_path)
    assert loaded == pointer
    assert (tmp_path / "latest.json").is_file()


def test_pointer_requires_existing_model(tmp_path):
    with pytest.raises(ModelPointerError, match="missing"):
        write_model_pointer(tmp_path, tmp_path / "nope.onnx", weight_version=1)


def test_watcher_hot_reloads_between_games(tmp_path):
    """Pointer mtime+size fingerprint -> adopt v1, then v2; cache between."""
    watcher = ModelWatcher(tmp_path)
    assert watcher.current() is None, "no pointer published yet"

    v1 = tmp_path / "model-1.onnx"
    v1.write_bytes(b"weights-one")
    write_model_pointer(tmp_path, v1, weight_version=1)
    first = watcher.current()
    assert first is not None and first.weight_version == 1
    assert watcher.current() is first, "unchanged pointer fingerprint serves the cache"

    v2 = tmp_path / "model-2.onnx"
    v2.write_bytes(b"weights-two-longer")
    write_model_pointer(tmp_path, v2, weight_version=2)
    second = watcher.current()
    assert second.weight_version == 2
    assert second.path.name == "model-2.onnx"


def test_watcher_rejects_hash_mismatch(tmp_path):
    """A model file that does not match the pointer hash is rejected loudly."""
    model_file = tmp_path / "model-1.onnx"
    model_file.write_bytes(b"weights")
    write_model_pointer(tmp_path, model_file, weight_version=1)
    watcher = ModelWatcher(tmp_path)
    assert watcher.current().weight_version == 1
    # The trainer never creates this state (the model file is replaced
    # atomically BEFORE the pointer publishes), but a corrupted/torn file
    # under a published pointer must fail the worker, not poison a game.
    model_file.write_bytes(b"tampered")
    with pytest.raises(ModelPointerError, match="hash mismatch"):
        ModelWatcher(tmp_path).current()


def test_export_model_to_onnx_publishes_pointer(tmp_path):
    """The checkpoint-hook export: file + pointer land atomically."""
    torch = pytest.importorskip("torch")
    from training.model import GambitNet

    model = GambitNet(NIGHT_V2_PROFILE.model)
    model.train()
    pointer = export_model_to_onnx(model, tmp_path, weight_version=3)
    assert pointer.weight_version == 3
    assert pointer.path.is_file() and pointer.path.name == "model-3.onnx"
    assert not list(tmp_path.glob(".tmp-*")), "no temp artifacts left behind"
    loaded = read_model_pointer(tmp_path)
    assert loaded.sha256 == pointer.sha256
    # model restored to its prior training state
    assert model.training is True
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(pointer.path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": torch.zeros(2, 104, 8, 8).numpy()})
    assert out[0].shape == (2, 4672)


# --- end-to-end trainer wiring (fake spool, real pipeline) --------------------


def test_run_training_rust_backend_consumes_spool(tmp_path):
    """H4/H5 wiring: profile flag -> spool drain -> buffer -> train -> export."""
    torch = pytest.importorskip("torch")
    from training.model import ModelConfig
    from training.pipeline import run_training

    profile = TrainingProfile(
        name="rust-smoke",
        model=ModelConfig(
            in_channels=104, num_blocks=1, filters=8, value_channels=4, value_fc=16, value_head="wdl"
        ),
        regime="scratch",
        iterations=1,
        games_per_iteration=1,
        max_plies=4,
        sims=2,
        train_steps=1,
        batch_size=4,
        min_examples=2,
        buffer_size=100,
        seed=0,
        snapshot_every_iteration=True,
        selfplay_backend="rust_workers",
        spool_dir=str(tmp_path / "spool"),
        spool_timeout_s=5.0,  # keep the wiring test fast on a bad spool
    )
    write_game_record(tmp_path / "spool", _fake_record(weight_version=9))

    summary = run_training(profile, run_id="rust-wiring", base_dir=tmp_path, data_dir=tmp_path)
    assert summary.games_played == 1, "the spooled game entered the pipeline"
    assert summary.steps >= 1, "the drained examples trained"
    # the checkpoint-hook export published a pointer inside the run lineage
    onnx_dir = tmp_path / "scratch" / "rust-wiring" / "onnx"
    pointer = read_model_pointer(onnx_dir)
    assert pointer is not None, "per-iteration ONNX export ran"
    assert pointer.weight_version == summary.steps
    assert pointer.path.is_file()
    spool_inbox = tmp_path / "spool" / "inbox"
    assert not list(spool_inbox.glob("game-*.jsonl")), "game claimed from the inbox"
    assert len(list((spool_inbox / "processed").glob("game-*.jsonl"))) == 1
    assert not list((spool_inbox / "rejected").glob("game-*.jsonl"))
    # and the exported ONNX actually loads
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(pointer.path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": torch.zeros(1, 104, 8, 8).numpy()})
    assert out[0].shape == (1, 4672)


# --- rust-gated: real selfplay_game through the spool + H6(a) parity ---------


def _find_dylib():
    env = os.environ.get("ORT_DYLIB_PATH")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    for hit in sorted((ROOT / "target").glob("**/onnxruntime.dll")):
        return hit
    return None


@pytest.fixture(scope="module")
def selfplay_env():
    if not hasattr(gambit_core, "selfplay_game"):
        pytest.skip("wheel built without --features onnx (no selfplay_game)")
    if not MODEL_PATH.is_file():
        pytest.skip("models/gambitnet-v2.onnx missing (run scripts/export_onnx.py)")
    dylib = _find_dylib()
    if dylib is None:
        pytest.skip("onnxruntime dylib not found under target/ (set ORT_DYLIB_PATH)")
    old = os.environ.get("ORT_DYLIB_PATH")
    os.environ["ORT_DYLIB_PATH"] = str(dylib)
    yield dylib
    if old is None:
        os.environ.pop("ORT_DYLIB_PATH", None)
    else:
        os.environ["ORT_DYLIB_PATH"] = old


RUST_GAME_KWARGS = dict(
    seed=11,
    sims=4,
    max_plies=6,
    temperature=1.0,
    temperature_plies=6,
    opening_seed=4242,
    weight_version=42,
)


def test_rust_game_spool_roundtrip_bit_exact(selfplay_env, tmp_path):
    """A REAL gambit_core.selfplay_game survives the spool bit-exactly."""
    game = gambit_core.selfplay_game(str(MODEL_PATH), **RUST_GAME_KWARGS)
    record = game_record_from_rust(
        game,
        weight_version=42,
        worker_id="w9",
        seed=RUST_GAME_KWARGS["seed"],
        opening_seed=RUST_GAME_KWARGS["opening_seed"],
        sims=RUST_GAME_KWARGS["sims"],
        max_plies=RUST_GAME_KWARGS["max_plies"],
        temperature=RUST_GAME_KWARGS["temperature"],
        temperature_plies=RUST_GAME_KWARGS["temperature_plies"],
    )
    decoded = read_game_record(write_game_record(tmp_path, record))
    assert decoded["weight_version"] == 42
    assert decoded["opening_fen"] == game["opening_fen"]
    assert decoded["moves"] == game["moves"]
    assert decoded["ucis"] == game["ucis"]
    assert decoded["fens"] == game["fens"]
    assert decoded["result"] == game["result"]
    assert decoded["termination"] == game["termination"]
    for ex_decoded, ex_game in zip(decoded["examples"], game["examples"]):
        assert np.array_equal(ex_decoded["board_tensor"], np.asarray(ex_game["board_tensor"]))
        assert np.array_equal(ex_decoded["policy"], np.asarray(ex_game["policy"]))
        assert ex_decoded["value"] == float(ex_game["value"])


def _replay_check(opening_fen, ucis, sans, fens, examples):
    """python-chess oracle: legality, SAN, FEN, label schema/ranges."""
    board = chess.Board(opening_fen)
    for i, uci in enumerate(ucis):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"illegal move {uci} at ply {i}"
        assert sans[i] == board.san(move), f"SAN mismatch at ply {i}"
        ex = examples[i]
        bt = np.asarray(ex["board_tensor"])
        pol = np.asarray(ex["policy"])
        assert bt.shape == (104, 8, 8) and bt.dtype == np.float32
        assert pol.shape == (4672,) and pol.dtype == np.float32
        assert np.isfinite(bt).all() and np.isfinite(pol).all()
        assert float(pol.min()) >= 0.0
        total = float(pol.sum())
        assert total == pytest.approx(1.0, abs=1e-3) or total == 0.0
        assert ex["value"] in (-1.0, 0.0, 1.0)
        board.push(move)
        assert fens[i] == board.fen(), f"FEN mismatch at ply {i}"


def test_example_parity_rust_vs_python_same_seed(selfplay_env, tmp_path):
    """H6(a): rust spool record vs python self_play_game - schema-identical."""
    from training.selfplay import SelfPlayConfig, SearchConfig, self_play_game

    # python side (stub evaluator: schema is the contract, not trajectories)
    class _StubEvaluator:
        def evaluate(self, boards):
            n = len(boards)
            return np.zeros((n, 4672), dtype=np.float32), np.zeros(n, dtype=np.float32)

    import numpy as _np

    py = self_play_game(
        _StubEvaluator(),
        config=SelfPlayConfig(
            search=SearchConfig(num_simulations=4, search_backend="python"),
            max_plies=6,
            temperature_plies=6,
            encoder_version=2,
        ),
        rng=_np.random.default_rng(11),
    )
    assert py.ply_count == len(py.examples) > 0
    py_examples = [
        {"board_tensor": ex.planes, "policy": ex.policy, "value": ex.value} for ex in py.examples
    ]

    # rust side: same seed/config, through the SPOOL (what the trainer eats)
    game = gambit_core.selfplay_game(str(MODEL_PATH), **RUST_GAME_KWARGS)
    record = game_record_from_rust(
        game,
        weight_version=42,
        worker_id="w0",
        seed=RUST_GAME_KWARGS["seed"],
        opening_seed=RUST_GAME_KWARGS["opening_seed"],
        sims=RUST_GAME_KWARGS["sims"],
        max_plies=RUST_GAME_KWARGS["max_plies"],
        temperature=RUST_GAME_KWARGS["temperature"],
        temperature_plies=RUST_GAME_KWARGS["temperature_plies"],
    )
    decoded = read_game_record(write_game_record(tmp_path, record))
    rust_examples = record_to_buffer_examples(decoded)

    # identical record schema on both sides
    assert set(rust_examples[0]) == set(py_examples[0]) == {"board_tensor", "policy", "value"}
    assert len(rust_examples) == len(py_examples)
    for examples in (rust_examples, py_examples):
        for ex in examples:
            bt, pol = np.asarray(ex["board_tensor"]), np.asarray(ex["policy"])
            assert bt.shape == (104, 8, 8) and bt.dtype.kind == "f"
            assert pol.shape == (4672,) and pol.dtype.kind == "f"
            assert float(pol.min()) >= 0.0
            total = float(pol.sum())
            assert total == pytest.approx(1.0, abs=1e-3) or total == 0.0
            assert float(ex["value"]) in (-1.0, 0.0, 1.0)
    # legality/SAN/FEN replay on the RUST record (python-chess oracle)
    _replay_check(
        decoded["opening_fen"], decoded["ucis"], decoded["moves"], decoded["fens"], rust_examples
    )
    # the python record replays too (start-position game, SAN-driven)
    board = chess.Board()
    for i, san in enumerate(py.moves):
        move = board.parse_san(san)
        ex = py_examples[i]
        bt, pol = np.asarray(ex["board_tensor"]), np.asarray(ex["policy"])
        assert bt.shape == (104, 8, 8) and bt.dtype.kind == "f"
        assert pol.shape == (4672,) and pol.dtype.kind == "f"
        assert float(ex["value"]) in (-1.0, 0.0, 1.0)
        board.push(move)
