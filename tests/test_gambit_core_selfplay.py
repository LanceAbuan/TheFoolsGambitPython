"""WP-H H3 gates: ``gambit_core.selfplay_game`` (the Rust self-play game loop).

Gate shape follows H6(a): one Rust game and one Python game under the same
config produce SCHEMA-identical example records — keys, shapes, dtypes
(float kind; the buffer casts to f32 via ``torch.FloatTensor`` so the trained
schema is f32 both ways), legality, and target ranges. NOT per-move value
parity: Dirichlet + move-sampling draws ride different rng streams (rust
StdRng vs numpy PCG64), so trajectories legitimately diverge; same-seed RUST
replays are bit-identical (asserted here).

What is compared bit-exactly where the streams allow it:
- the SAN strings vs the python-chess oracle on every recorded move
- the recorded v2 planes vs ``training.encoder.encode_state_v2`` on every
  recorded ply (proves the last-7 history truncation end-to-end)
- the FENs vs python-chess ``board.fen()`` (legal-ep semantics)
- result/termination consistency vs python-chess ``result(claim_draw=True)``

Skips cleanly when the wheel lacks ``selfplay_game`` (built without
``--features onnx``) or when the model / dylib artifacts are missing.
"""

import os
import pathlib
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gambit_core  # noqa: E402
from training.encoder import encode_state_v2  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gambitnet-v2.onnx"

GAME_KWARGS = dict(
    seed=11,
    sims=8,
    max_plies=24,
    temperature=1.0,
    temperature_plies=10,
    weight_version=7,
)


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
    # The rust ort loader resolves the dylib itself; pin it for determinism.
    # (python onnxruntime is unaffected: it ships and loads its own dll.)
    old = os.environ.get("ORT_DYLIB_PATH")
    os.environ["ORT_DYLIB_PATH"] = str(dylib)
    fastpath = os.environ.pop("GAMBIT_CORE_FASTPATH", None)
    yield dylib
    if old is None:
        os.environ.pop("ORT_DYLIB_PATH", None)
    else:
        os.environ["ORT_DYLIB_PATH"] = old
    if fastpath is not None:
        os.environ["GAMBIT_CORE_FASTPATH"] = fastpath


def _play_game(**overrides):
    kwargs = dict(GAME_KWARGS)
    kwargs.update(overrides)
    return gambit_core.selfplay_game(str(MODEL_PATH), **kwargs)


def test_selfplay_record_schema(selfplay_env):
    """buffer.add schema EXACTLY: board_tensor (104,8,8) f32, policy (4672,)
    f32, value f32 in {-1,0,1} (mover's perspective); one record per ply."""
    out = _play_game(opening_seed=4242)
    assert out["weight_version"] == 7
    assert out["termination"] in {"game-over", "max-plies"}
    assert out["result"] in {"1-0", "0-1", "1/2-1/2"}
    n = out["ply_count"]
    assert n > 0
    assert n == len(out["moves"]) == len(out["ucis"]) == len(out["fens"]) == len(out["examples"])
    for ex in out["examples"]:
        assert set(ex.keys()) == {"board_tensor", "policy", "value"}
        bt, pol = np.asarray(ex["board_tensor"]), np.asarray(ex["policy"])
        assert bt.shape == (104, 8, 8) and bt.dtype == np.float32
        assert pol.shape == (4672,) and pol.dtype == np.float32
        assert np.isfinite(bt).all() and np.isfinite(pol).all()
        assert float(pol.min()) >= 0.0
        assert abs(float(pol.sum()) - 1.0) < 1e-4, "policy = normalized visit counts"
        assert ex["value"] in (-1.0, 0.0, 1.0)

def test_selfplay_replay_sans_planes_fens(selfplay_env):
    """Full replay gate: ucis legal under python-chess, SANs bit-equal to
    board.san, example planes bit-equal to encode_state_v2 (the last-7
    truncation, end to end), fens == board.fen(), values signed by the mover,
    result/termination consistent with the final position."""
    out = _play_game(opening_seed=4242)
    board = chess.Board(out["opening_fen"])
    outcome = {"1-0": 1.0, "0-1": -1.0}.get(out["result"], 0.0)
    for i, uci in enumerate(out["ucis"]):
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"illegal recorded move {uci} at ply {i}"
        assert out["moves"][i] == board.san(move), f"SAN mismatch at ply {i}"
        ex = out["examples"][i]
        expected_planes = encode_state_v2(board)
        assert np.array_equal(np.asarray(ex["board_tensor"]), expected_planes), (
            f"v2 planes diverge from python at ply {i} (history truncation?)"
        )
        expected_value = outcome if board.turn == chess.WHITE else -outcome
        assert float(ex["value"]) == expected_value, f"value label wrong at ply {i}"
        board.push(move)
        assert out["fens"][i] == board.fen(), f"FEN mismatch at ply {i}"
    if out["termination"] == "game-over":
        assert board.is_game_over(claim_draw=True)
        py_result = board.result(claim_draw=True)
        if py_result == "*":
            py_result = "1/2-1/2"
        assert out["result"] == py_result
    else:
        assert len(out["ucis"]) == GAME_KWARGS["max_plies"], "cap must exhaust exactly"


def test_selfplay_same_seed_replay_is_bit_identical(selfplay_env):
    a = _play_game(opening_seed=4242)
    b = _play_game(opening_seed=4242)
    assert a["ucis"] == b["ucis"]
    assert a["moves"] == b["moves"]
    assert a["fens"] == b["fens"]
    assert a["result"] == b["result"] and a["termination"] == b["termination"]
    for ea, eb in zip(a["examples"], b["examples"]):
        assert np.array_equal(np.asarray(ea["board_tensor"]), np.asarray(eb["board_tensor"]))
        assert np.array_equal(np.asarray(ea["policy"]), np.asarray(eb["policy"]))
        assert ea["value"] == eb["value"]


def test_selfplay_startpos_root_when_no_opening_seed(selfplay_env):
    out = _play_game(opening_seed=None)
    assert out["opening_fen"] == chess.STARTING_FEN


def test_selfplay_different_opening_seeds_diverge(selfplay_env):
    a = _play_game(opening_seed=4242)
    b = _play_game(opening_seed=4243)
    assert a["opening_fen"] != b["opening_fen"]

def test_selfplay_structural_parity_vs_python(selfplay_env):
    """H6(a) cross-language gate: the python ``self_play_game`` path and the
    rust ``selfplay_game`` path produce schema-identical records under the
    same config (sims/plies/temperature/encoder v2, startpos root). Values and
    trajectories MAY differ (different rng streams); the record schema, count
    relationships, target ranges and result consistency must not."""
    onnxruntime = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")  # training.selfplay imports torch

    from training.model import value_head_scalar
    from training.selfplay import SelfPlayConfig, SearchConfig, self_play_game

    session = onnxruntime.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )

    class OrtEvaluator:
        """training Evaluator protocol over the H1 ONNX artifact (CPU EP)."""

        def evaluate(self, boards):
            batch = np.stack([encode_state_v2(b) for b in boards]).astype(np.float32)
            policy, value = session.run(["policy", "value"], {"input": batch})
            scalars = value_head_scalar(torch.from_numpy(value)).numpy().astype(np.float32)
            return policy.astype(np.float32), scalars

    py = self_play_game(
        OrtEvaluator(),
        config=SelfPlayConfig(
            search=SearchConfig(num_simulations=GAME_KWARGS["sims"]),
            max_plies=GAME_KWARGS["max_plies"],
            temperature=GAME_KWARGS["temperature"],
            temperature_plies=GAME_KWARGS["temperature_plies"],
            encoder_version=2,
        ),
        rng=np.random.default_rng(GAME_KWARGS["seed"]),
    )
    rs = _play_game(opening_seed=None)

    # --- count relationships (one example per played ply, both sides) -------
    assert len(py.examples) == py.ply_count == len(py.moves)
    assert len(rs["examples"]) == rs["ply_count"] == len(rs["moves"]) == len(rs["ucis"])

    # --- schema: keys, shapes, float kinds, target ranges -------------------
    for side, examples in (("python", py.examples), ("rust", rs["examples"])):
        for i, ex in enumerate(examples):
            if side == "python":
                bt, pol, value = ex.planes, ex.policy, ex.value
            else:
                bt, pol, value = (
                    np.asarray(ex["board_tensor"]),
                    np.asarray(ex["policy"]),
                    ex["value"],
                )
            assert bt.shape == (104, 8, 8), (side, i)
            assert pol.shape == (4672,), (side, i)
            assert bt.dtype.kind == "f" and pol.dtype.kind == "f", (side, i)
            assert float(np.asarray(pol).min()) >= 0.0, (side, i)
            total = float(np.asarray(pol).sum())
            assert abs(total - 1.0) < 1e-4, (side, i, total)
            assert value in (-1.0, 0.0, 1.0), (side, i, value)

    # --- both games consistent with their own final position ---------------
    assert py.result in {"1-0", "0-1", "1/2-1/2"}
    assert rs["result"] in {"1-0", "0-1", "1/2-1/2"}
    py_termination = "game-over" if py.ply_count < GAME_KWARGS["max_plies"] else "max-plies"
    assert rs["termination"] in {"game-over", "max-plies"}
    assert py_termination == rs["termination"], (
        "same config must reach the same termination kind "
        f"(python={py_termination}, rust={rs['termination']})"
    )

    # the python-side SANs also satisfy the python-chess oracle (mirror of the
    # rust SAN gate above) — both languages agree with the oracle
    board = chess.Board()
    for i, san in enumerate(py.moves):
        move = board.parse_san(san)
        assert board.san(move) == san
        board.push(move)
