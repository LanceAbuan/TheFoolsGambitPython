"""WP3: deblunder rescorer + soft-Z/soft-resign plumbing tests."""
import numpy as np

from scripts.deblunder import deblunder_record, encode_record


def _record():
    """Two-ply game from the startpos: e2e4 then e7e5, low-share first move."""
    import chess

    from training.encoder import encode_move

    board = chess.Board()
    ucis = ["e2e4", "e7e5"]
    policies = []
    for uci in ucis:
        pol = np.zeros(4672, dtype=np.float32)
        mv = chess.Move.from_uci(uci)
        # Low visit share on ply 0 (temperature blunder), normal on ply 1.
        pol[encode_move(board, mv)] = 0.01 if len(policies) == 0 else 0.6
        policies.append(pol)
        board.push(mv)
    return {
        "schema": "gambit.selfplay.v2",
        "encoding": "f32.le.b64",
        "weight_version": 0,
        "worker_id": "w0",
        "seed": 1,
        "opening_seed": None,
        "sims": 64,
        "max_plies": 200,
        "temperature": 1.0,
        "temperature_plies": 30,
        "opening_fen": chess.STARTING_FEN,
        "result": "0-1",
        "termination": "game-over",
        "ply_count": 2,
        "moves": ["e4", "e5"],
        "ucis": ucis,
        "fens": [],
        "examples": [
            {"board_tensor": np.zeros((104, 8, 8), dtype=np.float32), "policy": policies[0],
             "value": -1.0, "policy_masked": False, "root_value": 0.3},
            {"board_tensor": np.zeros((104, 8, 8), dtype=np.float32), "policy": policies[1],
             "value": 1.0, "policy_masked": False, "root_value": -0.1},
        ],
    }


def test_deblunder_relabels_low_share_divergence():
    record = _record()
    # min_ply=0 so the ply-0 blunder row is eligible.
    changed = deblunder_record(record, min_ply=0, min_share=0.05, delta=0.5, blend=0.5)
    assert changed == 1
    # value blended halfway toward the search's estimate (+0.3).
    assert record["examples"][0]["value"] == (-1.0 * 0.5) + (0.3 * 0.5)
    # High-share, small-divergence rows are untouched.
    assert record["examples"][1]["value"] == 1.0


def test_deblunder_respects_min_ply_and_encode_roundtrip():
    record = _record()
    assert deblunder_record(record, min_ply=8) == 0
    encoded = encode_record(record)
    ex = encoded["examples"][0]
    assert set(ex) == {"board_tensor_b64", "policy_b64", "value", "policy_masked", "root_value"}
