"""Stateless game module — imports AI from root ai.py."""
import chess
import sys
import os

# Ensure root is on path so we can import ai.py from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ai import AIMoveGenerator  # noqa: E402


def new_game(ai_depth=3):
    """Return fresh game state."""
    board = chess.Board()
    return _state(board, [])


def make_move(fen, uci, legal=None):
    """Apply a single UCI move to the given FEN position.

    Args:
        fen: Current FEN string.
        uci: Move in UCI format (e.g. 'e2e4', 'e7e8q').
        legal: Optional list of legal UCI moves for pre-validation.

    Returns:
        Dict with updated game state or error.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)

    if legal and uci not in legal:
        return {"error": "Illegal move", "status": "active"}
    if move not in board.legal_moves:
        return {"error": "Illegal move", "status": "active"}

    san = board.san(move)
    board.push(move)
    return _state(board, [san])


def ai_move(fen, legal=None, ai_depth=3):
    """Generate and apply an AI move to the given FEN position."""
    gen = AIMoveGenerator(depth=ai_depth)

    board = chess.Board(fen)
    uci = gen.generate(board)
    if not uci:
        return _state(board, [])

    move = chess.Move.from_uci(uci)
    san = board.san(move)
    board.push(move)
    return _state(board, [san])


def undo(fen):
    """Rebuild state from a previous FEN (stateless undo).

    The frontend tracks FEN history and sends the FEN to revert to.
    """
    board = chess.Board(fen)
    return _state(board, [])


def _state(board, last_sans):
    """Build the state dict returned to the frontend."""
    status = "active"
    result = None

    if board.is_checkmate():
        status = "checkmate"
        winner = "Black" if board.turn == chess.WHITE else "White"
        result = f"Checkmate! {winner} wins."
    elif board.is_stalemate():
        status = "stalemate"
        result = "Stalemate. Draw."
    elif board.is_insufficient_material():
        status = "draw"
        result = "Draw — insufficient material."
    elif board.is_repetition():
        status = "draw"
        result = "Draw — repetition."
    elif board.halfmove_clock >= 100:
        status = "draw"
        result = "Draw — 50-move rule."

    pgn = " ".join(last_sans) if last_sans else ""

    return {
        "fen": board.fen(),
        "legal": [m.uci() for m in board.legal_moves],
        "turn": "white" if board.turn == chess.WHITE else "black",
        "pgn": pgn,
        "status": status,
        "result": result,
        "in_check": board.is_check(),
    }
