"""
Stateless game engine. Accepts FEN + UCI move, returns new state.
Designed for serverless — no persistent state between requests.
"""
import chess


def new_game():
    """Return fresh game state."""
    board = chess.Board()
    return _state(board, [])


def play_move(fen, uci, legal=None):
    """Apply a single UCI move to the given FEN position.

    Returns new state dict or error.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)

    if legal and uci not in legal:
        return {"error": "Illegal move", "status": "active"}
    if move not in board.legal_moves:
        return {"error": "Illegal move", "status": "active"}

    san = board.san(move)
    board.push(move)

    # Build PGN from the board
    pgn = board.board_fen()  # placeholder — frontend tracks full PGN
    return _state(board, [san])


def ai_move(fen, legal=None, depth=3):
    """Generate and apply an AI move to the given FEN position."""
    from ai import AIMoveGenerator
    gen = AIMoveGenerator(depth=depth)

    board = chess.Board(fen)
    uci = gen.generate(board, depth=depth)
    if not uci:
        return _state(board, [])

    move = chess.Move.from_uci(uci)
    san = board.san(move)
    board.push(move)
    return _state(board, [san])


def _state(board, last_sans):
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
