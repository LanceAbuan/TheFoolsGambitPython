import chess
import math

# Piece-square tables (from white's perspective; mirrored for black)
PSQT = {
    chess.PAWN: [
         0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
         5,   5,  10,  25,  25,  10,   5,   5,
         0,   0,   0,  20,  20,   0,   0,   0,
         5, -5,-10,   0,   0,-10,  -5,   5,
        5,  10,  10,-20,-20,  10,  10,   5,
         0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,   0,   0,   0,   0,-20,-40,
        -30,   0,  10,  15,  15,  10,   0,-30,
        -30,   5,  15,  20,  20,  15,   5,-30,
        -30,   0,  15,  20,  20,  15,   0,-30,
        -30,   5,  10,  15,  15,  10,   5,-30,
        -40,-20,   0,   5,   5,   0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,   0,   0,   0,   0,   0,   0,-10,
        -10,   0,   5,  10,  10,   5,   0,-10,
        -10,   5,   5,  10,  10,   5,   5,-10,
        -10,   0,  10,  10,  10,  10,   0,-10,
        -10,  10,  10,  10,  10,  10,  10,-10,
        -10,   5,   0,   0,   0,   0,   5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0,
    ],
    chess.QUEEN: [
        -20,-10,-10,  -5,  -5,-10,-10,-20,
        -10,   0,   0,   0,   0,   0,   0,-10,
        -10,   0,   5,   5,   5,   5,   0,-10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0,-10,
        -10,   0,   5,   0,   0,   0,   0,-10,
        -20,-10,-10,  -5,  -5,-10,-10,-20,
    ],
    chess.KING: [
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
    ],
}

# Piece values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def _eval_table(board, table):
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        row, col = chess.square_rank(sq), chess.square_file(sq)
        if piece.color == chess.WHITE:
            idx = row * 8 + col
        else:
            idx = (7 - row) * 8 + col
        val = table[idx]
        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val
    return score


def evaluate(board):
    """Material + positional evaluation (positive = white advantage)."""
    score = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type != piece_type:
                continue
            row, col = chess.square_rank(sq), chess.square_file(sq)
            if piece.color == chess.WHITE:
                idx = row * 8 + col
            else:
                idx = (7 - row) * 8 + col

            val = PIECE_VALUES[piece_type] + PSQT[piece_type][idx]
            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val

    # Mobility bonus
    white_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board_copy = chess.Board(board.fen())
    board_copy.turn = chess.WHITE
    w_mob = len(list(board_copy.legal_moves))
    board_copy2 = chess.Board(board.fen())
    board_copy2.turn = chess.BLACK
    b_mob = len(list(board_copy2.legal_moves))
    score += (w_mob - b_mob) * 2

    return score


def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0:
        return evaluate(board), None

    legal = list(board.legal_moves)
    if not legal:
        return evaluate(board), None

    # Move ordering: captures first
    def move_score(m):
        s = 0
        if board.is_capture(m):
            captured = board.piece_at(m.to_square)
            if captured:
                s += PIECE_VALUES.get(captured.piece_type, 0)
        if board.is_en_passant(m):
            s += 100
        return s

    legal.sort(key=move_score, reverse=True)

    best_move = legal[0]

    if maximizing:
        best_val = -math.inf
        for move in legal:
            board.push(move)
            val, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = math.inf
        for move in legal:
            board.push(move)
            val, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best_val, best_move


class AIMoveGenerator:
    def __init__(self, depth=3):
        self.depth = depth

    def generate(self, board, depth=None):
        legal = list(board.legal_moves)
        if not legal:
            return None

        depth = depth if depth is not None else self.depth

        # If game is nearly over (checkmate in 1), just take it
        for m in legal:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m.uci()
            board.pop()

        _, best = minimax(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
        if best:
            return best.uci()
        return legal[0].uci()
