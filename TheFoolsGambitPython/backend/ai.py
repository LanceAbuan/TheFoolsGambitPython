"""Minimax chess AI with alpha-beta pruning and PST evaluation."""
import chess
import math


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

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def evaluate(board):
    """Material + positional evaluation (positive = white advantage)."""
    score = 0
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type != piece_type:
                continue
            row, col = chess.square_rank(sq), chess.square_file(sq)
            idx = row * 8 + col if piece.color == chess.WHITE else (7 - row) * 8 + col

            val = PIECE_VALUES[piece_type] + PSQT[piece_type][idx]
            score += val if piece.color == chess.WHITE else -val

    # Mobility bonus
    board_w = chess.Board(board.fen())
    board_w.turn = chess.WHITE
    board_b = chess.Board(board.fen())
    board_b.turn = chess.BLACK
    score += (len(list(board_w.legal_moves)) - len(list(board_b.legal_moves))) * 2

    return score


def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0:
        return evaluate(board), None

    legal = list(board.legal_moves)
    if not legal:
        return evaluate(board), None

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

    def generate(self, board):
        legal = list(board.legal_moves)
        if not legal:
            return None

        for m in legal:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m.uci()
            board.pop()

        _, best = minimax(board, self.depth, -math.inf, math.inf, board.turn == chess.WHITE)
        if best:
            return best.uci()
        return legal[0].uci()
