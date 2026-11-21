"""Position encoder v2 — AlphaZero-canonical input planes + 73x64 policy encoding.

Input planes (29 x 8 x 8), all from the **side-to-move's perspective** (board
ranks are mirrored when black is to move, "our" = the side to move):

======================  =============================================
planes  0- 5            our   pieces  P N B R Q K
planes  6-11            their pieces  P N B R Q K
planes 12-17            our   pieces, previous position
planes 18-23            their pieces, previous position
plane  24               our   kingside castling right
plane  25               our   queenside castling right
plane  26               their kingside castling right
plane  27               their queenside castling right
plane  28               en-passant target square
======================  =============================================

Policy encoding (73 x 64 = 4672): one 73-plane stack per origin square, index
``from_square * 73 + plane`` where ``from_square`` is the origin square in
side-to-move perspective.

- planes  0-55: queen-like moves — 8 directions x 7 distances
  (direction order: N, NE, E, SE, S, SW, W, NW; plane = dir * 7 + dist - 1)
- planes 56-63: knight jumps (fixed offset table)
- planes 64-72: underpromotions — 3 file-deltas (-1, 0, +1) x 3 pieces
  (N, B, R); plane = 64 + (file_delta + 1) * 3 + piece_idx

Queen promotions ride the queen-like planes (a7a8q -> dir N dist 1, a7b8q ->
dir NE dist 1). Castling is the king's two-square move (e1g1 -> dir E dist 2).
En passant is the pawn's diagonal capture. This gives underpromotions real
planes — the §10 "promotion policy" bug (everything sharing one logit) cannot
reappear.

All plane/delta math happens in perspective coordinates on both encode and
decode, so the encoding round-trips for both colors and the network weights
stay color-symmetric.
"""
from __future__ import annotations

from collections.abc import Sequence

import chess
import numpy as np

NUM_PLANES = 29
POLICY_PLANES = 73
POLICY_MOVES = POLICY_PLANES * 64  # 4672

PIECE_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# Perspective direction table (dr toward the opponent, df toward larger file).
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
)

_KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
)

_UNDERPROMOTION_PIECES: tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

_FIRST_KNIGHT_PLANE = 56
_FIRST_UNDERPROMOTION_PLANE = 64


def _flip_square(sq: int) -> int:
    """Mirror a square's rank (file unchanged) — the perspective transform."""
    return sq ^ 56


def _perspective(board: chess.Board) -> bool:
    """True when planes must be mirrored (black to move)."""
    return board.turn == chess.BLACK


def _to_perspective(sq: int, flip: bool) -> int:
    return _flip_square(sq) if flip else sq


def _from_perspective(sq_p: int, flip: bool) -> int:
    return _flip_square(sq_p) if flip else sq_p


def _sq_rank(sq: int) -> int:
    return sq // 8


def _sq_file(sq: int) -> int:
    return sq % 8


def _piece_planes(
    board: chess.Board, flip: bool, offset: int, tensor: np.ndarray, our_color: chess.Color
) -> None:
    """Write the 12 piece planes for one position at channel `offset`.

    `our_color` is the CURRENT side to move — for history planes the previous
    position may have the opposite side to move, so it cannot be derived from
    `board.turn` there.
    """
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sq_p = _to_perspective(sq, flip)
        rank, file = _sq_rank(sq_p), _sq_file(sq_p)
        base = PIECE_ORDER.index(piece.piece_type)
        if piece.color == our_color:
            tensor[offset + base, rank, file] = 1.0
        else:
            tensor[offset + 6 + base, rank, file] = 1.0


def encode_state(board: chess.Board, prev_board: chess.Board | None = None) -> np.ndarray:
    """Encode one position into a (29, 8, 8) float32 tensor.

    `prev_board` is the position before the last move (1-deep history per plan
    §2); when omitted the history planes are zero.
    """
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    flip = _perspective(board)

    _piece_planes(board, flip, 0, tensor, board.turn)
    if prev_board is not None:
        _piece_planes(prev_board, flip, 12, tensor, board.turn)

    # Castling rights: our kingside/queenside, then theirs.
    ours, theirs = board.turn, not board.turn
    tensor[24, 0, 0] = 1.0 if board.has_kingside_castling_rights(ours) else 0.0
    tensor[25, 0, 0] = 1.0 if board.has_queenside_castling_rights(ours) else 0.0
    tensor[26, 0, 0] = 1.0 if board.has_kingside_castling_rights(theirs) else 0.0
    tensor[27, 0, 0] = 1.0 if board.has_queenside_castling_rights(theirs) else 0.0

    # En-passant target square (mirrored like every other square).
    if board.ep_square is not None:
        sq_p = _to_perspective(board.ep_square, flip)
        tensor[28, _sq_rank(sq_p), _sq_file(sq_p)] = 1.0

    return tensor


def encode_batch(
    boards: Sequence[chess.Board],
    prev_boards: Sequence[chess.Board | None] | None = None,
) -> np.ndarray:
    """Encode N positions into a (N, 29, 8, 8) float32 batch."""
    if prev_boards is None:
        prev_boards = [None] * len(boards)
    if len(prev_boards) != len(boards):
        raise ValueError("prev_boards length must match boards length")
    return np.stack([encode_state(b, p) for b, p in zip(boards, prev_boards)])


def _queen_like_plane(dr: int, df: int) -> int:
    """Plane index for a straight/diagonal move with deltas (dr, df)."""
    dist = max(abs(dr), abs(df))
    if not 1 <= dist <= 7:
        raise ValueError(f"queen-like distance out of range: ({dr}, {df})")
    if not (dr == 0 or df == 0 or abs(dr) == abs(df)):
        raise ValueError(f"not a queen-like delta: ({dr}, {df})")
    direction = _DIRECTIONS.index((int(np.sign(dr)), int(np.sign(df))))
    return direction * 7 + dist - 1


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal move into its 4672 policy index (perspective coords)."""
    flip = _perspective(board)
    from_p = _to_perspective(move.from_square, flip)
    to_p = _to_perspective(move.to_square, flip)
    dr = _sq_rank(to_p) - _sq_rank(from_p)
    df = _sq_file(to_p) - _sq_file(from_p)
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")

    plane: int
    if move.promotion is not None and move.promotion != chess.QUEEN:
        if piece.piece_type != chess.PAWN:
            raise ValueError("underpromotion plane used by a non-pawn")
        plane = _FIRST_UNDERPROMOTION_PLANE + (df + 1) * 3 + _UNDERPROMOTION_PIECES.index(
            move.promotion
        )
    elif piece.piece_type == chess.KNIGHT:
        plane = _FIRST_KNIGHT_PLANE + _KNIGHT_OFFSETS.index((dr, df))
    else:
        plane = _queen_like_plane(dr, df)
    return from_p * POLICY_PLANES + plane


def decode_index(board: chess.Board, index: int) -> chess.Move | None:
    """Decode a policy index back into a move for this position.

    Returns None when the index does not correspond to a legal move.
    """
    if not 0 <= index < POLICY_MOVES:
        return None
    flip = _perspective(board)
    from_p, plane = divmod(index, POLICY_PLANES)
    from_sq = _from_perspective(from_p, flip)
    piece = board.piece_at(from_sq)
    if piece is None:
        return None

    to_p: int
    promotion: chess.PieceType | None = None
    if plane >= _FIRST_UNDERPROMOTION_PLANE:
        if piece.piece_type != chess.PAWN:
            return None
        file_delta = (plane - _FIRST_UNDERPROMOTION_PLANE) // 3 - 1
        promotion = _UNDERPROMOTION_PIECES[(plane - _FIRST_UNDERPROMOTION_PLANE) % 3]
        to_p = from_p + 8 + file_delta
    elif plane >= _FIRST_KNIGHT_PLANE:
        dr, df = _KNIGHT_OFFSETS[plane - _FIRST_KNIGHT_PLANE]
        to_p = from_p + dr * 8 + df
    else:
        direction, dist = divmod(plane, 7)
        dr, df = _DIRECTIONS[direction]
        to_p = from_p + dr * (dist + 1) * 8 + df * (dist + 1)

    if not 0 <= to_p < 64:
        return None
    to_sq = _from_perspective(to_p, flip)
    if piece.piece_type == chess.PAWN and _sq_rank(to_p) == 7 and promotion is None:
        promotion = chess.QUEEN
    move = chess.Move(from_sq, to_sq, promotion=promotion)
    return move if move in board.legal_moves else None


def legal_move_map(board: chess.Board) -> dict[int, chess.Move]:
    """Map every legal move's policy index to the move (bijective by design)."""
    mapping: dict[int, chess.Move] = {}
    for move in board.legal_moves:
        index = encode_move(board, move)
        existing = mapping.get(index)
        if existing is not None and existing != move:
            raise AssertionError(f"policy encoding collision: {index} for {existing} and {move}")
        mapping[index] = move
    return mapping


def legal_mask(board: chess.Board) -> np.ndarray:
    """Binary (4672,) float32 mask with 1.0 at every legal move's index."""
    mask = np.zeros(POLICY_MOVES, dtype=np.float32)
    for index in legal_move_map(board):
        mask[index] = 1.0
    return mask
