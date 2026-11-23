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
`from_square * 73 + plane` where `from_square` is the origin square in
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

v2 (WP-B): `encode_v2` extends this to 104 channels — the 29 v1 planes as a
strict prefix (hard requirement), then t-2..t-7 piece planes, two repetition
count planes and a side-to-move flag. See the layout block below `NUM_PLANES_V2`.
"""
from __future__ import annotations

from collections.abc import Sequence

import chess
import numpy as np

from .fastpath import fastpath_enabled, import_gambit_core

NUM_PLANES = 29
POLICY_PLANES = 73
POLICY_MOVES = POLICY_PLANES * 64  # 4672

# --- v2 layout (WP-B): 104 channels = v1 prefix + deep history + repetition ---
#
# The v1 29 planes are a strict PREFIX of v2 (hard requirement): the first 29
# output planes of `encode_v2` are exactly v1's 29 planes for the same
# (board, t-1) pair, in the same order — new planes are appended only. This is
# what makes the WP-D warm-start a straight weight copy on channels 0-28.
#
# v2 plane map:
#     0-11    t0  pieces (v1)            12-23   t-1 pieces (v1 history slot)
#     24-27   castling rights (v1)      28      en-passant target (v1)
#     29-100  t-2 .. t-7 pieces (6 x 12)
#     101     repetition count, own side to move   (capped at 3, /3 -> 0-1)
#     102     repetition count, opponent side to move (capped at 3, /3 -> 0-1)
#     103     side to move (1.0 white / 0.0 black)
NUM_PLANES_V2 = 104
ENCODER_VERSION = 2

V2_REP_CAP = 3                                     # counts saturate here
V2_DEEP_HISTORY_STEPS = 6                          # t-2 .. t-7 (t0, t-1 in prefix)
V2_DEEP_HISTORY_OFFSET = NUM_PLANES                # 29
V2_DEEP_HISTORY_PLANES = V2_DEEP_HISTORY_STEPS * 12  # 72 -> planes 29..100
V2_REP_OWN_PLANE = V2_DEEP_HISTORY_OFFSET + V2_DEEP_HISTORY_PLANES  # 101
V2_REP_OPP_PLANE = V2_REP_OWN_PLANE + 1            # 102
V2_SIDE_TO_MOVE_PLANE = V2_REP_OPP_PLANE + 1       # 103
assert V2_SIDE_TO_MOVE_PLANE == NUM_PLANES_V2 - 1

# Padding snapshot for missing deep-history slots (documented WP-B choice: the
# standard initial position, written from the CURRENT side's perspective like
# every other history plane). Only read, never mutated.
_INITIAL_BOARD = chess.Board()

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


# UCI -> chess.Move memo for the Rust legal-move path. chess.Move is a plain
# value type; the dict is bounded by the finite UCI vocabulary (~4.7k strings)
# and reads/writes are atomic under the GIL, so it is thread-safe for the
# parallel self-play callers (worst case: duplicate idempotent writes).
_UCI_MOVE_MEMO: dict[str, chess.Move] = {}


def _move_from_uci_memo(uci: str) -> chess.Move:
    """Module-level memoized chess.Move.from_uci (hot path: ~30 calls/expand)."""
    move = _UCI_MOVE_MEMO.get(uci)
    if move is None:
        move = chess.Move.from_uci(uci)
        _UCI_MOVE_MEMO[uci] = move
    return move


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
    if fastpath_enabled() and all(prev is None for prev in prev_boards):
        # M3 fast path: one FFI call, bit-exact with the loop below for
        # prev_board=None rows. The bridge FEN uses en_passant="fen" so the RAW
        # ep square survives (python-chess's default "legal" mode drops ep
        # squares with no legal capture, which would zero plane 28).
        gc = import_gambit_core()
        return gc.encode_batch([b.fen(en_passant="fen") for b in boards])
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
    """Map every legal move's policy index to the move (bijective by design).

    Insertion order follows `board.legal_moves` in BOTH paths: the Rust path
    enumerates moves in python-chess order (`gambit_core.legal_move_pairs`,
    order-differential-fuzzed), and MCTS child order / tie-breaks depend on it.
    """
    if fastpath_enabled():
        gc = import_gambit_core()
        return {
            index: _move_from_uci_memo(uci)
            for index, uci in gc.legal_move_pairs(board.fen(en_passant="fen"))
        }
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


# --- v2 encoder (WP-B) --------------------------------------------------------


def _history_transposition_key(board: chess.Board) -> tuple:
    """python-chess 1.11.2 transposition key of one position.

    The private method is pinned via uv.lock and already load-bearing for the
    M3 FastBoard (`training/selfplay.py`); using it keeps the repetition
    planes consistent with the search's threefold-repetition semantics.
    """
    return board._transposition_key()  # type: ignore[attr-defined]


def repetition_counts(
    board: chess.Board, history: Sequence[chess.Board]
) -> tuple[int, int]:
    """(own, opp) PRIOR occurrences of the current position within `history`.

    Authoritative WP-B layout (docs/OPTIMIZATION_PLAN.md, resolved): the
    CURRENT position is EXCLUDED, both counts start at 0, and only history
    entries contribute. `own` counts entries identical to `board` (same piece
    placement AND side to move, python-chess transposition-key semantics);
    `opp` counts entries with the SAME piece placement but the OPPOSITE side
    to move. Counts run over the FULL history list, beyond the 7-step plane
    window, so long repetition cycles still register.
    """
    key = _history_transposition_key(board)
    flipped = board.copy(stack=False)
    flipped.turn = not board.turn  # same placement, opposite mover (hypothetical key)
    opp_key = _history_transposition_key(flipped)
    own = 0
    opp = 0
    for previous in history:
        previous_key = _history_transposition_key(previous)
        if previous_key == key:
            own += 1
        elif previous_key == opp_key:
            opp += 1
    return own, opp


def encode_v2(board: chess.Board, history: Sequence[chess.Board] | None = None) -> np.ndarray:
    """Encode one position into the 104-channel v2 tensor, (104, 8, 8) float32.

    Plane map is in the module docstring block (WP-B): v1's 29 planes as a
    strict prefix, then t-2..t-7 piece planes (29-100), two repetition-count
    planes (101 own / 102 opp, capped at 3, normalized 0-1 by /3) and the
    side-to-move flag (103, 1.0 white — a global flag, never mirrored).

    `history[0]` is t-1 (the position immediately before `board`), `history[1]`
    is t-2, ... Piece planes consume at most the first 7 entries; repetition
    counts consume the full list.

    Padding (authoritative WP-B layout): a NON-EMPTY history shorter than 7
    pads its missing slots with the standard initial position snapshot,
    written from the current side perspective like every other history
    plane. An EMPTY history (or None) zeroes ALL history slots instead:
    planes 12-23 (t-1, inside the v1 prefix) AND planes 29-100, which is
    v1 encode_state(board, prev_board=None) parity and holds the hard
    prefix rule at ply 0.

    Prefix rule (hard requirement, regression-tested)::

        encode_v2(b, h)[:29] == encode_state(b, h[0] if h else None)
    """
    tensor = np.zeros((NUM_PLANES_V2, 8, 8), dtype=np.float32)
    flip = _perspective(board)
    hist = list(history) if history is not None else []

    # Planes 0-28: bit-identical v1 prefix (t-1 from history[0], else zeros).
    tensor[:NUM_PLANES] = encode_state(board, hist[0] if hist else None)

    # Planes 29-100: t-2..t-7 piece planes. Empty history -> all zero (v1
    # parity); non-empty short history -> startpos-snapshot padding.
    if hist:
        for step in range(V2_DEEP_HISTORY_STEPS):  # step 0 -> t-2
            offset = V2_DEEP_HISTORY_OFFSET + step * 12
            index = step + 1  # history index of t-(step+2)
            source = hist[index] if index < len(hist) else _INITIAL_BOARD
            _piece_planes(source, flip, offset, tensor, board.turn)


    # Planes 101-102: repetition counts, capped and normalized to 0-1.
    own, opp = repetition_counts(board, hist)
    tensor[V2_REP_OWN_PLANE, 0, 0] = min(own, V2_REP_CAP) / V2_REP_CAP
    tensor[V2_REP_OPP_PLANE, 0, 0] = min(opp, V2_REP_CAP) / V2_REP_CAP

    # Plane 103: side to move (global flag, not perspective-mirrored).
    tensor[V2_SIDE_TO_MOVE_PLANE, 0, 0] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor


def v2_history_from_stack(board: chess.Board, max_steps: int = V2_DEEP_HISTORY_STEPS + 1) -> list[chess.Board]:
    """History boards [t-1, t-2, ...] recovered from `board.move_stack`.

    The search carries full game stacks on every board (`Board.copy()` keeps
    `move_stack`), so history is derived position-locally: pop a copy back one
    ply at a time. Returns [] for an empty stack (game root -> encode_v2's
    zeroed history slots, v1 parity). `max_steps` bounds the walk; repetition
    counting then sees at most the last `max_steps` plies — the SAME bound
    applies at eval and example-recording time, which is what consistency
    requires.
    """
    if not board.move_stack:
        return []
    walked = board.copy()
    out: list[chess.Board] = []
    for _ in range(min(max_steps, len(board.move_stack))):
        walked.pop()
        out.append(walked.copy())
    return out


def encode_state_v2(board: chess.Board) -> np.ndarray:
    """encode_v2 with the history derived from `board.move_stack`.

    Single source of truth for the v2 encoding of a position that carries its
    own history (self-play search boards, game example boards): eval time and
    example time derive the identical tensor for the identical stack.
    """
    return encode_v2(board, v2_history_from_stack(board))


def encode_batch_v2(boards: Sequence[chess.Board]) -> np.ndarray:
    """N positions -> (N, 104, 8, 8) float32, histories from move stacks.

    Python fallback path (per-board `encode_state_v2`). The gambit_core
    `encode_batch_v2` bridge is intentionally NOT used here: its contract
    requires each game's rows to appear consecutively and grow by exactly one
    move (forward replay from an empty-stack anchor), which search-time eval
    batches (single mid-tree rows) cannot satisfy.
    """
    if not boards:
        return np.zeros((0, NUM_PLANES_V2, 8, 8), dtype=np.float32)
    return np.stack([encode_state_v2(b) for b in boards])


def encoder_version_for_model(model: object) -> int:
    """Encoder layout version a model consumes, from its input plane count.

    `GambitNet` carries `config.in_channels`; legacy models (ChessNet, raw
    test doubles) have no config and are treated as v1 consumers, preserving
    their exact pre-WP-D behavior.
    """
    config = getattr(model, "config", None)
    in_channels = getattr(config, "in_channels", None)
    if in_channels is None:
        return 1
    planes = int(in_channels)
    if planes == NUM_PLANES:
        return 1
    if planes == NUM_PLANES_V2:
        return ENCODER_VERSION
    raise ValueError(
        f"unknown encoder plane count {planes}; expected {NUM_PLANES} (v1) or {NUM_PLANES_V2} (v2)"
    )
