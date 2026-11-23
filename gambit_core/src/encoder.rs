//! Bit-exact port of `training/encoder.py` (the ground truth): 29 input planes
//! and the 73x64 policy encoding, all in side-to-move perspective.
//!
//! Plane layout (29 x 8 x 8), mirrored when black is to move (square ^ 56):
//! - 0-5   our pieces   P N B R Q K
//! - 6-11  their pieces P N B R Q K
//! - 12-23 history planes (zeros — the FEN-based API carries no history)
//! - 24    our   kingside  castling right  (at [0, 0])
//! - 25    our   queenside castling right  (at [0, 0])
//! - 26    their kingside  castling right  (at [0, 0])
//! - 27    their queenside castling right  (at [0, 0])
//! - 28    en-passant target square
//!
//! Policy index = from_square_perspective * 73 + plane:
//! - 0-55  queen-like: 8 directions (N, NE, E, SE, S, SW, W, NW) x 7 distances,
//!   plane = dir * 7 + dist - 1
//! - 56-63 knight jumps (fixed offset table)
//! - 64-72 underpromotions: (file_delta + 1) * 3 + piece_idx (N, B, R)
//!
//! Queen promotions ride the queen-like planes; castling is the king's
//! two-square move; en passant is the pawn's diagonal capture.

use chess::{ChessMove, Color, Piece};

use crate::position::Position;

pub const NUM_PLANES: usize = 29;
pub const POLICY_PLANES: usize = 73;
pub const POLICY_MOVES: usize = POLICY_PLANES * 64; // 4672

pub const FIRST_KNIGHT_PLANE: i32 = 56;
pub const FIRST_UNDERPROMOTION_PLANE: i32 = 64;

/// Perspective direction table (dr toward the opponent, df toward larger file).
const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),   // N
    (1, 1),   // NE
    (0, 1),   // E
    (-1, 1),  // SE
    (-1, 0),  // S
    (-1, -1), // SW
    (0, -1),  // W
    (1, -1),  // NW
];

const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
];

/// Underpromotion plane piece order: knight, bishop, rook.
fn underpromotion_index(piece: Piece) -> i32 {
    match piece {
        Piece::Knight => 0,
        Piece::Bishop => 1,
        Piece::Rook => 2,
        _ => panic!("underpromotion plane used by a non-minor/rook piece"),
    }
}

#[inline]
fn flip_square(sq: usize) -> usize {
    sq ^ 56
}

#[inline]
fn to_perspective(sq: usize, flip: bool) -> usize {
    if flip {
        flip_square(sq)
    } else {
        sq
    }
}

/// Encode one position into a row-major (29, 8, 8) float32 block
/// (index = (plane * 8 + rank) * 8 + file). Bit-exact with
/// `training.encoder.encode_state(board, prev_board=None)`.
pub fn encode_planes(pos: &Position) -> [f32; NUM_PLANES * 64] {
    let mut tensor = [0f32; NUM_PLANES * 64];
    let board = pos.board();
    let flip = board.side_to_move() == Color::Black;
    let stm = board.side_to_move();

    // Planes 0-11: current pieces. Piece index order P N B R Q K matches
    // training.encoder.PIECE_ORDER and the chess crate's Piece::to_index().
    write_piece_planes(pos, flip, stm, 0, &mut tensor);

    // Planes 12-23: history planes stay zero (no prev_board in this API).

    // Planes 24-27: castling rights (cleaned like python-chess), all at [0, 0].
    let (our_ks, our_qs) = pos.clean_castling_rights(stm);
    let (their_ks, their_qs) = pos.clean_castling_rights(!stm);
    tensor[24 * 64] = our_ks as u8 as f32;
    tensor[25 * 64] = our_qs as u8 as f32;
    tensor[26 * 64] = their_ks as u8 as f32;
    tensor[27 * 64] = their_qs as u8 as f32;

    // Plane 28: en-passant target square (mirrored like every other square).
    if let Some(ep) = pos.raw_ep() {
        set_plane_square(&mut tensor, 28, to_perspective(ep as usize, flip));
    }

    tensor
}

#[inline]
fn set_plane_square(tensor: &mut [f32], plane: usize, sq_p: usize) {
    let rank = sq_p / 8;
    let file = sq_p % 8;
    tensor[(plane * 8 + rank) * 8 + file] = 1.0;
}

/// Write the 12 piece planes of `pos` at channel `offset`, classified by
/// `our_color` into our (offset+0..6) and their (offset+6..12) blocks, in
/// `flip` perspective. `our_color` is the CURRENT side to move — for history
/// planes the source position may have the opposite mover, exactly like
/// python `_piece_planes(board, flip, offset, tensor, our_color)`.
fn write_piece_planes(pos: &Position, flip: bool, our_color: Color, offset: usize, tensor: &mut [f32]) {
    let board = pos.board();
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let base = piece.to_index(); // Pawn=0 .. King=5
        for sq in *board.pieces(piece) & *board.color_combined(our_color) {
            set_plane_square(tensor, offset + base, to_perspective(sq.to_index(), flip));
        }
        for sq in *board.pieces(piece) & *board.color_combined(!our_color) {
            set_plane_square(tensor, offset + 6 + base, to_perspective(sq.to_index(), flip));
        }
    }
}

// --- v2 encoder (WP-F C4): bit-exact port of `training.encoder.encode_v2` -----
//
// Authoritative WP-B layout (docs/OPTIMIZATION_PLAN.md §2): the v1 29 planes
// are a strict prefix, then t-2..t-7 piece planes at 29-100 (block t-k at
// 29 + 12*(k-2)), repetition-own 101, repetition-opp 102 (capped at 3,
// normalized /3), side-to-move flag 103 (1.0 white — a global, never-mirrored
// flag).
//
// History contract (`history[0]` = t-1, `history[1]` = t-2, ...):
// - EMPTY history zeroes planes 12-23 AND 29-100 (v1 `prev_board=None`
//   parity, ply-0 prefix rule).
// - Non-empty history shorter than 7 pads missing deep slots with the
//   standard initial position snapshot, written from the CURRENT side's
//   perspective.
// - Repetition counts run over the FULL history list (beyond the 7-step
//   plane window), prior occurrences only (the current position is excluded).

/// Queen-like plane for a (dr, df) delta in perspective coordinates.
fn queen_like_plane(dr: i32, df: i32) -> usize {
    let dist = dr.abs().max(df.abs());
    assert!((1..=7).contains(&dist), "queen-like distance out of range: ({dr}, {df})");
    assert!(dr == 0 || df == 0 || dr.abs() == df.abs(), "not a queen-like delta: ({dr}, {df})");
    let sign = |v: i32| v.signum() as i8;
    let (dr_s, df_s) = (sign(dr), sign(df));
    let direction = DIRECTIONS
        .iter()
        .position(|&d| d == (dr_s, df_s))
        .expect("sign pair is always a valid direction");
    direction * 7 + (dist - 1) as usize
}

/// Port of `training.encoder.encode_move`: the 4672 policy index of a legal move.
pub fn policy_index(pos: &Position, m: &ChessMove) -> usize {
    let board = pos.board();
    let flip = board.side_to_move() == Color::Black;
    let from_p = to_perspective(m.get_source().to_index(), flip);
    let to_p = to_perspective(m.get_dest().to_index(), flip);
    let dr = (to_p / 8) as i32 - (from_p / 8) as i32;
    let df = (to_p % 8) as i32 - (from_p % 8) as i32;
    let piece = board
        .piece_on(m.get_source())
        .expect("legal move source square holds a piece");

    let plane: usize = if let Some(promo) = m.get_promotion() {
        if promo != Piece::Queen {
            // Underpromotion planes 64-72.
            assert!(piece == Piece::Pawn, "underpromotion plane used by a non-pawn");
            (FIRST_UNDERPROMOTION_PLANE + (df + 1) * 3 + underpromotion_index(promo)) as usize
        } else {
            // Queen promotion rides the queen-like planes.
            queen_like_plane(dr, df)
        }
    } else if piece == Piece::Knight {
        let idx = KNIGHT_OFFSETS
            .iter()
            .position(|&o| o == (dr as i8, df as i8))
            .unwrap_or_else(|| panic!("knight move with non-knight delta ({dr}, {df})"));
        (FIRST_KNIGHT_PLANE + idx as i32) as usize
    } else {
        queen_like_plane(dr, df)
    };
    from_p * POLICY_PLANES + plane
}

/// Binary (4672,) float32 mask with 1.0 at every legal move's index —
/// bit-exact with `training.encoder.legal_mask(board)`.
pub fn encode_mask(pos: &Position) -> [f32; POLICY_MOVES] {
    let mut mask = [0f32; POLICY_MOVES];
    for m in pos.legal_moves() {
        mask[policy_index(pos, &m)] = 1.0;
    }
    mask
}

// --- v2 (WP-F C4) -------------------------------------------------------------

pub const NUM_PLANES_V2: usize = 104;
/// Deep-history window: t-2 .. t-7 (t0 and t-1 live in the v1 prefix).
pub const V2_DEEP_HISTORY_STEPS: usize = 6;
/// First deep-history channel (block t-k starts at 29 + 12*(k-2)).
pub const V2_DEEP_HISTORY_OFFSET: usize = 29;
pub const V2_REP_OWN_PLANE: usize = 101;
pub const V2_REP_OPP_PLANE: usize = 102;
pub const V2_SIDE_TO_MOVE_PLANE: usize = 103;
/// Repetition counts saturate here, then normalize to 0-1 by /3.
pub const V2_REP_CAP: f32 = 3.0;

/// Bit-exact port of `training.encoder.encode_v2(board, history)`.
///
/// `history` is most-recent-first (`history[0]` = t-1, `history[1]` = t-2,
/// ...) — the python function's `history` parameter, already materialized as
/// positions. See the module block above for the layout and padding rules.
pub fn encode_planes_v2(pos: &Position, history: &[Position]) -> [f32; NUM_PLANES_V2 * 64] {
    let mut tensor = [0f32; NUM_PLANES_V2 * 64];
    let board = pos.board();
    let flip = board.side_to_move() == Color::Black;
    let stm = board.side_to_move();

    // Planes 0-28: bit-identical v1 prefix (t-1 from history[0], else zeros).
    write_piece_planes(pos, flip, stm, 0, &mut tensor);
    if let Some(prev) = history.first() {
        write_piece_planes(prev, flip, stm, 12, &mut tensor);
    }
    let (our_ks, our_qs) = pos.clean_castling_rights(stm);
    let (their_ks, their_qs) = pos.clean_castling_rights(!stm);
    tensor[24 * 64] = our_ks as u8 as f32;
    tensor[25 * 64] = our_qs as u8 as f32;
    tensor[26 * 64] = their_ks as u8 as f32;
    tensor[27 * 64] = their_qs as u8 as f32;
    if let Some(ep) = pos.raw_ep() {
        set_plane_square(&mut tensor, 28, to_perspective(ep as usize, flip));
    }

    // Planes 29-100: t-2..t-7 piece planes. Empty history -> all zero (v1
    // parity); non-empty short history -> startpos-snapshot padding.
    if !history.is_empty() {
        let start = Position::starting();
        for step in 0..V2_DEEP_HISTORY_STEPS {
            let offset = V2_DEEP_HISTORY_OFFSET + step * 12;
            match history.get(step + 1) {
                Some(src) => write_piece_planes(src, flip, stm, offset, &mut tensor),
                None => write_piece_planes(&start, flip, stm, offset, &mut tensor),
            }
        }
    }

    // Planes 101-102: repetition counts over the FULL history (prior
    // occurrences only), capped at 3 and normalized to 0-1.
    let cur_key = pos.transposition_key();
    let opp_key = cur_key.opponent_turn_key();
    let mut own = 0u32;
    let mut opp = 0u32;
    for prev in history {
        let key = prev.transposition_key();
        if key == cur_key {
            own += 1;
        } else if key == opp_key {
            opp += 1;
        }
    }
    tensor[V2_REP_OWN_PLANE * 64] = own.min(3) as f32 / V2_REP_CAP;
    tensor[V2_REP_OPP_PLANE * 64] = opp.min(3) as f32 / V2_REP_CAP;

    // Plane 103: side to move (global flag, not perspective-mirrored).
    tensor[V2_SIDE_TO_MOVE_PLANE * 64] = u8::from(stm == Color::White) as f32;
    tensor
}

/// Replay + encode a whole `(fen, uci-stack)` batch into (N, 104, 8, 8)
/// blocks — the Rust side of the WP-B parity contract
/// (`gambit_core.encode_batch_v2(fens, histories)`).
///
/// Each row carries the CURRENT position's FEN (raw en-passant field, i.e.
/// python-chess `fen(en_passant="fen")`) and the full UCI move stack from the
/// game root to this position. History positions are REPLAYED, because a
/// position's past is not recoverable from the position alone (a captured
/// piece's type is gone). Rows of one game must be consecutive and grow the
/// stack by exactly one move; a row with an empty stack starts a new game and
/// anchors the replay. Every row is verified — the replayed position must
/// FEN-match the given FEN exactly (placement, turn, castling, raw ep,
/// clocks) — and mismatches raise (ValueError on the python side).
pub fn encode_batch_v2_rows(
    fens: &[String],
    stacks: &[Vec<String>],
) -> Result<Vec<[f32; NUM_PLANES_V2 * 64]>, String> {
    if fens.len() != stacks.len() {
        return Err(format!(
            "fens and histories must have the same length ({} != {})",
            fens.len(),
            stacks.len()
        ));
    }
    let mut out = Vec::with_capacity(fens.len());
    // Chronological replay of the current game: P_0..P_j with the move that
    // produced each entry.
    let mut replay: Vec<Position> = Vec::new();
    for (row, (fen, stack)) in fens.iter().zip(stacks.iter()).enumerate() {
        let bad = |msg: String| format!("encode_batch_v2 row {row}: {msg}");
        let cur = Position::from_fen(fen).map_err(|e| bad(e))?;
        if stack.is_empty() {
            // New game anchor: the given FEN IS the root.
            replay.clear();
            replay.push(cur);
        } else {
            if replay.len() != stack.len() {
                return Err(bad(format!(
                    "history stack of {} moves does not continue the current game rows \
                     (expected a row with an empty stack to anchor the game)",
                    stack.len()
                )));
            }
            let prev = replay[replay.len() - 1];
            let next = prev
                .make_move_uci(stack[stack.len() - 1].trim())
                .map_err(|e| bad(e))?;
            replay.push(next);
        }
        let replayed = replay[replay.len() - 1];
        let replayed_fen = replayed.to_fen();
        if replayed_fen != *fen {
            return Err(bad(format!(
                "history replay mismatch: replayed {replayed_fen} != given {fen}"
            )));
        }
        // history = [P_{L-1}, ..., P_0] = python [t-1, t-2, ...].
        let l = stack.len();
        let history: Vec<Position> = replay[..l].iter().rev().copied().collect();
        out.push(encode_planes_v2(&cur, &history));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use chess::Square;

    #[test]
    fn startpos_planes() {
        let pos = Position::starting();
        let t = encode_planes(&pos);
        let plane0: usize = 0; // silence clippy::erasing_op while keeping plane math readable
        // White to move: no flip. Pawns on rank 1 (perspective) => plane 0, rank 1.
        for file in 0..8 {
            assert_eq!(t[(plane0 * 8 + 1) * 8 + file], 1.0);
            assert_eq!(t[(6 * 8 + 6) * 8 + file], 1.0); // their pawns, plane 6, rank 6
        }
        // Knights: our plane 1 at b1/g1 => perspective (0,1) and (0,6).
        assert_eq!(t[64 + 1], 1.0);
        assert_eq!(t[64 + 6], 1.0);
        // Kings on e1/e8 (plane * 64 + rank * 8 + file).
        assert_eq!(t[5 * 64 + 4], 1.0);
        assert_eq!(t[(11 * 8 + 7) * 8 + 4], 1.0);
        // Castling: all four rights.
        assert_eq!(t[24 * 64], 1.0);
        assert_eq!(t[25 * 64], 1.0);
        assert_eq!(t[26 * 64], 1.0);
        assert_eq!(t[27 * 64], 1.0);
        // No ep at start.
        assert_eq!(t[28 * 64..29 * 64].iter().sum::<f32>(), 0.0);
        // History planes zero.
        for plane in 12..24 {
            assert_eq!(t[plane * 64..(plane + 1) * 64].iter().sum::<f32>(), 0.0);
        }
    }

    #[test]
    fn black_perspective_flip() {
        // After 1. e4 black is to move. White pawn on e4 (28) is "their" pawn
        // at perspective 28 ^ 56 = 36 (rank 4, file 4); the ep target e3 (20)
        // mirrors to 20 ^ 56 = 44.
        let pos = Position::starting().make_move_uci("e2e4").unwrap();
        assert_eq!(pos.board().side_to_move(), Color::Black);
        let t = encode_planes(&pos);
        let plane0: usize = 0;
        assert_eq!(t[28 * 64 + (20usize ^ 56)], 1.0); // ep plane
        assert_eq!(t[(6 * 8 + 4) * 8 + 4], 1.0); // their pawn on e4
        assert_eq!(t[(plane0 * 8 + 1) * 8 + 4], 1.0); // our (black) pawn on e7 -> perspective rank 1
    }

    #[test]
    fn policy_index_startpos_e2e4() {
        let pos = Position::starting();
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::E2 && m.get_dest() == Square::E4)
            .unwrap();
        // e2 -> e4: dr = 2, df = 0 -> dir N(0), dist 2 -> plane 1; from_p = 12.
        assert_eq!(policy_index(&pos, &m), 12 * 73 + 1);
        // Knight: g1 -> f3.
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::G1 && m.get_dest() == Square::F3)
            .unwrap();
        // g1 -> f3: dr = 2, df = -1 -> offset (2, -1) = index 1.
        assert_eq!(policy_index(&pos, &m), 6 * 73 + 56 + 1);
    }

    #[test]
    fn policy_index_castling_and_promotions() {
        // Castling: e1g1 -> dir E (index 2), dist 2 -> plane 15.
        let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::E1 && m.get_dest() == Square::G1)
            .unwrap();
        assert_eq!(policy_index(&pos, &m), 4 * 73 + 15);
        // Queenside: e1c1 -> dir W (index 6), dist 2 -> plane 43.
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::E1 && m.get_dest() == Square::C1)
            .unwrap();
        assert_eq!(policy_index(&pos, &m), 4 * 73 + 43);
    }

    #[test]
    fn policy_index_underpromotions() {
        // Black knight on b8 and rook on d8 give both capture underpromotions;
        // the empty c8 square gives the straight promotions.
        let pos = Position::from_fen("kn1r4/2P5/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let find = |dst: Square, promo: Option<Piece>| {
            pos.legal_moves()
                .into_iter()
                .find(|m| m.get_source() == Square::C7 && m.get_dest() == dst && m.get_promotion() == promo)
                .unwrap()
        };
        // c7c8q: queen-like, dir N dist 1 -> plane 0.
        assert_eq!(policy_index(&pos, &find(Square::C8, Some(Piece::Queen))), 50 * 73);
        // c7c8n: underpromotion, df = 0, knight idx 0 -> plane 64 + 3 + 0 = 67.
        assert_eq!(policy_index(&pos, &find(Square::C8, Some(Piece::Knight))), 50 * 73 + 67);
        // c7b8r: capture underpromotion, df = -1, rook idx 2 -> plane 64 + 0 + 2 = 66.
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::C7 && m.get_dest() == Square::B8 && m.get_promotion() == Some(Piece::Rook))
            .unwrap();
        assert_eq!(policy_index(&pos, &m), 50 * 73 + 66);
        // c7d8b: capture underpromotion, df = +1, bishop idx 1 -> plane 64 + 6 + 1 = 71.
        let m = pos
            .legal_moves()
            .into_iter()
            .find(|m| m.get_source() == Square::C7 && m.get_dest() == Square::D8 && m.get_promotion() == Some(Piece::Bishop))
            .unwrap();
        assert_eq!(policy_index(&pos, &m), 50 * 73 + 71);
    }

    #[test]
    fn mask_startpos_matches_twenty_moves() {
        let pos = Position::starting();
        let mask = encode_mask(&pos);
        assert_eq!(mask.iter().filter(|&&v| v == 1.0).count(), 20);
    }
}
