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
//!         plane = dir * 7 + dist - 1
//! - 56-63 knight jumps (fixed offset table)
//! - 64-72 underpromotions: (file_delta + 1) * 3 + piece_idx (N, B, R)
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

    let write = |tensor: &mut [f32; NUM_PLANES * 64], plane: usize, sq_p: usize| {
        let rank = sq_p / 8;
        let file = sq_p % 8;
        tensor[(plane * 8 + rank) * 8 + file] = 1.0;
    };

    // Planes 0-11: current pieces. Piece index order P N B R Q K matches
    // training.encoder.PIECE_ORDER and the chess crate's Piece::to_index().
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let base = piece.to_index(); // Pawn=0 .. King=5
        for sq in *board.pieces(piece) & *board.color_combined(stm) {
            write(&mut tensor, base, to_perspective(sq.to_index(), flip));
        }
        for sq in *board.pieces(piece) & *board.color_combined(!stm) {
            write(&mut tensor, 6 + base, to_perspective(sq.to_index(), flip));
        }
    }

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
        write(&mut tensor, 28, to_perspective(ep as usize, flip));
    }

    tensor
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use chess::Square;

    #[test]
    fn startpos_planes() {
        let pos = Position::starting();
        let t = encode_planes(&pos);
        // White to move: no flip. Pawns on rank 1 (perspective) => plane 0, rank 1.
        for file in 0..8 {
            assert_eq!(t[(0 * 8 + 1) * 8 + file], 1.0);
            assert_eq!(t[(6 * 8 + 6) * 8 + file], 1.0); // their pawns, plane 6, rank 6
        }
        // Knights: our plane 1 at b1/g1 => perspective (0,1) and (0,6).
        assert_eq!(t[(1 * 8 + 0) * 8 + 1], 1.0);
        assert_eq!(t[(1 * 8 + 0) * 8 + 6], 1.0);
        // Kings on e1/e8.
        assert_eq!(t[(5 * 8 + 0) * 8 + 4], 1.0);
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
        assert_eq!(t[28 * 64 + (20usize ^ 56)], 1.0); // ep plane
        assert_eq!(t[(6 * 8 + 4) * 8 + 4], 1.0); // their pawn on e4
        assert_eq!(t[(0 * 8 + 1) * 8 + 4], 1.0); // our (black) pawn on e7 -> perspective rank 1
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
        assert_eq!(policy_index(&pos, &find(Square::C8, Some(Piece::Queen))), 50 * 73 + 0);
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
