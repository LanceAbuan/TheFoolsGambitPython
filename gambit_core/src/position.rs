//! Position wrapper: the `chess` crate (jordanbray) bitboard rules core plus the
//! python-chess-compatible state that its movegen does not expose:
//!
//! - `raw_ep`: the en-passant **target** square, stored unconditionally after any
//!   double pawn push (python-chess `Board.ep_square` semantics). The chess crate
//!   only stores its internal ep square when a capture is pseudo-possible, so the
//!   encoder cannot use the crate's field.
//! - halfmove clock / fullmove number (needed for the fifty-move rule and FEN IO).
//!
//! Castling rights live in the crate board and are maintained identically to
//! python-chess `push()` (bits cleared at from/to squares of every move; a king
//! move clears the mover's whole backrank); read-time cleaning mirrors
//! python-chess `clean_castling_rights()` for the encoder.

use std::str::FromStr;

use chess::{BitBoard, Board, CastleRights, ChessMove, Color, File, MoveGen, Piece, Rank, Square};

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// python-chess `BB_DARK_SQUARES`: a1 is dark, dark iff (rank + file) even.
fn square_is_dark(sq: Square) -> bool {
    let i = sq.to_index();
    (i / 8 + i % 8) % 2 == 0
}

fn square_from_index(i: usize) -> Square {
    Square::make_square(Rank::from_index(i / 8), File::from_index(i % 8))
}

fn square_name(sq: Square) -> String {
    let i = sq.to_index();
    let mut s = String::with_capacity(2);
    s.push((b'a' + (i % 8) as u8) as char);
    s.push((b'1' + (i / 8) as u8) as char);
    s
}

fn parse_square(name: &str) -> Result<Square, String> {
    let b = name.as_bytes();
    if b.len() != 2 || !b[0].is_ascii_lowercase() || !b[1].is_ascii_digit() {
        return Err(format!("bad square name: {name:?}"));
    }
    let file = b[0] - b'a';
    let rank = b[1] - b'1';
    if file > 7 || rank > 7 {
        return Err(format!("bad square name: {name:?}"));
    }
    Ok(square_from_index((rank * 8 + file) as usize))
}

fn parse_promo(c: &str) -> Result<Piece, String> {
    match c {
        "n" => Ok(Piece::Knight),
        "b" => Ok(Piece::Bishop),
        "r" => Ok(Piece::Rook),
        "q" => Ok(Piece::Queen),
        _ => Err(format!("bad promotion piece: {c:?}")),
    }
}

fn piece_char(piece: Piece, color: Color) -> char {
    let lower = match piece {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    };
    if color == Color::White {
        lower.to_ascii_uppercase()
    } else {
        lower
    }
}

/// A chess position with python-chess-compatible observable state.
#[derive(Clone, Copy, Debug)]
pub struct Position {
    board: Board,
    /// En-passant *target* square (the square behind a double-pushed pawn),
    /// python-chess semantics: set unconditionally on any double push, dropped
    /// on every other move.
    raw_ep: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Position {
    /// Parse a FEN (4-6 fields; halfmove defaults to 0, fullmove to 1).
    ///
    /// Input contract: FENs of positions reachable in legal play (or FENs
    /// python-chess accepts). The chess crate rejects unsane boards (missing
    /// kings, overlapping pieces, castling rights without rook/king on their
    /// home squares); python-chess also rejects those in practice.
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut parts: Vec<String> = fen.split_whitespace().map(str::to_string).collect();
        if parts.len() < 4 || parts.len() > 6 {
            return Err(format!(
                "FEN must have 4-6 fields (placement turn castling ep [halfmove fullmove]): {fen:?}"
            ));
        }
        if parts.len() == 4 {
            parts.push("0".into());
        }
        if parts.len() == 5 {
            parts.push("1".into());
        }
        let board = Board::from_str(&parts.join(" "))
            .map_err(|e| format!("invalid FEN {fen:?}: {e}"))?;
        let raw_ep = if parts[3] == "-" {
            None
        } else {
            Some(parse_square(&parts[3])?.to_index() as u8)
        };
        let halfmove_clock: u32 = parts[4].parse().map_err(|_| format!("bad halfmove clock in {fen:?}"))?;
        let fullmove_number: u32 = parts[5].parse().map_err(|_| format!("bad fullmove number in {fen:?}"))?;
        Ok(Position {
            board,
            raw_ep,
            halfmove_clock,
            fullmove_number,
        })
    }

    #[allow(dead_code)] // convenience API; used by unit tests
    pub fn starting() -> Position {
        Position::from_fen(START_FEN).expect("start FEN is valid")
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn raw_ep(&self) -> Option<u8> {
        self.raw_ep
    }

    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    pub fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    pub fn side_to_move(&self) -> Color {
        self.board.side_to_move()
    }

    /// All legal moves as UCI strings (castling as the king's two-square move,
    /// e.g. `e1g1`; promotions as `a7a8q`, including underpromotions), matching
    /// python-chess `Move.uci()` in standard (non-960) mode.
    pub fn legal_moves_uci(&self) -> Vec<String> {
        MoveGen::new_legal(&self.board)
            .map(|m| move_uci(&m))
            .collect()
    }

    pub fn legal_moves(&self) -> Vec<ChessMove> {
        MoveGen::new_legal(&self.board).collect()
    }

    /// Apply a legal move (copy-on-make). The move MUST be legal; use
    /// [`make_move_uci`] for validated input.
    pub fn make(&self, m: ChessMove) -> Position {
        let mover = self.board.side_to_move();
        let moved = self
            .board
            .piece_on(m.get_source())
            .expect("legal move source square holds a piece");
        let captured = self.board.piece_on(m.get_dest());
        let is_pawn = moved == Piece::Pawn;

        // python-chess push(): ep target set on ANY double pawn push.
        let raw_ep = if is_pawn {
            let diff = m.get_dest().to_index() as i32 - m.get_source().to_index() as i32;
            if diff.abs() == 16 {
                Some(((m.get_source().to_index() + m.get_dest().to_index()) / 2) as u8)
            } else {
                None
            }
        } else {
            None
        };

        let new_board = self.board.make_move_new(m);

        // python-chess is_zeroing(move): pawn move or capture.
        let halfmove_clock = if is_pawn || captured.is_some() {
            0
        } else {
            self.halfmove_clock + 1
        };
        let fullmove_number = if mover == Color::Black {
            self.fullmove_number + 1
        } else {
            self.fullmove_number
        };

        Position {
            board: new_board,
            raw_ep,
            halfmove_clock,
            fullmove_number,
        }
    }

    /// Validate a UCI string against the legal move list and apply it.
    pub fn make_move_uci(&self, uci: &str) -> Result<Position, String> {
        let uci = uci.trim().to_ascii_lowercase();
        if uci.len() != 4 && uci.len() != 5 {
            return Err(format!("bad UCI move: {uci:?}"));
        }
        let src = parse_square(&uci[0..2])?;
        let dst = parse_square(&uci[2..4])?;
        let promo = if uci.len() == 5 {
            Some(parse_promo(&uci[4..5])?)
        } else {
            None
        };
        for m in MoveGen::new_legal(&self.board) {
            if m.get_source() == src && m.get_dest() == dst && m.get_promotion() == promo {
                return Ok(self.make(m));
            }
        }
        Err(format!(
            "illegal move {uci} in position {}",
            self.to_fen()
        ))
    }

    /// FEN with the raw en-passant field (always written when set — i.e.
    /// python-chess `board.fen(en_passant="fen")`, which round-trips internal
    /// state; python-chess's default `"legal"` mode *drops* ep squares that no
    /// enemy pawn can capture).
    pub fn to_fen(&self) -> String {
        let mut placement = String::with_capacity(80);
        for rank in (0..8).rev() {
            let mut empty = 0u32;
            for file in 0..8u32 {
                let sq = square_from_index((rank * 8 + file) as usize);
                match self.board.piece_on(sq) {
                    Some(piece) => {
                        if empty > 0 {
                            placement.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let color = self.board.color_on(sq).expect("piece has a color");
                        placement.push(piece_char(piece, color));
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                placement.push_str(&empty.to_string());
            }
            if rank > 0 {
                placement.push('/');
            }
        }
        let turn = if self.board.side_to_move() == Color::White { "w" } else { "b" };
        let mut castling = String::new();
        let white = self.board.castle_rights(Color::White);
        let black = self.board.castle_rights(Color::Black);
        if white.has_kingside() { castling.push('K'); }
        if white.has_queenside() { castling.push('Q'); }
        if black.has_kingside() { castling.push('k'); }
        if black.has_queenside() { castling.push('q'); }
        if castling.is_empty() {
            castling.push('-');
        }
        let ep = self
            .raw_ep
            .map(|sq| square_name(square_from_index(sq as usize)))
            .unwrap_or_else(|| "-".to_string());
        format!(
            "{placement} {turn} {castling} {ep} {} {}",
            self.halfmove_clock, self.fullmove_number
        )
    }

    /// Is the side to move in check?
    pub fn in_check(&self) -> bool {
        *self.board.checkers() != BitBoard(0)
    }

    /// True when the game is over: no legal moves (checkmate/stalemate),
    /// fifty-move rule, or insufficient material.
    ///
    /// Matches python-chess `is_game_over()` for the history-independent
    /// conditions; threefold/fivefold repetition (which needs the move stack)
    /// is intentionally not covered — the MCTS layer owns repetition state.
    pub fn is_terminal(&self) -> bool {
        if MoveGen::new_legal(&self.board).next().is_none() {
            return true; // checkmate or stalemate (these take precedence)
        }
        if self.halfmove_clock >= 100 {
            return true; // python-chess is_fifty_moves() (mate already excluded)
        }
        self.is_insufficient_material()
    }

    /// `"1-0"`, `"0-1"` or `"1/2-1/2"` when the game is over, else `None`.
    /// Semantics as [`Position::is_terminal`] (no repetition claims).
    pub fn outcome(&self) -> Option<&'static str> {
        if !self.is_terminal() {
            return None;
        }
        if MoveGen::new_legal(&self.board).next().is_none() {
            if self.in_check() {
                // Side to move is checkmated.
                return Some(if self.board.side_to_move() == Color::White {
                    "0-1"
                } else {
                    "1-0"
                });
            }
            return Some("1/2-1/2"); // stalemate
        }
        Some("1/2-1/2") // fifty-move or insufficient material
    }

    /// Exact port of python-chess `Board.is_insufficient_material()`.
    pub fn is_insufficient_material(&self) -> bool {
        self.has_insufficient_material(Color::White) && self.has_insufficient_material(Color::Black)
    }

    fn has_insufficient_material(&self, color: Color) -> bool {
        let b = &self.board;
        let own = *b.color_combined(color);
        let opp = *b.color_combined(!color);
        let pawns = *b.pieces(Piece::Pawn);
        let knights = *b.pieces(Piece::Knight);
        let bishops = *b.pieces(Piece::Bishop);
        let rooks = *b.pieces(Piece::Rook);
        let queens = *b.pieces(Piece::Queen);
        let kings = *b.pieces(Piece::King);

        if own & (pawns | rooks | queens) != BitBoard(0) {
            return false;
        }
        // Knights: only insufficient if we have no other pieces (incl. >1 knight)
        // and the opponent has no pawns, knights, bishops or rooks.
        if own & knights != BitBoard(0) {
            return own.popcnt() <= 2 && (opp & !kings & !queens) == BitBoard(0);
        }
        // Bishops: only insufficient if all bishops (both sides) share one square
        // color and nobody has pawns or knights.
        if own & bishops != BitBoard(0) {
            let mut dark = false;
            let mut light = false;
            for sq in bishops {
                if square_is_dark(sq) {
                    dark = true;
                } else {
                    light = true;
                }
            }
            let same_color = !(dark && light);
            return same_color && pawns == BitBoard(0) && knights == BitBoard(0);
        }
        true
    }

    /// Does `color` have the kingside/queenside castling right after
    /// python-chess `clean_castling_rights()` (rook actually on its corner
    /// square, king on e1/e8)? Returns `(kingside, queenside)`.
    pub fn clean_castling_rights(&self, color: Color) -> (bool, bool) {
        let b = &self.board;
        let rights: CastleRights = b.castle_rights(color);
        if rights == CastleRights::NoRights {
            return (false, false);
        }
        let backrank = color.to_my_backrank();
        let king_ok = *b.pieces(Piece::King) & *b.color_combined(color)
            == BitBoard::set(backrank, File::E);
        let pieces = *b.pieces(Piece::Rook) & *b.color_combined(color);
        let ks = rights.has_kingside()
            && king_ok
            && pieces & BitBoard::set(backrank, File::H) != BitBoard(0);
        let qs = rights.has_queenside()
            && king_ok
            && pieces & BitBoard::set(backrank, File::A) != BitBoard(0);
        (ks, qs)
    }
}

pub fn move_uci(m: &ChessMove) -> String {
    let mut s = String::with_capacity(5);
    s.push_str(&square_name(m.get_source()));
    s.push_str(&square_name(m.get_dest()));
    if let Some(promo) = m.get_promotion() {
        s.push(match promo {
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            _ => panic!("invalid promotion piece"),
        });
    }
    s
}

/// Standard perft over the crate's legal movegen.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    for m in MoveGen::new_legal(board) {
        if depth == 1 {
            nodes += 1;
        } else {
            nodes += perft(&board.make_move_new(m), depth - 1);
        }
    }
    nodes
}

/// `perft(fen, depth)` entry point used by tests and the Python binding.
pub fn perft_fen(fen: &str, depth: u32) -> Result<u64, String> {
    let pos = Position::from_fen(fen)?;
    Ok(perft(&pos.board, depth))
}

#[cfg(test)]
mod perft_tests {
    use super::*;

    fn check(fen: &str, expected: &[u64]) {
        let board = Board::from_str(fen).expect("valid test FEN");
        for (depth, &nodes) in expected.iter().enumerate() {
            assert_eq!(perft(&board, depth as u32 + 1), nodes, "{fen} at depth {}", depth + 1);
        }
    }

    #[test]
    fn perft_startpos() {
        check(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[20, 400, 8902, 197_281, 4_865_609],
        );
    }

    #[test]
    fn perft_kiwipete() {
        check(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2039, 97_862],
        );
    }

    #[test]
    fn perft_position3() {
        check("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43_238, 674_624]);
    }

    #[test]
    fn perft_position4() {
        check(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9467, 422_333],
        );
    }

    #[test]
    fn perft_position5() {
        check(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1486, 62_379],
        );
    }

    #[test]
    fn perft_position6() {
        check(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2079, 89_890],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::Square;

    #[test]
    fn fen_roundtrip_start() {
        let pos = Position::from_fen(START_FEN).unwrap();
        assert_eq!(pos.to_fen(), START_FEN);
    }

    #[test]
    fn raw_ep_set_unconditionally() {
        // 1. e4 (no black pawn anywhere near d/e files) — python-chess keeps
        // ep_square = e3; the chess crate would drop it.
        let pos = Position::from_fen(START_FEN).unwrap().make_move_uci("e2e4").unwrap();
        assert_eq!(pos.raw_ep(), Some(Square::E3.to_index() as u8));
        // 1. e4 a6 2. e5?? not a double push — ep must be gone after any other move.
        let pos2 = pos.make_move_uci("a7a6").unwrap();
        assert_eq!(pos2.raw_ep(), None);
        assert!(pos2.to_fen().ends_with(" - 0 2"));
    }

    #[test]
    fn make_move_state() {
        // 1. a2a3 a7a6 2. Ra1-a2 Ra8-a7 (both lose queenside) 3. h2-h3 h7-h6
        // 4. Rh1-h2 (white loses kingside).
        let pos = Position::from_fen(START_FEN).unwrap();
        assert_eq!(pos.to_fen().split(' ').nth(2), Some("KQkq"));
        let pos = pos.make_move_uci("a2a3").unwrap();
        let pos = pos.make_move_uci("a7a6").unwrap();
        let pos = pos.make_move_uci("a1a2").unwrap();
        assert_eq!(pos.to_fen().split(' ').nth(2), Some("Kkq"));
        let pos = pos.make_move_uci("a8a7").unwrap();
        assert_eq!(pos.to_fen().split(' ').nth(2), Some("Kk"));
        assert_eq!(pos.board.piece_on(Square::A7), Some(Piece::Rook));
        assert_eq!(pos.board.color_on(Square::A7), Some(Color::Black));
        let pos = pos.make_move_uci("h2h3").unwrap();
        let pos = pos.make_move_uci("h7h6").unwrap();
        let pos = pos.make_move_uci("h1h2").unwrap();
        assert_eq!(pos.to_fen().split(' ').nth(2), Some("k"));
    }

    #[test]
    fn halfmove_clock() {
        let pos = Position::from_fen(START_FEN).unwrap();
        let pos = pos.make_move_uci("g1f3").unwrap();
        assert_eq!(pos.halfmove_clock(), 1);
        let pos = pos.make_move_uci("g8f6").unwrap();
        assert_eq!(pos.halfmove_clock(), 2);
        let pos = pos.make_move_uci("e2e4").unwrap();
        assert_eq!(pos.halfmove_clock(), 0); // pawn move zeroes
        assert_eq!(pos.fullmove_number(), 2);
    }
}
