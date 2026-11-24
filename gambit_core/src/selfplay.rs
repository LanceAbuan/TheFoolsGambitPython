//! WP-H H3 — the Rust self-play game loop (zero Python per move).
//!
//! Ports `training/selfplay.py::self_play_game` (PUCT path) end-to-end over
//! the C2 search + WP-B encoder + H2 ort evaluator:
//! - seeded opening sampling mirroring `training/openings.py::random_openings`
//!   (6 uniformly random half-moves per candidate, fresh rng per attempt
//!   `seed + attempt`, unique-FEN retry with the 200x bound)
//! - example records in the EXACT `pipeline.py` buffer.add schema:
//!   board_tensor (104, 8, 8) f32 (encoder v2, history truncated to the LAST
//!   7 plies — python `v2_history_from_stack` semantics, the identical
//!   derivation the search evaluator uses for this position), policy
//!   (4672,) f32 normalized visit counts, value f32 in {-1, 0, 1} from the
//!   mover's perspective (mate decisive, everything else 1/2-1/2)
//! - temperature schedule (`ply < temperature_plies` samples, argmax after)
//!   with visit-count sampling (`select_move_from_visits` semantics)
//! - termination detection: mate / stalemate / fifty-move / threefold
//!   repetition claims / insufficient material
//!   (`is_game_over(claim_draw=True)`), or the ply cap ('max-plies',
//!   scored as a draw)
//! - SAN strings via a python-chess `board.san(move)` port (minimal
//!   disambiguation, castling, promotions, en passant, '+'/'#' suffixes)
//!
//! RNG contract: python consumes a numpy PCG64 stream (Dirichlet draws and
//! move sampling); this loop runs ONE rand StdRng stream (per-search
//! Dirichlet seeds + temperature draws). Same-seed rust vs python games are
//! therefore compared structurally (H6(a): shapes, dtypes, legality, label
//! ranges), not trajectory-wise; same-seed RUST replays are bit-identical.
//! The opening stream is likewise a structural mirror (rand StdRng, not
//! PCG64), deterministic for a fixed `opening_seed`.

use std::collections::HashSet;

use chess::Color;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::encoder::{NUM_PLANES_V2, POLICY_MOVES, encode_planes_v2};
use crate::mcts::{self, Evaluator, GameHistory, SearchConfig, select_by_temperature};
use crate::position::{self, Position};

/// `training/openings.py::OPENING_PLIES` — half-moves of random opening play.
pub const OPENING_PLIES: usize = 6;

/// `training/openings.py::_MAX_OPENING_ATTEMPTS_FACTOR` (unique-FEN retry bound).
const MAX_OPENING_ATTEMPTS_FACTOR: u64 = 200;

/// Driver knobs for one game — the rust mirror of python `SelfPlayConfig`
/// (PUCT path; resign and playout-cap flags stay off — they remain
/// python-side until C3/C5/C6 land).
#[derive(Clone, Debug)]
pub struct SelfPlayParams {
    /// Game rng seed (per-search Dirichlet seeds + move-sampling draws).
    pub seed: u64,
    /// Simulations per ply (python `SearchConfig.num_simulations`).
    pub num_simulations: u32,
    /// Ply cap; capped games end 'max-plies' and score as draws.
    pub max_plies: u32,
    /// Sampling temperature for the first `temperature_plies` plies.
    pub temperature: f64,
    /// Number of plies sampled at `temperature` (argmax afterwards).
    pub temperature_plies: u32,
    /// WP1: KataGo forced playouts + policy target pruning at the root.
    pub forced_playouts: bool,
    /// WP1: forced-playout scale k in sqrt(k * P * sum(N)).
    pub fp_k: f64,
    /// WP2: probability a ply runs the FULL search (KataGo PCR, 0.25).
    /// <= 0 disables PCR entirely (old behavior bit-identical).
    pub pcr_p_full: f64,
    /// WP2: simulations for the cheap (value-only) PCR searches.
    pub pcr_fast_sims: u32,
}

impl Default for SelfPlayParams {
    fn default() -> Self {
        SelfPlayParams {
            seed: 0,
            num_simulations: 64,
            max_plies: 200,
            temperature: 1.0,
            temperature_plies: 30,
            // WP1: off by default = canonical AlphaZero (pre-WP1 behavior).
            forced_playouts: false,
            fp_k: 2.0,
            // WP2: PCR off by default (old behavior bit-identical).
            pcr_p_full: 0.0,
            pcr_fast_sims: 100,
        }
    }
}

/// One training example — the pipeline.py buffer.add record payload.
#[derive(Clone, Debug)]
pub struct SelfPlayExample {
    /// (104, 8, 8) row-major f32 planes (encoder v2, last-7 history).
    pub planes: [f32; NUM_PLANES_V2 * 64],
    /// (4672,) f32 normalized visit counts (sums to ~1; all zeros when the
    /// search ran 0 simulations — python `basis / total if total > 0`).
    pub policy: Vec<f32>,
    /// Game outcome from the MOVER's perspective: -1.0 / 0.0 / +1.0.
    pub value: f32,
    /// WP2: true when this row came from a cheap PCR search and must not
    /// contribute to the policy loss (policy is all zeros; value still trains).
    pub policy_masked: bool,
}

/// One finished game — the rust mirror of python `SelfPlayResult`.
#[derive(Clone, Debug)]
pub struct SelfPlayGame {
    pub examples: Vec<SelfPlayExample>,
    /// SAN per played move (python-chess `board.san` port).
    pub sans: Vec<String>,
    /// UCI per played move.
    pub ucis: Vec<String>,
    /// FEN after each played move (python-chess `board.fen()` semantics:
    /// the ep square only when a legal en-passant capture exists).
    pub fens: Vec<String>,
    /// The game root FEN (sampled opening, or the standard start position).
    pub opening_fen: String,
    /// '1-0' | '0-1' | '1/2-1/2' (mate decisive; capped games are draws).
    pub result: String,
    /// 'game-over' (claimable terminal at a loop boundary) | 'max-plies'.
    pub termination: String,
}

/// FEN in python-chess `board.fen()` semantics: like [`Position::to_fen`] but
/// the ep square is shown ONLY when a legal en-passant capture exists (the
/// raw state still round-trips: parsing the hidden form yields the identical
/// playable position python-chess reconstructs from the same string).
pub fn fen_legal_ep(pos: &Position) -> String {
    let mut parts: Vec<String> = pos.to_fen().split(' ').map(str::to_string).collect();
    if parts[3] != "-" && pos.legal_ep().is_none() {
        parts[3] = "-".to_string();
    }
    parts.join(" ")
}

/// Mirror of `training/openings.random_openings(count=1, seed, plies=6)`:
/// replay 6 uniformly random half-moves from the start position under a fresh
/// rng seeded `seed + attempt` (attempts count from 1), dedupe on the FEN,
/// retry within 200x the count. The rng stream is rand StdRng (NOT numpy's
/// PCG64) — structurally identical, not bit-identical to python; same-seed
/// rust replays are deterministic. Returns the FEN in `board.fen()`
/// (legal-ep) semantics — exactly the string python stores/feeds back.
pub fn random_opening_fen(opening_seed: u64) -> Result<String, String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut attempts: u64 = 0;
    // count is always 1 in the H3 entry point (one opening per game).
    let max_attempts = MAX_OPENING_ATTEMPTS_FACTOR;
    loop {
        if attempts >= max_attempts {
            return Err(format!(
                "random_openings could not find 1 unique position after {attempts} attempts ({OPENING_PLIES} plies each)"
            ));
        }
        attempts += 1;
        let mut rng = StdRng::seed_from_u64(opening_seed.wrapping_add(attempts));
        let mut pos = Position::starting();
        // A uniform 6-ply walk can terminate EARLY: checkmate/stalemate is
        // reachable inside the window (seed 125 -> 1.e3 f6 2.h4 g5 3.Qh5#),
        // where legal_moves() is empty and the old `gen_range(0..0)` panicked
        // the whole worker. python's mirror crashes the same way
        // (`rng.integers(0)` -> ValueError), so retrying is a strictly safer
        // superset: every seed python serves today replays identically, and
        // terminal walks retry under the same 200x bound.
        let mut terminated_early = false;
        for _ in 0..OPENING_PLIES {
            let legal = pos.legal_moves();
            if legal.is_empty() {
                terminated_early = true;
                break;
            }
            let pick: usize = rng.gen_range(0..legal.len());
            pos = pos.make(legal[pick]);
        }
        if terminated_early {
            continue;
        }
        let fen = fen_legal_ep(&pos);
        if seen.insert(fen.clone()) {
            return Ok(fen);
        }
    }
}

fn file_char(idx: usize) -> char {
    (b'a' + (idx % 8) as u8) as char
}

fn rank_char(idx: usize) -> char {
    (b'1' + (idx / 8) as u8) as char
}

/// python-chess `Board.is_capture`: an occupied destination or an en-passant
/// capture (pawn, diagonal, ep target square set).
fn is_capture(pos: &Position, m: &chess::ChessMove) -> bool {
    let board = pos.board();
    if board.piece_on(m.get_dest()).is_some() {
        return true;
    }
    let from = m.get_source().to_index();
    let to = m.get_dest().to_index();
    board.piece_on(m.get_source()) == Some(chess::Piece::Pawn)
        && pos.raw_ep() == Some(to as u8)
        && (from % 8) != (to % 8)
}

/// python-chess `board.san(move)` — castling (`O-O` / `O-O-O`), piece
/// letter with MINIMAL disambiguation under python-chess's exact rules (rank
/// candidate —> file candidate —> fallback), pawn-capture file prefix,
/// promotion suffix, and the '+'/'#' look-ahead on the resulting position.
pub fn san(pos: &Position, m: &chess::ChessMove) -> String {
    let board = pos.board();
    let from = m.get_source().to_index();
    let to = m.get_dest().to_index();
    let piece = board
        .piece_on(m.get_source())
        .expect("san: source square of a legal move holds a piece");
    let capture = is_capture(pos, m);

    let mut san = String::new();
    let is_castle = piece == chess::Piece::King && (from as i32 % 8 - to as i32 % 8).abs() == 2;
    if is_castle {
        // python-chess: queenside iff the destination file is the SMALLER one.
        san.push_str(if to % 8 < from % 8 { "O-O-O" } else { "O-O" });
    } else {
        if piece != chess::Piece::Pawn {
            san.push(position::piece_char(piece, Color::White));
            // Ambiguity candidates: OTHER fully-legal moves of the same piece
            // type onto the same square (python: generate_legal_moves with
            // from_mask = own pieces of this type minus the mover, to_mask).
            let mut others: u64 = 0;
            for cand in pos.legal_moves() {
                if cand.get_dest() == m.get_dest()
                    && cand.get_source() != m.get_source()
                    && board.piece_on(cand.get_source()) == Some(piece)
                {
                    others |= 1u64 << cand.get_source().to_index();
                }
            }
            if others != 0 {
                let mut row = false;
                let mut column = false;
                let rank_mask = 0xFFu64 << (8 * (from / 8));
                let file_mask = 0x0101010101010101u64 << (from % 8);
                if others & rank_mask != 0 {
                    column = true;
                }
                if others & file_mask != 0 {
                    row = true;
                } else {
                    column = true;
                }
                if column {
                    san.push(file_char(from));
                }
                if row {
                    san.push(rank_char(from));
                }
            }
        } else if capture {
            san.push(file_char(from));
        }
        if capture {
            san.push('x');
        }
        san.push_str(&position::square_name(m.get_dest()));
        if let Some(promo) = m.get_promotion() {
            san.push('=');
            san.push(position::piece_char(promo, Color::White));
        }
    }
    let next = pos.make(*m);
    if next.is_checkmate() {
        san.push('#');
    } else if next.in_check() {
        san.push('+');
    }
    san
}

/// python `game_outcome`, white's perspective: mate is decisive (the mated
/// side is the side to move); stalemate, insufficient material, fifty-move /
/// threefold CLAIMS and the ply cap all score as draws (python folds the
/// '*' result into '1/2-1/2' with outcome 0.0). Resign is off (H3 scope).
fn outcome_white(pos: &Position) -> f64 {
    if pos.is_checkmate() {
        if pos.side_to_move() == Color::Black { 1.0 } else { -1.0 }
    } else {
        0.0
    }
}

/// python `game_outcome`'s result string: '1-0'/'0-1' on mate, '1/2-1/2'
/// for every terminal draw AND for the ply cap (python maps '*' to the draw).
fn result_string(pos: &Position) -> String {
    if pos.is_checkmate() {
        if pos.side_to_move() == Color::Black { "1-0" } else { "0-1" }.to_string()
    } else {
        "1/2-1/2".to_string()
    }
}

/// Play ONE self-play game from `opening_root` (fresh search tree per ply,
/// root noise on — python `SearchConfig` defaults). The recorded examples
/// follow python `self_play_game` exactly: one per played ply, v2 planes with
/// the evaluator-identical last-7 history, normalized visit counts as the
/// policy target, game outcome signed by the mover as the value target.
pub fn play_game(
    evaluator: &mut dyn Evaluator,
    opening_root: &Position,
    params: &SelfPlayParams,
) -> SelfPlayGame {
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut history = GameHistory::empty();
    let mut cur = *opening_root;
    let mut pending: Vec<([f32; NUM_PLANES_V2 * 64], Vec<f32>, bool, bool)> = Vec::new();
    let mut sans: Vec<String> = Vec::new();
    let mut ucis: Vec<String> = Vec::new();
    let mut fens: Vec<String> = Vec::new();
    let mut termination = "max-plies";

    for ply in 0..params.max_plies {
        if mcts::is_game_over_claim_draw(&cur, &history) {
            termination = "game-over";
            break;
        }

        // WP2 (KataGo playout cap randomization): most plies run a cheap
        // value-only search; a fraction (pcr_p_full) run the full search and
        // record the policy target. PCR off (<=0) and the first 8 plies
        // always run full, so opening targets stay high quality.
        let full = params.pcr_p_full <= 0.0 || ply < 8 || rng.gen::<f64>() < params.pcr_p_full;
        let sims = if full { params.num_simulations } else { params.pcr_fast_sims.max(1) };

        // Root-noise-on search with a fresh Dirichlet seed per ply drawn from
        // the single game stream (python passes one rng to the search).
        let cfg = SearchConfig {
            num_simulations: sims,
            root_noise: full,
            seed: rng.gen::<u64>(),
            forced_playouts: params.forced_playouts,
            fp_k: params.fp_k,
            ..SearchConfig::default()
        };
        let result = mcts::search(&cur, &history, &cfg, evaluator);
        let legal_pairs = cur.legal_moves_python_order();

        // Policy target: normalized visit counts (python
        // `policy = basis / total if total > 0 else np.zeros_like(basis)`).
        // Fast PCR rows are masked out of the policy loss -> zeros.
        let total: f32 = result.visit_counts.iter().sum();
        let policy: Vec<f32> = if !full {
            vec![0.0; POLICY_MOVES]
        } else if total > 0.0 {
            result.visit_counts.iter().map(|c| c / total).collect()
        } else {
            vec![0.0; POLICY_MOVES]
        };

        // Record the example: v2 planes over the SAME last-7 truncation the
        // search evaluator used (`history.positions` is the chronological
        // pre-root list [P0..P(k-1)]; the game root's own stack is empty,
        // exactly like python's Board(opening_fen) at ply 0).
        let hist7: Vec<Position> = history.positions.iter().rev().take(7).copied().collect();
        pending.push((
            encode_planes_v2(&cur, &hist7),
            policy,
            cur.side_to_move() == Color::White,
            !full,
        ));

        // Temperature schedule + visit-count sampling (python
        // `select_move_from_visits`; tau <= 1e-6 degenerates to argmax).
        let temperature = if ply < params.temperature_plies { params.temperature } else { 0.0 };
        let indices: Vec<usize> = legal_pairs.iter().map(|(idx, _)| *idx).collect();
        let chosen = select_by_temperature(&indices, &result.visit_counts, temperature, &mut rng);
        let mv = legal_pairs
            .iter()
            .find(|(idx, _)| *idx == chosen)
            .map(|(_, m)| *m)
            .expect("temperature selection returned a legal policy index");

        sans.push(san(&cur, &mv));
        ucis.push(position::move_uci(&mv));

        // Advance one ply: Pk leaves the root slot into the history (carrying
        // the OLD root_move as its producing move); the chosen move becomes
        // the next root's `root_move`.
        history.moves.push(history.root_move.take());
        history.positions.push(cur);
        history.root_move = Some(mv);
        cur = cur.make(mv);
        fens.push(fen_legal_ep(&cur));
    }

    let ow = outcome_white(&cur);
    let examples: Vec<SelfPlayExample> = pending
        .into_iter()
        .map(|(planes, policy, white, masked)| {
            let value = if white { ow } else { -ow };
            SelfPlayExample {
                planes,
                policy,
                value: value as f32,
                policy_masked: masked,
            }
        })
        .collect();

    SelfPlayGame {
        examples,
        sans,
        ucis,
        fens,
        opening_fen: fen_legal_ep(opening_root),
        result: result_string(&cur),
        termination: termination.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::policy_index;
    use crate::mcts::find_legal_uci;

    /// Constant-logit stub evaluator: uniform priors, zero values; `bias`
    /// concentrates priors on one policy index (forces a chosen move).
    struct StubEvaluator {
        bias: Option<usize>,
    }

    impl Evaluator for StubEvaluator {
        fn evaluate(&mut self, batch: &[Position]) -> (Vec<f32>, Vec<f32>) {
            let n = batch.len();
            let mut logits = vec![0f32; n * POLICY_MOVES];
            if let Some(idx) = self.bias {
                for row in 0..n {
                    logits[row * POLICY_MOVES + idx] = 10.0;
                }
            }
            (logits, vec![0.0; n])
        }
    }

    fn san_of_fen(fen: &str, uci: &str) -> String {
        let pos = Position::from_fen(fen).expect("test FEN parses");
        let mv = find_legal_uci(&pos, uci).expect("test UCI is legal");
        san(&pos, &mv)
    }

    /// SAN port vs python-chess-verified ground truth (each expectation was
    /// produced by python-chess 1.11 `board.san`, see the WP-H H3 transcript).
    #[test]
    fn san_battery() {
        let cases = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", "e4"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", "Nf3"),
            ("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3", "b1d2", "Nbd2"),
            ("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3", "f3d2", "Nfd2"),
            ("7k/8/8/R7/8/8/8/R3K3 w - - 0 1", "a1a3", "R1a3"),
            ("7k/8/8/R7/8/8/8/R3K3 w - - 0 1", "a5a3", "R5a3"),
            ("8/k7/8/8/4Q2Q/8/K7/7Q w - - 0 1", "e4e1", "Qee1"),
            ("8/k7/8/8/4Q2Q/8/K7/7Q w - - 0 1", "h4e1", "Qh4e1"),
            ("8/k7/8/8/4Q2Q/8/K7/7Q w - - 0 1", "h1e1", "Q1e1"),
            ("8/8/8/3p4/4P3/8/8/K6k w - - 0 1", "e4d5", "exd5"),
            ("rn1qkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", "exd6"),
            ("rnbq1bnk/pppP1ppp/8/8/8/8/PPP1PPPP/RNBQKB1R w KQ - 0 1", "d7c8q", "dxc8=Q"),
            ("8/3P4/5k2/8/8/8/8/4K3 w - - 0 1", "d7d8q", "d8=Q+"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "O-O"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", "O-O-O"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "O-O"),
            ("6k1/5ppp/8/8/8/8/8/K2R4 w - - 0 1", "d1d8", "Rd8#"),
            ("k7/8/8/8/8/8/8/K6R w - - 0 1", "h1h8", "Rh8+"),
            ("4k3/8/8/8/8/8/3R4/3K4 w - - 0 1", "d2d7", "Rd7"),
        ];
        for (fen, uci, expected) in cases {
            assert_eq!(san_of_fen(fen, uci), expected, "san({fen}, {uci})");
        }
    }

    /// `fen_legal_ep`: python-chess `board.fen()` hides a non-capturable ep
    /// square while the raw state keeps it.
    #[test]
    fn fen_legal_ep_hides_uncapturable_ep() {
        let pos = Position::starting().make_move_uci("e2e4").expect("e4");
        assert!(pos.raw_ep().is_some());
        assert!(pos.legal_ep().is_none());
        let raw = pos.to_fen();
        let legal = fen_legal_ep(&pos);
        assert!(raw.contains(" e3 "), "raw FEN keeps the ep square: {raw}");
        assert!(legal.contains(" - "), "board.fen() hides it: {legal}");
        // Round-trip: parsing the legal-ep form reproduces the position.
        let reparsed = Position::from_fen(&legal).expect("legal-ep FEN parses");
        assert_eq!(fen_legal_ep(&reparsed), legal);
    }

    /// Opening sampling: deterministic per seed, structurally mirroring
    /// `random_openings` (6 half-moves, white to move, fullmove 4).
    #[test]
    fn opening_fen_deterministic_and_valid() {
        let a = random_opening_fen(7).expect("opening samples");
        let b = random_opening_fen(7).expect("opening samples");
        assert_eq!(a, b, "same seed must replay the same FEN");
        let pos = Position::from_fen(&a).expect("opening FEN parses");
        assert_eq!(pos.side_to_move(), Color::White, "6 plies -> white to move");
        assert_eq!(pos.fullmove_number(), 4, "three full moves played");
        assert_eq!(fen_legal_ep(&pos), a, "opening FEN round-trips");
        let mut distinct = HashSet::new();
        for seed in 0..20u64 {
            distinct.insert(random_opening_fen(seed).expect("opening samples"));
        }
        assert!(distinct.len() >= 2, "20 seeds must not collapse to one FEN");
    }

    /// Threefold-repetition claims need the FULL game history: the repetition
    /// walk must cross the search-root boundary via `GameHistory::root_move`
    /// (this is the regression gate for the root_move fix — without it the
    /// walk stops at the root and P0/P4 are never counted).
    #[test]
    fn threefold_claim_needs_full_history() {
        let start = Position::starting();
        let knight_round = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let mut ucis: Vec<&str> = Vec::new();
        for _ in 0..2 {
            ucis.extend_from_slice(&knight_round);
        }
        let (history8, pos8) = GameHistory::replay(&start, &ucis).expect("replay");
        assert_eq!(history8.positions.len(), 8, "P0..P7 strictly before the root");
        assert!(history8.root_move.is_some());
        assert!(
            mcts::is_game_over_claim_draw(&pos8, &history8),
            "P0/P4/P8 threefold must be claimable at P8 (python-chess verified)"
        );
        let (history4, pos4) = GameHistory::replay(&start, &knight_round).expect("replay");
        assert!(
            !mcts::is_game_over_claim_draw(&pos4, &history4),
            "only two occurrences at P4 (python-chess verified)"
        );
    }

    /// Fifty-move CLAIM: clock 99 with a quiet move reaching 100 is already a
    /// drawn claim (python `can_claim_fifty_moves` extension; verified against
    /// python-chess). Clock 98 is not.
    #[test]
    fn fifty_move_claim_at_clock_99() {
        for (clock, expected) in [(99u32, true), (98, false)] {
            let fen = format!("k7/8/8/8/1q6/8/6K1/8 w - - {clock} 80");
            let pos = Position::from_fen(&fen).expect("FEN parses");
            assert_eq!(
                mcts::is_game_over_claim_draw(&pos, &GameHistory::empty()),
                expected,
                "clock {clock}"
            );
        }
    }

    /// A capped game: draw result, 'max-plies' termination, one example per
    /// played ply, zero values, normalized policies, legal move chain.
    #[test]
    fn playout_cap_randomization_masks_fast_rows() {
        let root = Position::starting();
        let params = SelfPlayParams {
            seed: 9,
            num_simulations: 8,
            max_plies: 24,
            temperature: 0.0,
            temperature_plies: 30,
            pcr_p_full: 0.5,
            ..SelfPlayParams::default()
        };
        let game = play_game(&mut StubEvaluator { bias: None }, &root, &params);
        assert_eq!(game.examples.len(), 24);
        // Opening guard: the first 8 plies always run the full search.
        assert!(game.examples[..8].iter().all(|e| !e.policy_masked));
        let masked = game.examples.iter().filter(|e| e.policy_masked).count();
        assert!(masked > 0, "PCR p_full=0.5 should produce fast rows");
        for ex in game.examples.iter().filter(|e| e.policy_masked) {
            assert!(ex.policy.iter().all(|c| *c == 0.0), "fast rows carry no policy");
        }
        // PCR disabled (default): every row trains the policy.
        let off = SelfPlayParams { seed: 9, max_plies: 24, num_simulations: 8, ..SelfPlayParams::default() };
        let game_off = play_game(&mut StubEvaluator { bias: None }, &root, &off);
        assert!(game_off.examples.iter().all(|e| !e.policy_masked));
    }

    #[test]
    fn play_game_draws_at_ply_cap() {
        let mut evaluator = StubEvaluator { bias: None };
        let root = Position::starting();
        let params = SelfPlayParams {
            seed: 5,
            num_simulations: 8,
            max_plies: 3,
            temperature: 1.0,
            temperature_plies: 1,
            ..SelfPlayParams::default()
        };
        let game = play_game(&mut evaluator, &root, &params);
        assert_eq!(game.termination, "max-plies");
        assert_eq!(game.result, "1/2-1/2");
        assert_eq!(game.opening_fen, position::START_FEN);
        assert_eq!(game.ucis.len(), 3);
        assert_eq!(game.sans.len(), 3);
        assert_eq!(game.fens.len(), 3);
        assert_eq!(game.examples.len(), 3, "one example per played ply");
        for (i, ex) in game.examples.iter().enumerate() {
            assert_eq!(ex.planes.len(), NUM_PLANES_V2 * 64);
            assert_eq!(ex.policy.len(), POLICY_MOVES);
            let total: f32 = ex.policy.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "policy normalized, got {total}");
            assert_eq!(ex.value, 0.0, "capped games are draws");
            let _ = i;
        }
        // The move chain must replay legally from the opening root.
        let mut pos = Position::from_fen(&game.opening_fen).expect("opening FEN");
        for uci in &game.ucis {
            let mv = find_legal_uci(&pos, uci).expect("recorded move is legal");
            pos = pos.make(mv);
        }
        assert_eq!(fen_legal_ep(&pos), game.fens[2], "final FEN matches the replay");
    }

    /// A forced mate: decisive result, 'game-over' termination, mover-perspective
    /// value +1, and the SAN carries the '#' suffix.
    #[test]
    fn play_game_mate_is_decisive() {
        let root = Position::from_fen("6k1/5ppp/8/8/8/8/8/K2R4 w - - 0 1").expect("FEN");
        let mating = find_legal_uci(&root, "d1d8").expect("Rd8 is legal");
        let mut evaluator = StubEvaluator {
            bias: Some(policy_index(&root, &mating)),
        };
        let params = SelfPlayParams {
            seed: 1,
            num_simulations: 16,
            max_plies: 4,
            temperature: 0.0,
            temperature_plies: 30,
            ..SelfPlayParams::default()
        };
        let game = play_game(&mut evaluator, &root, &params);
        assert_eq!(game.ucis, vec!["d1d8".to_string()]);
        assert_eq!(game.sans, vec!["Rd8#".to_string()]);
        assert_eq!(game.termination, "game-over");
        assert_eq!(game.result, "1-0");
        assert_eq!(game.examples.len(), 1);
        assert_eq!(game.examples[0].value, 1.0, "white delivered mate: +1");
        assert_eq!(game.fens[0], fen_legal_ep(&root.make(mating)));
    }
}

#[cfg(test)]
mod opening_walk_regression {
    //! H4 worker regression: the seeded opening walk must NEVER panic or
    //! return a terminal position (worker seed sweep hit StdRng(126) ->
    //! 1.e3 f6 2.h4 g5 3.Qh5# inside the 6-ply window).

    use super::*;

    #[test]
    fn opening_walk_survives_terminal_reachability() {
        for seed in 0..20_000u64 {
            // No panic (the old code panicked on StdRng(126) == seed 125).
            let fen = random_opening_fen(seed).expect("opening samples within the retry bound");
            let pos = Position::from_fen(&fen).expect("opening FEN parses");
            assert!(
                !pos.legal_moves().is_empty(),
                "seed {seed} returned a terminal opening: {fen}"
            );
        }
    }
}
