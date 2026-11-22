//! `gambit_core` — Rust hot-path core for Fool's Gambit v2.
//!
//! Python extension module exposing:
//! - `Board` — python-chess-compatible legal movegen (via the `chess` crate
//!   bitboard engine): `legal_moves()`, `make_move(uci)`, `in_check()`,
//!   `is_terminal()`, `fen()`, `perft(d)`, `encode()`, `legal_mask()`
//! - `perft(fen, depth)` — perft harness
//! - `encode_batch(fens) -> (N, 29, 8, 8) float32` — bit-exact port of
//!   `training/encoder.py` (`encode_batch(boards, prev_boards=None)`)
//! - `legal_masks(fens) -> (N, 4672) float32` — bit-exact `legal_mask`
//! - `legal_moves(fen)`, `make_move(fen, uci)`, `step_batch(fens, ucis)`,
//!   `legal_move_pairs(fen)` (python-chess-ordered `(policy_index, uci)` pairs
//!   — the M3 expansion fast path), `zobrist(fen)`, `outcome(fen)` — flat
//!   helpers for the later hot-path slice
//!
//! python-chess remains the oracle; see `tests/test_gambit_core.py` for the
//! differential fuzzing gates.

mod encoder;
mod position;

use numpy::{ndarray::Array, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use position::Position;

/// FEN parse helper: `String` error -> Python `ValueError`.
fn parse(fen: &str) -> PyResult<Position> {
    Position::from_fen(fen).map_err(PyValueError::new_err)
}

/// A chess position with python-chess-compatible observable state.
///
/// Construct with `Board(fen)` (defaults to the standard start position) or
/// `Board.from_fen(fen)`. Moves are made copy-on-make: `board.make_move(uci)`
/// returns a NEW Board (the original is untouched), matching the §4.1
/// `play(pos, move) -> pos` contract.
#[pyclass(name = "Board")]
pub struct PyBoard {
    pos: Position,
}

#[pymethods]
impl PyBoard {
    /// Build a Board from a FEN (4-6 fields; halfmove defaults to 0, fullmove
    /// to 1). Raises ValueError on invalid FENs.
    #[new]
    #[pyo3(signature = (fen = position::START_FEN))]
    fn new(fen: &str) -> PyResult<Self> {
        Ok(PyBoard { pos: parse(fen)? })
    }

    /// Same as the constructor; mirrors the §4.1 `from_fen` contract.
    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        Ok(PyBoard { pos: parse(fen)? })
    }

    /// All legal moves as UCI strings (castling = king two-square move,
    /// promotions incl. underpromotions, en passant included).
    fn legal_moves(&self) -> Vec<String> {
        self.pos.legal_moves_uci()
    }

    /// Apply a legal UCI move and return the resulting position. Raises
    /// ValueError when the move is not legal here.
    fn make_move(&self, uci: &str) -> PyResult<Self> {
        self.pos
            .make_move_uci(uci)
            .map(|pos| PyBoard { pos })
            .map_err(PyValueError::new_err)
    }

    /// True when the side to move is in check.
    fn in_check(&self) -> bool {
        self.pos.in_check()
    }

    /// True when the game is over: no legal moves (checkmate/stalemate),
    /// fifty-move rule, or insufficient material. Repetition draws are NOT
    /// covered (the move stack is not tracked; the MCTS layer owns them).
    fn is_terminal(&self) -> bool {
        self.pos.is_terminal()
    }

    /// FEN of this position. The en-passant field is always written when set
    /// (python-chess `fen(en_passant="fen")` semantics) so the raw ep state
    /// round-trips.
    fn fen(&self) -> String {
        self.pos.to_fen()
    }

    /// Count leaf nodes of the legal-move tree to `depth`.
    fn perft(&self, depth: u32) -> u64 {
        position::perft(self.pos.board(), depth)
    }

    /// Encode this position into a (29, 8, 8) float32 tensor
    /// (bit-exact `training.encoder.encode_state(board)`).
    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let planes = encoder::encode_planes(&self.pos);
        let arr = Array::from_shape_vec((encoder::NUM_PLANES, 8, 8), planes.to_vec())
            .expect("29*8*8 shape is exact");
        Ok(arr.into_pyarray(py))
    }

    /// (4672,) float32 mask with 1.0 at every legal move's policy index
    /// (bit-exact `training.encoder.legal_mask(board)`).
    fn legal_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mask = encoder::encode_mask(&self.pos);
        Ok(mask.to_vec().into_pyarray(py))
    }

    /// True when white is to move.
    #[getter]
    fn turn(&self) -> bool {
        self.pos.side_to_move() == chess::Color::White
    }

    /// Halfmove clock since the last capture or pawn move.
    #[getter]
    fn halfmove_clock(&self) -> u32 {
        self.pos.halfmove_clock()
    }

    /// Fullmove number (1-based, increments after black's move).
    #[getter]
    fn fullmove_number(&self) -> u32 {
        self.pos.fullmove_number()
    }

    fn __repr__(&self) -> String {
        format!("Board({})", self.pos.to_fen())
    }
}

/// `perft(fen, depth)` — node count of the legal-move tree from a FEN.
#[pyfunction]
fn perft(fen: &str, depth: u32) -> PyResult<u64> {
    position::perft_fen(fen, depth).map_err(PyValueError::new_err)
}

/// `encode_batch(fens) -> (N, 29, 8, 8) float32` — bit-exact port of
/// `training.encoder.encode_batch(boards, prev_boards=None)` (history planes
/// are zero; the FEN-based API carries no history).
#[pyfunction]
fn encode_batch<'py>(py: Python<'py>, fens: Vec<String>) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let n = fens.len();
    let mut data = vec![0f32; n * encoder::NUM_PLANES * 64];
    let block = encoder::NUM_PLANES * 64;
    let parsed: Result<Vec<[f32; encoder::NUM_PLANES * 64]>, String> = py.detach(|| {
        fens.iter()
            .map(|fen| Position::from_fen(fen).map(|pos| encoder::encode_planes(&pos)))
            .collect()
    });
    for (i, planes) in parsed.map_err(PyValueError::new_err)?.into_iter().enumerate() {
        data[i * block..(i + 1) * block].copy_from_slice(&planes);
    }
    let arr = Array::from_shape_vec((n, encoder::NUM_PLANES, 8, 8), data)
        .expect("N*29*8*8 shape is exact");
    Ok(arr.into_pyarray(py))
}

/// `legal_masks(fens) -> (N, 4672) float32` — binary masks, 1.0 at every legal
/// move's 73x64 policy index (bit-exact port of `training.encoder.legal_mask`).
#[pyfunction]
fn legal_masks<'py>(py: Python<'py>, fens: Vec<String>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n = fens.len();
    let mut data = vec![0f32; n * encoder::POLICY_MOVES];
    let parsed: Result<Vec<[f32; encoder::POLICY_MOVES]>, String> = py.detach(|| {
        fens.iter()
            .map(|fen| Position::from_fen(fen).map(|pos| encoder::encode_mask(&pos)))
            .collect()
    });
    for (i, mask) in parsed.map_err(PyValueError::new_err)?.into_iter().enumerate() {
        data[i * encoder::POLICY_MOVES..(i + 1) * encoder::POLICY_MOVES].copy_from_slice(&mask);
    }
    let arr = Array::from_shape_vec((n, encoder::POLICY_MOVES), data)
        .expect("N*4672 shape is exact");
    Ok(arr.into_pyarray(py))
}

/// `legal_moves(fen)` — legal moves as UCI strings (flat helper).
#[pyfunction]
fn legal_moves(fen: &str) -> PyResult<Vec<String>> {
    Ok(parse(fen)?.legal_moves_uci())
}

/// `legal_move_pairs(fen)` — `(policy_index, uci)` pairs for every legal move,
/// in **python-chess `board.legal_moves` order** (see
/// `Position::legal_moves_python_order`). The M3 self-play fast path builds
/// `training.encoder.legal_move_map` from this in one FFI call; child order
/// drives MCTS tie-breaks, so the order is part of the parity contract.
#[pyfunction]
fn legal_move_pairs(fen: &str) -> PyResult<Vec<(usize, String)>> {
    position::Position::legal_move_pairs_fen(fen).map_err(PyValueError::new_err)
}

/// `make_move(fen, uci)` — apply a legal UCI move, return the new FEN.
#[pyfunction]
fn make_move(fen: &str, uci: &str) -> PyResult<String> {
    parse(fen)?
        .make_move_uci(uci)
        .map(|pos| pos.to_fen())
        .map_err(PyValueError::new_err)
}

/// `step_batch(fens, ucis)` — apply one UCI move to each of N positions in a
/// single FFI call (§4.1 `step_batch`), returning the new FENs. Raises
/// ValueError on the first illegal move.
#[pyfunction]
fn step_batch<'py>(py: Python<'py>, fens: Vec<String>, ucis: Vec<String>) -> PyResult<Vec<String>> {
    if fens.len() != ucis.len() {
        return Err(PyValueError::new_err("fens and ucis must have the same length"));
    }
    let out: Result<Vec<String>, String> = py.detach(|| {
        fens.iter()
            .zip(ucis.iter())
            .map(|(fen, uci)| {
                Position::from_fen(fen)?
                    .make_move_uci(uci)
                    .map(|pos| pos.to_fen())
            })
            .collect()
    });
    out.map_err(PyValueError::new_err)
}

/// `zobrist(fen)` — the chess crate's Zobrist hash of the position
/// (transposition keys / dedupe; stable within this crate version).
#[pyfunction]
fn zobrist(fen: &str) -> PyResult<u64> {
    Ok(parse(fen)?.board().get_hash())
}

/// `outcome(fen)` — `"1-0"`, `"0-1"` or `"1/2-1/2"` when the game is over
/// (checkmate/stalemate/fifty-move/insufficient material), else `None`.
/// Repetition claims are not covered (no move stack).
#[pyfunction]
fn outcome(fen: &str) -> PyResult<Option<&'static str>> {
    Ok(parse(fen)?.outcome())
}

#[pymodule]
fn gambit_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch, m)?)?;
    m.add_function(wrap_pyfunction!(legal_masks, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(legal_move_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(make_move, m)?)?;
    m.add_function(wrap_pyfunction!(step_batch, m)?)?;
    m.add_function(wrap_pyfunction!(zobrist, m)?)?;
    m.add_function(wrap_pyfunction!(outcome, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
