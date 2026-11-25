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

// The C2/C4 search+encoder API surface below is wired to Python by C3/WP-H
// (mcts is not exposed yet); until then their pub items are intentionally
// unreachable from the python module and clippy's dead-code lint is off for
// exactly these four modules (pre-existing on main, kept scoped).
#[allow(dead_code)]
mod encoder;
#[allow(dead_code)]
mod mcts;
#[allow(dead_code)]
mod position;
#[allow(dead_code)]
mod tree;

/// WP-H H2: ONNX Runtime inference (`ort` crate) - behind the `onnx` feature
/// so default builds (cargo test / CI / plain maturin) stay hermetic.
#[cfg(feature = "onnx")]
mod ort_eval;

/// WP-H H3: the Rust self-play game loop (opening sampling, temperature
/// schedule, SAN port, buffer-schema example records). Feature-free core
/// (unit-tested hermetically); the ort-backed python entry point is
/// `selfplay_game` below.
#[allow(dead_code)]
mod selfplay;

use numpy::{ndarray::Array, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayLike};
#[cfg(feature = "onnx")]
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::encoder::POLICY_MOVES;
use crate::mcts::{Evaluator, GameHistory, SearchConfig as RustSearchConfig};
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

/// `encode_batch_v2(fens, histories) -> (N, 104, 8, 8) float32` - bit-exact
/// port of `training.encoder.encode_v2` over replayed histories (WP-B v2
/// layout: 29-plane v1 prefix + 6 deep-history steps + repetition counts +
/// side-to-move flag). Each `histories[i]` is the UCI move stack from the
/// game root to `fens[i]`; rows of one game must be consecutive (see
/// `encoder::encode_batch_v2_rows` for the replay/verification contract).
#[pyfunction]
fn encode_batch_v2<'py>(
    py: Python<'py>,
    fens: Vec<String>,
    histories: Vec<Vec<String>>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let n = fens.len();
    let rows: Vec<[f32; encoder::NUM_PLANES_V2 * 64]> = py
        .detach(|| encoder::encode_batch_v2_rows(&fens, &histories))
        .map_err(PyValueError::new_err)?;
    let mut data = Vec::with_capacity(n * encoder::NUM_PLANES_V2 * 64);
    for row in &rows {
        data.extend_from_slice(row);
    }
    let arr = Array::from_shape_vec((n, encoder::NUM_PLANES_V2, 8, 8), data)
        .expect("N*104*8*8 shape is exact");
    Ok(arr.into_pyarray(py))
}

/// WP-H H3: play ONE full self-play game in pure Rust (rules + encoder v2 +
/// MCTS + ort inference — zero Python per move).
///
/// Mirrors python `self_play_game` (PUCT path, root noise on, temperature
/// schedule, one normalized-visit-count policy target and a mover-perspective
/// {-1, 0, 1} value per ply — the exact `pipeline.py` buffer.add record
/// schema: board_tensor (104, 8, 8) f32, policy (4672,) f32, value f32).
/// `opening_seed` samples the game-root FEN like
/// `training/openings.random_openings` (None = standard start position).
///
/// RNG streams are rust-side (see the `selfplay` module docs): same-seed rust
/// replays are bit-identical; rust vs python games compare structurally
/// (H6(a)). The loop runs with the GIL released, on the CPU EP
/// (`prefer_cuda=false` — CUDA EP selection stays an H4 worker decision).
// The dispatch contract pins this signature (H4 workers call it positionally);
// 9 parameters is the interface, not a smell.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "onnx")]
#[pyfunction]
#[pyo3(signature = (model_path, seed = 0, sims = 64, max_plies = 200, temperature = 1.0, temperature_plies = 30, opening_seed = None, weight_version = 0, forced_playouts = false, fp_k = 2.0, pcr_p_full = 0.0, pcr_fast_sims = 100, soft_resign = false, search_batch = 1))]
fn selfplay_game<'py>(
    py: Python<'py>,
    model_path: &str,
    seed: u64,
    sims: u32,
    max_plies: u32,
    temperature: f64,
    temperature_plies: u32,
    opening_seed: Option<u64>,
    weight_version: u64,
    forced_playouts: bool,
    fp_k: f64,
    pcr_p_full: f64,
    pcr_fast_sims: u32,
    soft_resign: bool,
    search_batch: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let game = py
        .detach(|| -> Result<(selfplay::SelfPlayGame, bool), String> {
            let model = ort_eval::OrtModel::load(
                std::path::Path::new(model_path),
                // A60: GPU inference is the difference between 4.16 ms and
                // 0.13 ms per position (batch 16). Default ON, with the load
                // path's loud CPU fallback when the CUDA EP/dylibs are absent;
                // GAMBIT_ORT_CUDA=0 forces CPU (parity tests, CPU-only boxes).
                std::env::var("GAMBIT_ORT_CUDA").map(|v| v != "0").unwrap_or(true),
            )?;
            let used_cuda = model.used_cuda();
            let mut evaluator = ort_eval::OrtMctsEvaluator::new(model);
            let opening_fen = match opening_seed {
                Some(os) => selfplay::random_opening_fen(os)?,
                None => position::START_FEN.to_string(),
            };
            let root = Position::from_fen(&opening_fen)?;
            let params = selfplay::SelfPlayParams {
                seed,
                num_simulations: sims,
                max_plies,
                temperature,
                temperature_plies,
                forced_playouts,
                fp_k,
                pcr_p_full,
                pcr_fast_sims,
                soft_resign,
                search_batch,
            };
            Ok((selfplay::play_game(&mut evaluator, &root, &params), used_cuda))
        })
        .map_err(PyValueError::new_err)?;
    let (game, used_cuda) = game;

    let dict = PyDict::new(py);
    let examples = PyList::empty(py);
    for ex in &game.examples {
        let record = PyDict::new(py);
        let tensor = Array::from_shape_vec((encoder::NUM_PLANES_V2, 8, 8), ex.planes.to_vec())
            .expect("104*8*8 shape is exact");
        record.set_item("board_tensor", tensor.into_pyarray(py))?;
        record.set_item("policy", ex.policy.clone().into_pyarray(py))?;
        record.set_item("value", ex.value)?;
        record.set_item("policy_masked", ex.policy_masked)?;
        record.set_item("root_value", ex.root_value)?;
        examples.append(record)?;
    }
    dict.set_item("examples", examples)?;
    dict.set_item("moves", &game.sans)?;
    dict.set_item("ucis", &game.ucis)?;
    dict.set_item("fens", &game.fens)?;
    dict.set_item("opening_fen", &game.opening_fen)?;
    dict.set_item("result", &game.result)?;
    dict.set_item("termination", &game.termination)?;
    dict.set_item("weight_version", weight_version)?;
    dict.set_item("ply_count", game.ucis.len())?;
    // A62: requested-vs-actual execution provider, so a CPU fallback is
    // visible in every worker log instead of silently halving throughput.
    dict.set_item("used_cuda", used_cuda)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// WP-F C3: PyO3 search bridge.
// ---------------------------------------------------------------------------

/// Python-facing search config - mirrors `mcts::SearchConfig` (WP-F C2)
/// field-for-field. Construct with keyword defaults; every field is readable
/// and writable so tests can tweak one knob.
#[pyclass(name = "SearchConfig")]
#[derive(Clone)]
pub struct PySearchConfig {
    #[pyo3(get, set)]
    pub cpuct: f64,
    #[pyo3(get, set)]
    pub dirichlet_alpha: f64,
    #[pyo3(get, set)]
    pub dirichlet_epsilon: f64,
    #[pyo3(get, set)]
    pub root_noise: bool,
    #[pyo3(get, set)]
    pub num_simulations: u32,
    #[pyo3(get, set)]
    pub virtual_loss: f64,
    /// Selections per wave (1 = the python-sequential parity mode).
    #[pyo3(get, set)]
    pub batch_size: usize,
    /// Seed for the root Dirichlet draw.
    #[pyo3(get, set)]
    pub seed: u64,
    /// WP1: KataGo forced playouts + policy target pruning at the root.
    #[pyo3(get, set)]
    pub forced_playouts: bool,
    /// WP1: forced-playout scale k in sqrt(k * P * sum(N)).
    #[pyo3(get, set)]
    pub fp_k: f64,
}

#[pymethods]
impl PySearchConfig {
    #[new]
    #[pyo3(signature = (
        *,
        cpuct = 1.5,
        dirichlet_alpha = 0.3,
        dirichlet_epsilon = 0.25,
        root_noise = true,
        num_simulations = 64,
        virtual_loss = 1.0,
        batch_size = 1,
        seed = 0,
        forced_playouts = false,
        fp_k = 2.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        cpuct: f64,
        dirichlet_alpha: f64,
        dirichlet_epsilon: f64,
        root_noise: bool,
        num_simulations: u32,
        virtual_loss: f64,
        batch_size: usize,
        seed: u64,
        forced_playouts: bool,
        fp_k: f64,
    ) -> Self {
        PySearchConfig {
            cpuct,
            dirichlet_alpha,
            dirichlet_epsilon,
            root_noise,
            num_simulations,
            virtual_loss,
            batch_size,
            seed,
            forced_playouts,
            fp_k,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchConfig(cpuct={}, alpha={}, epsilon={}, noise={}, sims={}, vloss={}, batch={}, seed={})",
            self.cpuct,
            self.dirichlet_alpha,
            self.dirichlet_epsilon,
            self.root_noise,
            self.num_simulations,
            self.virtual_loss,
            self.batch_size,
            self.seed
        )
    }
}

impl From<&PySearchConfig> for RustSearchConfig {
    fn from(cfg: &PySearchConfig) -> Self {
        RustSearchConfig {
            cpuct: cfg.cpuct,
            dirichlet_alpha: cfg.dirichlet_alpha,
            dirichlet_epsilon: cfg.dirichlet_epsilon,
            root_noise: cfg.root_noise,
            num_simulations: cfg.num_simulations,
            virtual_loss: cfg.virtual_loss,
            batch_size: cfg.batch_size,
            seed: cfg.seed,
            forced_playouts: cfg.forced_playouts,
            fp_k: cfg.fp_k,
        }
    }
}

/// Extract a flat f32 vector (row-major) from a numpy array or a float list.
///
/// ndarray iteration is in C-order logical order regardless of strides, so
/// (N, 4672) logit batches flatten row-major exactly like python
/// `array.reshape(-1)`. float64 arrays and plain lists are accepted as a
/// convenience for test doubles.
fn extract_f32_vec(obj: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<PyArrayLike<f32, numpy::IxDyn>>() {
        let mut out = Vec::with_capacity(arr.as_array().len());
        for v in arr.as_array().iter() {
            out.push(*v);
        }
        return Ok(out);
    }
    if let Ok(arr) = obj.extract::<PyArrayLike<f64, numpy::IxDyn>>() {
        let mut out = Vec::with_capacity(arr.as_array().len());
        for v in arr.as_array().iter() {
            out.push(*v as f32);
        }
        return Ok(out);
    }
    obj.extract::<Vec<f32>>()
}

/// Bridges the Rust `Evaluator` trait to a Python callable (WP-F C3).
///
/// The callback receives ONE list of FENs per expansion wave (the wave leaves
/// batched - mirroring how `BatchNNEvaluator` aggregates) and returns
/// `(logits, values)`: raw UNMASKED policy logits, row-major (N, 4672), and a
/// scalar value per position from the side to move perspective.
///
/// Python exceptions cannot cross the `Evaluator` trait (it returns plain
/// tuples), so the first failure is captured in `error`, neutral zero rows
/// keep the tree walk well-defined, and the pyfunction re-raises the original
/// error once the search unwinds.
struct PyBatchCallback {
    callback: Py<PyAny>,
    error: Option<PyErr>,
    eval_calls: usize,
    eval_positions: usize,
}

impl PyBatchCallback {
    fn evaluate_via_python(&mut self, batch: &[Position]) -> PyResult<(Vec<f32>, Vec<f32>)> {
        Python::attach(|py| {
            let fens: Vec<String> = batch.iter().map(Position::to_fen).collect();
            let result = self.callback.bind(py).call1((fens,))?;
            let (logits_obj, values_obj): (Bound<PyAny>, Bound<PyAny>) = result.extract()?;
            let logits = extract_f32_vec(&logits_obj)?;
            let values = extract_f32_vec(&values_obj)?;
            Ok((logits, values))
        })
    }
}

impl Evaluator for PyBatchCallback {
    fn evaluate(&mut self, batch: &[Position]) -> (Vec<f32>, Vec<f32>) {
        let n = batch.len();
        if self.error.is_some() {
            // A previous wave already failed; feed neutral rows until the
            // search unwinds and the captured error is re-raised.
            return (vec![0.0; n * POLICY_MOVES], vec![0.0; n]);
        }
        match self.evaluate_via_python(batch) {
            Ok((logits, values)) => {
                if values.len() != n || logits.len() < n * POLICY_MOVES {
                    self.error = Some(PyValueError::new_err(format!(
                        "eval callback returned {} logit floats / {} values for {} positions; expected (N, {POLICY_MOVES}) logits and (N,) values",
                        logits.len(),
                        values.len(),
                        n
                    )));
                    (vec![0.0; n * POLICY_MOVES], vec![0.0; n])
                } else {
                    self.eval_calls += 1;
                    self.eval_positions += n;
                    (logits, values)
                }
            }
            Err(err) => {
                self.error = Some(err);
                (vec![0.0; n * POLICY_MOVES], vec![0.0; n])
            }
        }
    }
}

/// `board` -> (game history, root position). See [`search`].
fn parse_search_root(board: &Bound<PyAny>) -> PyResult<(GameHistory, Position)> {
    if let Ok(fen) = board.extract::<String>() {
        return Ok((GameHistory::empty(), parse(&fen)?));
    }
    let (start_fen, moves): (String, Vec<String>) = board.extract().map_err(|_| {
        PyValueError::new_err("board must be a FEN string or a (start_fen, [uci, ...]) pair")
    })?;
    let start = parse(&start_fen)?;
    let ucis: Vec<&str> = moves.iter().map(String::as_str).collect();
    GameHistory::replay(&start, &ucis).map_err(PyValueError::new_err)
}

/// `search(board, eval_callback, cfg) -> (visit_counts, root_value)` (WP-F C3).
///
/// `board` is the search root:
/// - a FEN string (no game history), or
/// - a `(start_fen, [uci, ...])` pair: the position reached by playing the
///   UCI move list from `start_fen`. The replayed history powers repetition
///   and fifty-move claims exactly like python-chess `claim_draw=True`.
///
/// `eval_callback(fens: list[str]) -> (logits, values)` is called once per
/// expansion wave with the wave leaves batched (always a batch of 1 in the
/// deterministic single-thread mode, `batch_size=1`). Tree work runs under
/// `py.detach` - the GIL is only re-acquired inside the callback, and a
/// callback exception is re-raised after the search unwinds.
///
/// Returns `(visit_counts, root_value)`: a (4672,) float32 visit-count array
/// over the 73x64 policy encoding (python-chess `MCTSSearch.search` shape
/// contract) and the root value from the root side-to-move perspective.
#[pyfunction]
#[pyo3(signature = (board, eval_callback, cfg = None))]
fn search(
    py: Python,
    board: &Bound<PyAny>,
    eval_callback: Bound<PyAny>,
    cfg: Option<&PySearchConfig>,
) -> PyResult<(Py<PyArray1<f32>>, f64)> {
    let (history, root) = parse_search_root(board)?;
    let rust_cfg = match cfg {
        Some(c) => RustSearchConfig::from(c),
        None => RustSearchConfig::default(),
    };
    let mut evaluator = PyBatchCallback {
        callback: eval_callback.unbind(),
        error: None,
        eval_calls: 0,
        eval_positions: 0,
    };
    let result = py.detach(|| mcts::search(&root, &history, &rust_cfg, &mut evaluator));
    if let Some(err) = evaluator.error {
        return Err(err);
    }
    let counts = Array::from_shape_vec((POLICY_MOVES,), result.visit_counts)
        .expect("visit counts are exactly POLICY_MOVES long");
    Ok((counts.into_pyarray(py).unbind(), result.root_value))
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
    m.add_function(wrap_pyfunction!(encode_batch_v2, m)?)?;
    m.add_class::<PySearchConfig>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    #[cfg(feature = "onnx")]
    {
        m.add_class::<ort_eval::PyOrtModel>()?;
        m.add_function(wrap_pyfunction!(ort_eval::ort_dylib_path, m)?)?;
        m.add_function(wrap_pyfunction!(selfplay_game, m)?)?;
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
