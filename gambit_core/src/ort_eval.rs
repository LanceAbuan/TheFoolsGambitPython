//! WP-H H2 - ONNX Runtime inference for the v2 `GambitNet` (the `ort` crate,
//! `load-dynamic` so the extension module has no link-time dependency).
//!
//! Two faces:
//! - [`OrtModel`]: load `models/gambitnet-v2.onnx`, batched forward on
//!   pre-encoded (N, 104, 8, 8) f32 planes -> (policy logits (N, 4672),
//!   W/D/L value logits (N, 3)). Python bridge: `gambit_core.OrtModel`
//!   (`evaluate`, `evaluate_v2`, `used_cuda`). The H1 parity fixture
//!   (`tests/fixtures/onnx_parity_v2.npz`) is the validation gate (<= 1e-5).
//! - [`OrtMctsEvaluator`]: the [`crate::mcts::Evaluator`] the H3 game loop
//!   searches with - v2-encodes each leaf from its ancestry, truncated to the
//!   LAST 7 positions exactly like python `v2_history_from_stack` (python's
//!   repetition planes only see a 7-ply window; `encode_batch_v2_rows`'s
//!   full-stack bridge is a DIFFERENT contract and must not be used here).
//!
//! CUDA EP status is decided at load time: `prefer_cuda=true` tries
//! [`ort::ep::CUDA::register`] and falls back to CPU (loudly) on any error -
//! ONNX Runtime would otherwise silently fall back, which is exactly what the
//! WP-H report must not do.

use std::path::{Path, PathBuf};

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray4};
use ort::environment::Environment;
use ort::ep::cuda::CUDA;
use ort::ep::{ExecutionProvider, CPU};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::encoder::{encode_planes_v2, encode_batch_v2_rows, NUM_PLANES_V2, POLICY_MOVES};

/// The `(policy (N, 4672), value (N, 3))` pyarray pair `PyOrtModel::evaluate*`
/// returns (factored for clippy::type_complexity).
type PolicyValueArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>);
use crate::mcts::Evaluator;
use crate::position::Position;

/// ONNX graph I/O names as exported by `scripts/export_onnx.py` (H1).
pub const INPUT_NAME: &str = "input";
pub const POLICY_NAME: &str = "policy";
pub const VALUE_NAME: &str = "value";

/// python `v2_history_from_stack` max_steps: the repetition-count window is
/// the LAST 7 prior positions, not the full move stack.
const V2_ENCODE_HISTORY_STEPS: usize = 7;

/// Locate the ONNX Runtime shared library for `load-dynamic` builds.
///
/// Resolution order: `ORT_DYLIB_PATH` (respected untouched) -> the ort-sys
/// build cache under `<crate>/../target/<profile>/build/ort-sys-*/out/`
/// (covers `cargo test` and in-tree wheels) -> `None` (ort then falls back to
/// its default loader search; adjacent-to-.pyd wheel installs should set
/// `ORT_DYLIB_PATH` from python, see `gambit_core.ort_dylib_path`).
pub fn resolve_dylib() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent()?;
    let mut profiles: Vec<String> = Vec::new();
    for extra in ["debug", "release"] {
        if profiles.iter().all(|p| p != extra) {
            profiles.push(extra.to_string());
        }
    }
    let names = [
        "onnxruntime.dll",
        "libonnxruntime.so",
        "libonnxruntime.dylib",
    ];
    for profile in profiles {
        // The copied-dylib layout (WP-H H2): target/<profile>/onnxruntime.dll
        let flat = root.join("target").join(profile.as_str()).join(names[0]);
        if flat.is_file() {
            return Some(flat);
        }
        let build_dir = root.join("target").join(profile).join("build");
        let Ok(entries) = std::fs::read_dir(&build_dir) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("ort-sys-"))
                    .unwrap_or(false)
            })
            .collect();
        dirs.sort();
        for dir in dirs.into_iter().rev() {
            for name in names {
                let candidate = dir.join("out").join(name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// python `value_head_scalar` for a W/D/L head: softmax over the 3 logits,
/// then `p_win - p_loss` (side-to-move perspective), f64 internally.
pub fn wdl_scalar(logits: &[f32; 3]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let ex: Vec<f64> = logits.iter().map(|l| ((*l - max) as f64).exp()).collect();
    let sum: f64 = ex.iter().sum();
    ((ex[0] - ex[2]) / sum) as f32
}

fn wdl_to_scalar_batch(wdl: &[f32]) -> Vec<f32> {
    wdl.as_chunks::<3>()
        .0
        .iter()
        .map(|c| wdl_scalar(&[c[0], c[1], c[2]]))
        .collect()
}

fn ort_err(context: &str, e: impl std::fmt::Display) -> String {
    format!("ort_eval {context}: {e}")
}

/// ONNX Runtime allows exactly ONE Environment per process; every
/// `OrtModel::load` shares it (fresh SESSIONS are cheap and per-load, so H4
/// weight hot-reload keeps working — only the API init + environment are
/// process-global). The dylib is resolved once, on first use.
fn shared_environment() -> Result<Environment, String> {
    static ORT_ENVIRONMENT: std::sync::OnceLock<Result<Environment, String>> =
        std::sync::OnceLock::new();
    ORT_ENVIRONMENT
        .get_or_init(|| {
            let env_builder = match resolve_dylib() {
                Some(path) => ort::init_from(&path).map_err(|e| ort_err("dylib load", e))?,
                None => ort::init(),
            };
            env_builder
                .with_name("gambit-core")
                .build()
                .map_err(|e| ort_err("environment", e))
        })
        .clone()
}

/// A loaded ONNX model + its execution-provider status.
pub struct OrtModel {
    // The environment owns ORT's global thread pool; sessions reference it.
    _environment: Environment,
    session: Session,
    used_cuda: bool,
}

impl OrtModel {
    /// Load `model_path`. With `prefer_cuda`, the CUDA EP is attempted FIRST;
    /// a failed load/registration falls back to the CPU EP with a loud
    /// stderr note (the fallback is ALWAYS visible in the returned flag).
    pub fn load(model_path: &Path, prefer_cuda: bool) -> Result<OrtModel, String> {
        // load-dynamic (always on for our `onnx` feature): the dylib + API
        // init + environment are shared per process (see `shared_environment`);
        // each load builds its own session on top.
        let environment = shared_environment()?;
        let builder = Session::builder(&environment).map_err(|e| ort_err("builder", e))?;
        let (builder, used_cuda) = configure_execution_providers(builder, prefer_cuda)?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ort_err("optimization level", e))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| ort_err(format!("commit {}", model_path.display()).as_str(), e))?;
        Ok(OrtModel {
            _environment: environment,
            session,
            used_cuda,
        })
    }

    /// True when the session runs on the CUDA execution provider.
    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    /// Registered execution provider names (reporting only).
    pub fn providers(&self) -> Vec<&'static str> {
        if self.used_cuda {
            vec!["CUDAExecutionProvider", "CPUExecutionProvider"]
        } else {
            vec!["CPUExecutionProvider"]
        }
    }

    /// Batched forward: `planes` is `n` row-major (104, 8, 8) f32 rows.
    /// Returns (policy logits `n * 4672`, W/D/L value logits `n * 3`).
    pub fn forward(&mut self, planes: &[f32], n: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        debug_assert_eq!(planes.len(), n * NUM_PLANES_V2 * 64);
        let input = Tensor::from_array((
            vec![n as i64, NUM_PLANES_V2 as i64, 8, 8],
            planes.to_vec(),
        ))
        .map_err(|e| ort_err("input tensor", e))?;
        let outputs = self
            .session
            .run(ort::inputs![INPUT_NAME => input])
            .map_err(|e| ort_err("run", e))?;
        let (_, policy) = outputs[POLICY_NAME]
            .try_extract_tensor::<f32>()
            .map_err(|e| ort_err("policy output", e))?;
        let policy = policy.to_vec();
        let (_, value) = outputs[VALUE_NAME]
            .try_extract_tensor::<f32>()
            .map_err(|e| ort_err("value output", e))?;
        let value = value.to_vec();
        Ok((policy, value))
    }

    /// `gambit_core.encode_batch_v2` rows (fen + full UCI stack, the REPLAY
    /// contract) -> forward. Fixture/test bridge; the self-play search uses
    /// [`OrtMctsEvaluator`] instead (last-7 history, not full stacks).
    pub fn evaluate_v2_rows(
        &mut self,
        fens: &[String],
        stacks: &[Vec<String>],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let rows = encode_batch_v2_rows(fens, stacks)?;
        let n = rows.len();
        let mut flat = Vec::with_capacity(n * NUM_PLANES_V2 * 64);
        for row in &rows {
            flat.extend_from_slice(row);
        }
        self.forward(&flat, n)
    }
}

/// CUDA-first registration with explicit, loud CPU fallback (ONNX Runtime's
/// own EP fallback is silent, which is what the WP-H report must not be).
fn configure_execution_providers(
    mut builder: SessionBuilder,
    prefer_cuda: bool,
) -> Result<(SessionBuilder, bool), String> {
    if prefer_cuda {
        match CUDA::default().register(&mut builder) {
            Ok(()) => return Ok((builder, true)),
            Err(e) => {
                eprintln!(
                    "[ort_eval] CUDA EP FAILED to load ({e}); falling back to the CPU EP \
                     (flag this in the WP-H report)"
                );
            }
        }
    }
    CPU::default()
        .register(&mut builder)
        .map_err(|e| ort_err("cpu ep", e))?;
    Ok((builder, false))
}

/// `mcts::Evaluator` over an ONNX model - the H3 game loop's embedded
/// inference. History-aware: each leaf is v2-encoded from its ancestry,
/// truncated to the LAST 7 positions (python `v2_history_from_stack`
/// semantics - repetition planes see a 7-ply window, NOT the full stack).
pub struct OrtMctsEvaluator {
    model: OrtModel,
}

impl OrtMctsEvaluator {
    pub fn new(model: OrtModel) -> OrtMctsEvaluator {
        OrtMctsEvaluator { model }
    }
}

/// Last-7 prior positions of `path` (chronological root..=leaf), most-recent
/// first - python `v2_history_from_stack(board)`.
fn encode_history(path: &[Position]) -> Vec<Position> {
    path[..path.len() - 1]
        .iter()
        .rev()
        .take(V2_ENCODE_HISTORY_STEPS)
        .copied()
        .collect()
}

impl Evaluator for OrtMctsEvaluator {
    fn evaluate(&mut self, batch: &[Position]) -> (Vec<f32>, Vec<f32>) {
        // History-free fallback (empty history: history planes zero). The H3
        // game loop always drives `evaluate_history`; this path only exists
        // so v1-style batch callers keep a defined encoding.
        let n = batch.len();
        let mut flat = vec![0f32; n * NUM_PLANES_V2 * 64];
        for (i, pos) in batch.iter().enumerate() {
            let planes = encode_planes_v2(pos, &[]);
            flat[i * NUM_PLANES_V2 * 64..(i + 1) * NUM_PLANES_V2 * 64].copy_from_slice(&planes);
        }
        let (logits, wdl) = self
            .model
            .forward(&flat, n)
            .unwrap_or_else(|e| panic!("ort_eval inference failed: {e}"));
        (logits, wdl_to_scalar_batch(&wdl))
    }

    fn evaluate_history(&mut self, batch: &[Position], paths: &[Vec<Position>]) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(batch.len(), paths.len());
        let n = batch.len();
        let mut flat = vec![0f32; n * NUM_PLANES_V2 * 64];
        for (i, pos) in batch.iter().enumerate() {
            let history = encode_history(&paths[i]);
            let planes = encode_planes_v2(pos, &history);
            flat[i * NUM_PLANES_V2 * 64..(i + 1) * NUM_PLANES_V2 * 64].copy_from_slice(&planes);
        }
        let (logits, wdl) = self
            .model
            .forward(&flat, n)
            .unwrap_or_else(|e| panic!("ort_eval inference failed: {e}"));
        (logits, wdl_to_scalar_batch(&wdl))
    }
}

// ---------------------------------------------------------------------------
// Python bridge
// ---------------------------------------------------------------------------

/// ONNX Runtime evaluator for the v2 `GambitNet` (WP-H H2).
///
/// Load `models/gambitnet-v2.onnx` and run batched forwards in Rust. The H1
/// parity fixture (`tests/fixtures/onnx_parity_v2.npz`) is the validation
/// gate: outputs must match PyTorch to 1e-5.
#[pyclass(name = "OrtModel")]
pub struct PyOrtModel {
    inner: OrtModel,
}

#[pymethods]
impl PyOrtModel {
    /// Load an ONNX model. `prefer_cuda` tries the CUDA EP first and falls
    /// back to CPU (loudly) when it fails to load.
    #[new]
    #[pyo3(signature = (model_path, prefer_cuda = false))]
    fn new(model_path: &str, prefer_cuda: bool) -> PyResult<Self> {
        let inner = OrtModel::load(Path::new(model_path), prefer_cuda)
            .map_err(PyValueError::new_err)?;
        Ok(PyOrtModel { inner })
    }

    /// True when the session runs on the CUDA EP.
    #[getter]
    fn used_cuda(&self) -> bool {
        self.inner.used_cuda()
    }

    /// Registered execution providers (reporting only).
    #[getter]
    fn providers(&self) -> Vec<&'static str> {
        self.inner.providers()
    }

    /// Batched forward on pre-encoded (N, 104, 8, 8) float32 planes.
    /// Returns (policy_logits (N, 4672) float32, value_logits (N, 3) float32).
    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        planes: PyReadonlyArray4<f32>,
    ) -> PyResult<PolicyValueArrays<'py>> {
        let slice = planes
            .as_slice()
            .map_err(|_| PyValueError::new_err("planes must be C-contiguous"))?;
        let n = planes.as_array().shape()[0];
        let (policy, value) = py
            .detach(|| self.inner.forward(slice, n))
            .map_err(PyValueError::new_err)?;
        let policy =
            Array2::from_shape_vec((n, POLICY_MOVES), policy).expect("n * 4672 is exact");
        let value = Array2::from_shape_vec((n, 3), value).expect("n * 3 is exact");
        Ok((policy.into_pyarray(py), value.into_pyarray(py)))
    }

    /// Batched forward on (fen, full-UCI-stack) rows - the
    /// `gambit_core.encode_batch_v2` replay contract. Same outputs as
    /// `evaluate`.
    fn evaluate_v2<'py>(
        &mut self,
        py: Python<'py>,
        fens: Vec<String>,
        histories: Vec<Vec<String>>,
    ) -> PyResult<PolicyValueArrays<'py>> {
        let (policy, value) = py
            .detach(|| self.inner.evaluate_v2_rows(&fens, &histories))
            .map_err(PyValueError::new_err)?;
        let n = fens.len();
        let policy =
            Array2::from_shape_vec((n, POLICY_MOVES), policy).expect("n * 4672 is exact");
        let value = Array2::from_shape_vec((n, 3), value).expect("n * 3 is exact");
        Ok((policy.into_pyarray(py), value.into_pyarray(py)))
    }
}

/// The resolved ONNX Runtime dylib path (diagnostics for `ORT_DYLIB_PATH`).
#[pyfunction]
pub fn ort_dylib_path() -> Option<String> {
    resolve_dylib().map(|p| p.to_string_lossy().into_owned())
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;

    /// Minimal .npz reader for the committed H1 fixture (np.savez writes
    /// ZIP_STORED members of raw little-endian .npy - no compression, no
    /// external crates needed).
    fn npz_f32_member(archive: &[u8], member: &str) -> (Vec<usize>, Vec<f32>) {
        let wanted = format!("{member}.npy");
        // np.savez entries use data descriptors (sizes live in the central
        // directory), so walk the CENTRAL DIRECTORY, not the local headers.
        let eocd = archive
            .windows(4)
            .rposition(|w| w == [0x50, 0x4b, 0x05, 0x06])
            .expect("zip EOCD");
        let entries = u16::from_le_bytes(archive[eocd + 10..eocd + 12].try_into().unwrap());
        let cd_offset = u32::from_le_bytes(archive[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
        let mut cursor = cd_offset;
        for _ in 0..entries {
            let h = &archive[cursor..];
            assert_eq!(&h[0..4], &[0x50, 0x4b, 0x01, 0x02], "central header");
            let method = u16::from_le_bytes(h[10..12].try_into().unwrap());
            let compressed_size = u32::from_le_bytes(h[20..24].try_into().unwrap()) as usize;
            let name_len = u16::from_le_bytes(h[28..30].try_into().unwrap()) as usize;
            let extra_len = u16::from_le_bytes(h[30..32].try_into().unwrap()) as usize;
            let comment_len = u16::from_le_bytes(h[32..34].try_into().unwrap()) as usize;
            let local_offset = u32::from_le_bytes(h[42..46].try_into().unwrap()) as usize;
            let name = String::from_utf8(h[46..46 + name_len].to_vec()).unwrap();
            if name == wanted {
                assert_eq!(method, 0, "fixture member {member} must be ZIP_STORED");
                let local = &archive[local_offset..];
                let l_name = u16::from_le_bytes(local[26..28].try_into().unwrap()) as usize;
                let l_extra = u16::from_le_bytes(local[28..30].try_into().unwrap()) as usize;
                let data_start = local_offset + 30 + l_name + l_extra;
                return read_npy_f32(&archive[data_start..data_start + compressed_size]);
            }
            cursor += 46 + name_len + extra_len + comment_len;
        }
        panic!("npz member {member} not found");
    }

    fn read_npy_f32(npy: &[u8]) -> (Vec<usize>, Vec<f32>) {
        assert_eq!(&npy[0..6], b"\x93NUMPY", "not an npy member");
        assert_eq!(npy[6], 1, "npy major version");
        let header_len = u16::from_le_bytes(npy[8..10].try_into().unwrap()) as usize;
        let header = String::from_utf8_lossy(&npy[10..10 + header_len]).to_string();
        assert!(header.contains("'<f4'"), "expected f4 dtype: {header}");
        assert!(header.contains("False"), "fortran_order expected False: {header}");
        let shape_part = header.split("'shape':").nth(1).expect("shape key");
        let shape: Vec<usize> = shape_part
            .split('(')
            .nth(1)
            .expect("shape tuple")
            .split(')')
            .next()
            .unwrap()
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = 10 + header_len;
        let mut out = Vec::with_capacity((npy.len() - data_start) / 4);
        for chunk in npy[data_start..].as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(*chunk));
        }
        (shape, out)
    }

    /// H2 gate: the rust ort evaluator reproduces the committed H1 fixture
    /// (PyTorch reference outputs) within 1e-5.
    #[test]
    fn fixture_parity_gate_1e5() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let model_path = root.join("models").join("gambitnet-v2.onnx");
        let fixture_path = root.join("tests").join("fixtures").join("onnx_parity_v2.npz");
        assert!(
            model_path.exists() && fixture_path.exists(),
            "H1 artifacts missing ({model_path:?}, {fixture_path:?}); run scripts/export_onnx.py"
        );
        let mut model = OrtModel::load(&model_path, false).expect("cpu ort session");
        println!(
            "[ort_eval] dylib: {:?}, providers: {:?}",
            resolve_dylib(),
            model.providers()
        );
        let bytes = std::fs::read(&fixture_path).unwrap();
        let (in_shape, inputs) = npz_f32_member(&bytes, "inputs");
        assert_eq!(&in_shape[1..], &[NUM_PLANES_V2, 8, 8]);
        let n = in_shape[0];
        let (pol_shape, policy_ref) = npz_f32_member(&bytes, "policy");
        assert_eq!(pol_shape, vec![n, POLICY_MOVES]);
        let (val_shape, value_ref) = npz_f32_member(&bytes, "value");
        assert_eq!(val_shape, vec![n, 3]);

        let (policy, value) = model.forward(&inputs, n).expect("forward");
        assert_eq!(policy.len(), n * POLICY_MOVES);
        assert_eq!(value.len(), n * 3);
        let dp = policy
            .iter()
            .zip(policy_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let dv = value
            .iter()
            .zip(value_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("[ort_eval] fixture parity: max|dPolicy|={dp:.3e} max|dValue|={dv:.3e} over {n} rows");
        assert!(dp <= 1e-5, "policy parity {dp} > 1e-5");
        assert!(dv <= 1e-5, "value parity {dv} > 1e-5");
    }

    /// CUDA EP status probe (run explicitly: `cargo test --features onnx --
    /// --ignored cuda_ep_probe`). Passes either way; prints the status the
    /// WP-H report needs. Kept #[ignore] so routine test runs never touch the
    /// GPU while training owns it.
    #[test]
    #[ignore]
    fn cuda_ep_probe() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let model_path = root.join("models").join("gambitnet-v2.onnx");
        let model = OrtModel::load(&model_path, true).expect("session load");
        println!(
            "[ort_eval] CUDA EP probe: used_cuda={} providers={:?}",
            model.used_cuda(),
            model.providers()
        );
    }
}
