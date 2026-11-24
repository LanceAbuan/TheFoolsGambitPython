# Fool's Gambit — v2 Training Optimization Plan (Subagent Dispatch)

**Written:** 2026-11-22 · **Status:** PLANNED — nothing applied. Restart is a separate, user-gated step (WP-G).
**Read this if you are a subagent:** your entire brief is §0 (preamble) + your WP block + §6 metrics. Do not expand scope into other WPs.

---

## 0. Dispatcher preamble — paste verbatim into every subagent prompt

```
Repo: E:\github repos\TheFoolsGambitPython (Windows, PowerShell). Python: run
`.venv\Scripts\python.exe` from repo root with $env:PYTHONPATH set to the repo root.
Git: work on the branch named in your WP; conventional commits; push and open a PR to main;
never push to main. gh CLI: prepend `C:\Program Files\GitHub CLI` to $env:Path (here-strings
are blocked — write PR bodies to a temp file and use --body-file).

HARD BOUNDARIES (all WPs):
- A live training run is in progress (PIDs change; find via Get-CimInstance Win32_Process
  filtering CommandLine like '%train.py%'). NEVER kill, restart, or slow it.
- NEVER write anything under runs\ or to data\train_log.jsonl / train.err.log / train.out.log.
- NEVER touch opencode.json (uncommitted local changes, stay out of every commit).
- GPU work allowed (smoke tests, unit tests) but keep runs < a few minutes; no benches that
  pin the GPU for long periods.
- Tests live in tests\ (e.g. test_training_encoder.py, test_training_mcts.py,
  test_training_pipeline.py, test_gambit_core.py). Run `.venv\Scripts\python.exe -m pytest
  tests\<relevant>.py -x` plus any test you add. Extend the suite for every behavior change.

Report back: files changed, commands run with outcomes, acceptance evidence per criterion,
any deviation from spec and why.
```

## 0b. Dispatch order (dependency-safe)

| Wave | Who | WPs | Why this order |
|---|---|---|---|
| 1 | training-dev | WP-A → WP-B → WP-C → WP-E (sequential; they touch the same files) | avoids merge conflicts |
| 1 (parallel) | rust-dev | WP-F (tree/search immediately; encoder-v2 mirror after WP-B merges) | independent files |
| 2 | training-dev | WP-D (needs WP-B + WP-C merged) | warm-start touches both |
| 3 | orchestrator + **user** | WP-G | needs everything merged + explicit user go |
| 4 | rust-dev | WP-H (needs WP-F merged **and measured**) | full-Rust self-play builds on proven search parity |

---

## 1. Current state — measured facts (2026-11-22)

| Fact | Value | Source |
|---|---|---|
| Live run | `second-light`, iter ~30+/1000, ~1,200 games, champion `ckpt-001536-0768` | `runs/scratch/second-light/` |
| Model | `ModelConfig(in_channels=NUM_PLANES, num_blocks=10, filters=128, value_channels=32, value_fc=512)`; single tanh scalar value head | `training/pipeline.py:112` |
| Input planes | 29 total: t0 pieces ×12 + t−1 pieces ×12 + castling ×4 + en-passant ×1. No repetition planes, 1 history step | `training/encoder.py` |
| Policy | 4,672 moves (73 planes/origin square) | `training/model.py` |
| Profile `night` | iterations=1000, games_per_iteration=32, max_plies=200, sims=400, train_steps=64, batch_size=256, min_examples=64, buffer=100k, parallel_games=16; arena: 128 sims / 16 games | `training/pipeline.py:110-124` |
| Regime | `MIN_PARALLEL_GAMES=1`, `MAX_PARALLEL_GAMES=32`, `learning_rate=1e-3` (constant), batch_size=256 | `training/regimes.py:57-58,135-136` |
| Loss | policy cross-entropy + value MSE + L2, weighted `POLICY_WEIGHT`/`VALUE_WEIGHT` | `training/trainer.py:267-272` |
| Optimizer | `torch.optim.Adam(model.parameters(), lr=regime.learning_rate)` | `training/pipeline.py:303` |
| Eval batching | `QueuedNNEvaluator`, `MAX_BATCH_SIZE=256` | `training/evaluator.py:41` |
| Search | `MCTSSearch`, `NUM_MCTS_THREADS=8`, ThreadPoolExecutor per search | `training/selfplay.py:43,188` |
| GPU utilization | 37–41% — GIL-starved, not GPU-limited | nvidia-smi during run |
| Rust crate | `gambit_core/` has `encoder.rs` + `position.rs`, PyO3 wired; MCTS not ported | `gambit_core/src/` |
| Absolute strength | baseline vs SF recorded in `data/sf_bench.jsonl`; SF ladder calibration (8 node-limited rungs) running in background; `scripts/bench_sf.py --mode bench\|calibrate\|fit` | `scripts/bench_sf.py` |

---

## 2. Work packages

### WP-A — Throughput knobs + `night-v2` profile  ·  owner: training-dev · branch: `feat/opt-a-knobs`
**Files:** `training/pipeline.py` (NIGHT_PROFILE block ~lines 110-124; add LR schedule near the Adam construction at line 303), `training/regimes.py` if a cap needs raising.

Spec — add `NIGHT_V2_PROFILE = TrainingProfile(...)` **alongside** NIGHT_PROFILE (do not edit `night` in place; the live run's config must stay byte-identical for resume safety):
1. `parallel_games: 16 → 32` (exactly at `MAX_PARALLEL_GAMES=32` — no regime change needed).
2. `sims: 400 → 600`; keep arena at 128 sims / 16 games.
3. `train_steps: 64 → 96`; `batch_size: 256 → 512`; `min_examples: 64 → 128`.
4. LR schedule: step decay ×0.5 at iterations 200/400/600/800, gated on absolute iteration number (must be stable under `--resume` from any iteration). Implement as a small helper used by the trainer loop; default profiles keep current constant behavior unless the schedule is enabled.
5. `scripts/train.py`: add/verify a `--profile night-v2` path so the new profile is launchable; smoke profile untouched.

**Acceptance:**
- `python -c "from training.pipeline import NIGHT_V2_PROFILE as p; assert (p.parallel_games, p.sims, p.train_steps, p.batch_size) == (32, 600, 96, 512)"` passes.
- LR helper unit test: iteration 199→1e-3, 200→5e-4, 400→2.5e-4; resume from 350 lands on the right rung.
- `pytest tests/test_training_pipeline.py` green; smoke profile still runs end-to-end.

---

### WP-B — Encoder v2: 104 channels (history + repetition)  ·  owner: training-dev · branch: `feat/opt-b-encoder-v2`
**Files:** `training/encoder.py`, `tests/test_training_encoder.py` (extend), `training/pipeline.py` (NUM_PLANES handling — introduce `NUM_PLANES_V2 = 104` and `ENCODER_VERSION = 2`, stamp both into checkpoint metadata/journal records).

Spec:
1. **Prefix rule (hard requirement):** the first 29 output planes are exactly v1's 29 planes in the same order for the same position. New planes are appended only.
2. v2 layout (**resolved, authoritative** — the prefix rule forces v1's 29 planes first): v1 prefix = t0 pieces 0–11, t−1 pieces 12–23, castling 24–27, en-passant 28; then appended: t−2…t−7 piece planes at 29–100 (block t−k at `29 + 12·(k−2)`), repetition-own 101, repetition-opp 102, side-to-move 103. Inventory check: 96 piece + 2 rep + 4 castling + 1 ep + 1 stm = 104. **Rust mirror (WP-F C4) must use exactly this indexing.**
   - History is most-recent-first (`history[0]` = t−1). Prefix guaranteed by construction: `encode_v2(x)[:29] == encode(x)` always.
   - Padding: non-empty history shorter than 7 pads older slots with the startpos snapshot; **empty** history zeroes all history slots (v1 `prev_board=None` parity, holds the prefix rule at ply 0).
   - Repetition planes: prior occurrences of the current position in history (current excluded), split own-turn (101) vs opp-turn (102), capped at 3, normalized ÷3. Plane 103 = 1.0 when white to move.
3. Keep the v1 `encode` function intact and exported (WP-D's parity proof and any old-checkpoint tooling depend on it).
4. Add `encode_v2(board, history: list[Board]) -> np.ndarray` (or equivalent); history shorter than 7 plies pads with the initial position snapshot (document the choice).

**Acceptance:**
- New test: v1-prefix property — `encode_v2(x)[:29] == encode(x)` for ≥ 50 random positions from real play.
- New test: shape `(104, 8, 8)`; repetition planes increase after a position repeats.
- `pytest tests/test_training_encoder.py` green; rust mirror parity test (`tests/test_gambit_core.py`) may xfail until WP-F lands — mark it explicitly, do not delete it.

---

### WP-C — W/D/L value head  ·  owner: training-dev · branch: `feat/opt-c-wdl-head`
**Files:** `training/model.py` (head), `training/trainer.py` (loss, ~lines 267-272), `training/selfplay.py` (scalar derivation), `tests/test_training_model.py`, `tests/test_training_mcts.py`.

Spec:
1. Value head becomes 3 logits (W, D, L) from side-to-move's perspective; config flag `value_head: "wdl" | "scalar"` — **default `"scalar"`** so every existing checkpoint and test path is untouched.
2. Loss for `wdl`: cross-entropy on the one-hot game result; drop the MSE term for the value when `wdl` (keep policy CE + L2).
3. Search interface unchanged: expose `value = p_win − p_loss` from the softmax wherever `MCTSSearch` currently consumes the tanh scalar.
4. Targets: the game result is already on every self-play example — no new plumbing beyond mapping to the 3-way label.
5. Checkpoint metadata gains `value_head`; loader branches on it (old `scalar` checkpoints load as before).

**Acceptance:**
- Unit test: `wdl` model forward → `(N, 4672)` policy + `(N, 3)` value logits; scalar path identical to today.
- Unit test: loss decreases over 20 steps on a fixed toy batch for `wdl` mode.
- `pytest tests/test_training_model.py tests/test_training_mcts.py` green with default config.

---

### WP-E — Gate fairness (FB-1) + champion.json integrity (FB-2)  ·  owner: training-dev · branch: `feat/opt-e-gate-fixes`
**Files:** `training/selfplay.py` (`play_match_game` call path), `training/pipeline.py` (gate call site ~line 640 region; champion writer), `tests/test_training_gate_promotion.py`, `tests/test_training_pipeline.py`.

Spec:
1. **FB-1:** gate openings are currently effectively deterministic across the 16 games → effective sample size ≈ 2 → promotion on noise (gate 22's 8W/8D/0L). Fix: seed a fresh opening set **per gate event** with `seed = f(run_seed, gate_index)`, balanced colors, and log every opening FEN in the gate record for reproducibility.
2. **FB-2:** `runs/scratch/champion.json` has exhibited tag `ckpt-001536-0768` with `path` pointing at a different file. Fix the writer to derive tag and path from the same resolved checkpoint object, and assert `metadata.tag == Path(metadata.path).stem` before writing.

**Acceptance:**
- Test: two gate events with different `gate_index` produce different opening sets; the same index reproduces the same set (determinism per event).
- Test: champion writer round-trips tag/path consistently; deliberate mismatch raises.
- `pytest tests/test_training_gate_promotion.py tests/test_training_pipeline.py` green.

---

### WP-D — Capacity 12×160 + warm-start loader  ·  owner: training-dev · branch: `feat/opt-d-warmstart` (after WP-B + WP-C merge)
**Files:** new `scripts/warmstart_v2.py`, `training/pipeline.py` (v2 model config), `tests/test_training_model.py`.

Spec:
1. `NIGHT_V2_PROFILE` model becomes `ModelConfig(in_channels=104, num_blocks=12, filters=160, value_channels=48, value_fc=512, value_head="wdl")` (WP-A's profile).
2. `scripts/warmstart_v2.py --src runs/scratch/second-light/checkpoints/ckpt-001536-0768.pt --dst runs/v2-warmstart/ckpt-v2-0000.pt`:
   - Build the v2 model; copy residual tower + policy head weights verbatim (shapes match).
   - Input conv `[128,29,3,3]` → `[128,104,3,3]`: copy old into channels `0:29`, zero-init `29:104` (prefix rule makes this a straight copy).
   - W/D/L head: fresh init (no v1 equivalent). Copy Adam moments for all matching params.
   - Write metadata: `warm_started_from`, `encoder_version: 2`, `value_head: "wdl"`.
   - **Write only to `runs/v2-warmstart/`** — never into `runs/scratch/`.
3. Default path `--src` resolves to the explicit champion file above (not `champion.json` — FB-2 divergence quirk; irrelevant post-WP-E but be explicit).

**Acceptance:**
- Script runs on a **copy** of a v1 checkpoint (test fixture or `runs/v2-warmstart/` only), produces loadable output.
- Proof-of-transfer test: 10 train steps on a small fixed buffer with the warm-started model → policy_loss < 5.0 (fresh start is ≈ ln(4672) ≈ 8.4; live run sits ≈ 3.x).
- `pytest tests/test_training_model.py` green.

---

### WP-F — Rust MCTS backend (`gambit_core`)  ·  owner: rust-dev · branch: `feat/opt-f-rust-mcts`
**Files:** `gambit_core/src/tree.rs` (new), `gambit_core/src/mcts.rs` (new), `gambit_core/src/encoder.rs` (extend to v2 after WP-B merges), `gambit_core/src/lib.rs`, `training/selfplay.py` or new `training/rust_search.py` (PyO3 wrapper), `SearchConfig` flag, `tests/test_gambit_core.py`, `tests/test_training_mcts.py`.

Spec:
1. **C1 tree:** arena-based (`Vec<Node>`), node = {visit count, total value, prior, children, move}. No GIL.
2. **C2 search:** PUCT selection, expansion, backup, virtual loss for in-search parallelism, root Dirichlet noise, temperature + visit-count export — output contract identical to Python `MCTSSearch.search` (visit counts + nn_value tuple).
3. **C3 PyO3 wrapper:** `search(position, eval_callback, cfg)`; tree work runs under `py.allow_threads()`; NN evaluation calls back into Python's `QueuedNNEvaluator` in batches (one callback per expansion wave, batch of requested leaves — mirror how `BatchNNEvaluator` aggregates).
4. **C4 encoder:** after WP-B merges, extend `encoder.rs` to the 104-plane v2 layout and make the py↔rust parity test green (plane-for-plane equality on ≥ 100 positions).
5. **C5 flag:** `search_backend: "python" | "rust"` on `SearchConfig`/profile. **Default `"python"`** — flip only after C6.
6. **C6 gates:** (a) parity: same seeds/fixed positions → Rust root visit distribution matches Python's within tolerance in deterministic single-thread mode; (b) perf: node throughput ≥ 10× Python; (c) e2e smoke: 1 profile-smoke run with backend=rust completes and produces legal games.

**Out of scope:** transposition tables, multi-GPU, replacing `QueuedNNEvaluator`. Python search path stays untouched as fallback. `NUM_MCTS_THREADS` is irrelevant under the rust backend — do not repurpose it.

**Acceptance:** `pytest tests/test_gambit_core.py tests/test_training_mcts.py` green; `cargo test` green in `gambit_core/`; flag default remains `"python"`; PR includes a throughput comparison table (python vs rust, nodes/sec).

**Scope note:** WP-F removes the GIL from the search hot loop only. The per-move game driver and NN inference stay Python. WP-H below is the *complete* shift.

---

### WP-H — Full Rust self-play backend (zero Python in the per-move path)  ·  owner: rust-dev · branch: `feat/opt-h-rust-backend` (after WP-F merges and its numbers are recorded)

**Goal:** self-play workers become pure Rust processes — rules (`position.rs`), encoder v2, MCTS, **and NN inference** — running full games end-to-end with no Python call per move. Python keeps only what it is genuinely the right tool for: the PyTorch training loop, replay buffer, gates, persistence, orchestration, and the web/serve stack. That is the Lc0/KataGo shape (fast native engine + Python training harness). **Principle: Rust owns everything per-move; Python owns everything per-iteration.**

Spec:
1. **H1 — ONNX export:** `scripts/export_onnx.py` — export the v2 `GambitNet` (`torch.onnx.export`) after folding batch-norm. The net is plain conv/BN/ReLU/tanh/linear — no custom ops, so export is clean. Parity test: ONNX vs PyTorch outputs agree to 1e-5 on ≥ 1,000 random positions (policy after softmax + value logits).
2. **H2 — Rust inference:** `gambit_core` gains an `ort`-crate (ONNX Runtime) evaluator loading the exported model. VRAM/device selection via ONNX Runtime CUDA EP. Validate: identical outputs to H1 parity fixture.
3. **H3 — Rust game driver:** `gambit_core/src/selfplay.rs` — complete game loop in Rust: opening sampling (seeded), encoder v2, MCTS with embedded inference, temperature schedule, root noise, result + termination detection, example/position records with visit-count targets matching the Python buffer schema exactly.
4. **H4 — Worker topology + IPC:** `N` independent Rust worker **processes** (true parallelism — no GIL question at all), each streaming finished games/examples to the Python trainer over a local socket or shared-memory ring as JSONL/bytes with a versioned schema. Python trainer consumes unchanged (buffer already takes plain arrays).
   **Weight distribution:** the trainer exports the current model to ONNX (H1 script) every iteration; workers watch the artifact (mtime + hash) and hot-reload **between games** — a game never mixes weights; every game record tags the weight version it used.
5. **H5 — Config flip:** profile gains `selfplay_backend: "python" | "rust"`. Default `"python"`. Flip after H6.
6. **H6 — Gates:** (a) example-parity: one Rust game and one Python game from the same seed/position produce schema-identical example records (values may differ from thread nondeterminism — compare shapes, dtypes, legality, target ranges); (b) perf: games/hr ≥ 2× the WP-F result and GPU ≥ 85% — if GPU < 85% with per-worker embedded inference, the first optimization to try is one shared inference process (workers ship eval requests over IPC), nothing else; (c) 24 h soak: no worker crashes, buffer intake rate stable.

**Explicitly still Python (correctly, permanently):** training loop (PyTorch — GPU-bound, ops release the GIL, single process), replay buffer, gates/ladder, checkpoints, FS-5 resume, server/ + web/. Migrating those buys nothing.

**Risk register:** ONNX runtime version pinning (Windows + CUDA EP); example-schema drift (versioned schema + parity test is the guard); worker crash isolation (supervisor restarts a dead worker; trainer keeps buffered examples).

**Acceptance:** `cargo test` green; parity + perf + soak evidence in the PR; flag default stays `"python"`; measured games/hr and GPU util table vs WP-F numbers in the PR description.
**Status (2026-11-23, WP-H H1-H4 dispatch):**

- **H1 DONE** - `scripts/export_onnx.py`: `models/gambitnet-v2.onnx` (opset 17, dynamic batch) + committed fixture `tests/fixtures/onnx_parity_v2.npz`; parity gate <= 1e-5 on 1,000 positions. Now extended with `--out` / `--checkpoint` (trained-checkpoint export for manual cadence); the default invocation is byte-identical.
- **H2 DONE** - `gambit_core/src/ort_eval.rs` (`--features onnx`): `OrtModel` + `OrtMctsEvaluator`, fixture-validated; CPU EP is the working default, the CUDA probe stays (`cargo test --features onnx`: 33 passed, 1 ignored).
- **H3 DONE** - `gambit_core/src/selfplay.rs` + `gambit_core.selfplay_game` pyfunction: seeded openings, temperature schedule, SAN/FEN python-chess semantics, buffer-schema records, bit-exact same-seed rust replay. PR #150.
- **H3 DONE addendum (found by the H4 worker soak smoke):** the seeded opening walk can reach a TERMINAL position inside the 6-ply window (seed 125 -> 1.e3 f6 2.h4 g5 3.Qh5#) - `random_opening_fen` panicked on `gen_range(0..0)`, killing the worker at game ~121 (`pyo3.PanicException` is a BaseException). Fixed: the rust sampler retries terminal walks under the existing 200x bound (regression test sweeps seeds 0..20,000); `training/openings.py` gets the identical guard (bit-identical for every seed whose walk never terminates early - python previously raised `rng.integers(0)` ValueError there); `scripts/rust_worker.py` catches PanicException explicitly. Post-fix: `selfplay_game` seed sweep 0..2000 zero failures; worker smoke 833 games / exit 0 / zero panics.
- **H4 DONE (flag-off)** - worker topology landed:
  - **Spool IPC** (`training/rust_workers.py`): one JSONL per finished game under `<run-root>/spool/inbox/`, written atomically (temp + `os.replace`); versioned schema `gambit.selfplay.v1` (base64 raw-f32 tensors => bit-exact round-trip); every game tagged `weight_version`; trainer drain claims to `processed/`, quarantines malformed records to `rejected/` (never crashes training).
  - **Worker** (`scripts/rust_worker.py`): pointer watch -> ONE `gambit_core.selfplay_game` -> atomic spool write; 3-strikes exit. **Fleet** (`scripts/run_rust_workers.py`): N processes, per-slot restart budget + backoff, disjoint seed ranges (worker i: `seed_base + i*1e9 + game_index`).  - **Weight hot-reload**: the trainer exports the current net to `<run-root>/onnx/model-<weight_version>.onnx` and publishes `<run-root>/onnx/latest.json` (path + sha256 + version) EVERY iteration on the rust path (after the optimize block); workers watch the POINTER (mtime+size), verify the model sha256 once, and reload BETWEEN games only - a game never mixes weights. Manual cadence alternative: `python scripts/export_onnx.py --checkpoint <ckpt>.pt --out <run-root>/onnx/model-<v>.onnx` + republish pointer; the worker picks it up before its next game.
  - **Requires the `onnx` pip package in the trainer env** (torch 2.15 exporter dependency); the export hook degrades to a log error (workers keep the last good model) if it is missing.
- **H5 DONE (default "python", NOT flipped)** - `TrainingProfile.selfplay_backend: "python" | "rust_workers"` (+ `spool_dir`, `spool_timeout_s`); `run_training` swaps ONLY the game source on the rust path (spool -> `TrainingExample` -> the SAME buffer/train/gate/journal loop). Python path byte-identical; the flag is deliberately NOT part of the FS-5 resume signature (additive, non-strict).
- **H6 gates**:
  - (a) example-parity GREEN - `tests/test_rust_workers.py` (16 tests): rust game through the spool decodes bit-exact; schema parity vs python same-seed `self_play_game` (shapes/dtypes/legality/SAN/FEN/ranges, python-chess oracle on both sides); full `run_training(selfplay_backend="rust_workers")` wiring: spool -> buffer -> train -> export -> pointer.
  - (b) perf MEASURED (CPU EP both sides, same ONNX, identical 160-ply samples, live training co-running - indicative, not the flip gate): `scripts/bench_selfplay_backends.py --games 4 --sims 16 --max-plies 40`: python 63,580 plies/hr (1,590 games/hr); rust 1 proc 165,149 plies/hr (4,129 games/hr, **2.60x**); rust x2 procs 162,728 plies/hr (2.68x, earlier run); rust x4 procs 82,207 plies/hr (1.29x) - ORT intra-op oversubscription (each worker session spawns an all-core thread pool; per-worker thread pinning or the plan's "shared inference process" lever is the next optimization if the flip is taken). The real H6(b) flip gate stays: >= 2x WP-F games/hr on the night-v2 config with GPU >= 85%.
  - (c) 24h soak - OUT OF SCOPE for the dispatch, command documented: `python scripts/run_rust_workers.py --workers 8 --duration-seconds 86400 --models-dir runs/<regime>/<run-id>/onnx --spool-dir runs/<regime>/<run-id>/spool --sims 600 --max-plies 200` (pass: zero restart-exhausted slots, stable `[RUST-SPOOL]` intake rate in the trainer journal).
- **Flip procedure** (orchestrator, after (c) passes): (1) install `onnx` in the training env; (2) stop the current run at an iteration boundary (WP-G pre-flight rules apply); (3) new run (or resume) with a `NIGHT_V2_PROFILE`-derived profile + `selfplay_backend="rust_workers"`, `spool_timeout_s` tuned to the fleet; (4) start the fleet launcher against the run's `onnx/` + `spool/` dirs; (5) watch the first iterations' journal lines for `[RUST-SPOOL]`/`[RUST-EXPORT]` health; (6) keep `search_backend` untouched (arena stays python).

---

### WP-G — Restart + validation  ·  owner: orchestrator (me) · **gated on explicit user go**
**Pre-flight (all must be true):**
- [ ] WP-A…WP-F merged; full `pytest tests/` green.
- [ ] SF calibration finished; `scripts/bench_sf.py --mode fit` baseline Elo recorded.
- [ ] Backup: `Copy-Item -Recurse runs\scratch runs\scratch-backup-<date>`.
- [ ] User said "go" for the restart **at that moment**.

**Procedure:**
1. Wait for an iteration boundary in `data/train_log.jsonl`; `taskkill /PID <worker>` then `<launcher>`; verify last checkpoint's `.pt` + optimizer state exist on disk.
2. `python scripts/warmstart_v2.py` → `runs/v2-warmstart/ckpt-v2-0000.pt`.
3. Relaunch with `--profile night-v2` and the warm-start checkpoint (same launcher pattern as the current run).
4. First 30 min: 1 iteration completes, journal fresh, `policy_loss ≈ 3.x` (not 8.4), GPU ≥ 55%, VRAM stable.
5. After the first v2 gate: `scripts/bench_sf.py --mode bench` + `--mode fit`; compare to baseline (§6).

---

## 6. Success metrics (decision table)

| Metric | Baseline (today) | After WP-A(+D) | After WP-F |
|---|---|---|---|
| GPU utilization | 37–41% | ≥ 55% | ≥ 80% |
| Self-play games/hr | ~65 | ≥ 80 | ≥ 160 |
| Fitted SF-ladder Elo | TBD (calibration running) | ≥ baseline | ≥ baseline |
| Policy loss at resume | ~3.x and falling | starts ~3.x (warm-start proof) | unchanged |
| Gate decision quality | noise-prone (FB-1) | opening-randomized, logged | unchanged |

**Revert rules:** games/hr < 1.1× after 24 h → drop `sims` 600→400, re-measure. Fitted Elo drops after first v2 gate → re-check warm-start loss trajectory before blaming capacity; knobs revert before model reverts.

## 7. What we are NOT doing (and why)

- `sf-distill` regime switch — abandons from-scratch AlphaZero premise; stays available in `regimes.py`.
- Porting the **trainer** to Rust (`tch-rs`/`burn`) — the training step is GPU-bound and single-process, so the GIL is irrelevant there; trading PyTorch's autograd/AMP/debugging ecosystem for zero throughput gain is a net loss. The optimal boundary: **Rust owns everything per-move, Python owns everything per-iteration.** Inference itself does NOT need Python — WP-H moves it into Rust via ONNX Runtime, which is faster than PyTorch eager for this net.
- Transposition tables, nets bigger than 12×160, multi-GPU — scope creep until the above lands and is measured.
- Any UI change — the 20-pass review loop closed the UI overhaul.

## 8. Unrelated open threads (unchanged)

- HF checkpoint backfill — waiting on user's Write token (`training/hf.py`, `LanceAbuan/gambit-scratch`).
- FS-9 Cloudflare tunnel — user-gated.
- SF calibration background job → report fitted Elo when it lands.
