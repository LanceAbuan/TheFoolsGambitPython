# Fool's Gambit — Verified Implementation Plan (post-research)

**Status:** verified against code on 2026-11-24 (main @ `7835a51`). Every
research finding below cites the code evidence that confirms it. Nothing in
this plan touches the live run (`second-light-v2`); restarts are gated and
bundled per §5.

---

## 0. Verification audit (finding → code evidence)

| Research finding | Code evidence (verified) | Status |
|---|---|---|
| Policy target = raw visit counts (noise included) | `training/selfplay.py:208-216` — `visit_counts[idx] = child['visit_count']`; rust `mcts.rs:393/536` exports raw counts; only a forced-move shortcut, no forced playouts/pruning | CONFIRMED |
| Ply-capped games are labelled draws | `selfplay.py:1370` "hit the ply cap - score as a draw"; arena variant `selfplay.py:1675` | CONFIRMED |
| Value = single game outcome, WDL CE | `pipeline.py:795-796` `cross_entropy(pred_value, wdl_targets(values))` | CONFIRMED |
| Policy/value loss at parity (1.0/1.0) | `trainer.py:36-37` `POLICY_WEIGHT=1.0 VALUE_WEIGHT=1.0` (v2 optimize loop uses the same helpers) | CONFIRMED |
| Buffer = uniform sampling, no priorities | `trainer.py:126-146` `TrainingBuffer`, `np.random.choice(...)` uniform | CONFIRMED |
| No warmup, no EMA anywhere | `pipeline.py` `LRStepSchedule` = step-decay only; grep for `warmup|ema` → none | CONFIRMED |
| Encoder lacks halfmove clock / material planes | `training/encoder.py` grep `halfmove|clock|fullmove` → none; 104 planes = pieces+history+rep+castling+ep+stm | CONFIRMED |
| Rust self-play rebuilds the tree every move | `gambit_core/src/selfplay.rs:306` `mcts::search(&cur, ...)` per ply, no retention | CONFIRMED |
| ORT sessions use default thread pools | `gambit_core/src/ort_eval.rs` — `SessionBuilder` with EP choice, **no `intra_op_num_threads` config** → ORT default = threads per physical core per session | CONFIRMED |
| CUDA EP code path already exists | `ort_eval.rs:161` `prefer_cuda` tries CUDA first, falls back to CPU — only the local dylib is CPU-only | CONFIRMED (better than assumed) |
| Gate = 16 games @ 0.55, 8-way parallel, paired colors | `selfplay.py` `ArenaConfig(games=16, promote_win_rate=0.55)`, `pipeline.py` night-v2 `parallel=8`, `ladder.py` color-paired openings | CONFIRMED |
| No Syzygy usage; reader available | venv `chess/syzygy.py` exists (python-chess); no repo usage | CONFIRMED |
| Metrics = loss/lr only | `pipeline.py` journal events; no entropy / holdout / value-accuracy | CONFIRMED |
| SE blocks absent; 10×128 = Lc0 canonical minus SE | `training/model.py` `ResidualBlock` = conv-BN-ReLU×2 + add | CONFIRMED |
| Search constants unverified numerically | `mcts.rs` `SearchConfig{cpuct, dirichlet_alpha, dirichlet_epsilon, virtual_loss, batch_size}`; concrete values live in `selfplay.py` defaults — verify in WP1 | TO CHECK in WP1 |

**Verdict: all findings are consistent with the code.** Two upgrades vs the
research assumptions: CUDA inference is one build away, and Syzygy comes free
with the installed python-chess.

---

## 1. Ground rules

- Branch per WP; conventional commits; PR to main; merge when acceptance green.
- **Never** restart/kill the live run mid-iteration except at a WP-approved
  restart point (§5). Never write `runs/<live-run>`, `data/train_*`.
- Every WP: unit tests + the relevant existing suite + a measurable acceptance
  number. Feature flags default OFF (old behavior bit-identical) until gated.
- Worktrees for parallel agents (see `docs/SUBAGENT_PROTOCOL.md`).

## 2. Architecture invariants (do not break)

1. **Model boundary = ONNX file** (`<run>/onnx/model-<v>.onnx` + `latest.json`).
   Rust workers hot-reload between games; never mid-game.
2. **Spool contract** `gambit.selfplay.v1`: one JSONL per game, atomic
   `os.replace`, b64-f32 tensors, `weight_version` tagged. Schema changes bump
   the version string.
3. **Encoder versions are prefixed**: v1 29 planes ⊂ v2 104 planes (first 29);
   v3 must keep v2's 104 as a prefix (see WP11) so warm starts stay trivial.
4. **Gate math suited to samples**: SPRT decides; fixed-N numbers get quoted.
5. **Python owns**: trainer, buffer, gates, checkpoints, resume, server/web.

---

## 3. Implementation waves

| Wave | WPs | Restart needed? |
|---|---|---|
| W1 — search & data quality | WP1, WP2, WP3 | WP1/WP2: yes (rust binary + profile); WP3: no (next iteration) |
| W2 — visibility & free Elo | WP4 (EMA), WP5 (metrics) | WP4: next resume/restart; WP5: live |
| W3 — gate overhaul | WP6 | no (next gate event) |
| W4 — throughput | WP7, WP12 | WP7: restart fleet only |
| W5 — data & measurement | WP8 (Syzygy), WP9 (puzzles) | no |
| W6 — next generation | WP10 (optimizer), WP11 (model) | one bundled restart |

---

## 4. Work packages (code-level)

### WP1 — Policy-target pruning + forced playouts  *(search, both backends)*
**Evidence:** raw visit export confirmed; KataGo ablation ~4% Elo-equivalent.
**Files:** `gambit_core/src/mcts.rs`, `gambit_core/src/selfplay.rs`,
`training/selfplay.py` (python MCTS parity path), `tests/test_training_mcts.py`,
`tests/test_gambit_core.py`.
**Design:**
- `SearchConfig` += `forced_playouts: bool = false`, `fp_k: f64 = 2.0`.
- At root expansion, for every child with `N>0`: quota
  `n_forced = sqrt(fp_k * P(c) * ΣN)`; while `N(c) < n_forced`, selection sets
  `PUCT(c)=∞` (forced visit) — implement in the selection score, not the tree.
- **Target pruning before export:** after search, for each non-best child,
  remove up to `n_forced` visits while it stays below the best child's PUCT;
  drop children pruned to 1 visit. Export the pruned counts.
- Add `SearchResult.visits_raw` and `visits_pruned` for tests.
- Audit current `cpuct`, `dirichlet_alpha/epsilon`, virtual loss values during
  this WP; record them in `docs/RESEARCH_TRAINING.md` (ref: c_puct≈1.1,
  FPU=V−0.2√ΣP, root temp ≥1).
**Tests:** a hand position with a noise-loaded bad child; assert pruned target
mass removed, best move unchanged, python/rust parity on fixed seeds.
**Acceptance:** unit tests; `cargo test`; parity gate exact on fixed seeds;
feature flag off → bit-identical old export.
**Risk/rollback:** flag-off default; one-line config back.

### WP2 — Playout cap randomization (PCR)
**Evidence:** ~2.7× games & value samples at p=0.25/fast=100 (KataGo ablation
1.37×). Confirmed no PCR in code.
**Files:** `gambit_core/src/selfplay.rs`, `scripts/rust_worker.py` args,
`training/rust_workers.py` (schema: add `search_kind: "full"|"fast"` per row),
`training/pipeline.py` (buffer intake ignores policy on fast rows).
**Design:**
- Per move: `full = rng < p_full` (default 0.25) → full sims + noise on;
  else fast sims (100) + **no root noise** (maximize strength).
- Only full-search rows record policy targets; fast rows record
  value-only rows (schema: `policy: null` or zeroed + `policy_masked: true`).
- Buffer: value loss trains on all rows; policy CE only on unmasked rows.
- Opening guard: first 8 plies always full search (target quality matters most
  in the opening).
**Tests:** same-seed determinism; masked rows excluded from policy loss;
throughput bench (games/hr) documented in the PR.
**Acceptance:** games/hr and value-rows/hr both ≥ ~2× current; schema-version
bump `v1`→`v2` documented; `cargo test` + pytest green.
**Risk:** fast rows in isolation were "very surprising" only per KataGo — we
keep them value-only; flag to disable PCR.

### WP3 — Value-target hygiene (three fixes, no restart)
**Evidence:** capped games = "draw" (confirmed); no deblunder; no soft resign.
**Files:** `training/pipeline.py` (`game_outcome` call site),
`training/selfplay.py` (`game_outcome`, `play_match_game`), new
`scripts/deblunder.py`, `training/rust_workers.py` (record fields).
**Design:**
1. **Capped-game target:** when a rust game ends at `max_plies`, the worker
   stores `final_root_value` (MCTS root Q at the last searched position) in the
   record; the trainer uses `value_target = final_root_value` for those rows
   instead of 0. Python path: compute root Q in `play_match_game` and return it.
2. **Soft resignation:** if losing-side root win-prob < 5% for 5 consecutive
   turns → scale sims by `λ=p/0.05` toward fast cap; write rows with
   probability `0.1+0.9λ` (stochastic downweight). Config on profile.
3. **Deblunder (offline):** `scripts/deblunder.py` walks a spool game; for each
   ply, if `|played_move_Q − max_child_Q| > δ` (δ≈0.15) re-label that row's
   value with the search's `max_child_Q` (clamped by 0.99^k decay), policy
   unchanged. Runs before buffer admission (in the drain path, feature flag).
**Tests:** capped-game target uses stored Q; soft-resign weighting
deterministic under seed; deblunder on a synthetic blunder game.
**Acceptance:** unit tests; loss trajectory unchanged in first iteration;
journal records `[TARGETS]` counts per iteration.
**Risk:** deblunder can over-correct; flag on at p<1.0 (e.g. 50% of games) and
compare gate results.

### WP4 — EMA weights (free Elo)
**Evidence:** +30–65 Elo in self-play RL; Lc0 ships SWA nets. No EMA in code.
**Files:** `training/pipeline.py`, `training/resume.py` (state), `scripts/export_onnx.py`.
**Design:**
- `EmaWeights` helper: `theta_ema = decay*theta_ema + (1-decay)*theta` after
  every optimizer step (`decay=0.999`, configurable; enabled flag).
- Persist EMA in run state (resume-safe). After each iteration, export ONNX
  **from EMA weights**; workers play with EMA; gate/benchmark evaluates EMA vs
  the current champion.
- Keep `champion.json` pointing at the EMA checkpoint tag (`-ema` suffix) so
  the lineage is explicit. `scripts/warmstart_v2.py` unaffected.
**Tests:** EMA math; resume roundtrip; export-from-EMA parity on the fixture.
**Acceptance:** ONNX export from EMA verified to 1e-5 vs torch EMA; first two
gates show no regression; tag naming documented.
**Risk:** early in a run EMA lags (research: EMA behind for first few thousand
iterations) → make the gate use `max(ema, current)` score for promotion.

### WP5 — Training-health metrics
**Evidence:** loss/lr only; Lc0 watches entropy, P UL, value accuracy,
train/test divergence.
**Files:** `training/pipeline.py` (per-iteration metrics), `server/api/train.py`
(metrics snapshot), `web-react/src/pages/Watch.tsx` (panel), journal schema.
**Design:** compute on a **held-out slice** (e.g. last 2% of buffer frozen at
iteration start): policy CE, policy entropy, **P UL** (CE vs uniform),
value CE, value accuracy (WDL argmax match), `train_test_gap`. Journal event
`{"kind":"metrics", ...}`. UI: a "Health" card with 4 sparklines + red flag if
`train_test_gap` grows 2 iterations straight.
**Acceptance:** metrics present per iteration; UI renders on mobile; no
measurable slowdown (<2 s/iteration).
**Risk:** none material (read-only).

### WP6 — Two-stage gate + pentanomial ledger
**Evidence:** 16-game gates blind (~785 games for 55%); paired colors already.
**Files:** `training/selfplay.py` (`ArenaConfig`), `training/ladder.py`
(`Arena.match`), `training/pipeline.py` (gate orchestration), `data/arena.jsonl`
schema, `docs/OPTIMIZATION_PLAN.md` note.
**Design:**
- `ArenaConfig` += `screen_games: int = 64`, `screen_threshold: float = 0.52`,
  `confirm_games: int = 0` (off initially), `confirm_lower_bound: float = 0.50`.
- Gate event: play `screen_games` (8-way parallel). If score ≥ threshold and
  **Wilson 95% lower bound > 0.5**, promote (stage 1 only initially).
  Optional stage 2: `confirm_games` on **fresh openings** (new seed namespace)
  requiring LB > `confirm_lower_bound`.
- Ledger row: add `pentanomial: [nLL,nLD,nWLDD,nWD,nWW]` counts (computed from
  paired results) + `wilson_lb`. Keep old fields for compatibility.
- Keep `promote_rd_cap` logic intact.
**Acceptance:** gate duration measured (8-way, 64 games ≈ 10–15 min); a
same-strength candidate does not promote in 20 simulated gates (unit test with
stub evaluators); pentanomial math tested against known vectors.
**Risk:** gates cost ~4× compute; compensate by gating every 2 iterations
(config), never both.

### WP7 — Worker threading + fleet scaling
**Evidence:** ORT default = threads/physical core per session; measured 1.7×
from 1-thread sessions + process-level concurrency.
**Files:** `gambit_core/src/ort_eval.rs`, `scripts/rust_worker.py`,
`scripts/run_rust_workers.py`, docs.
**Design:** builder config `intra_op_num_threads = 1` (configurable),
`session.intra_op.allow_spinning=0`, `inter_op=1`, `ORT_SEQUENTIAL`; worker
count default = physical cores (8 on this box); stagger worker start by 2 s.
**Acceptance:** fleet matrix bench (1/2/4/8 workers × 1/2/4 threads) in PR;
games/hr improvement documented; no quality change (same weights/seeds).
**Risk:** CPU contention with the trainer; cap fleet by observed utilization.

### WP8 — Syzygy targets & endgame harness
**Evidence:** reader present (`chess.syzygy`); ~1 GB 3-4-5 WDL.
**Files:** new `training/tablebase.py`, `training/rust_workers.py` (optional TB
correction at drain), `scripts/tb_bench.py` (endgame accuracy), docs.
**Design:** download 3-4-5 WDL to `data/syzygy/`; at drain time, if
`piece_count ≤ 5`, override `value_target` with TB WDL (perfect label);
optional sampled TB positions added to buffer (perfect-label rows, config).
`tb_bench.py`: enumerate common ≤5-man endgames, score bot move correctness.
**Acceptance:** TB override changes ≤2% of rows in practice; endgame benchmark
numbers recorded; tests on known KRK/KQK positions.
**Risk:** storage; gate behind `tb_enabled` flag.

### WP9 — Puzzle benchmark harness
**Evidence:** puzzle accuracy moves before Elo; Lichess DB is public.
**Files:** new `scripts/bench_puzzles.py`, `data/puzzles.jsonl` (500–1000 from
Lichess DB, 50 themes × difficulty bands), docs.
**Design:** for each puzzle: play bot's move at 128 sims; accept if it equals
Lichess solution or gives mate; Lichess replies played; report %.
**Acceptance:** baseline recorded on current champion; script < 5 min.
**Risk:** none (offline bench).

### WP10 — Optimizer package
**Evidence:** AdamW+wd+warmup+cosine+clip (loss 2.49→1.67 in one study); √b LR
rule; our LR has no warmup and may be hot.
**Files:** `training/pipeline.py` (`LRStepSchedule` → `LRSchedule` with
warmup; optimizer construction), `scripts/train.py` (flags for A/B),
`tests/test_training_pipeline.py`.
**Design:** schedule = `warmup_steps` (500) linear → base_lr → cosine or
step-decay to `min_lr`; `AdamW(weight_decay=1e-4, fused=True)`; grad-clip 1.0.
A/B matrix (config-only, temp runs in `runs/ab-*`): {1e-3, 14.1e-4, 3e-4} ×
{warmup 0/500}. Adopt by lowest loss + first gate score.
**Acceptance:** resume-stable (pure function of absolute step); A/B table in PR;
default profile unchanged until result.
**Risk:** keep legacy schedule flag for rollback.

### WP11 — Next-generation model bundle (ONE restart)
**Evidence:** +96 Elo from planes; SE standard; moves-left anti-shuffle;
attention hybrid wins at scale; history 8 plies right + 5% masking.
**Files:** `training/encoder.py` (v3), `gambit_core/src/encoder.rs` (mirror),
`training/model.py` (SE, moves-left head, optional attention), `pipeline.py`
(profile `night-v3`), `scripts/warmstart_v3.py`, tests/fixtures.
**Design:**
- **Encoder v3:** keep planes 0–103 identical (v2 prefix). Append:
  `halfmove_clock` (normalized 0–1, broadcast), `fullmove_phase` (0–1),
  `material_diff` ×5 (P,N,B,R,Q normalized), `checkers` mask,
  `own_pieces_mask`, `opp_pieces_mask` → 104+9 = 113 planes.
- **Model:** `SqueezeExcitation` in each residual block (Lc0 topology);
  optional 2× square-token attention blocks after the tower (flag);
  `moves_left` head (scalar, Huber loss vs `plies_remaining/normalizer`).
- **Training:** random history masking (5% of rows zero history planes).
- **Warm start:** copy v2 weights into v3 (`strict=False`; new channels
  zero-init; SE/MLH fresh); transfer proof = v2-policy KL on 200 positions.
- **Profile:** `night-v3` = night-v2 + `in_channels=113`, SE on, MLH on,
  encoder_version=3; `value_head="wdl"` unchanged.
**Acceptance:** encoder prefix test (v2 == v3[:, :104]); rust mirror parity;
warm-start KL < 0.01; full suite; one smoke iteration in a temp runs dir.
**Risk:** highest-effort WP; behind flags; bundle only when W1–W3 are stable.

### WP12 — Infra (post-scaling)
- **Subtree reuse** (`selfplay.rs`): keep the played child as the new root,
  retain visits (target ~60%), reset virtual loss; acceptance = same games/hr
  with −20% compute or +sims at equal time.
- **Spool v2**: Arrow IPC memory-mapped records + sparse top-60 policies
  (schema v3); acceptance = ingest CPU −50%, memory −30× per row.
- **Shared inference option**: `ort-superserve` pattern if GPU EP lands and
  per-worker sessions underutilize; measure before adopting.
- **Gumbel A/B**: run search_kind=gumbel for one iteration pair; compare gate
  scores; keep only if ≥.

---

## 5. Restart matrix (bundling policy)

| Change | Mechanism | Restart? |
|---|---|---|
| WP3 target hygiene, WP5 metrics, WP6 gates, WP8/9 benches | hot: flags/config read per iteration or per gate | No |
| WP1/WP2 | rust binary + worker args → fleet restart (workers only; trainer resumes) | Fleet restart, trainer continues |
| WP4 EMA | trainer restart (resume) once | Resume restart |
| WP7 threading | fleet restart | Fleet only |
| WP10 optimizer A/B | temp runs in `runs/ab-*` (separate process) | None on live run |
| WP11 model | **one bundled restart** into `night-v3` (new run, warm-started) | Yes — deliberate generation boundary |

Rule: never two restart-requiring changes in flight; batch model changes per
WP11. Keep `runs/scratch-backup-*` before any restart.

## 6. Measurement protocol (quote-worthy numbers)

- **Primary**: fitted SF-ladder Elo (`scripts/bench_sf.py --mode bench/fit`)
  per promotion; puzzle accuracy (WP9); gate pentanomial rows.
- **Secondary**: games/hr, GPU %, value/policy loss, train/test gap.
- Never quote SPRT-stopped numbers as magnitudes; fixed-N only.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Promotion noise from small gates | WP6 two-stage + Wilson LB; pentanomial |
| Weight staleness in workers | existing sha256 pointer + between-games reload |
| CPU contention (trainer vs 8 workers) | WP7 matrix; cap by utilization; ORT 1-thread |
| EMA lag early | gate uses max(ema, current) until iteration ~2000 |
| Deblunder over-correction | 50% game coverage, gated by results |
| Next-gen regression | warm-start KL proof + temp-run smoke before adoption |

## 8. Task checklist

DONE (merged, acceptance green - PR numbers):
- [x] WP1 forced playouts + target pruning (#175) - audit: cpuct 1.5, Dirichlet 0.3/0.25, virtual loss 1.0 recorded
- [x] WP2 playout cap randomization (#176) - schema v2, masked policy loss
- [x] WP3 value-target hygiene (#177) - soft-Z caps, per-ply root_value, soft resign, deblunder script
- [x] WP4 EMA weights (#178) - export/checkpoint/gate on EMA past iteration 2000, sidecar resume
- [x] WP5 health metrics (#179) - policy CE, entropy, P UL, WDL accuracy journaled
- [x] WP6 two-stage gate (#180) - screen + fresh-opening confirmation, Wilson LB
- [x] WP7 ORT threading (#181) - intra-op 1, no spinning, fleet default 8 workers
- [x] WP8 Syzygy module (#183) - TB value override wired, graceful without tables
- [x] WP9 puzzle benchmark (#184) - Lichess-format scoring semantics
- [x] WP10 optimizer package (#182) - AdamW fused, wd, clip, warmup, cosine

DEFERRED (explicit reasons; ordered for the next session):
- [ ] WP11 next-gen bundle (encoder v3 + SE + MLH + attention flag + warmstart_v3 + night-v3):
      highest-effort package; needs python encoder + Rust mirror parity + warm-start KL proof +
      a bundled generation restart. W1-W3 are now stable, so this is the next work package.
      IMPORTANT: the Rust encoder mirror must land in the same PR set as the python encoder
      (workers cannot run night-v3 without encode_planes_v3).
- [ ] WP12 infra: subtree reuse (~60% retention in selfplay.rs) - next; spool v2 / Arrow IPC +
      sparse policies - after worker scaling measurements; shared inference (ort-superserve) -
      requires the CUDA ORT build first; Gumbel A/B - folded into a future experiment round.

## 8b. Operational notes for the next session
- The live run is `second-light-v2` (night-v2 profile, rust_workers). Enable the WP1/WP2/WP3
  fleet flags at launch: `--forced-playouts --pcr-p-full 0.25 --pcr-fast-sims 100 --soft-resign`.
- EMA + two-stage gate + TB + AdamW are profile-level: enable per run (ema_decay=0.999,
  screen_games=64, tb_path=data/syzygy, optimizer_kind=adamw + lr_warmup_steps=500).
- Syzygy tables: https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/ (~1 GB).
- Puzzle DB: https://database.lichess.org/#puzzles -> data/puzzles.jsonl (WP9 format).
- **Spool drain contract (2026-11-24, PRs #203-#205).** The trainer is a burst
  consumer and the fleet is the bottleneck, so the drain is `newest-first`, skips
  (never reads) games below a 3-iteration staleness horizon, and deletes only
  under disk pressure (`purge_stale_games`, high-water 512 files). If a drain
  yields < half a batch the loop keeps polling instead of training on stale
  replay. Rate facts on this box: fleet ~150 games/hr; a 64-game iteration is
  ~25 min of waiting + ~11 s of GPU; the gate adds minutes (rust backend, 90-min
  cadence). `games_per_iteration` is a drift-WARN resume key (not strict).
- **Staleness incident (2026-11-24).** Oldest-first draining + a 128-file purge
  per drain trained the net on 1056-1536-era games while the fleet was at 2496,
  and once the trainer's version counter raced ahead it deleted live output.
  Nothing was persisted past `state-002400`, so the rollback was free: republish
  `latest.json` -> `model-2496.onnx`, delete the junk ONNX, restart both. If it
  happens again, check `state-*.pt` mtimes before assuming damage.

