# Training Research Log — distilled (cycles 1–27)

Continuous research on training/server optimization for Fool's Gambit.
Autonomous loop: research → append → consolidate every 6 h. **Cycles 1–25 are
folded into the distillate; the newest cycles stay verbatim below.** Never
touch the live training run from this loop.

## Distilled knowledge

**Targets & data quality**
- Value targets: blend root-Q with game outcome (soft-Z / z→q falloff); TB-correct
  ≤5-man rows (implemented, WP8); capped games use the final search value (soft-Z,
  implemented WP3), never fake draws; post-game deblunder rescoring
  (`scripts/deblunder.py`, implemented) re-labels low-visit divergences.
- Policy targets: prune forced playouts from the visit distribution (implemented
  WP1, flag off); aux soft-policy head `policy^(1/4)` + moves-left head (night-v3);
  sparse top-60 rows (deferred).
- Sampling: freshness > size (100k window fine); ≤30 rows/game in AZ (we admit all
  plies — A72); 25% SF-labelled mix as volume (+0.05, 50% hurts — A35); policy-
  surprise weighting 70/30 (A11/A24); value-class rebalancing if WDL collapses (A75).
- Staleness is controlled on the SAMPLING side: bound accepted weight versions
  (K=2 gate, A86). Sublinear replay windows are the KataGo-style evolution.
- Search control: replay drawn/regret states (RGSC +47–89 Elo); 25% games vs past
  checkpoints (A27/A81); varied/curriculum openings; end-game-first curriculum is
  an N+1 experiment (A87).
- Termination: never hard-draw caps (done); soft resignation (implemented WP3,
  flag off); value head needs to SEE bad positions — no blunder filtering.

**Search & engine**
- Playout cap randomization: works in Go (~2.7× samples) but the 2026 chess paper
  measured it NET-NEGATIVE in chess (discarded targets, batching tail) — we run it
  ON; A73 requires a matched wall-clock A/B; fallback = staged sim caps 300→800.
- Subtree reuse across moves (~60% retention) is the cheapest sims multiplier
  (WP12, deferred). Graph/transposition search: measured net-negative in chess.
- Root recipe: forced playouts (done), prior temp ≥1, c_puct≈1.1–1.5, FPU=V−0.2√ΣP,
  0.99 per-ply discount (A49), Dirichlet 0.3@25% per search (matches ours).
- Per-tree parallel leaves cost ~6.4 Elo/doubling (235 at 100 sims) — prefer
  inter-game batching (our 8 workers); Gumbel ≈ PUCT at high sims.
- Play-time: tablebase root WDL probe (A80) + small human-derived opening book
  (A82, books help weak engines most); fixed 128 sims fine for humans.

**Optimization & training loop**
- AdamW(wd 1e-4, 2-D+ params only, BN/bias exempt) + warmup + cosine/WSD + clip
  1.0; at small scale lower LR wins (2e-4→2e-6 evidence); ours 1e-3 may be hot
  (A38); √b LR rule for AdamW when batch changes (A55).
- EMA weights (0.995–0.999, horizon ≈1% of budget) — implemented WP4, off; gate/
  export on EMA past iteration 2000 is the free 30–65 Elo.
- Outer self-play iterations subsume inner knobs (maximize iterations, keep games/
  epochs/sims per iteration low — A68); iterations > width; ring ~4 presentations
  per row (A74); sims 128 vs 48 no effect at small scale.
- Metrics: policy entropy, P UL, value accuracy are brittle as raw argmax (our
  live value_acc swings 0.00↔0.59) — use Brier/log-loss/mean-P(true) (A88);
  watch train/test divergence; verify per-head gradient norms once.
- Init hygiene: WDL head gain ~0.01 (flat start), no decay on BN/bias (A76).

**Architecture**
- 10×128 is Lc0's canonical ResNet minus SE. Input planes beat architecture
  (+96 Elo evidence): encoder v3 adds halfmove clock + material planes + masks,
  v2 as strict prefix (WP11). History 8 plies right (0→7 big, 7→31 none); mask
  history 5%. Hybrid conv→attention wins at bigger scale; attention mainly
  improves policy. Chess color-flip symmetry is structural in our encoder.
- For ResNets, RandomPad (zero-init new parts) matches/beats Net2Net — our v3 is
  additive: zero-init new channels/planes, SE near-identity, MLH fresh (A67).
- Small-model scaling: at 1.3M params we are data/search-bound, not capacity-
  bound; widening waits (A69).

**Evaluation & ops**
- 16-game gates are blind (~785 games for a true 55%); two-stage gate implemented
  (WP6, off by default — screen 64 needs enabling); pentanomial rows still TODO;
  SPRT decides, fixed-N measures.
- Puzzle benchmark implemented (WP9); Lichess DB download is the ops step.
  Validation-based checkpoint selection = our gate (correct); draw rate ~58%
  startpos is normal; expected score vs random is the honest sanity check.
- ORT: intra-op=1, no spinning, workers = physical cores (implemented WP7, fleet 8);
  CUDA EP is one `onnxruntime-gpu` install away (CUDA13/cuDNN9 matches torch cu132,
  A60) — log requested vs actual EP (A62).
- Spool v2 (Arrow/shm, sparse policies) for >100 games/hr (WP12, deferred).
- GPU for 24/7: ~280W power cap ≈ no throughput loss, much cooler (A84); TF32
  matmul flag is free on Ampere (A64); bf16 > fp16 if AMP (A65).
- Profiling: torch.profiler → nsys steady-state (GPU-gap triage) → ncu last (A61).

## Actionable queue (re-prioritized 2026-11-24 15:05 vs live state)

Live: iter 20, policy_ce 2.40, value_acc 0.00↔0.59 artifact; champion lineage
promotions working; PCR ON (contested, A73 pending A/B).

**Tier 1 — next code passes (highest measured value)**
- [ ] A88 value metrics fix — Brier/mean-P(true)/score-MAE instead of argmax acc
- [ ] A84 GPU 280W power cap (ops one-liner, reversible)
- [ ] A60+A62 worker CUDA EP + requested-vs-actual EP logging
- [ ] A73 PCR A/B at matched wall-clock (contested in chess) → staged sims 300→800
- [ ] A86 staleness version gate (K=2) at drain

**Tier 2 — data quality**
- [ ] A72 ≤30 rows/game subsample; A75 value rebalancing if collapse returns
- [ ] A11/A24 policy-surprise sampling; A12 sparse policies (with spool v2)
- [ ] A48 subtree reuse; A49 0.99 discount; A70 full-sims 600→800 (PCR p=0.25)
- [ ] A35 25% SF-labelled mix; A30 Syzygy tables download (code done)

**Tier 3 — night-v3 generation (one bundled restart, A54)**
- [ ] WP11: encoder v3 (halfmove/material/masks, v2 prefix) + SE + MLH +
      attention flag; warm-start zero-init (A67); cold-start control
- [ ] A28 optimizer package on (AdamW groups A78, warmup 500, cosine/WSD)
- [ ] A38/A55 LR A/B (1e-3 vs ~1.4e-3 vs 3e-4); A68 iteration-size A/B
- [ ] A63 source→dest policy head ablation; A66 aux value-horizon targets

**Tier 4 — play & infra**
- [ ] A80 play-server TB probe; A82 play opening book; A83 budget policy later
- [ ] A81 league sampling (stratified latest/rolling/archive)
- [ ] WP12 spool v2; shared inference (ort-superserve); Gumbel A/B
- [ ] A53 chess-mask kernels; M2CTS phase experts (far future)

Superseded/done: A19→WP6; A41→WP11; A42→WP1; A29 confirmed; A31 pending spool v2;
A76/A78 folded into WP10/WP11 scope; A79 no action (BN fine).

## Next-cycle topics (un-researched)
- MCTS root move selection at play time: argmax vs temperature for human feel
- SQLite/telemetry compression for long-run journals (ops)
- ONNX fp16 export for play-server latency (with A60)
- leaf batching (SearchConfig.batch_size > 1): quality cost at fixed SIMS
  (waves + virtual loss) vs the games/hr it buys — measure before defaulting
- replay window shape: uniform vs recency-weighted sampling at 100k rows

## Cycle 32 - 2026-11-24

Sources: "Beyond Search-Imitation: Prior-Directed Exploration for Searchless
Chess" (arXiv 2608.27757, entropy-adaptive temperature), "Uncertainty-Guided
Exploration for Efficient AlphaZero Training" (NeurIPS 2025, LCR branching),
"Engineering Efficient Self-Play Chess" (arXiv 2609.37447, resign calibration +
late-game poisoning), KataGo "Accelerating Self-Play Learning in Go" App. D,
LC0 training-data format v6 / adjudication + rescorer docs.

### Findings

1. **Entropy-adaptive temperature is a solved recipe and our WDL head already
   has the input.** tau(s) = H(p_WDL) in [0, ln3]; sample the *played* move from
   mu proportional to pi^(1/tau), so decided positions sharpen to a finish and
   genuinely contested ones keep their exploration. Because tau <= ln3 the
   exponent never drops below 0.91 - it only withdraws stochasticity, never
   flattens the policy. Our ply-indexed tau=1/30-plies is the coarse proxy this
   replaces.
2. **No off-policy correction needed in our loop** (the paper uses V-trace
   because it trains the *searchless* network on the sampled action). We distill
   the search policy pi as a supervised target, so changing how the move is
   sampled only shifts state visitation - no importance weights. This makes A98
   a ~40-line change in selfplay.rs, not a re-derivation.
3. **Uncertainty-weighted data (LCR = 2w(1-w)) is the strongest sample-efficiency
   result**: branching 10 variations/game from high-uncertainty positions gave
   +58.5% sample efficiency on 9x9 Go early and +47.3% on 19x19 late. Too
   invasive for us now (needs per-position branching), but it says the value
   head's own variance is the right signal to spend games on.
4. **Capped/adjudicated games must be flagged, not silently averaged in.** LC0
   stores per-row `adjudicated` / `max-length-exceeded` bits so training can
   filter; the 2026 chess paper *excludes capped-game evidence from resign
   calibration* because the natural outcome is unknown. Their poisoning case:
   27-32% capped games made 36-38% of admitted rows inherit a cutoff target,
   which self-reinforced into worse conversion. We already ship soft-Z +
   `root_value`, but we do not mark the reason a game ended.
5. **KataGo's soft alternative to resigning** (which we copied as WP3) records
   post-collapse samples with probability 0.1+0.9*lambda instead of dropping
   them - i.e. downweight, never hard-filter. Matches our design; worth keeping
   when A99 lands so we do not over-prune the "how to lose" examples.

### Actionable queue additions
- [ ] A98: entropy-adaptive play temperature (tau = H(WDL), D2 clamp at
      tau_min=0.05): log per-game tau mean + decided-position share; A/B vs
      fixed ply-30 schedule at matched wall clock
- [ ] A99: spool records carry an end-reason flag (checkmate / resign-soft /
      ply-cap / repetition) + `capped` exclusion switch for calibration-style
      analysis; never exclude by default

## Cycle 31 — 2026-11-24

Sources: "Reimagining Chess with AlphaZero" (opening-tree entropy, diverse
self-play sampling), qwen-chess walkthrough ch8 (mode-collapse instrumentation:
candidate mix + collapsed groups), engine-monoculture (collision entropy,
effective lines), Chess DNA phase analysis (opening accuracy tracks skill;
endgame accuracy is a decided-game illusion), phase-boundary conventions.

### Findings

1. **Opening diversity has a standard metric**: entropy over the first T plies
   (H(T), effective lines = exp H); collision entropy is the robust estimator
   at depth (unbiased, does not saturate). AlphaZero's variant studies had to
   sample the first 20 plies proportionally to visit counts or self-play games
   would be identical under fixed time controls.
2. **Mode collapse is detectable by instrumentation, not vibes**: log the
   candidate move-type mix and per-group variance; the qwen walkthrough caught
   a wing-pawn fixation (51-66% of candidates) that the reward could not see.
   For us the analogue is opening mix: one line of play = collapsed.
3. **Phase accuracy is a trap**: opening accuracy climbs cleanly with skill
   (70->84%), middlegame barely moves, and endgame accuracy is INFLATED for
   weak players (decided games) - stronger players post more endgame-technique
   mistakes because they reach contested endgames. Measure endgame strength by
   mistake counts on contested positions, never raw accuracy.
4. Phase boundaries (standard): opening = first 20 half-moves; endgame =
   combined non-pawn material <= 26 points; else middlegame.

### Actionable queue additions
- [ ] A96: opening-diversity health metric - collision entropy over the first
      12-20 plies of sampled self-play games (effective-lines), journaled;
      alert if it collapses toward a single line
- [ ] A97: phase-conditional SF-bench breakdown - outcomes split by phase
      reached + endgame-technique mistake counts on contested positions
      (report alongside the fitted Elo, never as raw phase accuracy)

## Cycle 30 — 2026-11-24

Sources: RL live-monitor toolkits (rl-doctor detectors: entropy collapse with
projected arrival, robust-slope plateau, gradient pathology), Lc0 training-data
adjudication/rescoring (DeepWiki), "Engineering Efficient Self-Play Chess"
§4.2.5-4.2.6 (resignation calibration, ply-cap targets, late-game poisoning).

### Findings

1. **Robust statistics beat OLS for curve alerts**: Theil-Sen slopes +
   Mann-Kendall significance survive outlier steps; conditional firing rules
   (entropy falling while raw score rises = convergence, not collapse) are what
   keep monitors from crying wolf.
2. **Ply-cap targets matter**: the 2026 chess run compared target estimators at
   capped positions - a dedicated FULL search at the actual cut position was
   best (Brier 0.444 vs 0.491 material heuristic; fresh search beat adjacent
   cheap-search values). We currently reuse the last in-game search value.
3. **Never exclude late-game rows**: excluding the endgame tail to save compute
   caused "late-game target poisoning" - conversion failures fed more capped
   games, a self-reinforcing loop (27-32 % of games capped, 36 % rows inherited
   cutoff targets). We admit all plies - already compliant.
4. Lc0's resignation calibration (20 % never-resign continuation games,
   false-resign UCB <= 2.5 %) is the principled way to tune soft resign; our
   soft_resign flag is not yet calibrated that way.

### Actionable queue additions
- [ ] A93: capped-game target = dedicated full search at the cut position
      (rust worker; replaces the reused in-game root value)
- [ ] A94: soft-resign calibration - log hypothetical-trigger outcomes on
      continuation games, require a low false-resign rate before enabling
- [ ] A95: optional robust plateau flag (Theil-Sen + conditional rules) on
      policy_ce/gate score in the health heartbeat

## Cycle 29 — 2026-11-24

Sources: Lc0 issue #67 (full temperature setting: tau 0.25 sweet spot for
games vs humans), Blundermind bot reference (Maia temperature presets, 70%
sampled / 30% top-move split), SQLite WAL docs + telemetry practice posts
(batched ingest, autocheckpoint, TRUNCATE, retention/downsampling).

### Findings

1. **For playing humans, a small temperature (~0.25-0.5) is the sweet spot**:
   tau->0 is strongest but robotic/deterministic (same game every time);
   tau=1 blunders too often (6% blunder-pick vs 0.006% at 0.25). Mixing
   presets (e.g. 70% sampled / 30% argmax) is common practice in human-facing
   bots. Our play server currently plays argmax at 128 sims - strong but
   predictable.
2. With temperature applied to MCTS visits (N^(1/tau)), the top
   nearly-equal moves keep their variety while low-visit (blunder) moves stay
   effectively unreachable - cheap variety without strength loss.
3. **SQLite telemetry is optional, not urgent**: WAL mode gives O(1)
   append-only writes with concurrent readers, but needs bounded
   autocheckpoint (~1000 pages), batched inserts (500 rows / 2 s), and a
   retention/downsampling pass to keep disk sane. JSONL append-only is the
   simpler correct form for our single-writer journal; migrate only if
   multi-week analytics queries become painful.

### Actionable queue additions
- [ ] A91: play-server variety - small root temperature (0.3-0.5) or
      "random move within a small eval window"; keeps near-full strength,
      removes the robotic determinism for human opponents
- [ ] A92: journal -> SQLite telemetry (WAL, batched inserts, retention)
      only when analytics queries justify it - recorded as low priority

## Cycle 28 — 2026-11-24

Sources: "Beyond Search-Imitation" §2 (entropy-adaptive temperature, WDL
entropy tau, sampled-vs-argmax ladders), its repo flags; ONNX Runtime fp16
docs (convert_float_to_float16, mixed precision), ORT issue #13838
(fp16 slower at batch 1 without tensor-core benefit), PyTorch fp16-ONNX
discrepancy thread.

### Findings

1. **WDL-entropy temperature is a better exploration schedule than ply
   cutoffs**: tau(s) = H(p_WDL) in [0, ln3] ~ [0, 1.1]; sample mu ∝ pi^(1/tau).
   Decided positions (near-one-hot WDL) -> tau->0 -> greedy finish (stops
   wasting rollouts and teaching stochastic noise); contested/draw-heavy
   positions (tau ~0.9) -> mu ~ pi (exploration preserved where the signal is
   richest).
2. Every adaptive-temperature arm outranked its fixed tau=1 twin; adaptive
   temperature alone gained +9.8 Elo (before any KL regularizer), and
   sharpening the base lifted its sampled Elo by +361 (entropy 1.69 -> 0.39).
3. The sampled-vs-argmax span (670 vs 30 Elo) shows how much of sampled
   strength is just policy concentration - our current "temp 1.0 for 30
   plies then argmax" is a crude proxy for the same idea.
4. **fp16 ONNX is NOT a free win at our batch sizes**: gains need tensor
   cores AND batch >= 8; at batch 1 fp16 was measurably SLOWER than fp32 in
   ORT issue #13838 (cast/kernel overhead), and CPU EP fp16 just casts to
   fp32. Relevant only if worker eval batches grow; skip for the play server.

### Actionable queue additions
- [ ] A89: WDL-entropy temperature in self-play (flag-gated rust selfplay):
      tau = H(WDL of the mover), replace the ply-30 cutoff; measure draw rate
      and gate strength vs the current schedule
- [ ] A90: fp16 ONNX only if measured batch >= 8 (workers, not play server);
      otherwise stay fp32 - recorded as likely-no-gain

## Cycle 27 (newest, brief) — 2026-11-24
Argmax accuracy is only optimal for equal-cost 0-1 decisions; probability quality
needs proper scoring rules — Brier score (bounded, decomposes into calibration +
resolution) or log loss (punishes extremes), plus calibration slope/intercept; ECE
is popular but flawed (max-prob only, binning pathologies, not proper). Our live
value_acc 0.00↔0.59 swings are exactly argmax brittleness on near-tied WDL classes
→ A88: switch health metrics to Brier/mean-P(true)/score-MAE, keep argmax as a
diagnostic only.

## Cycle 26 (brief) — 2026-11-24
Staleness must be bounded on the sampling side (version window; clipping cannot
fix stale data) → A86 K=2 weight-version gate at drain; sublinear replay windows
are the KataGo-style evolution; end-game-first curriculum is an N+1 experiment
(A87); TB rows only pay where reachable; M2CTS experts far-future.
