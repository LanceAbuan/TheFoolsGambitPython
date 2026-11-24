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
- uncertainty-driven temperature (WDL entropy → tau) as exploration policy
- MCTS root move selection at play time: argmax vs temperature for human feel
- replay admission by game quality (resign/cap quality filters)
- SQLite/telemetry compression for long-run journals (ops)
- ONNX fp16 export for play-server latency (with A60)

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
