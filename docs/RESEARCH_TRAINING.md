# Training Research Log (autonomous loop)

Purpose: continuous, incremental research on training/server optimization for
Fool's Gambit. Each cycle: read this file, research **new** topics (websearch /
webfetch), append findings + actionable items, commit. Cycles are listed
**newest first**. Focus = training quality, sample efficiency, throughput.
Never touch the live training run.

## Cycle 10 — 2026-11-24

Sources: KataGo Methods (root policy temperature, aux soft policy target),
KataGo paper §3.2 (forced playouts + policy target pruning, ablation table),
KataGo PUCT/FPU parameters, Lichess-puzzle benchmark writeups (chess-bench,
manoletre chess-puzzles-1k).

### Findings

1. **Forced playouts + policy target pruning** (KataGo §3.2): give every
   root child a minimum n_forced = sqrt(k·P·ΣN) playouts (k=2, PUCT=∞ until
   reached), then *subtract those forced playouts from the policy target*
   (prune children left with 1 playout). Ablation: removing it costs ~4 %
   Elo-equivalent (1276 vs 1329); it decouples the policy target from MCTS
   dispersion and noise. We currently export **raw visit counts** → we take
   noise-perturbed mass straight into training; this is the fix.
2. **Root policy softmax temperature ≥1** (1.03 in KataGo; 1.25→1.1 in SAI):
   a restoring force against PUCT's natural sharpening of near-equal moves;
   cheap one-liner if our root prior has no such temperature.
3. **Auxiliary soft policy head**: predict `policy^(1/T)` (T=4) as an extra
   head, nominally 8× weighted — "greatly improves the speed of learning of
   the policy"; helps after the early phase. Needs a model change (next gen).
4. KataGo's search constants for reference: c_puct=1.1, FPU =
   V(n) − 0.2·sqrt(ΣP_explored) (c_fpu=0 at root with noise),
   Dirichlet α ≈ 0.03 scaled by legal-move count.
5. **Puzzle benchmarks move before Elo does**: a Lichess-derived set (e.g.
   ~500–1,000 puzzles, 50 themes × difficulty bands, move-by-move scoring
   with Lichess replies) is a cheap tactical-accuracy metric — exactly the
   kind of signal that shows a net improving while gates still draw.

### Actionable queue additions
- [ ] A40: forced playouts + policy target pruning in our search exports
      (largest single search-side improvement left; applies to both python
      and rust backends)
- [ ] A41: auxiliary soft policy head (T=4, 8× weight) — next-gen model
- [ ] A42: audit root search params vs KataGo refs (c_puct, FPU, root temp,
      Dirichlet alpha scaling)
- [ ] A43: Lichess puzzle benchmark harness (~500–1,000 puzzles,
      move-by-move, Lichess replies) as the fast-moving tactical metric

## Cycle 9 — 2026-11-24

Sources: "Tiny AlphaZero" (PolyLabs, 6.5M params, 2026, pre-registered
rounds), lczero-training tfprocess (metrics), Dubslow's Test-10 overfitting
analysis, Leela-Zero #1480 (value_loss_weight).

### Findings

1. **Mixing 25 % Stockfish-labelled positions into each batch helped**
   (+0.05 score vs SF-skill-5, controlled); **doubling to 50 % hurt**
   (−0.044 while tracking the teacher's moves). A quarter is the sweet spot.
   → Our own `sf-distill` machinery already exists; a 25 % mix is a concrete,
   evidence-backed accelerator at our scale.
2. **Learning rate: much lower than ours was better** — 2e-4→2e-5 (+0.117),
   then 2e-6 (+0.075 more). At 1.3M params our constant-then-halving 1e-3 is
   plausibly hot; a 3e-4-vs-1e-3 A/B is cheap and evidence-aligned.
3. **Search coverage gap**: 66 % of their losing blunders "were never seen by
   the search" at 48 sims — sims count is not the lever they expected
   (128 vs 48 twice showed no effect). Ties to our A21: spend budget on
   games, not per-move sims.
4. **Value head overfits first** (Lc0 Train-10 analysis): train/test
   divergence starts early; symptom = loss spike-then-plunge on each new
   window; official fix = lower sampling rate (≈1×/position, not ~12×) and
   value_loss_weight 0.25. Our exposure: 49k samples/iter over a 100k
   buffer ≈ 0.5×/iter — acceptable, but worth monitoring.
5. **Metrics Lc0 actually watches**: policy loss, value CE+MSE, **policy
   entropy**, **policy-uniform loss (P UL)**, value accuracy, train/test
   divergence. We only journal loss — adding entropy + a held-out split is
   the cheapest early-warning system we lack.
6. A GPU-stack bug silently halved value-head gradients for many rounds
   before being found — verify gradient magnitudes / loss scale sanity once.

### Actionable queue additions
- [ ] A35: 25% Stockfish-labelled batch mix via the sf-distill path
      (teacher depth ~12, multi-PV → policy targets); A/B vs pure self-play
- [ ] A36: training-health metrics — policy entropy, value accuracy,
      train/test divergence on a held-out slice, P UL; surface in journal/UI
- [ ] A37: one-time sanity check of gradient norms per head (the
      half-gradient bug class) + confirm VALUE_WEIGHT/POLICY_WEIGHT split
- [ ] A38: LR A/B — 3e-4 vs current 1e-3 at 1.3M params (Tiny-AZ evidence)

## Cycle 8 — 2026-11-24

Sources: Chessformer (arXiv 2605.19091, 2026), pbaer/neural-chess v3,
"Distilling Stockfish with One Billion Positions" (2026), Lc0 WDL contempt
blogs + v0.30 rescale, Lc0 issue #237 (temperature & drawn endgames).

### Findings

1. **Deblunder rescorer** (Lc0's training-data post-processing): after a game,
   positions where the played move was an actively-induced blunder are
   re-labelled with corrected targets. Together with soft-Z this is Lc0's
   standard fix for "the average training row for a drawn position teaches a
   blunder". → Direct complement to A13/A25 for our draw-heavy self-play.
2. **Attention tower evidence (2026)**: Chessformer (square-token transformer
   + geometric attention bias) adds **+100 Elo to Lc0** at comparable params;
   pbaer v3 (pure square-token attention, no conv stem) beats a 2×-larger CNN
   by ~+150 Elo at 18M params. But a 79M-param distillation pilot found
   **hybrid ResNet→Transformer beats both** pure CNN and pure ViT — convs
   supply the board inductive bias, attention the whole-board reasoning.
3. **Attention mainly improves the policy**; the value head at matched data
   is a near-tie with CNN — which matches our current pain (value learning),
   so a swap alone would not fix our draw problem.
4. **Draw-rate context**: modern Lc0 nets draw ~58 % from startpos; higher
   draw rates at low strength are normal. Their temperature findings confirm
   our setup: temp-1.0 late-game blunders drawn positions (perpetual-check
   case: 36 % correct draws) — we already decay to 0 after 30 plies ✓.
5. **Scale caveat**: transformers are data-hungry; pbaer needs 100M positions
   and Chessformer 191M params. At our ~1.3M-param budget the safe version is
   a **small hybrid** (existing conv tower + a few attention/pooling blocks),
   validated in a short run, not a full swap.

### Actionable queue additions
- [ ] A32: deblunder post-processing — re-label rows where the played move's
      Q dropped vs the position's search value (Lc0 scheme; with A13/A25)
- [ ] A33: next-gen hybrid tower — conv tower + 2–4 square-token attention
      blocks with 2D relative-position bias; small-scale A/B before believing
- [ ] A34: track self-play draw rate as a health signal (Lc0 reference ~58 %
      from startpos); flat draw rate + flat gates = weak conversion

## Cycle 7 — 2026-11-24

Sources: Seer (McMonigle) TB-trained WDL net, ICGA 2021 "On the Road to
Perfection?" (Lc0 vs tablebases), chessprogramming Syzygy, Apache Arrow IPC
docs, arrow-ipc-bench, SPDL SharedMemoryArena case study (Meta), pymizu.

### Findings

1. **Syzygy labels are a cheap, perfect data source**: Seer trained a *small*
   net on several million ≤6-man TB-labelled positions and was effectively
   perfect in ≤5-man endgames. python-chess ships syzygy support; 3-4-5 WDL
   is ~1 GB from Lichess's mirror.
   → Three uses for us: (a) TB-correct the value target on any ≤5-man self-play
   position (removes endgame noise from game outcomes), (b) sample TB
   positions as perfect-label replay rows, (c) an endgame-accuracy eval
   harness vs TB.
2. **Endgames are a known AZ weakness worth measuring**: the ICGA study found
   Lc0 mistake rates in 3-4 piece endgames; search corrects many but not all.
   Our 200-ply caps/draw-heavy games likely hide endgame blunders — TB eval
   would expose them honestly.
3. **Our spool format will be the next throughput wall**: 9 MB/game of JSON +
   base64 f32 costs real CPU per game at both ends. Zero-copy options measured
   (arrow-ipc-bench): plasma export 14.5 GB/s, shared memory 8.7 GB/s, TCP
   4.3 GB/s; memory-mapped Arrow reads allocate nothing.
   → SPDL's SharedMemorySegmentPool pattern (producer writes once, consumer
   gets a zero-copy view) is the clean upgrade, and Arrow + sparse-policy
   columns also lands A12 naturally.

### Actionable queue additions
- [ ] A30: Syzygy integration — TB-corrected value targets for ≤5-man rows +
      sampled perfect-label replay + endgame-accuracy harness (needs the
      3-4-5 WDL download; python-chess has the reader built in)
- [ ] A31: spool v2 — Arrow IPC memory-mapped per-game records (or a
      shared-memory arena) replacing JSONL+b64; prerequisite for >~100
      games/hr and for sparse policy storage (A12)

## Cycle 6 — 2026-11-24

Sources: KataGo paper Appendix D + methods (soft resignation, game
randomization), KataGo playsettings, "Reproducing AlphaZero on Tablut"
(arXiv 2604.05476, 2026), ARENA AZ course (full-run hyperparameter sweep),
AlphaGateau optimization study (IEA/AIE 2026), d2l LR-schedule chapter.

### Findings

1. **Soft resignation** (KataGo): never resign, but when the losing side's
   MCTS winrate < 5% for 5 turns, cap visits to λn+(1−λ)N (λ=p/0.05) and
   record samples with prob 0.1+0.9λ. Cheaper than full play, no biased
   resignations. → A cleaner answer to our 200-ply "cap = draw label" bias
   than A13 alone.
2. **Opening initialization**: first r moves sampled from the raw policy
   (~exponential, mean 0.04·area ≈ 2–3 moves for chess), temperature decays
   smoothly 0.8 → 0.2 (halflife ≈ board width) — equivalent to AZ's 30-ply
   cutoff but smoother; 2.5–5% of games additionally branch into unusual
   openings for coverage.
3. **Opponent diversification**: the 2026 Tablut AZ reproduction played
   **25% of games vs randomly sampled past checkpoints** (plus C4 aug and a
   doubled replay) — that's what mitigated catastrophic forgetting; every
   other trick was secondary.
4. **Optimizer package** (Tablut + AlphaGateau, both 2026): AdamW (decoupled
   wd 1e-4) + warmup (500 steps) + cosine decay to ~1e-5, grad-clip 1.0;
   AlphaGateau's ablation: loss 2.49 → 1.67 and small Elo gain from exactly
   this package on a chess net. ARENA sweep: **all practical gains came from
   fresher self-play data** — capacity/loss/regularization were inert.
5. AZ original used momentum SGD 0.2 with 4 hard drops; WSD
   (warmup-stable-decay) beats cosine at fixed budget; we run Adam 1e-3 +
   step-halving — worth a measured upgrade to the 2026 package.

### Actionable queue additions
- [ ] A25: soft-resign / play-to-completion with visit scaling and sample
      downweighting (replaces hard-cap draw labels; pairs with A13)
- [ ] A26: policy-proportional opening init + smooth temperature decay
      (0.8→0.2, halflife ≈ 8 plies) — A/B vs the 30-ply hard cut
- [ ] A27: 25% of games vs sampled past checkpoints (worker code change;
      anti-forgetting, medium-term)
- [ ] A28: optimizer upgrade — AdamW(wd 1e-4) + 500-step warmup + cosine/WSD
      + grad-clip 1.0 (measure per-step loss trajectory before adopting)
- [ ] A29: confirmed: keep the replay window small/fresh (100k) — freshness
      beats size in the ARENA sweep

## Cycle 5 — 2026-11-24

Sources: Gumbel AlphaZero (Danihelka et al., ICLR 2022 + ARENA course notes),
MiniZero (arXiv 2310.11305), KataGo Methods (Playout Cap Randomization,
Policy Surprise Weighting, short-term value targets), "Value targets in
off-policy AlphaZero" (A0GB / soft-Z, 2021), "Lessons from AlphaZero pt4".

### Findings

1. **Playout Cap Randomization (KataGo) is the single biggest untapped lever
   for us**: per MOVE, with p≈0.25 run the full sim budget (keep the policy
   target); otherwise run a fast search (~100 sims) that only contributes
   value training (policy target masked). At 600-sim games this cuts average
   cost to ~0.37× → **~2.7× more distinct games AND 2.7× more value samples
   per wall-clock hour**. Directly attacks both our throughput and our
   value-head learning problem.
2. **Gumbel AlphaZero**: root Gumbel-top-k + sequential halving + improved
   policy target `softmax(logits + σ(completedQ))` — *guarantees* policy
   improvement per step; at low sims Gumbel-4 ≈ PUCT-16. We already ship a
   gumbel search option (`test_training_gumbel_search`) — cheap A/B at 600
   sims, but at high sims PUCT≈Gumbel (MiniZero), so lower priority.
3. **Value target: q (root MCTS value) beats/blends with z (game outcome)**:
   z is a single noisy sample (one bad move later poisons every label); q is
   a continuous average of the search's estimate. Averaging q+z or starting
   with z and linearly falling off to q measurably cut Connect-4 error.
   KataGo's version: auxiliary short-term value targets = λ-averaged future
   MCTS values (λ ≈ 6/16/50-turn horizons) as extra heads.
4. **Policy surprise weighting detail** (KataGo g170 onward, one of its
   larger wins): redistribute frequency weights — half uniform, half
   proportional to KL(prior ‖ target) — and write rows `floor(w)` times
   (+1 stochastically). This is the concrete implementation of A11.

### Actionable queue additions
- [ ] A21: playout cap randomization — p_full=0.25, fast_sims≈100,
      policy-masked fast rows (throughput + value-signal multiplier)
- [ ] A22: gumbel-root A/B for self-play once A21 lands (cheap, code exists)
- [ ] A23: value target blend — root-Q / outcome mix (start 50/50, or z→q
      falloff); long-term: auxiliary λ-horizon value heads
- [ ] A24: policy-surprise frequency weighting (half-uniform, half-KL,
      floor(w)+stochastic — KataGo's exact scheme)

## Cycle 4 — 2026-11-24

Sources: Fishtest math (pentanomial + GSPRT), chessprogramming SPRT/Match
Statistics, "Measuring Bot Strength" (gate design), NNUE docs (flip/rotate,
horizontal mirroring), reckless-nnue (+14 Elo mirroring, only past 1.4B pos).

### Findings

1. **Our 16-game gate is statistically blind to real improvement**: detecting
   a true 55% score needs ~785 games (95% conf, 80% power); 52% needs ~4,900.
   A 16-game gate at 0.55 can only see ~massive jumps — and with draw-heavy
   bots it almost never fires. This is *why* the Elo sits still: not a bug,
   the gate's sample size. → Raising gate games (now affordable at 8-way
   parallel) and/or gating every N iterations is the honest fix.
2. **Two-stage gates** (screen cheap → confirm on FRESH openings with a
   Wilson 95% lower bound > 0.5) cut false promotions from ~18% to <0.5%
   while keeping most real gains. Better than one long match.
3. **Pentanomial accounting**: paired color games on the same opening are
   correlated; the trinomial model overestimates variance. Fishtest uses
   pentanomial (LL/LD/WL+DD/WD/WW) — our gate already pairs colors, so the
   stats should switch to pentanomial for correct error bars.
4. **SPRT decides, it does not measure** — stopped tests read high (measured
   +124 vs +102 fixed-N on identical walk). Any Elo magnitude we quote must
   come from a fixed-N run; keep SPRT for go/no-go only.
5. **Chess symmetry**: color-flip (vertical mirror + color swap + stm) is a
   valid symmetry and standard augmentation (+14 Elo NNUE, but only past
   ~1.4B positions). Our encoder is side-to-move-relative, so this symmetry
   is **already structural** — no augmentation needed. Horizontal king
   mirroring does not apply to the policy head; not worth it at our scale.

### Actionable queue additions
- [ ] A17: gate statistics overhaul — two-stage gate (screen 64 @ ≥0.52 →
      confirm on fresh openings, Wilson lower bound > 0.5) + pentanomial
      variance; keeps false promotions near zero and catches real gains
- [ ] A18: gate compute budget — 64+ games per gate at 8-way parallel and/or
      gate every N iterations (same games/hour, honest statistics)
- [ ] A19: note in docs — the 0.55 single-stage 16-game gate was the "why no
      promotions" answer; replace per A17
- [ ] A20: record per-gate pentanomial counts + fixed-N point estimate in the
      ledger (SPRT only for the decision)

## Cycle 3 — 2026-11-24

Sources: "Engineering Efficient Self-Play Chess" (arXiv 2609.37447, 2026),
KataGo Methods (policy-surprise), RGSC paper details, PER-for-Expert-Iteration
(IEEE CoG 2020), ort-superserve / ort docs, onnxruntime issue #20494.

### Findings

1. **Policy-surprise sampling** (KataGo + the 2026 chess self-play paper):
   sample positions by how much search's policy disagrees with the net's prior
   — ~70 % priority + 30 % uniform floor, priority via *sampling frequency*
   (loss weights stay 1.0). Big sample-efficiency lever, cheap to compute at
   admission time from data we already store.
2. **Sparse policy rows**: the 2026 paper stores only the top ≤60 actions per
   position (records discarded visit mass). Ours stores the full 4672-float
   vector (~18 KB/row) → 36× memory saving, letting the same RAM hold a far
   larger, older replay window.
3. **Capped games must NOT get draw targets**: the reference implementation
   uses the *final search value estimate* as the outcome substitute for
   ply-capped games. → **We currently record 1/2-1/2 for every 200-ply cap
   (`game_outcome`) — a systematic value-label bias on long games. Worth fixing.**
4. **Restart-state selection**: half of games start from archived self-play
   states chosen for "meaningful unresolved alternatives" (search disagreed
   with value, untried branches exist); 30 % uniform fallback; replay capacity
   grows 600k → 20M rows with ~4 optimizer presentations per new row.
5. **Orange flag for GPU inference**: multiple ONNX sessions *in one process*
   don't scale throughput (ORT issue #20494 — rwlock on kernel launch); prefer
   multiple worker PROCESSES (we already do) or the `ort-superserve` pattern
   (session pool + dynamic batching actor). ort CUDA needs CUDA ≥13.2,
   cuDNN ≥9.23 on PATH.

### Actionable queue additions
- [ ] A11: policy-surprise sampling in `buffer.sample` (70/30, admission-time
      priority from stored visit targets vs net priors)
- [ ] A12: sparse policy storage (top-60 actions) — 36× memory per row
- [ ] A13: capped-game value target = final MCTS root value, not draw
- [ ] A14: replay restart archive (RGSC-lite restarts, λ ≈ 0.3)
- [ ] A15: replay capacity schedule (grow window with run age)
- [ ] A16: shared GPU inference via ort-superserve pattern once workers go CUDA

## Cycle 2 — 2026-11-24

Sources: Lc0 blog "AlphaZero paper & Lc0 v0.19.1" + issue #237 + #1480,
lc0 training docs, torch.compile docs + Geometric mode study (2026),
libreyolo CUDA-graph benchmarks (2026), PyTorch 2.15 docs.

### Findings

1. **Temperature only for the first ~30 plies** — the corrected AZ pseudocode
   sets temperature→0 after 15 moves; Lc0 found constant temperature blundered
   drawn endgames (perpetual-check draws fed as losses). → **We already do
   this** (`temperature_plies=30`, then 0). Good; keep.
2. **FPU = -1 "unvisited is lost"** (absolute) and **Cpuct 2.4 + growing term
   (base 19652)** are the real AZ values, not Lc0's parent-estimate/constant.
   → Verify our SearchConfig cpuct/fpu match the paper; cheap A/B.
3. **Value head overfits first** — Lc0 recovered from a strength regression by
   dropping `value_loss_weight` 1.0 → 0.25 (and later restored it). Policy
   overfits ~never (target is a distribution). → Tune VALUE_WEIGHT down if
   value loss improves while strength regresses; data-balancing (A2) is the
   principled version.
4. **Replay window in AZ = last ~1M games** (Lc0 used 500k). Our buffer is
   100k *examples* (~1k games) — fine early, but the recency window may matter
   as the run matures; revisit when buffer cycles < ~1 day.
5. **torch.compile is a big lever**: 2.3× on a toy train loop; 2026 study across
   41 workloads says `max-autotune-no-cudagraphs` or `default` beat
   `max-autotune`; `reduce-overhead` (CUDA graphs) wins for **small batches**
   where launch overhead dominates — exactly a 10×128 conv net at batch 512.
6. **AMP (fp16 autocast + GradScaler)**: 1.5–5.5× step speedup typical;
   ~115 ms/step today for 96 steps — worth trying once training step matters.
7. **Server-side**: fused AdamW, channels-last, and capturing the whole
   train_step in a CUDA graph (libreyolo measured 1.1–3.6× on the step, bigger
   on small batches); on Windows launch overhead is the worst of all platforms.

### Actionable queue additions
- [ ] A7: verify Cpuct/FPU against AZ paper values (config A/B, gated)
- [ ] A8: torch.compile(mode="reduce-overhead") on the train step + AMP —
      measure step time; enable only if stable with checkpoint load/save
- [ ] A9: fused AdamW (`torch.optim.AdamW(fused=True)`) — one-liner
- [ ] A10: buffer recency window review once iterations are <20 min



Sources: ameye.dev chess-engine notes, KataGo Methods + David Wu interview,
RGSC (arXiv 2602.20809), Uncertainty-guided exploration (NeurIPS '25).

### Findings

1. **Regret-Guided Search Control (RGSC)** — harvesting high-regret states
   (where the net's eval diverged most from the actual outcome) as *starting
   positions* for extra self-play gives +77 Elo vs AlphaZero in Go/Othello/Hex.
   → Our bot draws ~everything vs itself; drawn-but-tense positions are
   exactly the high-regret states worth re-playing. This is a sample-efficiency
   multiplier that works on top of our existing loop.
2. **KataGo data balancing** (Wu interview): value/policy imbalance is real —
   value gets one label per game while policy gets per-move signal. Fix:
   subsample policy rows for training while playing more games overall; give
   the value head proportionally more data. → Directly applicable: our policy
   loss dominates (≈2.5 vs value ≈0.19); we currently train every ply equally.
3. **Whole-board aggregation** (KataGo's first architectural win): AlphaZero's
   pure local convs need many layers for information to cross the board. A
   global-average-pool → FC branch (or light attention) added to the tower
   helped KataGo "a lot". → Our 10×128 GambitNet is pure local convs.
4. **Uncertainty-guided exploration** (NeurIPS '25): branch from high
   uncertainty positions (Label Change Rate metric), ensemble value labels
   across variants → +58% sample efficiency early (9×9 Go). Cognate of RGSC.
5. **Transformers now beat convs in KataGo v1.17+** (stronger + more
   efficient). Future-generation direction, not now.
6. **Classic engine perf** (ameye): bitboards + magic bitboards + perft are
   the standard movegen recipe. Our Rust movegen is already perft-verified;
   only worth revisiting if profiling shows movegen as a bottleneck (today
   inference dominates).

### Actionable queue (prioritized)
- [ ] A1: high-regret / high-uncertainty opening harvesting for self-play
      (RGSC-lite: replay drawn games from mid-game positions, or sample opening
      FENs from the buffer's worst-eval positions)
- [ ] A2: policy-row subsampling (train value on all plies, policy on a subset)
- [ ] A3: global-pooling block in GambitNet (next-gen net)
- [ ] A4: auxiliary heads (material count / phase) to densify value learning
- [ ] A5: playout-cap randomization (already spec'd as U2, unused)
- [ ] A6: transformer tower experiment (future generation)

### Next-cycle topics (queued)
- torch.compile / AMP / CUDA graphs for the 10×128 training step
- ONNX Runtime GPU EP for workers (CUDA 13 build) + shared inference process
- replay buffer sampling strategy (recency weighting, prioritized replay) in AlphaZero variants
- Leela Chess Zero training tricks (temperature schedules, FPU, symmetry aug)
- chess-specific data augmentation (vertical flip / color swap symmetry)
- gating statistics: SPRT ladders, Glicko vs BayesElo, thresholds at 16 games
