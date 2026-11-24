# Training Research Log — distilled (cycles 1–13 folded)

Continuous research on training/server optimization for Fool's Gambit.
Autonomous loop: research → append → consolidate every 6 h. **Cycles 1–13 are
folded into the distillate below; newest cycle kept in brief at the bottom.**
Never touch the live training run from this loop.

## Distilled knowledge

**Targets & data quality**
- Value targets: blend root-Q with game outcome (soft-Z / z→q falloff); TB-correct
  ≤5-man rows; capped games use final search value (never fake draws); post-game
  deblunder pass re-labels induced blunders. (Lc0/KataGo/2026 refs)
- Policy targets: prune forced playouts from the visit distribution (~4% Elo-eq);
  aux soft-policy head `policy^(1/4)` at 8× weight; sparse top-60 rows (36× memory).
- Sampling: replay **freshness > size** (100k window fine); ~1×/position per window
  (avoid overfit; value head overfits first); 25% Stockfish-labelled batch mix
  (+0.05 score; 50% hurts); policy-surprise frequency weighting (half uniform,
  half KL).
- Search control: replay drawn/high-regret states (RGSC +47–89 Elo; uncertainty
  branching +58% early sample efficiency); 25% games vs sampled past checkpoints
  (anti-forgetting); varied openings +0.24; policy-proportional opening init with
  smooth temp decay 0.8→0.2.
- Termination: never hard-draw ply caps; soft resignation (visit scaling +
  sample downweighting when win prob <5% for 5 turns).
- Symmetry: color-flip already structural in our stm encoder; horizontal mirroring
  not worth it below ~1.4B positions.

**Search & engine**
- Playout cap randomization: p=0.25 full + 100-sim value-only rows → ~2.7× games
  and value samples per hour.
- Subtree reuse across moves (retain ~60% of visits) — cheap 20–40% compute saving.
- Root search recipe: forced playouts `sqrt(k·P·ΣN)` k=2 + target pruning;
  prior softmax temp ≥1 (1.03); c_puct≈1.1; FPU = V−0.2·√ΣP.
- Per-tree parallel leaves cost ~6.4 Elo/doubling (≈235 at 100 sims) — batch
  across games instead; 0.99 per-ply value discount as anti-shuffle pressure.
- Graph/transposition search with exact chess identity: net-negative
  (0.02–0.18% evals saved, 8% slower) — do not pursue.
- Gumbel AZ: guaranteed policy improvement; Gumbel-4 ≈ PUCT-16 at low sims
  (at high sims PUCT≈Gumbel). Moves-left head: prefer faster wins, avoid shuffling.

**Optimization & training loop**
- AdamW (wd 1e-4) + warmup + cosine/WSD + grad-clip 1.0 (chess-net loss 2.49→1.67
  in one study); at small scale lower LR wins (2e-4→2e-6 evidence); our 1e-3 is hot.
- EMA weights (0.995–0.999) beat the live iterate by 30–65 Elo in self-play RL —
  gate/bench the EMA. Polyak/SWA is Lc0 standard.
- Buffer continuity across rounds removes performance dips; sims 128 vs 48 showed
  no effect at small scale — spend budget on games, not per-move sims.
- Metrics: policy entropy, P UL (CE vs uniform), value accuracy, train/test
  divergence; value-overfit alarm; verify per-head gradient norms once.

**Architecture (next-gen bundle)**
- 10×128 is Lc0's canonical ResNet *minus* Squeeze-and-Excitation; SE blocks are
  standard (GAP→FC→ReLU→FC(2F)→sigmoid(γ)x+β).
- Input planes beat architecture: +96.7 Elo from representation (halfmove clock,
  material-difference planes, grouped masks). Our encoder v2 lacks the halfmove
  clock. History 8 plies is right (0→7 big, 7→31 none); mask history 5% in training.
- Hybrid conv→attention beats pure CNN and pure ViT (+100–150 Elo at 18M+ params;
  attention mainly improves policy). Chess-mask kernels as structured regularizer.

**Evaluation & ops**
- 16-game 0.55 gates are statistically blind (~785 games needed for a true 55%);
  two-stage gate (screen 64 @ ≥0.52 → confirm on fresh openings, Wilson LB>0.5);
  pentanomial variance for paired colors; SPRT decides, fixed-N measures.
- Lichess puzzle benchmark (move-by-move, ~500–1000 puzzles) moves before Elo.
  Draw rate ~58% from startpos is normal for strong nets — track it as health.
- ORT threading: 1 thread/session, 1 process per physical core, allow_spinning=0
  (1.7× measured concurrency win). ORT CUDA needs CUDA ≥13.2 / cuDNN ≥9.23;
  multiple sessions per process do not scale.
- Spool v2: JSONL+b64 (9 MB/game) will wall; memory-mapped Arrow IPC or shared
  memory gives zero-copy reads and sparse policy columns.
- Syzygy 3-4-5 WDL (~1 GB) = perfect endgame labels for targets and eval.

## Actionable queue (re-prioritized 2026-11-24 07:20)

Live state: v2 run at iter 9 (loss 2.51, buffer 34.7k), **first promotion landed**
(ckpt-000864-0269, 3W-12D-1L) — gate cadence and promotions work.

**Tier 1 — next builds (biggest measured impact)**
- [ ] A21 playout cap randomization (2.7× games/value samples)
- [ ] A51 EMA weights + gate/bench on EMA (free 30–65 Elo)
- [ ] A40 forced playouts + policy target pruning (~4% Elo-eq, search-side)
- [ ] A44 worker threading: intra_op=1, allow_spinning=0, workers=physical cores
- [ ] A17+A18+A20 two-stage gate + 64-game budget + pentanomial ledger rows

**Tier 2 — data quality (cheap, compounding)**
- [ ] A13+A25 capped-game search-value targets + soft resignation
- [ ] A32 deblunder re-label pass
- [ ] A35 25% Stockfish-labelled batch mix (sf-distill path exists)
- [ ] A11+A24 policy-surprise sampling (half-uniform, half-KL)
- [ ] A12 sparse top-60 policy storage
- [ ] A48 subtree reuse; A49 0.99 discount; A43 Lichess puzzle harness
- [ ] A30 Syzygy targets + endgame eval (needs 1 GB WDL download)

**Tier 3 — next generation (one restart, bundle: A54)**
- [ ] A54 = A33 hybrid attention tower + A45 encoder v3 planes (halfmove clock,
      material diff, 5% history masking) + A47 moves-left head + A52 SE blocks
      (+ optional A53 chess-mask kernels); plus A28 optimizer package + A38 LR A/B
- [ ] A36 health metrics + A37 gradient sanity before the new-gen run

**Tier 4 — throughput/infra (after worker scaling)**
- [ ] A31 spool v2 (Arrow IPC / shared memory), A16 shared GPU inference
      (ort-superserve pattern), A22 Gumbel A/B, A46 worker-fleet matrix,
      A50 per-tree parallelism measurement, A26 opening init, A27 past-checkpoint
      opponents, A14 restart archive, A34 draw-rate metric

Superseded/folded: A19→A17; A41→A54; A42→A40; A29 confirmed; A1–A10 absorbed
above; A15 capacity schedule → revisit next gen; A39 → A36.

## Next-cycle topics (un-researched)
- batch-size ↔ LR scaling laws for small conv nets (match our 512/1e-3)
- SWA/EMA window timing (which checkpoints to average, when)
- CUDA EP end-to-end worker benchmark (onnxruntime-gpu build on the 3080)
- opening-book seeding from human master games (Lichess elite) vs policy init
- league/opponent-modeling depth (past-checkpoint sampling rates)
- profiling workflow for the trainer (py-spy/nsys) under rust_workers

## Cycle 23 — 2026-11-24

Sources: Lc0 classic search TB integration (root_probe/DTZ, in-search WDL
terminals), chessprogramming Syzygy (root vs search probing), "Best Policy
Tracking in RL" (2610.34501: last vs train-best vs validation-best),
Self-Play Algorithms framework (2006.04471: menagerie/curator), league/
PFSP practice notes (stratified latest + rolling + archive).

### Findings

1. **Play-time tablebases are a cheap, perfect endgame upgrade**: Lc0 probes
   WDL at every TB node (exact terminal) and filters root moves with DTZ when
   available. We already ship `TablebaseCorrector` (WP8) - wiring a root WDL
   probe into the PLAY server lets the bot convert/defend <=5-man positions
   instantly and perfectly against human friends (no sims wasted), with an
   optional online Syzygy (lichess API) fallback when tables are absent.
2. DTZ-at-root is subtle (50-move tensors: a win may only be blessed); WDL
   first, DTZ later - and only allow DTZ filtering in clearly winning paths.
3. **Checkpoint selection**: latest vs train-best vs validation-best are
   meaningfully different; our SPRT/Glicko gate already implements the
   validation-based champion choice - serving the champion stays correct.
4. **League play remains the big self-play gap**: stratified mixtures
   (latest ~60-75 % + rolling recent + geometric archive) with PFSP
   win-rate-weighted sampling mitigate cyclical overfitting; needs worker
   support for multiple model pointers (one hot-reloaded opponent slot).

### Actionable queue additions
- [ ] A80: play-server TB integration - root WDL probe for <=5-man positions
      (perfect endgame play vs humans); optional lichess API fallback
- [ ] A81: league sampling v1 - workers alternate self vs a rotated past
      checkpoint (25 % of games; stratified: latest/rolling/archive, PFSP
      weighting later) - extends A27 with concrete sampling ratios

## Cycle 22 — 2026-11-24

Sources: Group Normalization (1803.08494, BN small-batch degradation),
"Rethinking Batch in BatchNorm" (2105.07576: PreciseBN, RL moment staleness,
ELF OpenGO), Unified Normalization study (NeurIPS 2021), weight-decay pieces
(torchvision recipes, "Four Things..." 1906.03548, AdamW 1711.05101, d2l
AdamW).

### Findings

1. **BatchNorm stays the right choice at our batch 512**; GN's advantage only
   appears at batch 2-8 (BN error 34.7 % vs GN 24.1 % at batch 2). Our eval
   paths (gates, export) use frozen running stats, so small-batch inference
   is safe. Revisit GN only if training batch drops below ~32.
2. **RL has a BN-specific hazard: moment staleness** during training
   (ELF OpenGO used PreciseBN). Our EMA shadow covers ALL floating tensors -
   including BN running stats - and gate champions are frozen snapshots, so
   the exposure is low; noted for the record.
3. **If AdamW is enabled (WP10), decay must be grouped**: decay conv/linear
   weights only (2-D+), exempt BN gamma/beta and biases. With BN, decay acts
   mainly through the effective learning rate - tune it WITH the LR, and
   remember PyTorch's AdamW multiplies wd by the (scheduled) LR.
4. Our current pipeline loss has no explicit L2/weight decay (WDL branch =
   policy CE + value CE); the legacy trainer's all-parameter L2 does not
   touch the v2 loop - so the WDL collapse is not a decay artifact.

### Actionable queue additions
- [ ] A78: parameter groups when AdamW lands - decay 2-D+ weights only;
      BN gamma/beta and biases exempt; document the LR x wd interaction
- [ ] A79: no normalization switch needed at batch 512; keep BN, monitor
      running-stat freshness via EMA shadow (already covered)

## Cycle 21 — 2026-11-24

Sources: Neural Collapse under class imbalance (2309.09725, 2401.02058),
CappyChess implementation notes (value-class rebalancing, head-init gain),
Lc0 WDL+contempt blogs, Lc0 issue #1308 (temperature vs value), Lc0 TB
rescoring (retroactive WDL for <=5-man spans).

### Findings

1. **Class-imbalanced CE collapses minority classes** (Neural Collapse
   theory): classifier weights align to class means scaled by sqrt(n_k);
   with a draw-heavy distribution the WDL head degenerates toward one class.
   Our live value_acc = 0.00 across iterations 13-14 is the signature (likely
   "always argmax=Win while targets are mostly draws") - needs a histogram
   diagnostic before it is fixed blind.
2. **Value-class rebalancing is the standard remedy**: CappyChess samples the
   value loss by rebalancing to 1/3 W, 1/3 D, 1/3 L (default), with
   `--no-value-balance` (Leela-style uniform) as the alternative. Target
   class-prior bias init and a scaled-down head gain (0.01) at init are
   complementary anti-collapse details.
3. WDL calibration is separate from argmax collapse: Lc0's realistic-WDL
   machinery (draw-rate reference ~58% startpos, rescale) fixes prediction
   calibration, not class domination - do not conflate them.
4. Temperature/exploration blunders shape value targets (issue #1308): the
   value head needs to SEE bad positions; don't over-filter them.
5. TB rescoring (Lc0, server-side) retroactively rewrites WDL for all
   positions in a <=5-man span - matches our WP8 override; noted as
   confirmation.

### Actionable queue additions
- [ ] A75: value-class handling - log target/argmax class histograms in WP5
      metrics first (one line); then rebalanced value sampling (W/D/L 1/3) or
      class-weighted value CE if collapse confirmed
- [ ] A76: head-init hygiene - scale WDL output layer gain ~0.01 at init
      (flat start), verify no L2/weight-decay on BN/bias params
- [ ] A77: keep temperature-induced bad positions in training (no blunder
      filtering) - value head needs them

## Cycle 20 — 2026-11-24

Sources: AlphaZero PNAS (≤30 positions sampled per game; Dirichlet(0.3) 25 %),
"Assessing Game Balance with AlphaZero" (0.12 samples/position), ChessCoach
(800 sims + noise), "Engineering Efficient Self-Play Chess" §4.1 (PCR in
CHESS measured net-negative), AZ supplementary (root noise every search).

### Findings

1. **AlphaZero samples ≤ ~30 positions per GAME** ("subsequent positions are
   strongly correlated; including all may increase overfitting"). We admit
   every ply (~100–150 rows/game) - adjacent-row redundancy is real and is
   also a value-head overfitting vector. Simple knob: per-game subsample.
2. **Dirichlet(0.3) at 25 % weight on the root, applied on every search**
   (every ply) is the AZ-canonical setting - our implementation matches.
3. **⚠ PCR DID NOT TRANSFER TO CHESS** (2026 engineering paper §4.1.2): their
   fast/full scheme "discarded useful targets", cheap moves still consumed
   42.9 % of nominal search work, the tail hurt batching (128 active trees
   filling 86/320 batch slots), and the retained design searches EVERY
   recorded move to a staged cap (300→800). Their value objective was
   already learning, so extra finished games did not compensate for lost
   policy-target density.
4. Our run just launched with PCR p=0.25/fast=100 (WP2) - the contradiction
   demands an A/B, not a blind rollback: same wall-clock, gate score + target
   density, PCR-on vs PCR-off. Staged sim caps (300→400→600→800) are the
   measured alternative worth adopting regardless.

### Actionable queue additions
- [ ] A72: per-game row subsample (≤30 rows/game, uniform over plies) -
      cuts adjacent-position redundancy; A/B vs all-plies at matched compute
- [ ] A73: PCR A/B in chess (on vs off, matched wall-clock, WP5 metrics +
      gate scores); adopt staged sim caps (300→800) as the fallback design
- [ ] A74: track presentations-per-row (~4 in the reference run) alongside
      games/hr - reuse ratio is a first-class training knob

## Cycle 19 — 2026-11-24

Sources: Tiny AlphaZero round log (2026), "Engineering Efficient Self-Play
Chess" (2609.37447, 6.32M params -> 3,251 bench Elo in 2.5 days, 3.25M games),
scaling laws for AZ (2210.00849), pbaer v3 teaching models (capacity
saturation at 116k), Lc0 community notes (SL w/o policy ~200 Elo weaker).

### Findings

1. **At our 1.3M-param width we are data/search-bound, not capacity-bound**:
   scaling laws say Elo ~ log(params) only at fixed compute, and larger nets
   are more sample-efficient - but the pbaer teaching line shows tiny nets
   saturate on supervised signal while self-play keep improving, and the
   2026 engineering run hit 3,251 bench Elo with 6.3M params and a huge
   data/search budget. Their late-plateau levers: MORE search in self-play,
   MORE fresh games (lower replay reuse), larger variety window, then size.
2. **Order of operations for us**: iteration count > inner epochs/sims
   (cycle 18) and search-per-full-ply before width. Night-v3's SE/attention
   at the SAME 10x128 width is the cheap capacity step; widening waits.
3. **Human-position mixing helps as volume, not as teacher**: p2_ad's
   human-only positions scored 0.03 lower than older self-play
   position-for-position, yet the 31.9M mixed set produced the largest gain
   (+0.0875). Self-play targets remain the stronger per-position signal
   (SL-without-policy nets are ~200 Elo weaker per Lc0 lore).
4. Matched-compute comparison is the only fair yardstick for these choices -
   compare strength gained per wall-clock hour, never raw loss.

### Actionable queue additions
- [ ] A69: capacity audit before night-v3 - use WP5 train/holdout gap on
      night-v2; expect no saturation yet (1.3M params, small compute)
- [ ] A70: full-search sims 600 -> 800 on PCR full plies (p=0.25 makes it
      affordable); A/B vs current at matched wall-clock
- [ ] A71: future - human-position mixing as volume (Lichess Elite + SF
      labels) once a labeling path exists; keep self-play targets primary

## Cycle 18 — 2026-11-24

Sources: Net2Net (1511.05641), Net2Net growing-wiki (AutoGrow: RandomPad beats
Net2Net on ResNets), "Towards a More Complete Theory of Function Preserving
Transforms" (2410.11038, RandomPadWiden student 70.2 vs NetMorph 69.8),
"Analysis of Hyper-Parameters for Small Games" (2003.05988), MiniZero
hyperparameters, size-scaling studies (ESANN 2022, 2210.00849).

### Findings

1. **Net2WiderNet** = replicate channels/units by random mapping and DIVIDE
   the outgoing weights by the replication factor (+ small noise so replicas
   diverge); conv nets do it at channel level. **Net2DeeperNet** = insert
   identity layers (needs idempotent activations like ReLU; BN identity init
   needs data statistics).
2. **For ResNets, RandomPad beats classic Net2Net** (AutoGrow / 2410.11038):
   simply zero-padding new channels matches or beats function-preserving
   replication. Our v3 is architecture-ADDITIVE (SE + MLH + encoder planes),
   so zero-init of the new parts is effectively RandomPad - the cheap and
   often-better route; skip manual function-preserving surgery.
3. Warm-start advantage is **budget-dependent**: with short post-surgery
   budgets a cold-widen can win; warm-start wins once the teacher is well
   converged and the resume budget is long. Keep a cold-start A/B control.
4. **Outer self-play iterations subsume the inner knobs** (2003.05988): within
   a fixed budget, maximize iterations; LOWER games/epochs/sims per iteration.
   Too many inner epochs overfit and can reduce strength.
5. Reference shapes: MiniZero = 2000 games/iter, 200 steps/iter, 40k-game
   buffer, 300 iterations; we run 32 games / 96 steps over 1000 iterations -
   same shape, far smaller per-iteration.

### Actionable queue additions
- [ ] A67: WP11 warm start via additive zero-init (encoder planes 0, SE near
      identity via large positive gamma bias, new heads fresh); cold-start
      control run for the A/B
- [ ] A68: iteration-size A/B - more outer iterations with fewer inner steps
      (e.g. games 32->16, steps 96->64) vs current; compare gate score per
      wall-clock hour

## Cycle 17 — 2026-11-24

Sources: AZ supplementary data (73×64 encoding), "Mastering Chess with a
Transformer" (source-dest policy head + aux targets, arXiv 2409.12272),
graph-chess paper (2410.23753), PyTorch mixed-precision guide, BFloat16 study
(1905.12322), FP8/BF16 stability study (2405.18710), 16-bit training review.

### Findings

1. **Our 73×64 policy encoding is the AZ standard** and fine. Processor
   alternatives worth one ablation at night-v3: (a) source→destination
   factorized head (64×64 attention logits + promotion bias; Transformer
   paper reports clean training without the 73-plane waste); (b) graph
   encoding (only 1858 real edges). Keep 73×64 as the default.
2. **Auxiliary heads accelerate convergence** (same paper: "several auxiliary
   policy and value targets to increase convergence speed, following Go") -
   independent confirmation for A41/A47/A54; add value-horizon aux targets
   to the night-v3 scope.
3. **TF32 is free on Ampere**: PyTorch enables TF32 for convolutions by
   default but NOT matmuls - set `torch.backends.cuda.matmul.allow_tf32 =
   True` for a cheap speedup with negligible accuracy loss. One-liner.
4. **bf16 > fp16 for training**: bf16 has fp32's range (no grad scaler, no
   BN-variance underflow); FP16 needs loss scaling and its Adam-eps quirks.
   Keep fp32 master weights (AMP default) and BN stats in fp32.
5. Stability caution: a controlled study saw ~10 % BF16 divergence at short
   LLM runs vs TF32; conv chess nets are far less sensitive, but our WP5
   health metrics are exactly the alarm to keep on if bf16 is enabled.

### Actionable queue additions
- [ ] A63: source→destination factorized policy head ablation (night-v3
      experiment; 73×64 stays default)
- [ ] A64: enable TF32 matmul in the pipeline (free Ampere speedup)
- [ ] A65: bf16 autocast profile flag (fp32 master weights); A/B under the
      health-metrics alarm before any default change
- [ ] A66: aux value-horizon targets added to the night-v3 bundle scope

## Cycle 16 — 2026-11-24

Sources: ONNX Runtime CUDA EP docs (1.27+ = CUDA 13 / cuDNN 9 by default,
preload_dlls, cudnn_conv_use_max_workspace), onnxruntime-gpu 1.32 PyPI
metadata, rembg #841 (silent CPU fallback), Nsight Systems mastery + PyTorch
profiler decision tree, Ray "GPU starvation" data-ingest post.

### Findings

1. **The worker GPU path is one install away**: `onnxruntime-gpu` 1.27+ on
   PyPI is built with CUDA 13.0 + cuDNN 9.x - exactly our torch 2.15+cu132
   stack. Recipe: uninstall the CPU `onnxruntime` package (one runtime per
   env), install `onnxruntime-gpu`, `import torch` before creating sessions
   (or `onnxruntime.preload_dlls()`), then verify `session.get_providers()`
   actually starts with `CUDAExecutionProvider` (creation is the only real
   test - `get_available_providers()` lies).
2. **Silent-fallback pitfall**: with mismatched CUDA/cuDNN the session lists
   CUDA as available, fails to load `cublasLt64_13.dll`, and quietly runs on
   CPU (rembg #841). Log requested vs actual EP at worker startup.
3. Conv-heavy CUDA sessions want `cudnn_conv_use_max_workspace=1` (default
   since 1.14; matters most for fp16 tensor-core algorithms).
4. **Profiling decision tree**: steady-state step timing -> `torch.profiler`
   -> if the host dominates, fix the input path; only then `nsys` (kernel
   families); `ncu` last and scoped. Triage `nsys` GPU idle gaps: CPU busy =
   starvation (for us: spool JSON/b64 decode = the WP12 case), both idle =
   sync stall, cudaMalloc = allocator churn.
5. Capture steady state only (`cudaProfilerStart` after warm-up) so cuDNN
   autotune / compile does not pollute the trace.

### Actionable queue additions
- [ ] A60: CUDA EP switch - uninstall CPU ORT, install `onnxruntime-gpu`,
      import torch first, verify session providers, enable max-workspace;
      workers then use `prefer_cuda` (code path already exists)
- [ ] A61: profiling playbook - nsys steady-state runs with gap triage;
      `torch.profiler` before any kernel-level work
- [ ] A62: workers log requested vs actual EP on hot-reload (no silent CPU)

## Cycle 15 — 2026-11-24

Sources: "When, Where and Why to Average Weights?" (arXiv 2502.06761),
SWA revisited (2201.00519), SeWA (2502.10119), PyTorch SWA blog, Lightning
WeightAveraging docs.

### Findings

1. **Averaging horizon ≈ 1 % of the total training budget** is the broad
   optimum (LAWA measured across workloads; EMA similar). Our EMA at
   decay 0.999, updated every optimizer step (96 steps/iteration),
   averages ~1 000 steps ≈ 10 iterations ≈ 1 % of a 1 000-iteration run -
   the default is already on target; document it.
2. **Sparse-update EMA wants ν ∈ [64,128] steps between updates with
   β ≈ 0.9**; per-step updates pair with β ≈ 0.999. We update per step →
   keep 0.999 (0.995 only if the window proves too long).
3. **Averaging is a proxy for LR cooling**: its benefit shrinks when the LR
   is fully annealed to zero. Our step-decay schedule (no full anneal)
   keeps the EMA upside; if we move to cosine-to-zero, re-evaluate.
4. SWA proper needs a high-constant LR phase (last 25 %) - not our setup;
   **LAWA** (rolling window of the last ~20 checkpoints, updated every
   64-128 steps) is the fallback if EMA proves hyperparameter-sensitive.
5. EMA is more sensitive to its hyperparameters than LAWA - keep the decay
   a profile knob and A/B it once at the night-v3 generation.

### Actionable queue additions
- [ ] A59: keep EMA decay 0.999 (horizon ≈ 1 % of budget); log the horizon;
      consider LAWA (K≈20, update every 96 steps) as the fallback at night-v3

## Cycle 14 — 2026-11-24

Sources: d2l batch-size chapter (McCandlish critical batch size), Malladi
et al. 2022 (Adam SDE scaling), warmup-mechanisms paper (2406.09405),
LiteLLM-stack notes; Nemesis (Lichess Elite training, 2026), AZ paper Table 2
(human openings), AZ_db diversity paper (puzzle-start self-play).

### Findings

1. **Adam/AdamW scale with √b, not linearly**: SGD wants η∝b; adaptive
   optimizers want η∝√b up to the critical batch size (b_crit ≈ gradient
   noise scale). We doubled batch 256→512 but kept η=1e-3 — the rule says
   ~1.41e-3; our LR may be *conservative on the batch axis* while Tiny-AZ
   says it's *hot on the params axis*. Resolve with the planned A38 A/B.
2. **Warmup**: max(1000, 2% of total) steps from scratch; only 100–500 steps
   when starting from a checkpoint (our warm starts). If a run diverges early,
   lengthen warmup before lowering the peak LR — our LRStepSchedule has **no
   warmup**; fold one into the A28 optimizer package.
3. **Human-opening seeding**: Nemesis (2026) trains on 3.8M Lichess-Elite
   games and argues self-play from a random net "generates noisy positions
   that slow early convergence" — plans a precomputed opening book from the
   Elite DB. AZ independently rediscovered the 12 most popular human openings
   and beat SF from each. → seed a fraction of self-play from a small curated
   book (4–8 plies, popular positions) and measure early-game convergence.
4. **Puzzle-start self-play works as a curriculum**: AZ_db started self-play
   games from puzzle positions (solution withheld) and its solve rate rose
   11.8% → 76.5% on the hard Challenge set — directly training tactical
   solving via the normal RL loop.

### Actionable queue additions
- [ ] A55: fold √b LR rule into A38's A/B (batch 512 ↔ η 1e-3 vs ~1.4e-3)
- [ ] A56: curated opening-book seeding (500–2,000 positions ≤8 plies from
      masters/Elite) for a fraction of self-play games; measure convergence
- [ ] A57: periodic puzzle-position self-play curriculum (start games from
      benchmark positions; solutions withheld)
- [ ] A58: 500-step warmup in the A28 optimizer package (short, checkpoint-
      initialized); lengthen-before-lowering rule documented

## Cycle 13 (brief) — 2026-11-24
EMA of weights beat last iterate +65 Elo avg / +32 at 39k iters (Generals.io
2026); Lc0 ships SWA nets. GambitNet 10×128 = Lc0 canonical ResNet minus
Squeeze-and-Excitation; SE block standard in all modern Lc0 nets. Chess-mask
5×5 kernels (rook/bishop) act as structured regularizers. → A51 EMA, A52 SE,
A53 masks, A54 bundle next-gen into one restart.
