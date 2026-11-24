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
