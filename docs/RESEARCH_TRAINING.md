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

## Cycle 13 (newest, brief) — 2026-11-24
EMA of weights beat last iterate +65 Elo avg / +32 at 39k iters (Generals.io
2026); Lc0 ships SWA nets. GambitNet 10×128 = Lc0 canonical ResNet minus
Squeeze-and-Excitation; SE block standard in all modern Lc0 nets. Chess-mask
5×5 kernels (rook/bishop) act as structured regularizers. → A51 EMA, A52 SE,
A53 masks, A54 bundle next-gen into one restart.
