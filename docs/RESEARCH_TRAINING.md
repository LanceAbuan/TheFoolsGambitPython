# Training Research Log (autonomous loop)

Purpose: continuous, incremental research on training/server optimization for
Fool's Gambit. Each cycle: read this file, research **new** topics (websearch /
webfetch), append findings + actionable items, commit. Cycles are listed
**newest first**. Focus = training quality, sample efficiency, throughput.
Never touch the live training run.

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
