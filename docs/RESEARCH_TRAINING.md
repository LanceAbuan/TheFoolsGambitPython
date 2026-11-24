# Training Research Log (autonomous loop)

Purpose: continuous, incremental research on training/server optimization for
Fool's Gambit. Each cycle: read this file, research **new** topics (websearch /
webfetch), append findings + actionable items, commit. Cycles are listed
**newest first**. Focus = training quality, sample efficiency, throughput.
Never touch the live training run.

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
