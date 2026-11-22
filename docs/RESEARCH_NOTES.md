# Research notes — fresh-eyes companion, 2026-11-21

> **Provenance.** Composed by the fresh-eyes audit session from its own session evidence (findings F-1…F-14 live in `docs/AUDIT_2026-11-21.md`), merged with the SOTA survey session's ranked backlog. Backlog numbering follows the **dispatched plan of record** (U6 multiprocess shards and U9 Glicko-2 were already queued to the training worker in writing, so they own those numbers); audit-proposed items continue at U12. Authored from audit-time measurements on branch `feature/play-and-local-overhaul` @ `ed149ad`; notes landed @ `b36d82e`. Method: read-only review of the server, training, Rust, and web trees plus the full check battery, run against `docs/RESEARCH_STANDARD.md`. Two builder workers were active; `training/selfplay.py` grew ~1290 → ~1546 lines during the session (U1/U2 landing) — in-flight items are marked, not scored.

Measured this session: **pytest 346 passed / 0 failed / 0 flaky (53.0s)**; **svelte-check 0 errors / 0 warnings** (Node 22.23.5; the default-PATH Node 20.10 fails svelte-check on `node:util/styleText` — environment artifact); **vite build ✓** (JS 153.45 kB / 55.82 kB gzip; CSS 49.93 kB / 10.8 kB gzip); **cargo check + clippy clean**; **ruff 149 errors** (129 in legacy files); current `data/train_log.jsonl` 0.01 MB, `gambit.db` 0.13 MB (+WAL 3.95 MB), `runs/` 20.4 MB.

## 1. Focus areas

### 1.1 Correctness (training math, gates, chess rules, resume, races)

- **The arena gate is inert within a run (F-1, P1).** `training/pipeline.py:479` and `:508` set `champion_model = model` — the *live, still-training module object*. At the next gate `_arena_match(model, champion_model)` plays the model against itself; with argmax moves, no root noise, and shared weights, the 16-game match (8 per color) is deterministic and color-symmetric, so the score rate pins at 0.5 < 0.55 and **no checkpoint after the first can ever be promoted in one invocation**. Each such gate also burns ~16 games × 128 sims × ≤200 plies — roughly half a training iteration's search budget — for no information. The suite misses this because the only cross-champion test gates *across* runs (`tests/test_training_pipeline.py:79`), where the champion is loaded from a checkpoint file into a fresh object.
- **Candidate Elo resets every gate (F-5, P2).** `pipeline.py:461` passes `profile.ladder.base_elo` as the candidate's rating into `update_elo` (`ladder.py:154`), so the pipeline-local rating never accumulates; combined with F-1 the delta is exactly 0. `recompute_elos` (`ledger.py:90`) recomputes honestly for the metrics feed, but the ledger rows themselves store the reset values — and the ledger is what `server/db.py` mirrors and the Watch chart plots.
- **Chess-rules handling is in good shape.** The Rust core is perft-gated on positions 1–6 including Kiwipete (startpos d5 = 4,865,609; `gambit_core/src/position.rs:534-589`), FEN round-trips, and the raw-ep semantics match python-chess `fen(en_passant="fen")` (`position.rs:99-127,179-188`). Repetition claims (threefold/fivefold — the one thing a stackless core can't own) are correctly delegated: the MCTS layer uses `is_game_over(claim_draw=True)`, and the M3 `FastBoard` override walks the `_stack` snapshots instead of push/pop replay, with the empty-stack root cleaned via a faithful port of `clean_castling_rights` (`selfplay.py:532-706`). Castling through check, en passant legality, promotion/underpromotion, insufficient material (including the KB-vs-KB same-color and KN-vs-KN non-terminal cases, `position.rs:436-471`) all reduce to python-chess or its ported logic. The §10 mate-sign fixes (`terminal_value` = −1.0 for the mated side-to-move, `selfplay.py:858-867`) are regression-documented and consistent across legacy and v2 paths.
- **Resume is the strongest proof in the repo.** `tests/test_training_resume.py:194` (`test_resumed_loss_matches_fresh_run`) is a genuine resume-reproduces-fresh differential; profile-signature refusal on strict-key drift (`resume.py:78-91`) and corrupt-state degrade-to-fresh (`resume.py:154-168`) are both covered. The cost side is the problem, not the correctness side (F-2 below, §1.2).
- **Concurrency, checked case by case:** the reaper re-validates expiry *under the session lock* so a move racing the reaper always wins (`reaper.py:45-49`); `QueuedNNEvaluator` keeps per-request futures and slices fan-out rows — zero-cross-game-leakage is test-pinned (`tests/test_training_evaluator.py`); journal appends are lock-serialized whole lines with a persistent handle (`training/journal.py:58-67`); `Hub` seq allocation and subscription mutation are under one asyncio lock (`server/ws.py:44-77`). The real race-shaped risk is not data corruption but **head-of-line blocking** in `Hub.send` (F-6): serial `await ws.send_json` per connection means one stalled TCP peer backpressures the whole topic.

### 1.2 Performance honesty

- **Claims with numbers exist and carry caveats — the standard is visibly followed where it's visible.** The first-light ledger records the honest miss: 0 of 32 games finished in 71 min, GIL-bound to ~1.0 core (66.9 CPU-min in 71 wall-min with 16 game threads), throughput floor < 27 games/h against a 3k–20k prediction (RESEARCH_STANDARD, pre-registration table P3). The re-bench commit (`809eb5d`) records "p32 655 sims/s, 14.7k games/h; contention suspected". N1–N4 are pre-registered and pending. No decorated number pretending to be a measurement was found.
- **The two structural costs that produced the first-light diagnosis are still in the hot path.** The v2 `_simulate` does a full `board.copy()` per simulation (`selfplay.py:943-944` pre-U1 numbering; the copy carries the move stack under FastBoard), and MCTS is sequential per game — throughput currently comes from `parallel_games` sharing one queued evaluator, not from parallel sims. These are *hypotheses with a mechanism*, not findings — no fresh number exists yet; the M3/U1 benches (untracked `scripts/bench_m3_gate.py`) are the right instrument and must land with their numbers per §2.
- **State-size blowup is arithmetic, not speculation (F-2, P1).** One example ≈ planes 29·8·8·4 B ≈ 7.4 KB + policy 4672·4 B ≈ 18.7 KB ≈ 26 KB. Night profile: `buffer_size=100_000` ⇒ ~2.6 GB per `save_run_state` payload (`resume.py:123` materializes `list(buffer.buffer)` first — same again in RAM), one new unpruned file per ladder snapshot (5/10/15/20 min, then every 20), buffer filling after ~16 night iterations. This is the most likely way tonight's run dies by itself.
- **Unbounded growth inventory:** `train_log.jsonl` (append-only, per-ply `live` lines at p16 ≈ 6.4k lines/iteration), `runs/.../state-*.pt` (F-2), `_sessions` in `server/engine.py` (finished games are never evicted, F-7), `games`/`moves` SQLite tables (no retention — acceptable at friends scale, worth a note). Currently small (`train_log.jsonl` 0.01 MB), all of it prospective — which is exactly when nobody is watching.

### 1.3 Test quality

- **No assert-nothing tests found.** Spot-audited suites assert concrete values (perft counts, bit-exact differentials, ordering contracts). The 346 grew from 306 with the M3 differential tests included and green.
- **Highest-value proofs present:** resume-reproduces-fresh; evaluator zero-leakage with interleaved fake requests; WS journal tail (18.7 KB of tests covering partial lines, truncation/rotation, subscribe-time seek); policy-index collision assertion in `legal_move_map` (`encoder.py:286-292`).
- **CI-shaped gaps:** (a) no test exercises *within-run* multi-gate promotion — the exact hole F-1 lives in; (b) no budget assertion that a state file stays under N MB; (c) the U2 flag-off bit-identity golden test is promised in the docstring (`tests/test_training_playout_cap.py`) but absent at audit time (F-14); (d) CI has no Rust job at all, so the perft-proven core can regress CI-unnoticed (F-4); (e) CI's `ruff check .` is already red — 149 errors, 129 concentrated in `training/persistence.py` (58), `training/server.py` (36), `training/stockfish_engine.py` (26), `train_server.py` (9).

### 1.4 Security / robustness for hosting friends

- **Solid:** pydantic `Literal` enums + regex-validated squares on the play API (`server/api/play.py:19-30`); admin endpoints use `secrets.compare_digest` and fail 503-when-unconfigured (`server/api/train.py:51-55`); SPA fallback resolves and containment-checks paths (`server/main.py:80-81`) — no traversal; per-IP token bucket on POSTs with bounded bucket map (`ratelimit.py`); CORS default is localhost-only; SF processes pinned to 1 thread / 16 MB hash with hard-kill fallback (`server/stockfish.py`).
- **Gaps:** WS is unauthenticated and unthrottled, and each `train` subscribe triggered `journal.backlog()` → full-file read before tonight's F-3 fix (DoS amplification, worsening as the journal grows; the read is now O(last-50-lines) — the unauthenticated/unthrottled WS *surface* itself remains, worth per-connection subscribe caps before public exposure); `torch.load(..., weights_only=False)` at `model_server.py:83`, `resume.py:161`, `pipeline.py:177` is a pickle-RCE surface if `runs/` is ever poisoned (F-13); rate limit covers POST only. HF token handling is env-only with graceful no-token skips (`trainer.py:42-93`) — no secret found in the tree.

### 1.5 Architecture

- **Two MCTS stacks (assessment requested).** Verdict: v2 (`MCTSSearch` + `BatchNNEvaluator`/`QueuedNNEvaluator`) is the authoritative path — the pipeline, arena, and `server/model_server.py` all run it. Legacy (`MCTS`/`SelfPlayGame`/`BatchEvaluator` with Stockfish-leaf blending, 8 tree-lock threads) is still *wired* into `training/trainer.py` and `training/server.py`, still lint-red, and carries its own `estimate_elo` (`win_rate·400−200`, `trainer.py:378`) that is a §2-class unwarranted number if ever surfaced. Recommendation: quarantine/delete (F-8 → U15, L), not refactor. The module docstring of `selfplay.py` (lines 1-11) still describes the legacy path — misleading for the file's primary consumer.
- **Config sprawl (F-9):** `RegimeConfig.num_simulations / games_per_iteration / train_steps_per_iteration / batch_size` are dead — the pipeline takes all four from `TrainingProfile` (`pipeline.py:353-357`). Five regimes carry plausible values that change nothing. Single source of truth needed.
- **Clean boundaries worth keeping:** `training/journal.py` (writer) vs `server/journal.py` (reader) with no import cycle; `training/fastpath.py` as the single loud-failure gate for the Rust path (a silent fallback would poison the ablation — the docstring explicitly cites §2); `EventSink`/`Evaluator` structural protocols keeping selfplay decoupled.

### 1.6 Web client

- **The `$state(Map)` fix class is clean.** WatchView now uses a plain `Record<number, LiveSlot>` with per-cell newest-fen-wins derived state (`WatchView.svelte:169,291-299`); run-unique game ids (`iteration·N + k` from `pipeline.py:366` / `StreamContext.game_offset`) make the modulo addressing sound; MAX_STORE=64 eviction + game_end TTL + endTimers cleanup in the effect teardown. No siblings of the Map bug found.
- **Siblings of a different kind:** demo mode is hard-coded human-white — `tryMove` gates on `turn !== 'w'` (`game.svelte.ts:79`) while `scheduleDemo` plays black even when the human picked black (`:295`), so live-fallback-as-black is a self-playing board (F-10); live mode ignores `snap.status/winner` and re-derives from FEN, so threefold/50-move/reaper-finished games render "playing" until a 409 (F-11); `ws.ts` documents a `last_seq` resync the server never implements (F-12).
- **Sound:** reconnect with 1 s→15 s backoff, resubscribe-all, fresh-join slot reset on status `open`; journal DOM capped at 200 lines with follow-pause; a11y basics present (aria-labels/aria-pressed on live cells, focus-visible rings, aria-expanded primer). Bundle is healthy (56 kB gzip JS) with `chess.js` the only runtime dep of note.

### 1.7 Operational

- Server lifecycle artifacts exist (`data/server.out.log`, `server.err.log`, `heartbeat.log`, `patrol.log`, currently ~0 MB). The unattended-run killers ranked: F-1 (run produces nothing of value), F-2 (disk/stall), F-3 (WS feed degradation). Backup/recovery of lineage is layered (per-checkpoint `.pt`, `champion.json` pointer, per-regime HF upload, resumable state) — the design is right; only the state-file economics are wrong (F-2). WAL at 3.95 MB vs 0.13 MB db is normal SQLite behavior at this write rate.

## 2. Technique ledger

Format: what / citation / expected-effect / cost / risk / verdict. Principle 6 applies: techniques with no citation get none and say so.

- **M3 FastBoard stack-walk repetition claims** — landed this night (uncommitted at audit time).
  What: override `can_claim_threefold_repetition`/`is_repetition` to walk `_stack` snapshots (no push/pop replay), root-snapshot rights cleaning included.
  Citation: python-chess 1.11.2 internals (private `_BoardState`/`_stack`); profiling from `tmp_m3` profiles, 2026-11-21.
  Expected-effect: removes the largest per-sim sink (`is_game_over(claim_draw=True)` at O(ply+moves) python mutations, twice per sim); supports the N1 ≥3× e2e sims/s gate.
  Cost: S (already written + differential-tested).
  Risk: private-API dependency pinned by uv.lock; any python-chess bump must re-run the differential suite.
  Verdict: **land after the differential suite stays green; flip `GAMBIT_CORE_FASTPATH` via the orchestrator, never by default** (the gate's loud-failure policy is correct). N1 outcome recorded: gate MISS at honest 1.1× — the wireable share was ~4.5% of GIL demand; the slice lands as a free win + foundation for search-in-Rust (U12).
- **M3 FFI encode/legal-move batch** — landed this night.
  What: `gc.encode_batch(fens)` and `gc.legal_move_pairs(fen)` one-FFI-call paths replacing per-square Python loops; UCI→`chess.Move` memo (`encoder.py:139-148`).
  Citation: engineering (no paper); correctness contract is bit-exactness with the Python path, order-differential-fuzzed.
  Expected-effect: eliminates ~30 `chess.Move.from_uci` + plane-loop calls per expansion; part of N1.
  Cost: S (done).
  Risk: FEN bridge must use `en_passant="fen"` or plane 28 silently zeroes — verified handled at `encoder.py:188-194` with a comment; keep the differential fuzz in CI (currently only local).
  Verdict: **keep gated; measure before and after on the same box** (the re-bench's "contention suspected" caveat shows why).
- **U1 Gumbel AlphaZero search** — in-flight (code landed during this audit; ablation bench running).
  What: Gumbel-Top-k root sampling (m=16) + sequential halving; act by Eq. 8; train toward Eq. 11 improved policy; below-root stays PUCT (Full variant's Eq. 14 interior documented as a non-goal).
  Citation: Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 (Alg. 2; c_visit=50, c_scale=1.0 chess setting). Citation verified: ICLR 2022 spotlight, OpenReview `bERaNdoegnO`.
  Expected-effect: ~64–100 equally-good sims vs PUCT@400 ⇒ ~4× games/h (N2 gate: ≥4× games/h at win-rate ≥0.35).
  Cost: M (done, unbenched).
  Risk: improved policy replaces visit distribution as target — a training-distribution change riding along with a search change; the mixed-value imputation (Appendix D) is subtle; rng-stream coupling with U2's cap draws.
  Verdict: **ablation-gate exactly as pre-registered (N2); do not enable alongside M3's flip in the same run** — one change at a time (§ principle 4).
- **U2 Playout-cap randomization** — in-flight.
  What: share p=0.25 of turns searched fully (recorded); rest get a fast noise-off search (n = 0.25·sims, min 2), unrecorded by default (`playout_cap_record_fast=False`, matching the paper).
  Citation: Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2020, sec 3.1.
  Expected-effect: data diversity ⇒ strength-per-compute; stacks with U1.
  Cost: S (done, unbenched).
  Risk: flag-on changes rng consumption (documented, isolated to `cap_rng`); fewer recorded examples per wall-clock hour; the promised flag-off bit-identity golden test was absent at audit time (F-14) — **update, same night: `tests/test_training_playout_cap.py` + `tests/fixtures/search_golden.json` now present in the working tree; verification happens at integration.** Also: the Gumbel×playout-cap interaction (KataGo pruning presumes PUCT visit targets; Gumbel builds targets from completed-Q + logits) has been flagged to the training worker — the U2 flag must state which target construction it applies to; silent interaction = silently reshaped targets.
  Verdict: **block on the golden test verifying green; then orchestrator-gated after U1's gate** (sequenced per the standard).
- **FS-4 QueuedNNEvaluator micro-batching** — landed.
  What: one worker thread, 256-batch / 4 ms window, per-request futures, fail-loud close.
  Citation: plan §5 lever 2 (engineering).
  Expected-effect: GPU utilization at `parallel_games>1`; measured implicitly by every p16 run since.
  Cost: S (done).
  Risk: single-caller latency penalty (documented — sequential paths correctly use `BatchNNEvaluator` instead).
  Verdict: **keep.**
- **Champion gate snapshot (F-1 fix)** — in flight with the training worker.
  What: gate candidate vs a *frozen* champion loaded from the saved checkpoint, never the live module; accumulate candidate Elo from the ledger (F-5).
  Citation: standard §5.1 (the gate rule itself).
  Expected-effect: the ladder becomes real; U1's win-rate gate becomes meaningful; arena compute stops being wasted on self-matches.
  Cost: S.
  Risk: none identified beyond needing the within-run test.
  Verdict: **prerequisite for tonight's run being worth anything.**
- **Replay-state compaction (F-2 fix)** — in flight with the training worker.
  What: persist buffer as concatenated float16 npz (or subsample), prune `state-*.pt` keeping newest 1–2.
  Citation: none (engineering; principle 7 — simplest thing that meets the gate).
  Expected-effect: ~4× smaller state files, ~minutes-faster saves, no overnight disk exhaustion.
  Cost: M.
  Risk: fp16 planes/policy — training later re-widens to fp32; bit-identity of resume is relaxed unless tested, so keep a resume-reproduces-fresh variant for the compacted path.
  Verdict: **do before launch or early in the night.**
- **Journal bounded tail + rotation tolerance (F-3 fix)** — **LANDED 2026-11-21 night (`b36d82e`).**
  What: `backlog()` reads a byte-window via backward 64 KB-chunk newline scan — O(last-50-lines span), never O(file); cursor reconciles on size-shrink OR file-identity change (`st_dev`/`st_ino`); 9 new tests incl. a 50k-line subscribe and a poisoned-`read_text` regression gate.
  Citation: none (engineering).
  Expected-effect: constant-cost WS subscribes at any journal size; removes the self-DoS lever.
  Cost: S.
  Risk: rotation policy is writer-side (training worker) — the reader now tolerates any rotator behavior (truncate-in-place or delete-and-recreate).
  Verdict: **landed.** Remaining: per-connection subscribe caps on the unauthenticated WS surface before public exposure.
- **U12 Leaf-parallel MCTS with virtual loss** — candidate, not started.
  What: run sims concurrently against the shared tree, batching leaf evals through the queued evaluator.
  Citation: Silver et al., *AlphaZero*, Science 2018 (massively parallel self-play search); Silver et al. 2017 (virtual loss).
  Expected-effect: GPU saturation at p16 without relying on inter-game parallelism alone; compounds with U1/U2.
  Cost: M.
  Risk: tree-lock contention re-creates the first-light GIL problem in Python; confounds U1/U2 ablations if landed in the same run.
  Verdict: **defer until U1+U2 gates are scored; then ablate separately.**
- **U9 Glicko-2 deviation + SPRT + Stockfish anchors** — dispatched to the training worker's queue.
  What: RD-tracked ratings per checkpoint lineage; promotion via SPRT [0,10] α=β=0.05 or an RD-aware rule; periodic anchor matches at fixed SF skills.
  Citation: Glickman 1999 (Glicko-2); CPW pentanomial methodology; CCRL anchor methodology.
  Expected-effect: "847 ± 31" becomes a falsifiable statement; 2-game gates carry ≈ ±120 Elo @95% (347/√N arithmetic), so today's 55% promote threshold is noise-dominated; human-comparable equivalence via anchors.
  Cost: S/M.
  Risk: SF skill-level → Elo mapping is nonlinear and platform-variable — must be measured, not assumed (the anchor pool *is* the calibration). Supersedes the standard's original U3 line item; pairs with F-5 (stop fabricating candidate Elo history first).
  Verdict: **queued after U6 in the training worker's stack.**
- **U5 LR schedule + optimizer hygiene** — parked (per standard; no change).
  Citation: Silver et al. 2018 appendix.
  Verdict: fold into a U1 ablation only if a gate demands it.
- **MuZero-style learned model** — **rejected** (already documented in the standard; restated here so the ledger stays complete in one file). Exact fast simulator exists, perft-proven; principle 7.

## 3. Ranked backlog (merged: survey U5+ + audit additions)

In-flight findings serviced tonight (not backlog rows): **F-1** champion snapshot (pre-launch blocker, training worker), **F-2** state compaction (pre-launch, training worker), **F-3** journal tail (landed, `b36d82e`), **F-4** lint quarantine + Rust CI (orchestrator, this night).

| # | Upgrade | Type | Citation | Expected effect | Cost | Risk | Verdict/status |
|---|---------|------|----------|-----------------|------|------|----------------|
| U6 | **Multiprocess self-play shards** (3–4 procs × 4–6 threads each; per-process model replicas vs shared IPC evaluator — measure VRAM, pick with data) | process architecture | ELF OpenGo, arXiv:1902.04522 (worker/GPU split) | **3–4× games/h — the single biggest lever.** N1's honest 1.1× confirms native movegen alone can't move games/h against a measured 1-core GIL ceiling; the process level is where ELF/KataGo/lc0 all went | M | interleave timing vs determinism (resume-reproduces-fresh must stay green for state replay); config-gated `--selfplay-procs 1` default, flag-off bit-identical | **dispatched to training worker's queue** (after F-1/F-2/U1/U2/F-5); pre-register games/h gate |
| U7 | **SE blocks + moves-left aux head + policy-target pruning (PUCT path only)** | architecture | Hu et al., CVPR 2018 (SE gains *larger* on small nets — MobileNet −3.1 top-1); KataGo global-pooling ablation, arXiv:1902.10565 (removal = 1.60× slowdown); lc0 moves-left head | strength-per-compute at small scale; lc0's head is the chess-proven analogue | M | breaks checkpoint continuity; pruning presumes PUCT visit targets — PUCT-path only (see U2 interaction note) | adopt-after-overnight |
| U8 | **bf16 inference + `torch.compile(reduce-overhead)` with batch buckets {1,8,32,256}** | eval hot path | CUDA-graphs literature (survey cite arXiv:2605.30571) | −20–50% eval latency; batch-256 at 2.11 ms is launch-bound at small batches | S/M | recompile on legal-mask shape changes; dynamic shapes; bench-gated | bench-gated, after U6 |
| U9 | **Glicko-2 (r, RD, σ) + SPRT [0,10] α=β=0.05 on the arena ledger** | rating science | Glickman 1999; CPW pentanomial | 32-game gates carry ≈ ±120 Elo @95% — today's 55% threshold is noise-dominated; promotion becomes statistically meaningful; supersedes standard-U3 | S | gate match size vs overnight cost — find the honest middle, pre-register | **dispatched to training worker's queue** (pairs with F-5) |
| U10 | **WDL 3-head value** (replaces scalar tanh) | architecture | lc0; KataGo | better termination/draw handling | M | additive §14.0 payload fields needed | after U9/U6 |
| U11 | **Reanalyse** (re-search stored buffer positions with the current net for fresh targets) | data efficiency | MuZero Unplugged, NeurIPS 2021; EfficientZero, arXiv:2111.00210 | fresh targets at zero new games — pure compute re-allocation, scales down fine | M | compute tradeoff vs new games | queued |
| U12 | **Leaf-parallel MCTS + virtual loss** | search | Silver et al. 2018/2017 | GPU saturation at p16; compounds with U1/U2 | M | re-creates GIL contention in Python; confounds ablations if stacked same-run | deferred until U1+U2 scored, then ablated separately |
| U13 | **Cross-regime arena pool** (champions challenge across `runs/<regime>/`) | method | standard §5.2 | the regime matrix becomes an actual experiment; feeds U10 | S/M | regime Elo comparability needs U9 anchors first | queued behind U9 |
| U14 | **Data symmetry augmentation (horizontal mirror)** | data | **none verified** | potentially ~2× effective data if legal | S | castling/ep mirroring correctness; AlphaZero's own chess recipe did not use it | **declined pending a verified citation** (principle 6) |
| U15 | **Legacy stack quarantine** (`trainer.py`, `training/server.py`, `train_server.py`, `persistence.py`, `stockfish_engine.py`, `critic_game.py`) | infra | — | −100 KB divergent code, −129 lint errors, one MCTS truth; kills the fake `estimate_elo` (`win_rate·400−200`) | L | legacy HTTP trainer consumers must be ported or dropped | schedule after tonight |

Ranking rationale: U6 + the F-fixes are the difference between "the overnight run produces a result you can score" and "it produces a flat line and a full disk"; U7/U8/U11/U12 are strength-per-compute levers that only become measurable once the gates (F-1) and the ablation discipline (U1/U2 sequencing) are trustworthy.

Survey non-findings (checked, honestly uncited): no paper named "MCES"; no "Devin"/"Lila" chess projects found; Kolberg 2023 regularization unverifiable this session — none cited anywhere above (principle 6).

## 4. Contradictions (claims vs code, as-found)

1. `web/src/lib/api/ws.ts:5-9` — "last_seq … so the server can resync on reconnect" vs `server/ws.py:110-122`, which ignores `last_seq` entirely (snapshot + backlog instead). Behavior is coherent; the contract text is not. (F-12)
2. `training/regimes.py:1-13` — "the §5.2 experiment matrix as swappable configs" while four of its training knobs (`:68-71`) are dead code the pipeline never reads (`pipeline.py:353-357`). (F-9)
3. `training/trainer.py:378-396` — `estimate_elo` maps `win_rate·400−200` onto "ELO" vs standard §5 (ratings are paired-comparison estimates with uncertainty; this is neither).
4. `training/trainer.py:160` — constructs `StockfishPlayer(depth=11)` while the very next log line says "(depth=10)" (`:161`). Trivial, but it is a claim without a number agreeing with itself.
5. `selfplay.py:1-11` module docstring — describes Stockfish-leaf-blended rollouts, i.e. the legacy path, while the file's authoritative export (v2 `MCTSSearch`) is docstring-ed separately at `:709-716`. One file, two self-descriptions.
6. Watch primer copy (`WatchView.svelte:610-612`) — "A newly trained version only becomes the champion if it beats the previous champion" and "Its rating climbs" vs F-1: within a run, no version can ever beat the previous champion because they are the same object. The UI promises a mechanism that is currently broken. *(F-1 fix in flight — this contradiction expires when it lands.)*
7. `ledger.py:11-13` — "per-checkpoint Elo computed from the ledger" vs `pipeline.py:461,489-491` — pipeline-local Elo that resets the candidate each gate and is what actually gets written into ledger rows.
8. Git-history claim "final gate 306/306" (`5905131`) vs current repo state at audit time: ruff red (149), no Rust CI job, U2's promised golden test absent. The gate that was run was narrower than "the project is gated". *(F-4 services the ruff/CI half of this the same night.)*
9. `tests/test_training_playout_cap.py` — cited as the bit-identity gate in `selfplay.py:1231-1234`, did not exist at audit time (F-14). *Update: the file (plus `tests/fixtures/search_golden.json`) appeared in the working tree later the same night; the gate check happens at integration.*

## 5. Non-findings (checked, suspected, and cleared)

- **SPA path traversal** (`server/main.py:73-83`): `(WEB_DIST / full_path).resolve()` + `is_relative_to(WEB_DIST.resolve())` + `is_file()` — encoded-dot and backslash variants resolve into the check. Cleared.
- **FastBoard's restore-trick** (`selfplay.py:587-595,630-637`): saves a live `_BoardState`, restores the historical snapshot to evaluate `has_legal_en_passant()`, restores the live state in `finally`. Correct under exceptions; differential-tested. Cleared.
- **Gumbel rng discipline** (`selfplay.py:910-920,1062`): PUCT path and the global-rng Dirichlet draw do not consume the passed generator; only Gumbel draws do — seeded PUCT runs reproduce. Cleared (pending the U2 golden test for the cap-rng interplay).
- **BatchEvaluator OOM handling** (`selfplay.py:94-134`): half-split retry, `torch.cuda.empty_cache()`, zero-fill fallback sets *all* waiters (no partial-result deadlock). Cleared.
- **QueuedNNEvaluator lifecycle** (`evaluator.py:147-157`): close fails queued futures with exceptions rather than stranding callers. Cleared.
- **`parallel_self_play` partial failure** (`selfplay.py:1419-1487`): order-stable results, per-game failure logging, all-fail raises, `rng.spawn` children keep seeded runs deterministic. Cleared.
- **Encoder round-trip**: collision assertion on index reuse (`encoder.py:286-292`), decode boundary checks (`:239-270`), underpromotion planes distinct from queen promotions (the §10 bug cannot reappear by construction). Cleared.
- **SQLite hygiene** (`server/db.py`): WAL + single connection behind a lock with commit/rollback wrapper; arena backfill made idempotent by the `(ckpt, ts)` unique index; foreign keys on. Cleared. *(Night note: `busy_timeout` hardening dispatched separately.)*
- **Rate limiter bucket growth** (`ratelimit.py:36-56`): shed-on-overflow bounded at 10k entries, sorted by idle. Cleared. (The WS-side gap is F-3's, not the bucket's.)
- **Reaper/move race** (`reaper.py:45-49`): expiry re-checked under the session lock. Cleared.
- **Undo parity** (`engine.py:288-310`): pop-count trimming keeps the human on turn and never strands the game before the engine's opening reply; finished games refuse undo with 409. Cleared. *(Night note: undo-after-finish for casual play dispatched separately.)*
- **WS seq monotonicity** (`ws.py:58-77,103-107`): subscribe snapshots consume from the same per-topic counter as broadcasts, so a replaying client sees one monotonic stream. Cleared.
- **Node/npm environment**: the svelte-check failure reproduces *only* under the PATH Node v20.10.0 (`styleText` missing); under the project's Node 22.23.5 it is 0/0. Not a code defect; worth a `package.json#engines` note someday.
- **Champion hot-reload safety** (`server/model_server.py:137-160`): mtime-keyed, lock-guarded, failed reload keeps the previous model serving, vanished pointer keeps serving. Cleared.

## 6. Standing numbers (this session, single machine, contention caveats apply)

| Quantity | Value | Method |
|---|---|---|
| Test suite | 346 passed / 0 failed / 0 flaky | `uv run python -m pytest -q`, 53.0 s |
| Type check (web) | 0 errors / 0 warnings | `svelte-check` under Node 22.23.5 |
| Bundle | 153.45 kB JS (55.82 gzip), 49.93 kB CSS (10.8 gzip) | `vite build` |
| Rust | check + clippy clean | `cargo check`, `cargo clippy` |
| Lint | 149 errors (58 persistence.py, 36 training/server.py, 26 stockfish_engine.py, 9 train_server.py, 21 elsewhere) | `ruff check .` |
| Journal size now | 0.01 MB (`train_log.jsonl`) | `stat` |
| DB size now | 0.13 MB db / 3.95 MB WAL | `stat` |
| Projected state file at night-buffer fill | ~2.6 GB (100k × ~26 KB/example) | arithmetic from dtypes; not yet observed |

Everything above is either measured this session or read directly from the named file/line. The two numbers flagged as *not yet existing* and most needed: the U1 ablation under controlled contention (N2's gate — running as of this writing), and a state-file size budget test (F-2's acceptance metric).
