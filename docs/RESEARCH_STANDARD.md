# Research Standard — how this project does intellectual work

Adopted 2026-11-21 at the owner's direction: *everything we do should be as
intellectually serious as possible.* This document is the working agreement
that makes that checkable. It binds every worker and every future slice.

## Principles

1. **Proof over hope.** Correctness claims are pinned by tests that would
   catch the failure (perft counts, bit-exactness differentials,
   resume-reproduces-fresh, failing-then-passing proofs for bug fixes).
2. **Measure first.** No performance claim without a recorded number, its
   method, and its variance. Record the caveat with the number (see the
   sims/sec environment-variance note in §13).
3. **Falsifiable predictions.** Every real run gets pre-registered
   predictions *before* data arrives, scored afterwards (ledger below).
   Motivated reasoning dies in daylight.
4. **One change at a time.** Training-recipe changes are ablations by
   default: change one thing, gate on the acceptance metric, keep or revert.
   Stacked changes are only allowed when each was individually gated or the
   combination is the hypothesis being tested.
5. **Uncertainty is part of the result.** Ratings are reported with error
   bars; benchmarks with best-of-N and variance; "it improved" requires the
   interval, not the point estimate.
6. **Cite the literature; document the rejections.** Techniques enter the
   codebase with a citation to their paper. Equally important: techniques we
   *decline* get a written reason (see the MuZero note), so decisions are
   auditable and never re-litigated from scratch.
7. **Intellectual ≠ complicated.** The simplest design that meets the gate
   wins. Overengineering is a failure mode, not sophistication (the MuZero
   rejection is also this principle in action).

## Training-method upgrade backlog (literature-grounded, ranked by leverage)

| # | Upgrade | Source | Expected effect | Status |
|---|---------|--------|-----------------|--------|
| U1 | **Gumbel AlphaZero search** — Gumbel-Top-K root sampling + sequential halving replaces uniform PUCT sims; same policy-improvement guarantee at a fraction of the sims | Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 | ~400 → ~64–100 equally-good sims/move ⇒ **~4× games/hour** on the same GPU; biggest single lever | **in flight** (training-dev; ablation-gated: ≥4× games/h at win-rate ≥0.35 vs PUCT@400 baseline) |
| U2 | **Playout-cap randomization** — sample training positions from cheap searches too (mixed-cost data), not only full-cost searches | Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2020 | Large strength-per-compute gain from data diversity; stacks with U1 | **in flight** (same worker, sequenced after U1; flag-off identical by test; orchestrator-gated post-integration) |
| U3 | **Rating uncertainty + external anchor** — Glicko-2-style deviation on the ladder; periodic anchor evals vs Stockfish at fixed skills to report human-comparable equivalence | Glickman, *Parameter estimation in large dynamic paired comparison experiments*, 1999 (Glicko-2); CCRL methodology for engine anchor pools | "847 ± 31 internal" and "~1300 human-equivalent" become falsifiable statements | queued (pairs with the overnight run's first real arenas) |
| U4 | **Scaling study** — same recipe at TINY/SMALL/BASE, measure Elo-hour vs params/compute on one consumer GPU | — (no published equivalent for single-GPU chess; cf. Kaplan et al. 2020 / Hoffmann et al. 2022 methodology) | A genuine (small) research contribution: the measured answer to "what does a 3080 buy" | blocked on U1+U3 landing; needs ≥3 completed runs |
| U5 | LR schedule + optimizer hygiene — cosine decay, warmup, Adam betas per AlphaZero appendix | Silver et al., *AlphaZero*, Science 2018 | Modest, standard; fold into U1's ablation only if a gate demands it | parked |

## Documented rejections

- **MuZero-style learned model.** Chess has an exact, fast rules simulator
  (`gambit_core`, perft-proven). Learning frozen dynamics is wasted
  capacity; MuZero's contribution targets environments where the rules are
  unknown or expensive. Rejected on principle 7.
- **Massive parameter count as a strategy.** A 3080 cannot train a
  superhuman network by scale; it *can* run a disciplined small-scale
  recipe repeatedly. U4 turns that constraint into the contribution.

## Pre-registration ledger

Rules: predictions are written and committed *before* the relevant data
exists, scored honestly after, never edited. Score: ✓ (hit) / ✗ (miss) / —
(untestable, say why).

### first-light (test run: night profile, scratch regime, 3 iterations, seed 7 — launched 10:26 local, **cut 11:45** by owner decision to prioritize engine modernization)

| # | Prediction | Status |
|---|------------|--------|
| P1 | policy_loss falls monotonically across the 3 iterations | — untestable (run cut at 71 min, before any iteration event existed) |
| P2 | value_loss ends below 0.0173 | — untestable (same) |
| P3 | measured games/hour lands between 3k and 20k at parallel_games=16 with 400 sims | — untestable as stated, **but the throughput model is confirmed wrong** by direct observation: 0 of 32 games completed in 71 min ⇒ <27 games/h ⇒ far below the 3k floor. The bench numbers (15.5k games/h) were taken at tiny sims and do not extrapolate to night profile. |
| P4 | no promotion occurs in 3 iterations | — untestable (no arena ran; champion lineage unchanged, `ckpt-000004-0004` remains). **Retrospect:** the audit (F-1) later showed the gate itself was inert — `champion_model = model` aliasing meant promotion was *impossible* after iteration 0 and every historical arena result (win_rate 0.5, elo 800.0) was the model playing itself. P4 would have been "confirmed" for the wrong reason: the prediction was right, but the model of the system behind it was wrong. A reminder that a confirmed prediction is not evidence the mechanism works. |
| P5 | individual games last ≥60 plies on average | — untestable (no game completed) |

**Run yield (what the 71 minutes actually bought):** the baseline diagnostic the
optimization gates needed — the training process was **GIL-bound to ~1.0 core**
(66.9 CPU-min in 71 wall-min, 16 game threads), zero iteration completions, and
a measured lower bound on night-profile cost (≥71 min per 32-game iteration at
400 sims). These numbers are the control against which the Rust-wiring gate
(≥3× e2e sims/sec) and the Gumbel ablation (≥4× sims-efficiency at non-regressing
win-rate) will be judged.

### next run — pre-register before launch (fill in at launch time, do not edit after data arrives)

| # | Prediction | Status |
|---|------------|--------|
| N1 | (Rust gate) wired gambit_core yields ≥3× e2e sims/sec vs the first-light baseline at p16 | **✗ MISS — 1.1× measured** (interleaved paired arms, p16: 1.087×/1.110×/1.13× by game length). Attribution: wireable share (movegen/encoding) is ~4.5% of GIL demand; ~82% of per-sim cost lives in `_simulate`'s tree/PUCT ops, out of M3's no-search-change scope. Bit-exactness held everywhere (perft 4.87M, 5,990-position order fuzz, e2e identical visit counts/examples), so the slice lands as a 1.1× free win + the foundation for a search-in-Rust slice. N3's "combined <12 min" prediction now needs the search-policy lever (Gumbel) to carry it. |
| N2 | (Gumbel gate) gumbel@64–100 sims ≥4× games/h vs puct@400 with head-to-head win-rate ≥0.35 vs baseline | pending |
| N3 | (combined overnight) iteration time at the modern recipe < 12 min/iteration (vs ≥71 min baseline) | pending |
| N4 | (streaming) per-move live events flow end-to-end: journal → WS → Watch grid, at ≥1 event/s sustained during self-play | pending |

Scoring happens in this table, in place, with the actual numbers beside
each verdict. Missing a prediction is *good* science — it means the model of
the system was wrong and now it's less wrong.

## Review cadence

The standard is enforced where work happens: worker prompts carry the gate,
integration verifies it, and the standing review loop audits for
violations (a claim without a number, a change without an ablation, a
prediction scored softly).
