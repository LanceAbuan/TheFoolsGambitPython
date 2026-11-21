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
| U1 | **Gumbel AlphaZero search** — Gumbel-Top-K root sampling + sequential halving replaces uniform PUCT sims; same policy-improvement guarantee at a fraction of the sims | Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 | ~400 → ~64–100 equally-good sims/move ⇒ **~4× games/hour** on the same GPU; biggest single lever | **queued** (next training slice after FS-6b streaming lands; one-change ablation, gated on arena win-rate not regressing at equal games/hour) |
| U2 | **Playout-cap randomization** — sample training positions from cheap searches too (mixed-cost data), not only full-cost searches | Wu, *Accelerating Self-Play Learning in Go* (KataGo), 2020 | Large strength-per-compute gain from data diversity; stacks with U1 | queued behind U1 |
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

### first-light (test run: night profile, scratch regime, 3 iterations, seed 7 — launched 10:26 local)

| # | Prediction | Status |
|---|------------|--------|
| P1 | policy_loss falls monotonically across the 3 iterations (any increase = ✗) | pending |
| P2 | value_loss ends below 0.0173 (the smoke run's starting value) | pending |
| P3 | measured games/hour lands between 3k and 20k at parallel_games=16 with 400 sims (outside = ✗ and the throughput model is wrong) | pending |
| P4 | no promotion occurs in 3 iterations (near-random candidates don't beat the reigning champion in a 2-game arena; a promotion = ✗ and the arena is too weak a gate) | pending |
| P5 | individual games last ≥60 plies on average (random-ish MCTS produces long shuffles; shorter = ✗ and the game-length prior needs revising) | pending |

Scoring happens in this table, in place, with the actual numbers beside
each verdict. Missing a prediction is *good* science — it means the model of
the system was wrong and now it's less wrong.

## Review cadence

The standard is enforced where work happens: worker prompts carry the gate,
integration verifies it, and the standing review loop audits for
violations (a claim without a number, a change without an ablation, a
prediction scored softly).
