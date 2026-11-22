# Fool's Gambit — High-Level Plan

> The readable "what and why." For the detailed "how," see [`UPGRADE_PLAN.md`](./UPGRADE_PLAN.md) (living technical plan). Last updated: 2026-11-20.

## The idea (the only thing that must remain)

1. **Train a chess bot** — on the local RTX 3080, via self-play (AlphaZero-style).
2. **Have a website** — where anyone can watch it learn and play games live.
3. **Host it** — so the owner can play against it, and can send friends a link to play against it.

Everything else — code, frameworks, engines, even languages — is negotiable and judged by one bar: is it the best choice for those three lines on this machine?

## What the finished site does

- **Play tab** — you vs. the bot (or Stockfish at a chosen strength, or Random), either color, with promotion, undo, resign, rematch.
- **Watch tab** — a live lobby: training games and human games streaming in real time, pick one and spectate.
- **Training tab** — the existing dashboard: losses, metrics, live games as the bot trains.
- **Archive** — every game ever played (human or training) stored with full PGN; browse and replay.
- **A public link** — `play.lanceabuan.tech` via a free Cloudflare tunnel to your PC: friends play the *actual* model at full strength. Optional later: a tiny VPS keeps the site alive 24/7 when your PC is off.

## The five steps (full detail in the technical plan)

| # | Step | Outcome |
|---|---|---|
| **M0** | Toolchain | Python 3.13 + PyTorch CUDA verified on the 3080; repo cleaned of dead files |
| **M1** | The product | Play the bot in the browser, watch games live, everything archived; public tunnel link |
| **M2** | Training v2 | Fixed chess-logic bugs, redesigned model (proper move encoding), fast GPU inference; training runs end-to-end |
| **M3** | Custom engine core | A Rust extension replaces slow Python in the training hot path — target: many times more games per hour |
| **M4** | Ship it | Security hardening, one-click 24/7 deploy option, honest README, CI, PR |

## Honest expectations

- A single RTX 3080 will produce a bot that plays **sane, tactically reasonable chess** after days of training — not superhuman. That's hardware physics; AlphaZero used thousands of TPUs.
- Stockfish stays in the toolbox as the strength ladder and sparring partner while your bot grows.
- Every performance claim in the technical plan must be **measured**, not promised.

## Principles for the work

- **Complete overhaul allowed, product-first sequencing:** the playable site ships (M1) before deep performance engineering (M3), so there's always something working end-to-end.
- **Old bugs mean retraining:** the current checkpoints were trained by a pipeline with known chess-logic bugs — they get archived, and training restarts clean.
- **Everything is verified:** backend by tests (including a fuzz-tested Rust engine core), frontend by screenshots I take and review myself in a real browser.
- **Living docs:** the technical plan is updated at every execution step; code never diverges from it silently.
