# Fool's Gambit v2 — Full Overhaul Plan (living document)

> **Version:** 2.0 · **Status:** Approved, awaiting execution · **Branch:** `feature/play-and-local-overhaul` (created, cleanup staged, nothing committed) · **Last updated:** 2026-11-20
>
> **This is a living document.** Every phase execution updates the status markers here and appends to the Changelog (§13). Nothing in the repo should diverge from this file without a changelog entry. Companion docs: [`ARCHITECTURE.md`](./ARCHITECTURE.md) (current system, to be rewritten at ship) · [`README.md`](../README.md).

---

## 1. The Mission (non-negotiable)

Three things are fixed. Everything else — language, framework, engine, code — is negotiable and is judged against a single bar: **is this the optimal choice for the mission on this hardware?**

1. **Train a chess bot** on the local RTX 3080.
2. **Have a website** where anyone can watch it learn and play, live.
3. **Host the model** so the owner can play against it — and send other people a link to play against it.

Everything below is derived from those three lines.

---

## 2. Overhaul Verdicts (researched 2026-11-20)

The user mandate: *overhaul anything that is not 100% optimal, including changing languages or tech stack.* Here is every component, judged honestly. "Optimal" = best trade-off of performance, correctness, maintainability, and time-to-mission for a one-person project on one Windows box with one 3080.

| Component | Verdict | What replaces it / why |
|---|---|---|
| Pure-Python chess rules in the **training hot path** | **REPLACE** | New Rust extension **`gambit_core`** (PyO3 + `chess` crate): bitboard movegen, make/unmake, FEN, Zobrist, batched stepping, batched tensor encoding, perft. This is the "completely custom engine optimized for training." |
| `python-chess` | **DEMOTE, keep** | Demoted to: website game layer (correctness-critical, perf-irrelevant), PGN I/O, and the **test oracle** for `gambit_core` (perft + randomized game-equality fuzzing). |
| Flask + SSE server | **REPLACE** | **FastAPI + uvicorn + WebSockets.** Public play with spectators needs bidirectional realtime, typed messages, and async concurrency. SSE is one-directional and its Flask implementation is thread-based; WS is supported end-to-end through Cloudflare Tunnel. |
| `react-chessboard` (MIT, v5.12.1, active) | **REPLACE → DONE** | Replaced by a hand-built Svelte 5 SVG board (pointer drag + click-click, promotion picker, piece-ID glide, a11y labels). MIT-clean with zero board dependencies; matches the approved tarot design exactly. Trade-off logged in §12. |
| Model head: flat 4096 (from×to) | **REPLACE** | AlphaZero-canonical **73×64 = 4672** policy planes (56 queen-like + 8 knight + 9 underpromotion planes per origin square). Underpromotions get real logits; encoding is the standard every reference implementation uses. |
| Input encoder: 16 planes, no history | **REPLACE** | ~29 planes: 12 current pieces + 12 previous position + 4 castling + 1 en-passant. History gives repetition/draw awareness for ~zero cost. Encoder lives behind an interface so this stays a config knob, not a rewrite. |
| Training bugs (MCTS sign, instant-resign, promotion collision) | **FIX + regression tests** | §10. Prior checkpoints are tainted by these bugs → **archive old HF artifacts as `v1-legacy`, retrain from scratch** with the fixed pipeline. |
| `torch.compile` on Windows | **OPTIONAL, de-prioritized** | Research: only works via community `triton-windows` wheels with strict version pinning (torch 2.15 → triton 3.9) and the 260-char Windows path trap; measured gain on small ResNets ≈ **1.1–1.15x**. The real GPU wins are **TF32 + bf16 autocast + channels_last** (measured ≈ **2.3x** on comparable workloads). Compile is an optional flag behind env detection, never a dependency. |
| Batched NN inference | **KEEP, harden** | The existing GPU-batched evaluator idea is right. Rebuild it as a dedicated **NNEvaluator service**: request queue, micro-batching (size cap + 4 ms window), fp16, pinned memory, optional CUDA Graphs for fixed shapes. This — not movegen — is the true throughput lever. |
| Game storage: in-memory, 20-SAN truncation, no PGN | **REPLACE** | **SQLite** (`data/gambit.db`, WAL mode): games, moves, checkpoints tables; every human *and* training game archived with full PGN. |
| Training metrics storage: JSON files | **KEEP (deliberate)** | Works, isolated, zero-risk. Consolidating into SQLite touches every dashboard endpoint for no user-visible gain. Logged in §12 as a conscious "not now." |
| Stockfish wrapper | **KEEP, fix paths** | Windows binary in `tools/` (gitignored), path resolution: env → `tools/` → `PATH`. Local sparring partner + gate-eval ladder. **Never exposed to website visitors** (§9). |
| Hugging Face Hub checkpointing | **KEEP, extend** | Still the model upload/registry. Add `scripts/publish.py` (tag, push, attach model card) — this is the "upload the model" half of the mission. |
| Vercel/`vercel.json`/`Procfile`/serverless deploy | **DELETE** | Serverless cannot host GPU inference, WS, or training. Replaced by Cloudflare Tunnel (below) + optional VPS path (§8). |
| React 19 + Mantine 9 + Vite + TypeScript | **REPLACE → DONE** | Legacy `client/` deleted. New client: **Svelte 5 + Vite 7 + TypeScript** in `web/` — smaller bundle, runic reactivity without a vDOM, custom board fits naturally. UI is a 1:1 port of the user-approved static tarot mock (Cormorant Garamond + DM Mono, sepia palette). |
| SQLite for archive; Python for web/training; TS for UI | **KEEP** | §12 logs why each language stays: Python owns training (PyTorch is the irreplaceable asset) and the web layer; Rust owns only the hot path; TypeScript owns the UI. No other split pays for its coordination cost. |

---

## 3. Research Findings (with sources)

| # | Finding | Source | Consequence |
|---|---|---|---|
| R1 | `torch.compile` CUDA on Windows requires community `triton-windows`; torch 2.15 pairs with Triton 3.9; known Windows path-length failure mode; ~1.1x on small ResNets | github.com/triton-lang/triton-windows · pytorch/pytorch#167062 · b1ackjack12.github.io benchmark (Aug 2026) | Perf plan: TF32+bf16+channels_last first (≈2.3x), compile optional flag only |
| R2 | Rust `chess` crate (jordanbray): MIT, perft 5 ≈ 18.6 ms (faster than shakmaty 0.16's 24.1 ms on same bench), copy-on-make board, Zobrist built in | github.com/jordanbray/chess · docs.rs/crate/chess/latest | `gambit_core` uses `chess` crate; shakmaty rejected on GPL-3.0 license (§12) |
| R3 | PyO3 requires Rust ≥ 1.85, supports CPython 3.13; `maturin` builds wheels; `uv` handles PEP 517 maturin backends | pyo3.rs guide | Build chain: `rustup` + MSVC Build Tools → `uv sync` compiles `gambit_core` |
| R4 | chessground = GPL-3.0 ("combined work may be distributed only under the GPL"); react-chessboard = MIT, v5.12.1 released Aug 2026, 53k weekly downloads | github.com/lichess-org/chessground · npmjs.com/package/react-chessboard | Frontend stays MIT: react-chessboard |
| R5 | Cloudflare named tunnels: **free on every plan**, custom hostname, WebSockets proxied, cloudflared installs as a Windows service; quick tunnels (`trycloudflare.com`) are dev-only (no SSE, 200-conn cap) — irrelevant since we move to WS | developers.cloudflare.com/tunnel · flaviocopes.com guide (Sep 2026) | Public hosting = named tunnel on `lanceabuan.tech`, zero cost, GPU stays local |
| R6 | PyTorch 2.15.0 CUDA (cu132) supports Python 3.11–3.15, sm_86 verified | (prior session research, Nov 2026) | Locked: torch 2.15.0+cu132, Python 3.13 via uv |

---

## 4. Target Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                FastAPI + uvicorn (:8000)             │
                    │                                                     │
   Browser ────WS──▶│  ws/: one typed channel per client                  │
   (React 19)       │    subscribe: lobby | game:{id} | train             │
                    │  api/:                                              │
                    │    POST /api/play/new        (color, opponent,      │
                    │     /{id}/move /undo /resign  strength preset)      │
                    │    GET  /api/play/{id} /list                        │
                    │    GET  /api/games?kind=...  (archive + live)       │
                    │    GET  /api/checkpoints     POST /api/train/...    │
                    │        │                                            │
                    │  ┌─────▼───────────────────────────────┐            │
                    │  │  Engine interface (unchanged shape) │            │
                    │  │  NeuralNetEngine → MCTS(gambit_core │            │
                    │  │     or python-chess, NNEvaluator)   │            │
                    │  │  StockfishEngine (skill 1–20)       │            │
                    │  │  RandomEngine                       │            │
                    │  └─────┬───────────────────────────────┘            │
                    │        │                                            │
                    │  ┌─────▼──────────┐   ┌───────────────────────┐    │
                    │  │  PlaySessions  │   │ Trainer (self-play +  │    │
                    │  │  (GameManager) │   │ optimize + gate eval) │    │
                    │  └─────┬──────────┘   └──────────┬────────────┘    │
                    │        ▼                         ▼                 │
                    │  ┌─────────────────────────────────────────┐       │
                    │  │  SQLite data/gambit.db (WAL)            │       │
                    │  │  games · moves · checkpoints            │       │
                    │  └─────────────────────────────────────────┘       │
                    │        │  everything publishes through              │
                    │        ▼  the same lobby/game event bus           │
                    └────────┼────────────────────────────────────────────┘
                             ▼
              Watch tab: training games AND human games, live

   Optional (M4): same app in Docker with ONNX int8 inference on a $5 VPS
   Public exposure: Cloudflare named tunnel → play.lanceabuan.tech → :8000
```

**The one architectural principle:** trainer and play API publish through the **same event bus**, so a visitor watching the site sees training games and human games identically. No duplicate streaming code paths, ever.

### 4.1 `gambit_core` (Rust) contract

| Function | Purpose |
|---|---|
| `from_fen / to_fen` | Position I/O |
| `legal_moves(pos) -> [Move]` | Bitboard movegen |
| `play(pos, move) -> pos` | Copy-on-make step |
| `step_batch(fens, moves) -> [fens]` | Batched stepping for N parallel games in one FFI call |
| `encode_batch(fens, history) -> ndarray (N,C,8,8)` | Tensor encoding **in Rust** (kills the per-square Python loop in `board_to_tensor`) |
| `zobrist(pos) -> u64` | Transpositions / dedupe |
| `outcome(pos) -> ...` | Checkmate/stalemate/insufficient material/50-move |
| `perft(pos, depth)` | Used by the test suite against known perft numbers |

`python-chess` remains the oracle: randomized differential fuzzing (thousands of positions, movegen equality) + standard perft suite.

### 4.2 Database schema (initial)

```sql
games(id TEXT PK, kind TEXT CHECK(kind IN ('human','selfplay','eval')),
      white_name, black_name, engine_id, checkpoint_tag,
      result TEXT, termination TEXT, ply_count INT,
      pgn TEXT, meta JSON, started_at REAL, ended_at REAL)
moves(game_id TEXT, ply INT, uci TEXT, san TEXT,
      eval_cp REAL, winprob REAL, think_ms INT,
      PRIMARY KEY(game_id, ply))
checkpoints(tag TEXT PK, path TEXT, hf_rev TEXT,
      games_trained INT, elo_est REAL, is_current INT, created_at REAL)
-- indexes: games(ended_at DESC), games(kind, ended_at DESC), FK moves→games
```

### 4.3 WebSocket protocol (typed JSON, sequence-numbered)

```
C→S {type:"subscribe", channel:"lobby"|"game:<id>"|"train"}
C→S {type:"play/new", color, opponent:"nn"|"stockfish"|"random", strength}
C→S {type:"play/move", game_id, uci}          # server-side legality only
C→S {type:"play/undo"|"play/resign", game_id}
S→C {type:"game_state", game_id, fen, moves:[...], turn, result?, last_move}
S→C {type:"game_move", game_id, ply, uci, san, winprob}
S→C {type:"game_over", game_id, result, termination}
S→C {type:"lobby", games:[...]}               # live games for the Watch tab
S→C {type:"train_metrics", step, loss, ...}
-- reconnect: client sends last seq per channel; server resyncs state
```

---

## 5. Performance Engineering (measure-first, with gates)

No throughput claim ships without a measured baseline. `scripts/bench.py` reports **sims/sec, games/hour, NN batch latency, CPU vs GPU utilization**.

| Lever | Expected | Gate |
|---|---|---|
| TF32 + bf16 autocast + channels_last | ≈2.3x NN step (research: 21.4→9.1 ms on comparable ResNet) | default ON from M2 |
| NNEvaluator micro-batching (256-cap / 4 ms window), fp16, pinned memory | NN utilization from ~30% → 80%+ | default ON from M2 |
| `gambit_core` movegen + Rust batch encoding (replaces per-square Python loop) | 10–30x on the CPU portion of a tree step | M3 acceptance: **≥3x end-to-end games/hour** vs measured M2 baseline, else document why |
| Parallel self-play games (16–32) feeding one evaluator | near-linear until GPU saturates | tuned via bench |
| `torch.compile` | ≈1.1x on small ResNets, Windows friction | optional flag, off by default |
| CUDA Graphs (fixed batch shapes) | small; only if profiling shows launch overhead | stretch, M4 |

**Honest ceiling:** AlphaZero-scale strength used thousands of TPUs for 9 hours. One 3080 with a 1–6M-param net yields a sane, tactically reasonable club-level-ish bot after days of training — not superhuman. The bench numbers make this measurable instead of hand-wavy.

### 5.1 Checkpoint ladder + arena gating (WP-C — how "better" gets measured)

North-star metric: **Elo per wall-clock training hour**. Loss curves don't prove strength; games do. Protocol (user directive: multiple models, playing each other, on a time interval):

1. **Checkpoint ladder** — snapshot the training model on a wall-clock ladder: **5, 10, 15, 20 min**, then every 20 min. Time-based, not game-count-based, so "minutes of training" is the experiment axis. Configurable via dataclass.
2. **Arena gate** — each checkpoint plays a match vs the reigning champion: both colors, reduced sim budget (e.g. 128 sims vs training's 400–800) so arenas fit between training bursts on the single 3080, batched NN eval shared with the trainer.
3. **Promotion** — candidate replaces champion at **≥55% match win rate** (SPRT bounds optional later); otherwise the champion holds. The ladder IS the multi-model experiment: every checkpoint is a distinct model, the ledger records who beat whom.
4. **Elo ledger** — arena games persist to SQLite as `kind='arena'`; per-checkpoint Elo computed from the ledger, exposed via `GET /api/metrics` + WS `train` topic → the Watch page's "EST. ELO" becomes measured and live.
5. **Config benchmarking** — the ladder doubles as the A/B harness: change one training knob (sims, lr, batch size, parallel games), compare Elo-per-hour across runs, adopt winners (standing loop step 8).

### 5.2 Training regimes — the experiment matrix (WP-C)

Goal (user directive): end up with models trained *differently* and a measured answer to "which way wins".

**A regime is a named training recipe.** Each run records its regime; each regime produces its own model lineage (`runs/<regime>/<run-id>/`). First-class regimes:

| Regime | Data | Rationale |
|---|---|---|
| `scratch` | pure self-play (AlphaZero) | canonical run; highest ceiling, slowest start |
| `sf-games` | self-play + games vs Stockfish on a fixed skill ladder | real-opponent pressure; learns to punish mistakes |
| `sf-distill` | policy ← Stockfish move choice, value ← Stockfish eval on self-play positions | fastest early Elo; ceiling capped by SF style |
| `hybrid` | `sf-distill` warm start → pure self-play | likely best Elo-per-hour; the hypothesis to test |
| `finetune` | continue from a champion, or seed from a public HF net | measures transfer vs fresh init |

**Cross-regime arena:** every regime's champion plays a round-robin against the others (reduced sims, §5.1 mechanics) on the same `kind='arena'` ledger — one Elo table answers "from-scratch vs Stockfish-taught". Per-regime Elo flows to `/api/metrics` and the Watch page.

**Budget & storage:** one regime trains at a time at full speed; arenas run between bursts. The active schedule is dataclass config, not code. SQLite gains a `runs` table (regime id, config hash, checkpoint lineage); HF Hub publishes per regime as `<user>/gambit-<regime>` so every public model carries its provenance.

---

## 6. Milestones

### M0 — Toolchain & hygiene (half day)
| Task | Done when |
|---|---|
| `pyproject.toml` + `uv sync` (Python 3.13 managed; chess 1.11.2, fastapi, uvicorn, pydantic v2, torch 2.15.0+cu132, numpy, psutil, huggingface_hub, stockfish, pytest, ruff) | `uv run python -c "import torch; torch.cuda.is_available()"` → True on the 3080 |
| Stockfish Windows binary → `tools/`; path resolution env → `tools/` → PATH | wrapper evaluates a position on Windows |
| Repo hygiene commit (staged deletions: dead `ai.py`/`engine.py`, nested old repo, logs, `vercel.json`, `Procfile`) | clean root; `.gitignore` covers `.venv/ tools/ data/` |
| Install rustup + MSVC Build Tools (needed for M3; `winget` install, can run in background) | `cargo --version` works |

### M1 — Playable core + website (the product, ~2–3 days)
| Task | Done when |
|---|---|
| FastAPI app skeleton: lifespan, config, structured errors, CORS for dev | boots, serves client dist |
| Engine interface + `RandomEngine`, `StockfishEngine` (skill ladder), `NeuralNetEngine` (wraps existing MCTS over python-chess, bugs fixed from §10 inlined here so M1 is correct even before M2) | each engine returns a legal move on arbitrary positions; unit-tested |
| `PlaySession` + `GameManager` + play REST endpoints (`new/move/undo/resign/state/list`) | full lifecycle via API tests; illegal move → 4xx + reason; undo pops full move pair |
| SQLite archive (`games`, `moves`); every finished game saved with full PGN | PGN round-trips through `chess.pgn`; survives restart |
| WS layer: lobby + game channels, event bus shared by trainer; typed messages, seq + resync | a `/play` move appears in a WS subscriber's feed live |
| Client: `PlayView` (setup → board → promotion dialog → move list → result → rematch), `WatchView` (live lobby + game view), `TrainingView` ported off SSE onto WS | full game as White and as Black; watch tab shows both |
| Client dev proxy + built-dist serving both work | `npm run dev` and prod build both functional |
| **Browser self-review pass 1** (§7): screenshots of every flow, reviewed, defects fixed | sign-off recorded in changelog |
| Cloudflare named tunnel — **DEFERRED per user (2026-11-20): local hosting first.** Same app, exposure layer only; enabling later costs nothing | a phone on LTE can load the site and play the bot |

### M2 — Training pipeline v2 (~2–3 days)
| Task | Done when |
|---|---|
| Encoder v2 (29-plane, history) behind an interface; batch encoding vectorized | unit tests vs hand-built tensors |
| Model v2: 4672-plane policy head, masked softmax, ResNet presets (tiny 6-block/64-f ≈ 1M, base 10-block/128-f ≈ 6M) | forward pass shapes verified; param counts logged |
| Bug fixes + regression tests: MCTS terminal sign, 3-strike resign (default off for self-play; opt-in), promotion indexing | each bug has a failing-then-passing test |
| `NNEvaluator` service: queue, micro-batch, fp16, TF32, channels_last | bench: batch latency + utilization numbers recorded |
| Replay buffer: on-disk npz shards + sliding window; loader with pinned memory | `--profile smoke` (8×8 games, tiny net, 200 steps) runs end-to-end on GPU |
| Self-play driver: temperature schedule (τ=1 first 30 plies → argmax), Dirichlet α=0.3 ε=0.25, games archived to SQLite with full PGN | games/hour baseline measured and written into this file |
| Gate eval: candidate vs incumbent 200-game arena + Stockfish skill ladder + random; acceptance = ≥55% score | a gate decision is produced and logged |
| `scripts/train.py --profile smoke|night` orchestration: loop self-play → sample → optimize → eval → checkpoint → HF push | a full loop completes unattended; dashboard reflects it live |

### M3 — `gambit_core` (Rust, the custom training engine) (~2–4 days)
| Task | Done when |
|---|---|
| Crate scaffold: PyO3 + `chess` crate, maturin build wired into `uv` | `uv sync` builds the extension on Windows |
| FEN/movegen/play/Zobrist/outcome/perft bindings | perft(5) = 4,865,609 from startpos; matches known values |
| Batched `step_batch` + `encode_batch` (returns NumPy arrays) | differential fuzz vs python-chess: ≥10k random positions, zero mismatch |
| Integrate behind a `BoardBackend` interface: MCTS + self-play switch to `gambit_core`; python-chess stays for web/play/tests | all existing tests pass unchanged |
| Bench vs M2 baseline | **≥3x games/hour** recorded in this file (or documented reason + decision) |
| Stretch: tree itself in Rust with Python NN-eval callback via queue | only if profile shows Python tree-walk still dominates |

### M4 — Public hardening & durability (~1–2 days)
| Task | Done when |
|---|---|
| Security: `ADMIN_TOKEN` on train/reset/push endpoints; per-IP rate limits + concurrent-game cap on public surface; Stockfish never visitor-reachable; strict message validation on WS | unauth reset → 401; abuse tests pass |
| ONNX int8 export (`scripts/export_onnx.py`) + Dockerfile (CPU ONNX Runtime, small-sims profile) + deploy doc for a cheap VPS | exported model plays legal games; image runs locally |
| `scripts/publish.py`: tag checkpoint, push to HF Hub with model card | model card shows arch, training curves, Elo estimate |
| CI rewrite: pytest + ruff + `cargo test` + client build; Vercel removed | CI green on PR |
| README + ARCHITECTURE.md rewritten to match reality (play feature front and center, local quickstart, tunnel guide, honest Elo section) | docs match code |
| Conventional commits, push, PR to `main` (per `.opencode/build-prompt.md`) | PR URL reported |

### M5 — Live operations (continuous)
Kick off a real multi-day training run; watch it on the site; tune cpuct/FPU/Dirichlet from observed behavior; keep the tunnel up; friends play the current checkpoint via the site. Every material change updates this document.

---

## 7. UI Self-Review Protocol (mandatory for every UI change)

I do not claim a UI works unless I have watched it work:

1. Boot server (+ client dev server when iterating); open the site in the in-app browser.
2. Screenshot every surface: Play setup, in-game (both colors), promotion dialog, game-over/rematch, Watch lobby, live game view, Training metrics, Archive/replay.
3. Read each screenshot back and audit it myself: layout breaks, unstyled Mantine, blank boards, wrong orientation, missing highlights, overflow.
4. **Drive the flows myself:** start vs Stockfish skill 3 as White, play a real opening, promote a pawn, finish/resign, rematch; repeat as Black; open Watch in a second tab and confirm moves appear live.
5. Check console + network: JS errors, failed fetches, WS reconnect spam.
6. Fix → re-screenshot the affected flow → confirm visually. Screenshots are saved so the user can eyeball the same evidence.

(I read images natively — the screenshot *is* the OCR step.)

---

## 8. Hosting the Model (mission line 3)

| Path | Cost | Strength | Availability | When |
|---|---|---|---|---|
| **Cloudflare named tunnel → local 3080** | $0 (domain DNS on CF required) | Full (800+ sims) | While the PC is on | M1 — this is the primary |
| **VPS + Docker + ONNX int8** | ~$5/mo | Reduced (64–128 sims, still playable) | 24/7 | M4 — optional durability |
| HF Hub | $0 | (registry only) | — | already built; extended in M4 |

The tunnel gives visitors the exact model the owner trained, at full strength, for free. The VPS path exists so the site survives the PC being off. Both serve the same app.

**Do we need Docker/Linux? No.** Everything primary runs natively on Windows: CUDA training (PyTorch ships native Windows CUDA wheels — no WSL needed), `gambit_core` (compiles locally with MSVC via maturin), FastAPI/uvicorn, Stockfish, and cloudflared (installs as a Windows service). Docker appears **only** in the optional VPS path above, as packaging convenience: Linux VPSes are simply the cheapest 24/7 boxes, and a Docker image is the one-command way to ship the app there. Alternatives stay open (bare `uv` + systemd on the VPS, or a pricier Windows VPS). The model itself is platform-neutral — `.pt`/`.onnx` files on HF Hub; only the inference runtime differs per host. Linux *would* buy full official `torch.compile` support (§3 R1), but that lever is optional and worth ≈1.1x — not worth moving the primary environment off the GPU box where the 3080 lives. Note: GitHub Actions CI runners are Linux — that's free test infrastructure, unrelated to hosting.

**User action items:** (1) move/confirm `lanceabuan.tech` DNS on Cloudflare; (2) approve installs: VS Build Tools, rustup, cloudflared.

---

## 9. Security Posture (because the site becomes public)

- Destructive endpoints (`/api/train/reset`, HF push) require `ADMIN_TOKEN`; never routed through the tunnel.
- Public surface = play/watch/archive read endpoints only; per-IP rate limits; cap concurrent games per IP.
- Server-side legality on every move; WS messages schema-validated (pydantic); no eval of user strings; engine subprocesses are Stockfish-only, local-only.
- CORS: localhost dev origins; the tunnel serves same-origin so no wildcard CORS in prod.

---

## 10. Known Bug Ledger (each fixed with a regression test)

| Bug | Detail | Fix |
|---|---|---|
| MCTS sign | Terminal checkmate evaluates as a **win** for the mated side → poisons mate search + training labels | Terminal value from side-to-move perspective; test: mate-in-1 position → MCTS plays the mate |
| Resign | Single eval below threshold → instant resign, corrupting game outcomes | 3 consecutive evals below −0.9 past ply 40; **default off** for self-play |
| Promotion policy | All promotions share one 4096 logit | 73×64 encoding gives underpromotions real planes |
| Game history | In-memory, truncated to 20 SAN, no PGN | SQLite archive, full PGN, all game kinds |
| Stockfish path | Linux-only hardcode; also single shared instance w/o queue fairness | Windows resolution chain; keep lock, add health-check + auto-restart (exists) |

---

## 11. Decision Log (rejected alternatives — so we don't relitigate)

| Decision | Alternatives considered | Why rejected |
|---|---|---|
| Rust via PyO3 for hot path only | Full-Rust backend; fully-GPU board simulation; adopt LC0 wholesale | Full-Rust: loses PyTorch ecosystem, months of work, mission-neutral. GPU board sim: research-grade complexity, poor debuggability, 3080 saturates via batching anyway. LC0: opaque pipeline, not "your" bot, huge build |
| `chess` crate (MIT) | shakmaty (GPL-3.0, actively maintained, variants we don't need) | License cleanliness for a project that may be public; benchmarks equal or better for our needs |
| FastAPI + WS | Flask + SSE (existing); Socket.IO | Bidirectional realtime for play; async concurrency for spectators; one modern async stack; WS passes cleanly through Cloudflare |
| react-chessboard (MIT) | chessground (GPL-3.0) | Whole-frontend GPL contagion vs MIT; feature set sufficient for vs-AI play; revisit only if lichess-parity interactions become a hard requirement |
| TF32/AMP-first, compile optional | torch.compile default-on | Windows Triton friction + 1.1x payoff isn't worth build fragility (§3 R1) |
| SQLite for games; JSON metrics stay | Migrate metrics to SQLite too | Metrics storage works, is isolated, and consolidating it mid-flight risks dashboard endpoints for zero user-visible gain |
| Retrain from scratch | Fine-tune existing HF checkpoints | Existing labels were produced by the sign/resign/promotion-buggy pipeline — the data itself is suspect; fresh start with fixed pipeline |

---

## 12. Docs & References

| Topic | Reference |
|---|---|
| uv — env/Python management, PEP 517 builds | https://docs.astral.sh/uv/ |
| python-chess — rules, PGN, legality (oracle + web layer) | https://python-chess.readthedocs.io/ |
| PyTorch — CUDA, AMP, TF32, ONNX export | https://pytorch.org/docs/stable/ |
| triton-windows — Windows torch.compile (optional path) | https://github.com/triton-lang/triton-windows |
| PyO3 + maturin — Rust↔Python extension builds | https://pyo3.rs/ · https://www.maturin.rs/ |
| `chess` Rust crate — bitboard movegen core | https://github.com/jordanbray/chess · https://docs.rs/chess |
| AlphaZero (paper) — PUCT, Dirichlet, temperature, value targets | arXiv:1712.01815 |
| LC0 — cpuct/FPU reference values, network input conventions | https://lczero.org/ |
| FastAPI — async, WS, pydantic v2, testing | https://fastapi.tiangolo.com/ |
| WebSockets + FastAPI WS test client | https://fastapi.tiangolo.com/advanced/websockets/ |
| Cloudflare Tunnel — named tunnels, custom hostname, Windows service | https://developers.cloudflare.com/tunnel/ |
| SQLite (WAL mode, concurrency) | https://www.sqlite.org/wal.html |
| react-chessboard — board component, promotion, arrows | *(replaced — custom Svelte board)* |
| Svelte 5 — runes, $state/$derived/$effect, event attrs | https://svelte.dev/docs/svelte |
| Vite 7 — dev server, builds, @sveltejs/vite-plugin-svelte | https://vite.dev/ |
| pytest — fixtures, parametrization | https://docs.pytest.org/ |
| Stockfish PyPI wrapper | https://pypi.org/project/stockfish/ |

---

## 13. Changelog (execution log)

| Date | Change |
|---|---|
| 2026-11-20 | Plan v2 written: full-overhaul verdicts, Rust `gambit_core` added, FastAPI+WS replaces Flask+SSE, model v2 head, hosting strategy, research findings §3, decision log §11 |
| 2026-11-20 | §8: "no Docker/Linux needed" clarification added. Per user: focus on **local hosting first** — Cloudflare tunnel (M1) and VPS path (M4) deferred; they are exposure layers on the same app, zero rework to enable later |
| 2026-11-21 | Frontend milestone: legacy `client/` deleted; new Svelte 5 client in `web/` — custom SVG board (drag/click, promotion, glide), demo engine, **1:1 port of the user-approved static tarot mock** (Home landing added, Play/Watch/About restructured, CSS art, Cormorant Garamond + DM Mono, sepia palette). Verdicts §2 updated (React/Mantine → Svelte). `docs/TODO.md` added as the living checklist incl. the standing review loop |
| 2026-11-21 | §14 Subagent Delegation Map written (was referenced but missing): 14.0 pinned API contract v1, 14.1 work packages WP-A…WP-E. CI rewritten for `web/` (legacy Vercel-era checks dropped) |
| 2026-11-21 | Standing review loop automated: OpenCode automation `standing-review-loop` (cron 6h → fresh orchestrator session, dirty-tree safety gate), created paused then activated on user request. `experimental.automations` enabled |
| 2026-11-21 | WP-A slice 1: `server/` FastAPI + WS hub + SQLite (WAL) per §14.0 — play/new·move·undo·resign, games archive + PGN, metrics/checkpoints, admin-gated train stubs, demo Fool engine, `scripts/serve.py`, pytest suite. Stale `test_sample_returns_none_when_insufficient` updated to the adaptive contract (Trainer owns the MIN_BATCH_SIZE guard) |
| 2026-11-21 | §5.2 training regime matrix (user directive): five named regimes — `scratch`/`sf-games`/`sf-distill`/`hybrid`/`finetune` — per-regime lineage + cross-regime round-robin arena on the shared Elo ledger. Stockfish 19 installed (gitignored `tools/`). Overnight parallel build: WP-A s2 / WP-B s1 / WP-C s1 via backend-dev + client-dev + training-dev on disjoint ownership |
| 2026-11-21 | WP-A slice 2 (`2e12f19`): Stockfish 19 opponent (wandering/curious/enlightened skill presets, one UCI process per game, graceful lifecycle, crash degradation to Fool), WS train telemetry (10s, idle w/o subscribers), async per-session engine locks, hub seq-collision fix, strength enum (flagged deviation). Undo no-op contract pinned (`6e5542a`); §14.0 table already documented the reduced new-game shape |
| 2026-11-21 | WP-C slice 1 (`3e3b925`): 29-plane encoder (73×64 policy, underpromotion planes), GambitNet v2, MCTS v2 (PUCT + Dirichlet), regime configs w/ lineage, checkpoint ladder + arena gate + Elo ledger, pipeline/train/bench CLIs. §10 bugs fixed w/ failing-then-passing proof (terminal sign, instant resign, promotion policy). 108 new tests; bench: batch-256 NN **1.49 ms** on 3080, ~10.6k games/h est |
| 2026-11-21 | WP-B slice 1 (`258e84d`): web client wired to the live backend — typed REST/WS clients (seq + reconnect + resubscribe), Play live mode w/ offline demo fallback, Watch live metrics, promotion duplicate-key fix. Browser-verified headless: 229 API calls all 200 (promotion, undo, resign, as-black, reconnect); svelte-check 0 errors |
| 2026-11-21 | Cross-slice verification: **189/189 pytest**, ruff clean on all new code, live boot — health / black-undo-noop / SF turn 0.33 s / WS spontaneous push all PASS. Branch pushed → PR |
| 2026-11-21 | FS-1/2/6/9 landed (`04e40b7`): opponent `nn` (ModelServer, MCTS 32/96/256 sims, mtime hot-swap, pre-warm → 450 ms/move), arena_games SQLite ledger w/ idempotent backfill + `elo_by_regime`, WS journal feed (log frames), TTL reaper + rate limit + static single-origin mount. 25 new tests, 53 scoped green |
| 2026-11-21 | FS-7 landed (`f982dd5`): Rust `gambit_core` PyO3 crate — perft exact d1–5, encoder bit-exact (3,125-pos fuzz, zero mismatches), zero unsafe; tree-step CPU slice **9.0–11.1×** vs python-chess (330→30 µs). Hot-path wiring deferred to the gated §5 M3 slice |
| 2026-11-21 | FS-8 landed (`7fb1113`): client complete — NN opponent UI w/ live Elo, Watch journal + arena ledger + per-regime leaderboard, keyboard roving-cursor a11y, responsive 36/36 (320–1920), unicode toggle, single-origin relative base. svelte-check 0 errors; 6 browser verification stages all green |
| 2026-11-21 | FS-3/4/5/6 landed (`a5ba985`): all five regimes trainable (sf-games/sf-distill/hybrid/finetune smokes ~12 s each vs real Stockfish), queued GPU evaluator (**×8.8 sims/sec at p32; 16.8k games/h**), crash-resumable runs (found+fixed a real play-RNG threading bug via the resume gate), HF publish, live journal — **verified live on the Watch page**. Integration fixes `ea4d94a` (SPA 404 contract, journal-frame drain, latency flake from desktop GPU load). **Final gate: 306/306 pytest, svelte-check 0, build green. Finish-line build COMPLETE — remaining: FS-10 training start (user gate) + tunnel (user gate)** |
| 2026-11-21 | FS-6b landed (`306e9a5` server + `d946b5e` training): per-move game streaming — every self-play ply journals `live` (run-unique slot semantics, pinned by tests) + `game_end`; server routes journal lines to WS frames with no-replay subscribe and a coalesced 20-frame/2s cap; journal.append perf fix (persistent handle, flush-per-line; streaming A/B −0.9% @p16). +24 tests (196 scoped green), ruff clean |
| 2026-11-21 | FS-6c landed (`51e4117`): Watch **LIVE GAMES grid** — 16/9/6 responsive mini-boards keyed by run-unique game id (newest-fen-wins per cell, game_end shows result ~5 s then frees the cell), featured view w/ move list, keyboard/aria a11y; sample board kept inside the teaching empty state. Fixed a real Svelte 5 bug: Map-derived spread computed once, never re-invalidated (45 frames → 1 cell). Verified through the real server path (synthetic journal → tailer → WS → UI, 46/46 harness stages incl. 320/768/1920 zero overflow); svelte-check 0, build 153.45 kB |
| 2026-11-21 | Standing review loop (post-FS-6c sweep): responsive **40/40 clean** (4 pages × 320–1920, 0 overflow / 0 console errors / 0 failed requests, screenshots audited) — user-reported "odd sizes look broken" **CLOSED**; live-server probe **8/8** (health, SPA deep links, games/metrics shapes, WS train `metrics` snapshot + seq monotonic, live idle no-replay, invalid-topic reject); FS-6c grid idle empty-state verified. No new defects. playwright+chromium added to the venv as the standing headless sweeper |
| 2026-11-21 | Review loop converged → **optimization phase**: re-bench `bench_train.py` (tiny, 3080, idle-enough desktop) — p32 **655 sims/sec / 14.7k games/h** (×9.84 vs p1) vs recorded 808/16.8k (−12…−19%, desktop GPU contention suspected), NN batch 2.11 ms (vs 1.49 ms), streaming A/B +10.7% @p16 (vs −0.9%) — no winner to adopt; p32 stays the profile. Next levers filed in TODO: idle-GPU re-bench protocol + §5 M3 `gambit_core` hot-path wiring |

---

## 14. Subagent Delegation Map

How parallel work is delegated. Rules of engagement (enforced by `.opencode/agents/orchestrator.md`): workers never `git commit`/`git push` — only the orchestrator integrates; never run two live packages with overlapping file ownership.

### 14.0 Pinned API contract v1

WP-A (backend) implements this; WP-B (client) codes against it. Any deviation = version bump + §13 changelog row + both runbooks updated. Nothing drifts silently.

**REST — all JSON under `/api`:**

| Endpoint | Request → Response | Notes |
|---|---|---|
| `POST /play/new` | `{color: "white"\|"black"\|"random", opponent: "fool"\|"stockfish", strength}` → `{game_id, fen, orientation}` | orientation = the player's color |
| `GET /play/{id}` | → `{game_id, fen, moves: [san], status, winner?, in_check, last_move}` | full state |
| `POST /play/{id}/move` | `{from, to, promotion?}` → state | illegal ⇒ 422; server-side legality always |
| `POST /play/{id}/undo` | → state | pops both half-moves vs The Fool. **Since 2026-11-21:** take-backs after game over (200, was 409) — board-finish endings revert the finishing ply pair (same parity rule as mid-game), resignations un-resign with history untouched, result state cleared **and persisted** (what the rehydrate layer trusts — restart keeps the game active); abandoned/reaped games stay 409 |
| `POST /play/{id}/resign` | → state | |
| `GET /games` | `?kind=play\|selfplay&limit=` → `{games: [...]}` | archive listing |
| `GET /games/{id}/pgn` | → `text/pgn` | full PGN export |
| `GET /metrics` | → telemetry snapshot | elo, loss, win_rate, games_played |
| `GET /checkpoints` | → list | HF Hub + local |
| `POST /train/start` `/stop` `/reset` | → `{ok}` | `ADMIN_TOKEN` required |

**WebSocket `/ws` — one connection per client; typed JSON; sequence-numbered:**

- Client→server: `{"seq": n, "type": "subscribe", "topic": "lobby" | "game:{id}" | "train"}`
- Server→client: `{"seq": n, "topic": ..., "type": "position" | "state" | "metrics" | "log" | "lobby" | "error", "payload": {...}, "ts": iso8601}`
- `position.payload`: `{fen, last_move, move_number}` · `metrics.payload` mirrors `GET /metrics` · `log.payload`: `{level, message}`
- Heartbeat ping/pong every 30s; on reconnect the client resubscribes with its last seen seq.

### 14.1 Work packages

| WP | Scope | Owns (edit) | Deliverable |
|---|---|---|---|
| **WP-A** | Backend core | `server/**`, `tests/**`, `scripts/serve.py` | FastAPI + WS + SQLite per §14.0; self-play watcher feed; boot script |
| **WP-B** | Client integration | `web/**` | WS client; real telemetry/game slots replace demo data |
| **WP-C** | Training pipeline | `training/**`, `scripts/train*` | RTX 3080 self-play, checkpoints → HF Hub, metrics → WS `train` topic |
| **WP-D** | Rust core | `gambit_core/**`, `Cargo.toml`, parity tests in `tests/` | PyO3 movegen; perft + fuzz gates (§5) |
| **WP-E** | Hosting & hardening | deploy configs, cloudflared, security middlewares | tunnel, rate limits, backups |

Sequencing: **WP-A → WP-B** (client codes against the pinned contract once WP-A's schema lands). **WP-C and WP-D** start after WP-A and run in parallel. **WP-E** last.
