# TODO — The Fool's Gambit v2

Living checklist. Detailed design lives in `docs/UPGRADE_PLAN.md`; this file tracks what's next.
Rule: update on every execution step (same as the plan doc).

## Standing review loop (user mandate)

After every milestone — and on any "review" request — run the full loop (encoded in
`.opencode/agents/orchestrator.md`): visit **every** page in a real browser at mobile **and** desktop
sizes → screenshot each → actually read the screenshots and audit them → test the backend and real
functionality end-to-end → list **every** defect and improvement found → add to this file → fix →
repeat. When the loop converges (clean sweep), switch to optimization: benchmark training
algorithms/configs (sims/sec, games/hour, strength per wall-clock hour) and log results here + §13.

**Enforcement:** OpenCode automation `standing-review-loop` (`aut_1feadea1b001Flh6KarH8f58To`) fires
this loop **hourly** as a fresh orchestrator session — **ACTIVE** (user chose hourly; the heavy sweep
itself takes 20–40 min, which is why the fast cadence lives in the monitor below). Its prompt carries
a dirty-tree safety gate: if `git status --short` is not clean it aborts without touching anything.
Between sweeps, a second no-model automation `backend-health-monitor`
(`aut_1feacfef4001uYCa3ylqCP080l`) probes the site (:5173) + API (:8000) **every 5 minutes** and
alerts on up→down transitions (zero token cost). A third automation `code-patrol`
(`aut_1feae9f48001dJ0y5fJDAxI7pg`) fires a **model-driven** quick check **every 5 minutes** —
rotating: full test suite / lint one area / live perf bench vs the Hygiene baseline / single-file
improvement skim / web check+build — logging findings to gitignored `data/patrol.log` (never
committed) and committing small fixes immediately. The dirty-tree gate makes patrols self-skip
while a deep sweep holds the repo. Pause/resume any of the three via
`opencode.cmd api post /api/experimental/automation/{id}/pause` (or `/resume`).
Keep prompts in sync with the orchestrator runbook.

## Now / Next

- [ ] **Responsive audit & fixes** — odd screen sizes look broken (user report). Test every view
      (Home / Play / Watch / About) at ~320, 375, 540, 768, 850, 1024, 1180, 1280, 1440, 1920 px.
      Fix layout breakage, board sizing, grid collapses; then keep a screenshot sweep in the loop.
- [x] **Commit the web port** — LANDED (branch `feature/play-and-local-overhaul`, conventional commits; PR → main pending push):
      1. `chore!: remove legacy React client`
      2. `build: add python packaging and refresh ignore rules` (pyproject.toml, uv.lock, .gitignore)
      3. `feat(web): sepia-tarot Svelte client, 1:1 port of approved mock` (web/, incl. mock.css port)
      4. `docs: update upgrade plan and agent runbooks` (docs/UPGRADE_PLAN.md stack + §14, .opencode/agents/*)
      5. `ci: rewrite workflow for web client and drop Vercel-era checks`
- [x] **Rewrite CI** — DONE (was: builds the deleted `client/` and health-checks the
      old Vercel site — currently guaranteed red). New: web `npm ci && npm run build && npm run check`,
      Python `ruff check` + `pytest` (once tests exist), later `cargo test` for the Rust crate.

## Work packages (per docs/UPGRADE_PLAN.md)

- [x] **WP-A — Backend**: FastAPI + WebSocket + SQLite. **LANDED** — slice 1 (`9a9c1d9`) + slice 2
      (`2e12f19`): play/new·move·undo·resign, games archive + PGN, metrics, admin-gated train stubs,
      /ws hub (monotonic seq, lobby/train/game topics), SQLite WAL, demo Fool engine, Stockfish 19
      opponent (wandering/curious/enlightened skill ladder, one UCI process per game, graceful
      lifecycle, degrades to Fool on engine crash), WS train telemetry (10s, idle w/o subscribers).
      Live cross-slice check: SF turn 0.33s, spontaneous WS push verified. Slice-3 queue: TTL reaper.
- [x] **WP-B — Client wiring**: **LANDED** (`258e84d`): typed REST + WS clients (seq counter,
      reconnect backoff, auto-resubscribe), Play live mode with offline demo fallback
      (OFFLINE · DEMO chip), Watch live metrics over WS with illustrative fallback, promotion
      duplicate-key fix in Board.svelte. Browser-verified headless: 229 API calls all 200 —
      ~30-move live game incl. promotion via picker, undo, resign, as-black with opener, reconnect.
      svelte-check 0 errors.
- [ ] **WP-C — Training**: self-play pipeline on the RTX 3080 (torch 2.15 cu132 pinned in pyproject),
      checkpoints to HF Hub, resumable runs, metrics emitted to backend/WS.
      **Arena gating per §5.1 is binding**: checkpoint ladder 5/10/15/20 min wall-clock, each checkpoint
      arenas the champion at reduced sims, promotion ≥55%, arena games persisted `kind='arena'`,
      Elo ledger → `/api/metrics` + WS `train` topic.
      **Regime matrix per §5.2 is binding**: `scratch` / `sf-games` / `sf-distill` / `hybrid` / `finetune`,
      per-regime lineage (`runs/` + SQLite `runs` table + HF per-regime repos), cross-regime round-robin
      arena on one ledger.
      **Slice 1 LANDED** (`3e3b925`): 29-plane encoder (73×64 policy, underpromotion planes),
      GambitNet v2 (TINY 81k / SMALL / BASE), MCTS v2 (PUCT Q=−child.value_mean + Dirichlet),
      §10 bugs fixed with failing-then-passing proof (terminal sign, instant resign, promotion
      policy; mate-in-1 found at 96 sims), pipeline + `scripts/train.py` + `scripts/bench_train.py`,
      108 new tests. Bench (3080, TINY): NN batch-256 **1.49 ms**, train step 4.77 ms, ~10.6k games/h est.
      **Slice 2 next**: sf-* data providers (`NotImplementedError` stubs ready), queued GPU evaluator,
      SQLite arena backfill.
- [ ] **WP-D — Rust `gambit_core`**: fast movegen + perft harness via PyO3; replaces chess.js/chess
      hot paths for self-play speed.
- [ ] **Deploy/hosting**: host the site so the user + friends can play (replace dead Vercel flow);
      reverse proxy + backend service; decide box (local machine vs VPS).

## Finish-line plan (2026-11-21 · subagent-optimized · supersedes the WP list below for execution order)

**Definition of done:** a neural opponent trained by the §5.1 arena ladder on the 3080, playable in
the browser as opponent `nn` with live Elo on Watch, all five §5.2 regimes trainable, Rust hot path
benched, site hosted for friends. Ordered by impact; each slice has an acceptance gate.

| ID | Slice | Owner (file ownership — strict) | Acceptance gate |
|---|---|---|---|
| FS-1 | **Opponent `nn`** — `ModelServer` singleton loading `runs/<regime>/champion.json` → `GambitNet` + state_dict (device auto cuda>cpu, mtime reload); reply via `MCTSSearch` (sims: wandering 32 / curious 96 / enlightened 256), masked policy, async `to_thread` + per-session lock mirroring the stockfish path; `create_game(opponent="nn")` → 503 "no trained model yet" when no champion exists | backend-dev → `server/**`, new `tests/test_{api,ws}_*.py` | pytest green incl. no-checkpoint-503 + full-turn-with-tiny-checkpoint tests; NN move < 2 s at curious |
| FS-2 | **Arena ledger in SQLite** — new `arena_games` table (run_id, regime, ckpt, opponent, games, wins, draws, elo, ts), idempotent startup backfill from `data/arena.jsonl`, `/api/games?kind=arena`, metrics gains per-regime Elo map | backend-dev | double-boot backfill creates no dupes; tests |
| FS-3 | **sf-* + finetune data providers** — fix `training/stockfish_engine.py` path resolution (env → `tools/stockfish.exe` → PATH); implement `collect()` for `sf-games` (self-play mixed vs SF skill ladder 2→12), `sf-distill` (policy ← SF move, value ← SF eval), `hybrid` (distill warm-start → scratch chain), `finetune` (init from another champion / HF id) | training-dev → `training/**`, `scripts/train*.py`, `bench_train.py`, new `tests/test_training_*.py` | each regime's `collect()` returns examples; CPU smoke < 3 min per regime; tests |
| FS-4 | **Queued GPU evaluator + parallel self-play** — async batch service (queue from N games → ≤256 batch / 4 ms window → one forward → fan-out, no cross-game leakage); `parallel_games` 1–32 in `RegimeConfig`; wire `bench_train.py` to report old-vs-new | training-dev | ≥3× games/hour at 16 games vs slice-1 sequential baseline (record both in §13); queue correctness tests |
| FS-5 | **Resumable runs + HF publish** — replay buffer + optimizer state persisted per checkpoint; `run_training(resume=...)` exact rebuild; `huggingface_hub` per-regime upload, graceful no-op without `HF_TOKEN` | training-dev | kill mid-run → resume completes; tests |
| FS-6 | **Training journal on WS** — training appends structured events to `data/train_log.jsonl`; server tails it on `train`-topic subscribe (last 50) + pushes new lines every 2 s while subscribed | training-dev (writer) + backend-dev (tailer) | WS client receives live journal frames; tests |
| FS-7 | **Rust `gambit_core`** — PyO3 crate: bitboard movegen + perft d1–5 exact (20/400/8902/197281/4865609), batch encoder **bit-exact** vs `training/encoder.py` (1000 random positions cross-checked in pytest), legal-mask gen; bench vs python-chess recorded | rust-dev → `gambit_core/**`, `Cargo.toml`, crate lines of `pyproject.toml`/`uv.lock` | cargo test green; perft exact; encoder bit-exact; speedup number in §13. NOT wired into hot path (§5 M3 gate ≥3× e2e later). If cargo missing: `winget install Rustlang.Rustup` first |
| FS-8 | **Client: NN opponent + Watch journal + polish** — Play gains "The Fool (NN)" (disabled hint until `metrics.elo` non-null; opponent card shows live Elo); Watch: journal pane (WS log frames), arena results, per-regime leaderboard; keyboard nav (arrows+enter), responsive audit 320–1920, unicode-glyph toggle (persisted) | client-dev → `web/**` | svelte-check 0 errors; build green; browser-verified flows incl. vs-stockfish and vs-nn (tiny fixture checkpoint); screenshots read back |
| FS-9 | **Production hardening + hosting** — static mount of `web/dist` (single origin, CORS off), TTL reaper (`FSG_GAME_TTL_MIN` default 30 → auto-resign abandoned games), light per-IP rate limit; then cloudflared tunnel + service install + `ADMIN_TOKEN` | backend-dev (code) + orchestrator+user (tunnel/domain — user-gated, needs CF account) | reaper test; static mount test; tunnel runbook in docs |
| FS-10 | **First real training run** — night profile, scratch regime, ladder 5/10/15/20→every 20, arenas between bursts, Watch live | **USER-GATED** ("start training") | Elo ledger rows; champion promoted |

**Loop protocol:** workers never commit — the delegating loop integrates (scoped staging per the
ownership column, conventional commits, push feature branch). A 30-min orchestrator heartbeat
integrates finished slices (scoped verification first: pytest + ruff / svelte-check+build) and
spawns the next open slice's worker. Dirty tree from in-flight work = heartbeat skips. User gates:
FS-9 tunnel + FS-10 training start.

## Overnight parallel build — 2026-11-21 (user asleep, agents running)

Three workers on **disjoint file ownership**; they never commit — the orchestrator verifies and
commits each slice as it lands, then pushes + opens the PR:

- `backend-dev` → WP-A slice 2: owns `server/`, `scripts/serve.py`, new `tests/test_{api,ws}_*.py` — Stockfish opponent (env `FSG_STOCKFISH` → `tools/stockfish.exe` → 503), skill ladder per strength, WS train-topic telemetry every 10s.
- `client-dev` → WP-B slice 1: owns `web/` — WS client (seq + reconnect), REST client, Play live mode w/ offline fallback, Watch live metrics; verifies in browser on API :8001 / Vite :5173.
- `training-dev` → WP-C slice 1: owns `training/`, new `scripts/train*.py`/`scripts/bench_train.py`, new `tests/test_training_*.py` — encoder, ResNet v2, MCTS (§10 sign-bug regression), regime configs (§5.2), checkpoint ladder + arena skeleton (§5.1), JSONL arena ledger + `data/metrics.json` writer, CPU smoke run + bench.

**Integration queue (orchestrator, as each reports):**
- [x] Verify backend-dev: full pytest + ruff → commit `feat(backend): ...` (`2e12f19`)
- [x] Verify client-dev: `npm run check` + `npm run build` → commit `feat(web): ...` (`258e84d`)
- [x] Verify training-dev: pytest + ruff + smoke artifacts exist → commit `feat(training): ...` (`3e3b925`)
- [x] Cross-slice verify: 189/189 pytest, ruff clean, live boot — health / black-undo-noop /
      SF turn 0.33s / WS spontaneous push all PASS
- [x] §13 changelog rows + TODO checkboxes → push branch → `gh pr create` → report URL to user

## Client polish (post-port)

- [ ] Difficulty select → wire to real engine strength once backend exists (currently cosmetic).
- [ ] Opponent card ELO `—` → real number from backend.
- [ ] Consider unicode-glyph pieces option (mock look) vs Cburnett SVGs — user's call.
- [ ] Keyboard accessibility pass on board (arrow/enter movement), focus states review.
- [ ] Watch page: auto-follow live game when telemetry connects.

## Hygiene

- [ ] **Backend perf re-benchmarks** — slice-1 baseline (local, single client, uvicorn single worker):
      health 2.2 ms · play/new 27.9 ms · move 16–21 ms · games list 13.1 ms · PGN export 36.4 ms.
      Re-benchmark and record deltas after: (a) WS live feed wired (WP-B), (b) self-play telemetry
      streaming (WP-C), (c) 10 concurrent games. Targets: p95 move latency < 50 ms at 10 concurrent
      games; archive query < 100 ms at 10k games; zero WS message drops under load.

- [ ] **Backend slice 3 (queued)** — TTL reaper for abandoned still-active games (a stockfish process
      currently lives until app shutdown if its game is never finished); strength → real ELO once the
      model trains (WP-C).
- [ ] Remove `web/dist/` + stray build artifacts from working tree before commit (gitignored already).
- [x] Stockfish binary — SF 19 installed to gitignored `tools/stockfish.exe` (UCI verified). Legacy
      `training/stockfish_engine.py` still defaults `STOCKFISH_PATH` to a dead WSL path — WP-C adapts.
- [ ] Delete temp mock-server + extracted zip from `%TEMP%\opencode\fool-frontend` after mock retired.
- [ ] svelte-check clean run (`npm run check`) — add to CI once green.
- [ ] Rename session/docs references to old purple theme → sepia tarot (screenshots in plan doc).
