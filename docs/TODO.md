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

**Enforcement:** OpenCode automation `standing-review-loop` (`aut_1fcc6d34a0010FX5Pxly82OALk`) fires
this loop every 6 hours as a fresh orchestrator session — **ACTIVE** (user activated it; first fire
06:00 local). Its prompt carries a dirty-tree safety gate: if `git status --short` is not clean it
aborts without touching anything, so it never collides with in-progress work. Pause/resume:
`opencode.cmd api post /api/experimental/automation/aut_1fcc6d34a0010FX5Pxly82OALk/pause` (or `/resume`).
Keep its prompt in sync with the orchestrator runbook.

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
