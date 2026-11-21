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

**Enforcement:** automation roster trimmed at user request (2026-11-21) — the health monitor,
hourly review loop, and 5-min code patrol were **removed**. The one surviving automation is the
**`delegation-loop`** v2 (`aut_1fed98df3001bs0GXJENZPI6dA`, every 10 min): integrates finished worker
slices (scoped verification → scoped conventional commit → push → tick FS checkbox) when the tree
holds exactly one worker's files, and spawns the next open Finish-line slice's worker when the tree
is clean — FS-9 tunnel + FS-10 training start are hard-gated to the user in its prompt. **Every fire
appends an audit line to gitignored `data/heartbeat.log`** (watch: `Get-Content data\heartbeat.log -Wait`).

## Now / Next

- [x] **Responsive audit & fixes** — CLOSED (2026-11-21, post-FS-6c sweep): headless chromium
      sweep 4 pages × 10 sizes (320/375/540/768/850/1024/1180/1280/1440/1920) = **40/40 clean**
      (0 horizontal overflow, 0 console errors, 0 failed requests) + screenshot audit at
      320/850/1024/1920 across all pages. Root cause was the FS-8 `.chart-wrap` flex→grid fix.
      Sweeper: `uv run python` + playwright (chromium installed in venv 2026-11-21).
- [x] **Commit the web port** — LANDED (branch `feature/play-and-local-overhaul`, conventional commits; PR → main pending push):
      1. `chore!: remove legacy React client`
      2. `build: add python packaging and refresh ignore rules` (pyproject.toml, uv.lock, .gitignore)
      3. `feat(web): sepia-tarot Svelte client, 1:1 port of approved mock` (web/, incl. mock.css port)
      4. `docs: update upgrade plan and agent runbooks` (docs/UPGRADE_PLAN.md stack + §14, .opencode/agents/*)
      5. `ci: rewrite workflow for web client and drop Vercel-era checks`
- [x] **Rewrite CI** — DONE (was: builds the deleted `client/` and health-checks the
      old Vercel site — currently guaranteed red). New: web `npm ci && npm run build && npm run check`,
      Python `ruff check` + `pytest` (once tests exist), later `cargo test` for the Rust crate.

## Review-loop harvest (2026-11-21 functional sweep — API/WS/UI drive)

Phase B of the standing review loop: 34 API/WS checks + 10 UI-drive checks all pass (games
list/PGN/metrics/checkpoints, move/undo/resign incl. promotion-to-queen, illegal+404+429 paths,
WS subscribe snapshots, invalid/unsupported rejects, live position fan-out, per-topic seq
monotonicity, restart-persistence probe). Zero console errors, zero HTTP ≥400. Findings:

- [ ] **Next — restore active play sessions on boot.** Defect: games+moves persist to SQLite
      (row + plies verified in `data/gambit.db`) but `get_session` is memory-only and the app
      lifespan never rehydrates → any server restart/deploy 404s an in-progress game
      (GET `/api/play/{gid}`, WS `game:{gid}`, further moves) and drops it from the lobby.
      Implement: on `get_session` miss (or startup) rebuild `Session` from the DB by replaying
      stored UCIs; requires a `strength` column on `games` (small migration) for faithful
      Fool/NN strength; keep TTL reaper semantics unchanged.
- [ ] **Polish — unify archive game id field.** REST `/api/games` items use `id`, while the
      client's `GameArchiveEntry` (web/src/lib/api/http.ts) and the WS `lobby` payload use
      `game_id`. Benign today (UI only consumes arena entries via `run_id`) but a latent trap;
      align on one shape.
- [ ] **Idea — undo after game over.** Server 409s undo on finished games (UI matches by
      disabling Undo when over). Consider allowing take-backs after a loss for casual play.
- Note (not a defect): POST rate limit (20-token burst, 2/s refill, per IP) answers 429 with
      CORS headers — browsers are unaffected; script/API integrations must pace ~2 POST/s.
- Note: promotion verified via API (accepted; `=Q` in moves + PGN round-trip). The promo-picker
      UI path was browser-verified earlier by client-dev (WP-B) and isn't re-driven here only
      because reaching promotion through UI clicks needs a long manual game.

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
| FS-1 | ✅ **LANDED (`04e40b7`)** — Opponent `nn` — `ModelServer` singleton loading `runs/<regime>/champion.json` → `GambitNet` + state_dict (device auto cuda>cpu, mtime reload); reply via `MCTSSearch` (sims: wandering 32 / curious 96 / enlightened 256), masked policy, async `to_thread` + per-session lock mirroring the stockfish path; `create_game(opponent="nn")` → 503 "no trained model yet" when no champion exists | backend-dev → `server/**`, new `tests/test_{api,ws}_*.py` | 53 scoped tests green incl. no-checkpoint-503 + full-turn tests; NN move **450 ms** at curious (CUDA TINY, pre-warmed) |
| FS-2 | ✅ **LANDED (`04e40b7`)** — Arena ledger in SQLite — new `arena_games` table (run_id, regime, ckpt, opponent, games, wins, draws, elo, ts), idempotent startup backfill from `data/arena.jsonl`, `/api/games?kind=arena`, metrics gains per-regime Elo map | backend-dev | double-boot backfill creates no dupes (UNIQUE(ckpt,ts) + INSERT OR IGNORE); tests |
| FS-3 | ✅ **LANDED (`a5ba985`)** — **sf-* + finetune data providers** — path chain fixed (env → `tools/stockfish.exe` → PATH); `sf-games` (mixed vs SF skill 2→12), `sf-distill` (policy ← SF move, value ← SF eval), `hybrid` (distill warm-start → scratch chain), `finetune` (champion/HF id/path init) | training-dev → `training/**`, `scripts/train*.py`, `bench_train.py`, new `tests/test_training_*.py` | each regime `collect()` returns examples; smokes ~12 s each (≪3 min); 110 training-scoped tests green |
| FS-4 | ✅ **LANDED (`a5ba985`)** — Queued GPU evaluator + parallel self-play — `QueuedNNEvaluator` (≤256 batch / 4 ms window / Future fan-out), `parallel_games` 1–32, bench wired | training-dev | **sims/sec ×8.8** at p32 (92→808), games/h 2.4k→**16.8k** (×7.0); p8 reproduces the recorded slice-1 baseline; queue correctness tests green |
| FS-5 | ✅ **LANDED (`a5ba985`)** — Resumable runs + HF publish — model+optimizer+buffer+3 RNG streams+counters persisted per checkpoint; `--resume [RUN_ID]`; strict drift refusal; HF per-regime upload, no-op without `HF_TOKEN` | training-dev | CPU kill/resume reproduces fresh-run loss to 1e-5 (**found+fixed a real RNG-threading bug** — test failed before, passes after); mocked-upload tests |
| FS-6 | ✅ **LANDED (`04e40b7` + `a5ba985`)** — Training journal on WS — writer appends structured iteration/arena events to `data/train_log.jsonl`; server tails it (≤50 backlog on subscribe, 2s push); verified live in the browser | training-dev (writer) + backend-dev (tailer) | WS client receives live journal frames ✓ (seen rendering on Watch); reader+writer tests green |
| FS-7 | ✅ **LANDED (`f982dd5`)** — Rust `gambit_core` — PyO3 crate (chess 3.2.0 core, zero unsafe): bitboard movegen + perft d1–5 exact (20/400/8902/197281/4865609), batch encoder **bit-exact** vs `training/encoder.py` (3,125-position differential fuzz, zero mismatches), legal-mask gen; bench vs python-chess recorded | rust-dev → `gambit_core/**`, `Cargo.toml`, crate lines of `pyproject.toml`/`uv.lock` | cargo test 16 ✓; pytest 35 ✓; perft exact ✓; encoder bit-exact ✓; **9.0–11.1×** tree-step CPU slice (≥3× gate cleared). NOT wired into hot path (§5 M3 gate ≥3× e2e later) |
| FS-8 | ✅ **LANDED (`7fb1113`)** — Client: NN opponent + Watch journal + polish — Play gains "The Fool (NN)" (gated on `bestKnownElo()`, live EST. ELO card), Watch: WS journal (auto-follow, 200-line cap), arena ledger table, per-regime Elo leaderboard; keyboard roving-cursor a11y; responsive sweep 320–1920 = **36/36** (`.chart-wrap` flex→grid root-cause fix); unicode-glyph toggle persisted | client-dev → `web/**` | svelte-check 0 errors; build green (129 modules); browser stages 25/25 + 16/16 + 4/4 + 4/4 + 6/6 + 36/36, zero console errors |
| FS-9 | ✅ **CODE COMPLETE (`04e40b7` + `a7f2dff`)** — static single-origin serving (SPA fallback for deep links), TTL reaper (`FSG_GAME_TTL_MIN` 30), per-IP rate limit — **remaining: cloudflared tunnel + domain is USER-GATED** (needs the user's CF account; no worker will touch it) | backend-dev (code) + user (tunnel) | reaper test ✓; static mount + SPA-deep-link tests ✓; tunnel runbook pending the user gate |
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
