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

- [x] **Restore active play sessions on boot** — VERIFIED FIXED (2026-11-21 18:49 deploy):
      the pre-deploy game `5a01dfddf3a2` rehydrated after restart with full 6-ply state.
      (Original defect: games+moves persisted to SQLite but `get_session` was memory-only —
      any restart 404'd in-progress games.)
- [ ] **Polish — unify archive game id field.** REST `/api/games` items use `id`, while the
      client's `GameArchiveEntry` (web/src/lib/api/http.ts) and the WS `lobby` payload use
      `game_id`. Benign today (UI only consumes arena entries via `run_id`) but a latent trap;
      align on one shape.
- [x] **Idea — undo after game over.** SERVER-SIDE LANDED (2026-11-21, `51edb43`): take-backs on
      mate/stalemate/resigned games (finishing ply pair reverts, resignations un-resign; abandoned
      stays 409). Client slice (button + re-render) in flight tonight.
- Note (not a defect): POST rate limit (20-token burst, 2/s refill, per IP) answers 429 with
      CORS headers — browsers are unaffected; script/API integrations must pace ~2 POST/s.
- Note: promotion verified via API (accepted; `=Q` in moves + PGN round-trip). The promo-picker
      UI path was browser-verified earlier by client-dev (WP-B) and isn't re-driven here only
      because reaching promotion through UI clicks needs a long manual game.

## Live-run harvest (2026-11-21, `second-light` — findings from the first real overnight run)

First real training run on the modernized pipeline. Findings from live observation,
full-page screenshots, and the U9 audit — state at scoring time:

- [x] **Promotion ladder works in production** — first promotion in project history
      (13:55: `ckpt-000128-0064` dethroned `ckpt-000004-0004` 0.750 ≥ 0.55), then six
      consecutive successful defenses; 8/8 correct gate verdicts. Closes the F-1 audit
      finding with a live demonstration, not just a test. **Gate 8 (19:45) was the first
      *decisive* gate**: candidate `ckpt-000512-0256` scored 0.250 (+0 -8 =8) — the
      champion is now measurably stronger than fresh weights, not merely draw-equivalent.
      Lineage Elo rose 793.68 → 801.97 on the decisive result.
- [x] **U9 Glicko-2 + SPRT** — LANDED (`d7d1bff`): rating error bars, honest
      draw-saturated SPRT behavior (all-draw = zero information = undecided), RD-capped
      promotion rule. Additive ledger format; activates on next launch; tonight's state
      file verified loadable through the new code (crash-restart safe).
- [x] **Watch: false "Training runs offline" during title matches** — FIXED (`01ac676`,
      server key `b39c722`): three-state panel (IDLE / TITLE MATCH / LIVE) now driven
      primarily by server-truth `gate_in_progress` with the quiet-window inference kept as
      a pre-deploy fallback; gate line quotes `last_journal_ts` age for zero-frame joins.
      Client slice `01ac676`, verified on production.
- [x] **Rating surfaces contradict each other** — FIXED (`b39c722`, deployed 18:49 and
      verified live): `elo_by_regime` now derives from ledger JSONL replay (agreement with
      the trainer by construction); periodic 60s idempotent backfill replaces boot-only;
      honest `elo: null` preserved through the v2→v3 arena migration (6 rows repaired
      800.0 → NULL). Verified on production: all 9 second-light checkpoints with recomputed
      values (champion 806.32, dethroned 792.0), smoke-regime fossils gone from the map.
      Canonical rating = trainer-owned lineage series (headline); per-checkpoint maps are
      dated secondary diagnostics.
- [ ] **200-ply cap auto-adjudicates draws** — `training/selfplay.py:1301-1304`,
      `:1555-1569`, `:514-516` force `1/2-1/2` on truncation. Games now hit the cap
      routinely as nets sharpen: teaches shuffle=draw to the value head (self-play targets)
      and zeroes SPRT information (arena draw rate). Morning fix (NOT mid-run): KataGo-style
      eval adjudication — both sides' final root values |v| > θ≈0.9 agreeing in sign → ±1,
      else draw; no extra NN calls.
- [ ] **Gate economics degrade with strength** — gates 1–5 ≈ 8–10 min, gate 6 21.6 min,
      gate 7 45.8 min, gate 8 41.3 min; game lengths doubled (35–173 → 127–200 plies) as
      blunders vanished. Pre-registered response **P-U9-4** (next run): 32 games × 100 plies
      (same ply budget; fresh-candidate RD 121.6→88.8) + `promote_rd_cap` 150→100.
      Sequential SPRT is enabled and proven inert at N≤32 — it becomes decisive only in big
      anchored sessions. Confirmed again at gate 8: 41 min of GPU for 16 games is the
      dominant per-hour cost of the whole run loop.
- [x] **"BEST STREAK: —" dead stat on Watch** — REMOVED (`01ac676`), verified gone on
      production (no dead numbers per the research standard).
- [x] **Arena results panel: fossils oldest-first** — FIXED (`01ac676`): newest-first with
      the pre-F-1 self-vs-self fossils kept visible under an honest SHOW ALL footnote;
      verified on production (22 matches, legend, AS OF stamp).
- [x] **Journal panel read as broken during self-play** — FIXED (`01ac676`): `log` frames
      only exist for iteration/arena lines (~1/45 min), so the empty diary now shows a live
      heartbeat (PRACTICE MOVES · ~2/S · LAST MOVE just now) and reverts to the written
      journal when diary lines arrive; backlog `replayed: true` frames are ignored honestly.
- [ ] **U6 multiprocess shards** — the remaining 3–4× throughput lever (66–77 games/h is
      honest but modest vs the 15.5k bench that didn't extrapolate). Bench-gated: needs the
      idle GPU after the run.
- [ ] **Gumbel A/B at gate quality** — 6.14× bench throughput is real; strength evidence is
      not (draw-saturated 0.50 h2h, see N2 in the ledger). Needs the GPU.
- [x] **Production crash loop (discovered live 18:0x)** — the running pre-v3 server threw
      `TypeError: float() argument ... not 'NoneType'` inside `elo_by_regime` on every
      metrics tick (caught per-key, so headline metrics served; the regime map degraded to
      `{}`). Root cause: honest `elo: null` rows (F-5) flowing into a float coercion.
      FIXED by the v3 work (`b39c722`); deployed 18:49, zero tracebacks since.
- [x] **Deploy lost in-progress games** — VERIFIED FIXED by a real deploy (not just a
      test): the game played on :8000 before the 18:49 restart (`5a01dfddf3a2`, 6 plies)
      rehydrated after boot with full state, `status: active`, and the new additive
      `reason` field. Closes the review-loop "restore active play sessions on boot" item.
- [x] **Fresh-load deaf window on production** — FIXED in two halves (`2aedcc7` + `c0d7e97`).
      Server: the hub had one shared send path, so a single ping-answering non-reading
      consumer parked the whole fan-out (proven: 11→1.8→0 fps with one parked reader while
      a fresh mid-wedge reader got zero live frames). `server/ws.py` now gives every
      subscriber a bounded queue (512) with one pump per connection; overflow drops only the
      slow consumer and closes its transport with code 1013 (bounded 5 s) so the client
      reconnects instead of hanging. Ledger replay + journal scans moved to
      `asyncio.to_thread`, off the event loop. 7 regression tests in `tests/test_ws_hub.py`.
      Client: `web/src/lib/api/ws.ts` watchdog — 10 s tick, `train` topic, 25 s stale
      threshold, 15 s refocus threshold, 2 s half-open grace, one reconnect owner; verified
      in real Chrome (29.8 s recovery, one reconnect, clean teardown). Second, independent
      cause of the false "IDLE" copy: `GATE_QUIET_MS` parsed as `0` because
      `Number(null) === 0` when the query param was absent; now falls back to 90 000 ms.
      Fresh Watch loads now show LIVE · CONNECTED · STREAMING with no false idle.
- [x] **Owner-requested: smooth move animation** — LANDED (`0e38547`, deployed): pieces
      glide 210 ms on provable single-ply deltas (same piece id, origin matches lastUci,
      type unchanged); everything else snaps (coalesced frames, fake deltas, promotions,
      flips). Castling glides king+rook only on the exact castle pattern. Human's own
      Play-page moves snap (no artificial lag). `prefers-reduced-motion` respected.
      16/16 harness checks; mid-glide screenshot on record.
- [x] **Owner-requested: UI redesign** — LANDED (`c0d7e97`, deployed). Apple-grade minimal
      system (frosted nav, soft elevation, pill controls, generous whitespace) recolored to
      a classical chess identity — warm ivory/ebony surfaces, walnut/oxblood single accent,
      brass hairlines only. Three themes (Classical Light, Classical Dark, Refined Tarot)
      with per-theme board palettes, `<html data-theme>` hooks, localStorage persistence,
      and a footer radiogroup switcher (no floating pill). Verified by a 24-shot matrix
      (4 views × 3 themes × mobile/desktop at 390/1440 px): 24/24, zero overflow, zero
      console errors, switcher `static` and inside the footer everywhere; footer control
      measured at 390 px with no label clipping. `svelte-check` 0/0, build clean.
- [x] **Classical themes left the About card empty** — FIXED (`503a6fb`, deployed). Hiding
      the tarot scene under `classical-*` (correct: it is tarot artwork) left a 280 × 420 px
      bordered box with only "0" and "THE FOOL". Classical now renders an engraved piece
      medallion — brass outer ring, hairline inner ring, ♞ glyph, rule beneath — on the ivory
      or ebony card field; Refined Tarot keeps the full scene (`display: none` for the
      medallion). Verified 6/6 (2 sizes × 3 themes): medallion inside the card at 164 px
      desktop / 119 px mobile, zero overflow, zero console errors.
- [x] **Theme switcher keyboard semantics** — VERIFIED (Playwright, deployed bundle):
      exactly one tab stop (roving tabindex), ArrowLeft/Right/Up/Down cycle with wrap, Home
      and End jump to the ends, `aria-checked` and `<html data-theme>` move together, focus
      follows the selection, 2 px `:focus-visible` ring present on every step, and the choice
      persists after reload via `fsg:settings`.
- [x] **`/watch` said "IDLE" during a running gate** — FIXED (`b4a7f6f`, deployed). The
      gate hint required a journal write within 120 s, but a gate is *defined* by the journal
      going silent (40–80 min), so `gate_in_progress` was structurally false in exactly the
      window it existed for; a page opened mid-gate showed LIVE GAMES · IDLE while a title
      match ran. Now: `trainer_alive` (cached 30 s psutil probe for `scripts/train.py`,
      run-id scoped, never raises) + an existing arena row + >180 s journal silence, exposed
      as an additive `trainer_alive` key. Verified against the real process table
      (stale journal → true, streaming → false) and end-to-end on an isolated `:8001`
      harness: TITLE MATCH at 1440/390 px, zero overflow, zero console errors; live `:8000`
      during self-play still reads 11 IN PLAY. 11 new tests; full suite 490 passed.
- [ ] **Journal header says "LIVE FEED" during a 40–80 min gate** (open, low priority). The
      panel is connected and holds backlog, so the label is true about the socket but reads
      as "new entries are arriving" while the diary is silent by design. Proposal: add the
      newest entry's relative age to the footer (`N LINES · LAST 15M AGO · FOLLOW ON`) or
      switch the label to a neutral quiet state. Constraint found while scoping:
      `JournalLine.time` renders `clockTime(msg.ts)`, which is the *server push* time
      (backlog lines show load time), so the age must come from the line's own embedded JSON
      `ts` when present and fall back to the arrival time only for plain lines. Do not change
      the state machine without the same screenshot proof used above.

## Optimization bench log (standing review loop, converged → optimization phase)

2026-11-21 re-bench (`scripts/bench_train.py`, tiny config, 3080, torch 2.15.0+cu132) vs the
FS-4 acceptance record — same ballpark, no config change to adopt, p32 still the best point:

| Metric | Recorded (FS-4, `a5ba985`) | Re-bench | Δ |
|---|---|---|---|
| sims/sec p1 → p32 | 92 → 808 (×8.8) | 66.6 → 655.5 (×9.84) | −19% @p32 |
| games/hour p32 | 16.8k | 14.7k | −12% |
| NN batch-256 latency | 1.49 ms | 2.11 ms | +42% |
| journal streaming A/B @p16 | −0.9% (on faster) | +10.7% | noisier |

- [ ] **Next — re-bench on a verified-idle GPU.** Every number above is 10–25% below the FS-4
      record and the streaming A/B flipped sign (−0.9% → +10.7% @p16); the 2026-11-21 desktop
      session has browser/IDE GPU load, so this reads as contention, not regression. Pin a clean
      protocol before FS-10: close desktop apps (or bench headless), 3 runs, report median.
- [ ] **Next — wire `gambit_core` into the MCTS tree step (§5 M3, ≥3× e2e gate).** FS-7 proved
      9–11× on the isolated tree-step CPU slice but the hot path still uses python-chess; at
      p32 the CPU movegen side is the next throughput lever before the first real run.

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

- [ ] **policy_ce tracks time-since-restart, not (only) the learning rate
      (2026-11-25 overnight).** Observed twice at 1e-3 and once at 3e-4: CE sits
      at 0.65-0.80 for the first ~3 iterations after a trainer restart, then
      climbs over ~30 iterations to a plateau of ~2.1-2.2 while target entropy
      falls 2.65 -> 2.37. The LR cut helped the transient but not the plateau,
      so the plateau is most likely a **distribution shift** (early batches
      still contain backlog games with softer targets; steady-state batches are
      the model's own sharpened output) rather than divergence. Do NOT keep
      tuning hyper-parameters blind on this metric: the next step is a real
      strength measurement (scripts/bench_sf.py on two checkpoints ~50
      iterations apart) before changing anything else.

- [ ] **Gate is DISABLED on night-v2 (2026-11-25, unattended run).** Every gate
      attempt after the D: move failed: first ~2 h per gate (SPRT early-stop
      forced the sequential path, so `parallel=8` never applied), then stalling
      before the first `[GATE] progress` line and getting killed by the watchdog
      (each kill rolled back ~2 iterations). Since a 16-game gate cannot resolve
      55% anyway (~785 games needed per the research log), `markers_minutes` is
      now 1e6 = gate off, training uninterrupted, champion frozen at
      `ckpt-002016-0642`. **Re-enable** with a cheap parallel gate once someone
      can watch it: `arena=replace(..., games=32, parallel=8,
      sprt_early_stop=False)` and `markers_minutes=(90.0,)`. Suspect the stall
      is evaluator construction (8 torch/ONNX sessions) while the fleet holds
      the GPU — worth profiling `Arena.run` before trusting it again.

- [x] **Watchdog liveness was model-age based (2026-11-24, fixed).** A gate runs
      20–40 min without exporting ONNX, so "newest model > 25 min" restarted a
      healthy trainer mid-gate at 21:26 and rolled the run back three iterations
      (5568 -> 6144 discarded; durability checkpoint at 5568 limited the loss).
      Liveness is now "trainer burned > 2 CPU-seconds in 10 s", with a 30-min
      silent-log fallback — a gate is CPU-heavy, a hang is not.
- [x] **Fleet on stale reader code (2026-11-24).** Old workers parsed the new
      relative pointer (`model-N.onnx`) against the CWD, failed to load, kept an
      ancient in-memory model and tagged games with it; the staleness horizon
      then rejected those games (`0 games` iterations). Restarting the fleet onto
      the current code fixed it. Lesson: pointer-format changes need a fleet
      restart in the same window, not just a trainer restart.

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

## UI overhaul harvest (2026-11-22, `feature/ui-minimalist-overhaul` — 20-pass screenshot loop)

Defect ledger with full per-pass detail lives in `web-react/REVIEW_LOG.md` (passes 1–3+,
all read, fixes deployed to `web/dist` and verified on production each round).

- [ ] **Mobile nav strategy at ≤390px** — one cramped row (brand glyph + 5 routes + scheme
      toggle, ~12–13px type, sub-44px tap targets). Fits with zero overflow today; a
      menu/priority-nav is a feature decision, deliberately not made mid-overhaul.
- [ ] **Header vs page-container left edge at 390px** — hero/body content appears inset a
      few px differently than the header brand (needs a padding-token reconciliation).
- [ ] **Mantine `Badge`/micro-chips audit** — global `textTransform: none` now applied; any
      future chip that should be uppercase must be authored uppercase (mono mode chips are).
- [ ] **Watch: gate stat hint format** — currently `last result 0.500` (score); consider
      `0-16-0` W/D/L form to match the ledger rows.


## Findings 2026-11-23 (SF benchmark session)

| ID | Finding | Status |
|----|---------|--------|
| FB-1 | **Arena gate has an effective N of 2**: `play_match_game` is argmax-deterministic (no root noise, no opening variation) and colors alternate - a "16-game" match is the same two games repeated 8x each. The 0W/16D streaks and the 8W/8D promotion are structurally guaranteed; SPRT over duplicated games is meaningless. Fix: randomized (or book) openings per gate game, balanced across colors - makes SPRT legitimate and lets Elo move. | TODO (training-dev; pipeline change, takes effect next run) |
| FB-2 | **`champion.json` tag/path divergence**: the writer updates `checkpoint_path` on every save but `tag` only on promotion (currently tag `ckpt-001536-0768`, path -> `ckpt-001728-0864.pt`). The play server's "nn" opponent therefore plays the *newest* checkpoint, not the champion. Decide intended semantics; if "newest" is intentional, rename/document; if not, fix the writer. | TODO (training-dev) |
| FB-3 | **External SF axis landed**: `scripts/bench_sf.py` - champion vs server Stockfish presets (wandering/curious/enlightened), randomized seeded openings, argmax 128 sims (gate strength), results appended to `data/sf_bench.jsonl`. Baseline (ckpt-001536, 128 sims): 0-1-5 vs wandering, 0-1-11 vs curious, 0-2-10 vs enlightened (~4-8% score everywhere). Self-play ladder Elo (~814) is lineage-relative only; rerun the bench per promotion for the true strength curve. Bench openings are out-of-distribution for a start-position-only self-play run - consider a curated opening book for later runs. Wire into pipeline post-promotion when a restart is acceptable. | tool landed; pipeline wiring TODO |
