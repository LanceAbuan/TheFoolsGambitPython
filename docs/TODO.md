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

- [ ] **WP-A — Backend**: FastAPI + WebSocket + SQLite. Serves: live board streaming, training telemetry
      (ELO, loss, win-rate), training journal feed, game persistence. Swap the UI's mock data slots
      (Watch metrics, journal, ELO rating `—`) to real feeds. Self-play watcher process.
- [ ] **WP-C — Training**: self-play pipeline on the RTX 3080 (torch 2.15 cu132 pinned in pyproject),
      checkpoints to HF Hub, resumable runs, metrics emitted to backend/WS.
- [ ] **WP-D — Rust `gambit_core`**: fast movegen + perft harness via PyO3; replaces chess.js/chess
      hot paths for self-play speed.
- [ ] **Deploy/hosting**: host the site so the user + friends can play (replace dead Vercel flow);
      reverse proxy + backend service; decide box (local machine vs VPS).

## Client polish (post-port)

- [ ] Difficulty select → wire to real engine strength once backend exists (currently cosmetic).
- [ ] Opponent card ELO `—` → real number from backend.
- [ ] Consider unicode-glyph pieces option (mock look) vs Cburnett SVGs — user's call.
- [ ] Keyboard accessibility pass on board (arrow/enter movement), focus states review.
- [ ] Watch page: auto-follow live game when telemetry connects.

## Hygiene

- [ ] Remove `web/dist/` + stray build artifacts from working tree before commit (gitignored already).
- [ ] Delete temp mock-server + extracted zip from `%TEMP%\opencode\fool-frontend` after mock retired.
- [ ] svelte-check clean run (`npm run check`) — add to CI once green.
- [ ] Rename session/docs references to old purple theme → sepia tarot (screenshots in plan doc).
