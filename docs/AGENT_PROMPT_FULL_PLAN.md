# Agent prompt — execute the full implementation plan (shutdown → build → bring-up → research)

Paste this entire document as the first message to a coding agent session in
this repository.

---

You are the lead engineer for Fool's Gambit (a from-scratch AlphaZero-style
chess engine). You will execute a complete, staged operation: shut everything
down cleanly, implement the ENTIRE improvement plan, bring the full system back
up, then resume autonomous research. Work methodically; every phase has a
definition of done. Do not skip acceptance checks.

## ENVIRONMENT (Windows / PowerShell)

- Repo: `E:\github repos\TheFoolsGambitPython` (git repo; remote `LanceAbuan/fools-gambit`).
- Python: `.venv\Scripts\python.exe` from the repo root with `$env:PYTHONPATH = "E:\github repos\TheFoolsGambitPython"`.
- `uv`: `C:\Users\lance\.local\bin\uv.exe`. Node 22: prepend `C:\Users\lance\.local\node22\node-v22.23.5-win-x64` to `$env:Path`; use `npx.cmd` / `npm.cmd`.
- GitHub CLI: prepend `C:\Program Files\GitHub CLI` to `$env:Path`. HERE-STRINGS ARE BLOCKED — write PR bodies to a temp file and use `--body-file`.
- `gh` is authed as `LanceAbuan`; you may merge your own PRs once acceptance is green. NEVER push to `main` directly.
- `opencode.json` is the user's local config — NEVER stage or commit it. `data/` and `runs/` are gitignored operating state.
- Read `docs/SUBAGENT_PROTOCOL.md` before dispatching any subagents (worktree-per-agent, ID ledger, concise reports, permission-denial fallback).
- Machine notes: powercfg sleep is disabled on AC. GPU is an RTX 3080. ORT dylib lives at `.venv\Lib\site-packages\onnxruntime\capi\onnxruntime.dll` (set `ORT_DYLIB_PATH` for workers). 16 logical cores / 8 physical.

## PHASE 0 — SHUTDOWN (do this first, carefully)

1. Inventory what is running:
   `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py|rust_worker|replay_latest|uvicorn' } | Select-Object ProcessId, CommandLine`
   Expected: trainer (2 procs), rust worker fleet (fleet supervisor + workers), live-board feeder, uvicorn server on :8000.
2. Record the live state before stopping: newest file in `runs\scratch\second-light-v2\checkpoints\`, the last line of `data\train_log.jsonl`, and `runs\scratch\second-light-v2\state\` (iteration/step).
3. Stop at a safe point: wait for the current iteration to end if it is within ~10 minutes (`data\train.err.log` newest `[CKPT]` line), otherwise stop immediately — checkpoints are atomic and the resume state is iteration-granular.
4. `taskkill /F /PID <each>` for: trainer launcher, trainer worker, the whole rust worker fleet, the feeder, and the uvicorn process. Verify zero matches remain.
5. Back up: `Copy-Item -Recurse runs\scratch\second-light-v2 runs\backup-v2-<yyyyMMdd-HHmm>` (also keep the existing `runs\scratch-backup-*`). Verify the backup contains checkpoints/state/spool.
6. Do NOT delete: `runs\`, `data\` (except rotating logs you yourself write), backups, `runs\v2-warmstart\`.

Definition of done: no train/worker/feeder/server processes; backup verified; latest checkpoint tag + iteration recorded.

## PHASE 1 — EXECUTE THE FULL PLAN

Read `docs/IMPLEMENTATION_PLAN.md` end to end. It contains a verified audit
(§0), invariants (§2), 12 work packages with code-level designs and acceptance
criteria (§4), a restart matrix (§5), measurement protocol (§6), and risks
(§7). Implement ALL of it, in wave order:

- W1: WP1 (forced playouts + policy target pruning; also audit/record current cpuct, Dirichlet, virtual-loss values), WP2 (playout cap randomization p=0.25/fast=100, value-only fast rows), WP3 (capped-game final-Q value targets, soft resignation, deblunder pass).
- W2: WP4 (EMA weights; export/gate/bench the EMA; gate promotes on max(ema, current) early), WP5 (health metrics: policy entropy, P UL, value accuracy, held-out train/test gap + /watch card).
- W3: WP6 (two-stage gate: screen 64 @ ≥0.52 with Wilson LB > 0.5; pentanomial counts in the ledger; keep rd-cap logic).
- W4: WP7 (ORT intra_op=1, allow_spinning=0, workers = physical cores; fleet matrix bench).
- W5: WP8 (Syzygy 3-4-5 WDL: ≤5-man target override + perfect-label sampling + endgame harness), WP9 (Lichess puzzle benchmark, ~500–1000 puzzles, move-by-move scoring, record baseline).
- W6: WP10 (optimizer package: AdamW wd=1e-4 fused, 500-step warmup, cosine/step schedule, grad-clip 1.0; A/B in temp runs `runs/ab-*` — never the live run), WP11 (next-gen bundle: encoder v3 = v2 prefix + halfmove clock + material planes + masks; SE blocks; moves-left head; attention flag; 5% history masking; warm-start v2→v3 with strict=False + KL proof; new profile `night-v3`).
- WP12 (infra): subtree reuse in rust selfplay (~60% retention), spool v2 (Arrow IPC memory-mapped + sparse top-60 policies, schema v3), shared-inference evaluation, Gumbel A/B — implement behind flags, defaulting to current behavior until their acceptance numbers beat it.

DISCIPLINE (non-negotiable):
- One branch per WP (`feat/wpN-...`), conventional commits, PR to main, merge only when the WP's acceptance criteria pass. Run the full `pytest tests\` suite before merging each WP and `cargo test` (+ `--features onnx`) for Rust changes.
- Feature flags default OFF; old paths must remain bit-identical. Every schema change bumps the schema version string.
- Fix what breaks from the plan's own divergence — but log any deviation in the PR body and in `docs/IMPLEMENTATION_PLAN.md` §8 checkboxes.
- Keep reports concise (bullets). Redirect noisy command output to files and tail it.
- If a permission denial sticks twice, use the shell-write fallback from the subagent protocol; if it persists, report and continue with other WPs.

Definition of done: all 12 WPs merged (or explicitly deferred with a written reason in the PR/plan), full test suite green, flags documented, checklist in §8 updated.

## PHASE 2 — BRING EVERYTHING BACK UP

1. Resolve the warm-start source honestly: prefer the newest promoted champion checkpoint from the v2 lineage (cross-check `runs\scratch\second-light-v2\checkpoints\` newest by mtime and any `champion.json` under `runs\`); if WP11 landed, run `scripts/warmstart_v3.py`-equivalent to produce `runs\v3-warmstart\ckpt-v3-0000.pt` and start profile `night-v3`; otherwise resume `second-light-v2` with `--resume`.
2. Start training detached (survives session exit):
   `Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "scripts/train.py","--profile","night-v3"(or night-v2 --resume second-light-v2),... -WorkingDirectory <repo> -RedirectStandardOutput "data\train.out.log" -RedirectStandardError "data\train.err.log" -WindowStyle Hidden`
3. Start the worker fleet with `$env:ORT_DYLIB_PATH` set and the WP7 threading config; models dir `<run>\onnx`, spool dir `<run>\spool`.
4. Start the live-board feeder: `scripts\replay_latest_game.py --pacing 2 --run-id <run>`.
5. Start the web server: `.venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000` (firewall rule "FoolsGambit 8000" already exists; LAN UI at `http://<LAN-IP>:8000`).
6. Verify ALL of: trainer 2 procs; workers running with no restart-loop; spool `inbox\processed` growing; `data\train_log.jsonl` fresh (<1 min); `GET /api/health` = 200; `/watch` = 200; live board streaming; one full iteration completes (collect→train→export→gate) with the new features active in the logs; GPU utilization sane.

Definition of done: a full iteration + gate completes end-to-end on the new code with all features observable in logs, and the phone UI works on the LAN.

## PHASE 3 — RESUME RESEARCH

1. Run one research cycle now: read `docs/RESEARCH_TRAINING.md` (distilled section + next-cycle topics), pick 1–2 un-researched topics, websearch/webfetch, append a new `## Cycle N` section (≤25 lines, newest-first), commit via branch+PR+merge.
2. Re-arm the automations targeting YOUR OWN session (get the id via shell `$env:OPENCODE_SESSION_ID`):
   - **research-loop** — every 30 minutes, prompt = the research-cycle prompt (read RESEARCH_TRAINING.md → research top queued topics → append cycle ≤25 lines → commit/PR/merge → ≤3 bullets; research + doc edits only; never touch the live run).
   - **research-consolidation** — every 6 hours, prompt = fold cycles into the distillate, keep only newest 2 cycles verbatim, prune/re-prioritize the queue against live training state (read `data\train.err.log` tail + `data\arena.jsonl`), keep file ≤160 lines, commit/PR/merge.
   Use `opencode.automation_create` with `action.type="prompt"`, `to={session:<your session id>}`.

Definition of done: one cycle committed; both automations active with next-run times confirmed.

## HARD RULES (repeat)

- Never stage/commit `opencode.json`. Never push to main. Never delete `runs\`, `data\`, or backups.
- Never leave the trainer/workers/feeder/server down at the end of the operation.
- Don't quote SPRT-stopped numbers as magnitudes — fixed-N only (plan §6).
- Prefer bundled restarts: WP11 is the one generation boundary; don't drip restarts.
- Record every deviation in the PR body; keep the plan's checklist current.

## FINAL REPORT

Reply with: what was shut down (PIDs + checkpoint), per-WP status table (PR links, acceptance numbers), bring-up evidence (iteration+gate completed, service checks), research cycle summary, automation IDs, and anything deferred with reasons.

---
END OF PROMPT
