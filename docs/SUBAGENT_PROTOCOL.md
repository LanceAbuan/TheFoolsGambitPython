# Subagent Dispatch Protocol (Fool's Gambit)

Distilled from the v2 migration's incident log (2026-11-22). The orchestrator follows this for EVERY subagent dispatch. Deviations require a written reason in the session.

## 1. Worktree-per-agent (non-negotiable)
- Every agent works in its own git worktree, created BEFORE dispatch (`opencode.worktree_create`), path pasted into the prompt as the sole working directory.
- The main checkout belongs to whoever is doing last-mile glue (usually the orchestrator) or stays frozen. NEVER two agents in one tree.
- Agents `git checkout -B <their-branch> <base>` inside their worktree at start; push early, push often.

## 2. Git: orchestrator merges, agents push
- Agents commit + push their own branches. The orchestrator opens/merges PRs when acceptance evidence is attached (or lands git on the agent's behalf when its `git commit` is permission-blocked — known environment issue).
- `git branch --show-current` before every commit; stage explicit paths only; `opencode.json` is always foreign.
- When committing from a shared context, create the branch FIRST, then commit (a commit landed on local main once by skipping this).

## 3. Session-ID ledger (fill at dispatch time)
| Agent task | sessionID | Status |
|---|---|---|
| training-dev WP-A/B/C/E (round 1) | ses_df9fc0bbaffeH84qeTR01D1ORk | died: output limit |
| rust-dev WP-F round 1 | ses_df9fc0bbbffeEheWEldEe2BYo0 | died: output limit |
| training-dev WP-C/E round 2 | ses_df99f0627ffdvF9ai1g4RjW0Pz | ⚠ IDs were swapped — see §4 |
| rust-dev WP-F C1/C2/C4 round 2 | ses_df99f0627ffeM86OObO9TED2aa | ⚠ IDs were swapped — see §4 |
| training-dev WP-D | ses_df99f0627ffeM86OObO9TED2aa | died: output limit (work recovered) |
| rust-dev C2/C4 finish (worktree) | ses_df96bcc02ffesLB0uRBoqe30u2 | ✅ completed, merged #141 |
| training-dev WP-E | ses_df99f0627ffeM86OObO9TED2aa | ✅ completed, merged #142 |
| rust-dev WP-H H1–H3 | (fill at dispatch) | |
| training-dev WP-F C3/C5/C6 | (fill at dispatch) | |

## 4. Anti-misroute rule
Before EVERY follow-up to a running session: re-read the sessionID from the original dispatch result (or this ledger), never from memory. After dispatching, immediately append the returned ID to the ledger above.

## 5. Report limits (output-token deaths are real)
- Every prompt ends with: "Report concisely in bullets — files, acceptance evidence, deviations. No prose padding."
- Long design discussions belong in committed docs, not chat reports.

## 6. Permission-denial protocol
- Any tool denial twice in a row → STOP, report state immediately (staged-but-uncommitted is recoverable; the orchestrator lands git).
- Interactive re-approval does NOT reliably clear a stuck deny — a fresh session is the recovery.

## 7. Scope boundaries
- Never: kill/restart/slow a running training process; write `runs/<live-run>/`, `runs/scratch/`, `data/train_*`; touch `opencode.json`.
- GPU: short smoke tests only while training runs — training has priority.
- Last-mile glue (<1 hour, precisely specified) goes to the orchestrator, not a dispatch.

## 8. Verification gates
- Agent runs its own acceptance commands and pastes evidence.
- Orchestrator re-runs the relevant pytest subset after every merge before dispatching dependents.

| training-dev WP-F C3/C5/C6 (worktree wpf-wiring) | ses_df92fb5ccffegHVs32ZWJAQpcv | running |
| rust-dev WP-H H1-H3 (worktree wp-h) | ses_df92f6d6dffeL3f8vwVki1eEyw | running |
