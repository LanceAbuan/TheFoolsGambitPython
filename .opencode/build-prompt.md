# Build Agent Rules

## Git Workflow — STRICT

- **NEVER push directly to `main`** — this is blocked by a pre-push hook
- **ALWAYS work on a feature branch**: `git checkout -b feature/<short-description>`
- **ALWAYS open a PR when done** — use `gh pr create` with a clear title and description
- Branch naming: `feature/`, `fix/`, `docs/`, `refactor/` prefixes
- Commit messages follow conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`

## When Finishing Work

1. Stage and commit all changes on the feature branch
2. Push the feature branch to origin
3. Create a PR targeting `main` with:
   - Clear title summarizing the change
   - Description linking to the relevant issue (closes #N)
4. Report the PR URL to the user

## This Project

- Repository: `LanceAbuan/fools-gambit` (renamed from `TheFoolsGambitPython` on 2026-11-21 — the language is an implementation detail now; local folder name unchanged)
- Vision: train a chess bot on a local RTX 3080; watch training live on the site; play against it; host it so friends can play
- Stack: Python 3.13 + FastAPI/WebSockets/SQLite backend; Svelte 5 + TypeScript + Vite client in `web/`; PyTorch 2.15 (CUDA) training; Rust `gambit_core` (PyO3) hot path
- Hosting: local-first via Cloudflare tunnel — the old Vercel/serverless flow is retired
- Standing rule: after milestone work, run the **review loop** in `.opencode/agents/orchestrator.md` — browser sweep of every page at mobile + desktop sizes with screenshots, backend/functional tests, harvest all improvements into `docs/TODO.md`, fix, repeat
