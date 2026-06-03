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

- Repository: `LanceAbuan/TheFoolsGambitPython`
- Vision: Host a custom-trained chess bot with a website where people can play against it and watch training live-streams
- Current stack: Python (AI/engine), Node.js API (Vercel serverless), vanilla HTML/CSS/JS frontend with chessground
- Domain: `gambit.lanceabuan.tech` (Vercel)
