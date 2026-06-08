# DEV_POLICY.md — Autonomous Development Rules

This file defines how Arrodes (AI assistant) operates autonomously on this repository.

## Core Principles

1. **Safety first** — Never delete existing logic without a migration path. Use `trash` over `rm`.
2. **Test before shipping** — New features must have tests. Bug fixes must include a regression test.
3. **Small, atomic PRs** — One issue per branch. No mega-PRs.
4. **Branch naming** — `fix/<short-desc>`, `feat/<short-desc>`, `chore/<short-desc>`.
5. **Commit hygiene** — Clear messages, one logical change per commit.
6. **CI must pass** — No PR without green checks.

## Autonomous Workflow

1. **Audit** → Scan for TODOs, missing tests, linting issues, bugs, docs gaps, perf problems.
2. **Issue** → Create GitHub issues for discovered work. Label: `auto-generated`.
3. **Prioritize** → Pick the highest-value issue (bugs > tests > docs > chore).
4. **Branch** → Create feature branch from `main`.
5. **Implement** → Write code, add tests, verify locally.
6. **PR** → Push branch, open PR, link issue.
7. **Wait** → If Klein reviews and requests changes, implement them. If approved, merge.
8. **Repeat** → Return to step 1.

## What I Can Do Without Asking

- Close stale/duplicate issues
- Create new issues and PRs
- Branch, commit, and push to the repo
- Run tests, linters, and CI locally
- Update documentation
- Refactor code with equivalent behavior
- Add unit tests

## What I Must Ask First

- Delete files/modules
- Change public API endpoints
- Merge PRs to main (unless explicitly authorized)
- Push to Hugging Face with new checkpoints
- Change deployment configs (Vercel, Railway)
- Anything that affects live users

## Audit Profiles

Run these in rotation every scan cycle:

1. **Code Quality** — Linting, unused imports, dead code, TODO comments
2. **Test Coverage** — Find untested modules/functions
3. **Documentation** — Missing docstrings, outdated README sections
4. **Bug Hunting** — Type mismatches, edge cases, error handling gaps
5. **Performance** — N+1 queries, unoptimized loops, memory leaks
6. **Security** — Hardcoded secrets, exposed endpoints, dependency vulnerabilities

## Labels

Use these GitHub labels:
- `auto-generated` — Created by autonomous scan
- `bug` — Something is broken
- `enhancement` — Feature/improvement
- `test` — Missing or needed tests
- `docs` — Documentation update
- `chore` — Maintenance/cleanup
- `priority/high` — Blocking or critical
- `priority/medium` — Should do soon
- `priority/low` — Nice to have

## Audit Log (2026-06-08)

| # | Issue | Priority | Status |
|---|-------|----------|--------|
| [#87](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/87) | Add pytest suite for training modules | P0 | Open |
| [#86](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/86) | Replace 59 print() calls with logging | P1 | Open |
| [#88](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/88) | Split torch/web requirements | P1 | Open |
| [#91](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/91) | chess.js@0.13.x abandoned | P1 | Open |
| [#89](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/89) | Persistent eval cache unbounded | P2 | Open |
| [#90](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/90) | Duplicate PST tables JS/Python | P2 | Open |
| [#92](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/92) | Unused Stockfish instances at boot | P2 | Open |
| [#93](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/93) | Trainer requires 64 entries before training | P2 | Open |
| [#82](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/82) | UI Nits | P3 | Open |
| [#70](https://github.com/LanceAbuan/TheFoolsGambitPython/issues/70) | Best moves display | P3 | Open |
