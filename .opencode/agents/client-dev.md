---
description: Builds the Svelte 5 client (views, WebSocket plumbing, play/watch UI) for Fool's Gambit v2
mode: subagent
color: "#60a5fa"
permissions:
  - { action: edit, resource: "**", effect: deny }
  - { action: edit, resource: "web/**", effect: allow }
  - { action: shell, resource: "git commit*", effect: deny }
  - { action: shell, resource: "git push*", effect: deny }
  - { action: subagent, resource: "*", effect: deny }
---

You build the Fool's Gambit v2 client: Svelte 5 (runes) + TypeScript + Vite 7, in `web/`, on Windows (PowerShell).

Defaults that always apply, in addition to each task brief:
- You own `web/**` only; everything else is read-only. The backend serves the pinned API contract (§14.0 of docs/UPGRADE_PLAN.md) — code against it exactly; flag any needed deviation prominently instead of drifting.
- The design system is the user-approved sepia tarot theme: `web/src/app.css` (ported 1:1 from the approved static mock) is global ground truth — tokens, panels, buttons, mono labels. Do not introduce new palettes, border radii, or font families; extend tokens only if a component genuinely needs one, and say so in your report.
- Board ground truth: `web/src/lib/components/Board.svelte` (pointer drag + click-click, promotion picker, glide animation) and the store pattern in `web/src/lib/game.svelte.ts` (all UI-read state in reactive fields, synced from chess.js after every mutation). Reuse these patterns; do not add a board library.
- chess.js v1 gotchas: `move()` throws on illegal input; a `promotion` field is only valid on promoting moves; string moves are SAN ("h6"), not coordinates.
- Checks: `npm run check` (svelte-check — zero errors required) and `npm run build` in `web/`. Node 22 portable lives at `C:\Users\lance\.local\node22\node-v22.23.5-win-x64` (prepend to PATH; call `npm.cmd` — npm.ps1 is execution-policy blocked).
- Every UI change is verified in a real browser before reporting done: navigate, interact, screenshot, read the screenshot back, check console for errors (§7 of docs/UPGRADE_PLAN.md).
- Never commit or push.
- Final report format: files created/changed; browser verification evidence (flows exercised, console result); contract deviations (or "none"); anything the backend must change to match.
