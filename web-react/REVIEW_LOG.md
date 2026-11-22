# UI Review Log — minimalist overhaul

Method: every pass captures 20 full-page screenshots (5 routes × {1440×900, 390×844} × {light, dark})
on the **production build** served at `http://127.0.0.1:8000` (FastAPI serving `web/dist`),
then **every image is read** and defects are logged here with the fix. Shot rig:
`%TEMP%\opencode\ui_review\capture.py <base_url> <out_dir> <tag>`; shots in `%TEMP%\ui_shots\passNN\`.
Design spec: one emerald accent, neutral grays, type scale 12/14/16/20/28 at weights 400/600,
4/8px grid, flat 1px-border cards, no gradients/glows/ornament glyphs/tarot metaphors,
sentence case, tabular numerals for stats, ruthless subtraction.

## Pass 1 — baseline (tag `p01`, 2026-11-22)

20/20 captured, 0 horizontal overflow, 0 console errors. Verdict: functional but over-decorated
and internally inconsistent — multiple competing label styles, off-scale type, a hand-drawn tarot
centrepiece, and one real layout bug.

### Global
- **G1** `theme.ts`: h1 renders up to 68px at weight 650 (2.4× the 28px scale cap); weights 500/650/700 in use across pages. → type scale 12/14/16/20/28, weights 400/600 only.
- **G2** `styles.css` `.app-glow`: radial gradient light-source behind the Home hero — pure decoration. → removed.
- **G3** `.eyebrow` letter-spaced uppercase micro-label used on Home?/Play/Watch/Analysis/About headers — five different pages, one noisy pattern. → class deleted, all usages removed.
- **G4** Card system: theme default radius `lg` + shadow `sm`; pages explicitly mix padding `lg`/`md`/`0`/`{responsive}`. → flat cards, 1px border, radius `md`, padding 20 everywhere, no shadow.
- **G5** font-size one-offs everywhere: 9, 9.5, 10, 10.5, 11, 11.5, 12.5, 13, 17, 26, 30px. → consolidated onto the scale.
- **G6** colored left-borders: `.status-cell` 3px emerald border on About status cards. → removed with the About rewrite.

### Home (desktop + mobile)
- **H1** glow behind hero (G2); hero padding-top 80px pushes stats below the fold at 1440×900.
- **H2** stat cards: uppercase letter-spaced 12px labels + 30px/650 values — same "stat" role as Watch's 26px/650 with 10px labels (two treatments for one pattern). → one Stat pattern: 14/400 label, 28/600 tabular value, 12/400 hint.
- **H3** "TRAINING LIVE" badge: uppercase, size lg. → sentence case, sm.
- **H4** destination card copy longer than needed ("all server-side", "29-plane encoder…" belongs on About).

### Play
- **P1** **`.play-layout` has NO CSS anywhere in the project** (only `.board-layout` exists) → the intended two-column board+rail grid never applies; the board card renders ~1100px wide and the rail stacks under it at desktop. → switched Play.tsx to the existing `.board-layout`.
- **P2** decorative copy: "The game of chance" eyebrow, "Your journey" card label, "Begin a new journey" button, "Summoning The Fool…" pending state, "PAST STEPS" micro-label, "the fates are balanced" ×2, `✦` glyphs in checkmate/draw statuses, italic Lao-tzu quote footer. → all replaced with plain labels ("Start game", "Starting…", "Checkmate — you win.", etc.).
- **P3** "EST. ELO" 9px letter-spaced micro-label beside a 30px/650 Elo number. → 12px "Elo" label, 20px/600 value.
- **P4** board-card header: 700-weight uppercase "YOUR JOURNEY" + pulsing dot that duplicates the pulsing status line under the board. → removed label and dot; header is mode badge + keyboard hint only.

### Analysis
- **A1** "How the Fool sees it" eyebrow; "Evaluation" uppercase 700 card label; status word rendered uppercase ("ANALYZING… / UP TO DATE / QUEUED / PLACE BOTH KINGS"). → sentence case.
- **A2** turn badge text passed through `.toUpperCase()` ("WHITE TO MOVE"). → sentence case.
- **A3** "WHITE POV" 10px uppercase label repeated on both engine cards while the card footer already explains White-POV once. → removed from cards.
- **A4** eval score 30px/650; engine names 12.5px/650. → 28/600 and 14/600.
- **A5** Board.css: two-layer 32px board shadow (glow); keyboard-cursor squares stroked with a **gold** `--gold-bright` — a second accent color. → single-layer hairline shadow; cursor → emerald primary.
- **A6** engine-unavailable/waiting texts at 12.5/11.5px. → 12px.

### Watch
- **W1** **Gate stat defect (functional)**: hint reads "no gate recorded" while the arena ledger below lists matches. Root cause probed via API: `GET /api/games?kind=arena` rows carry `{ckpt, opponent, games, wins, draws, elo, glicko_r, glicko_rd, ts}` — **no `losses` field** — and `gateScore` requires `losses !== undefined`, so it is always `null`. → derive `losses = games − wins − draws` in one helper used by the stat, the ledger rows, and the gate-empty text.
- **W2** ArenaRow prints `{losses ?? 0}L`, so every row renders "0W 16D 0L" regardless of real losses. → same helper fix.
- **W3** the top status strip card (Run/Elo/Gate/Arena games) duplicates the "By the numbers" card (self-play/win-rate/loss/gate) on the same screen. → merged into one 8-stat grid beside the chart; top strip removed.
- **W4** uppercase micro-labels: "SAMPLE GAME", "REPLAY · ILLUSTRATIVE", "FEATURED GAME", "RECENT MOVES", "LIVE FEED", "TITLE MATCH", "UPDATED …", "AWAITING TELEMETRY", "N MATCHES / UNAVAILABLE / LOADING". → sentence case, no letter-spacing.
- **W5** Watch `Stat`: 10px/650 uppercase labels + 26px/650 values vs Home's 30px/650 + 12px labels. → shared pattern (G4-scale).
- **W6** centered "UPDATED X MIN AGO" footer line duplicates the chart card's "updated" stamp. → removed.
- **W7** journal level badges 9.5px/650 letter-spaced. → 10px mono/600, no letter-spacing.

### About
- **B1** CSS-drawn tarot medallion (bordered card, sun `✹`, moon `☾`, mountains clip-path, traveler `♟`, star row `✦ ✧ ✦`, gold palette `--gold*`) is the page's centrepiece — decoration, not information. → page rewritten without it.
- **B2** story-card icons are ornament glyphs `✧ ☼ ☾`; card labels are roman-numeral eyebrows ("I · The journey", "II · The mind", "III · The unknown"); header eyebrow "The card without a number"; chapter mark "0 · THE BEGINNING". → gone.
- **B3** ~600 words of tarot copy; the training-loop explanation is a single 140-word paragraph. → cut to ~120 words total across short sections + a 4-step loop list.
- **B4** `.about-title` 40px/500, `.about-copy` 17px — off-scale. → scale only.
- **B5** `.text-link` custom underline+offset link with arrow span. → normal Mantine anchor.

### Dark theme (pass 1, desktop + mobile) — read via subagent
- **D1** Home: green gradient wash behind hero (same as H1); hero H1 dimmer than the stat values below it (inverted emphasis). [both fixed with H1/H2]
- **D2** Footer: right-side "local-first · RTX 3080" silently missing at 390px (visibleFrom); both sides very dim. [fixed: always visible, xs, sentence case]
- **D3** Play: tarot/journey copy everywhere ("The game of chance", "Your journey", "Begin a new journey", "PAST STEPS", "Your story has yet to unfold." in MoveList, Lao-tzu quote, ✦ glyphs). [fixed incl. MoveList → "No moves yet."]
- **D4** Play: disabled "Undo" renders as bare borderless text while siblings are bordered buttons. [fixed: disabled default-buttons keep filled chrome, dimmed text]
- **D5** Play: missing-Elo placeholder renders as a stray floating em-dash line above "EST. ELO". [fixed: "No rating yet" text]
- **D6** Analysis: page header/badge/footer align to the 1180px container but the board grid is centered narrower — header and cards don't share edges. [mitigated: grid stretched to container]
- **D7** Analysis: FEN input truncates mid-token with no ellipsis (desktop and worse at 390px). [fixed: textOverflow ellipsis]
- **D8** Analysis: Stockfish top-move rows jam move+score ("e4+0.33") vs The Fool's separated format; "-0.00" negative zero shown. [fixed: min-width right-aligned meta; qText clamps to "0.00"]
- **D9** Analysis: icon-only flip button undecipherable. [fixed: labeled "Flip" button]
- **D10** Watch: chart has a green gradient area-fill; featured-game and journal cards stretch into ~300–400px empty voids; the two rows use different column splits (gutter jumps 757→703); identical ledger bars encode nothing (correct data — all-draw matches). [fill removed; align-items start + shorter feeds; gutters unified]
- **D11** Watch: lone centered "⏱ UPDATED 21 MIN AGO" strip floats above the footer. [removed]
- **D12** About: full amber tarot medallion off-palette (gold vs site green); hero centers content at 390px while everything else left-aligns; "Elo 810" here vs 801 elsewhere (bestKnownElo vs metrics.elo); letter-spaced links; stray rule from link arrow to card edge. [page rewritten; Elo now from metrics.elo; links are plain anchors]
- **D13** Nav at 390px: brand + five routes + toggle in one cramped row (~13px type). [mitigated by compact paddings; menu/overflow is a known follow-up, logged in docs/TODO]

### Tooltip defect (user-reported, reproduced pass 2)
- **T1** Mantine 9 inverts tooltips in dark scheme: near-white `gray-2` background, black text, **black border** — on dark pages it reads as a broken white slab (probed live: bg `rgb(233,236,239)`, color `#000`, border `#000`). [fixed in `theme.ts`: one dark tooltip style for both schemes — `dark-7` fill, `gray-2` text, `dark-5` hairline border, arrow inherits; re-probed: bg `rgb(36,36,36)`, text `#e9ecef`, border `#3b3b3b` in light AND dark]

**Pass 1 disposition:** G1–G6, H1–H4, P1–P4, A1–A6, W1–W7, B1–B5, D1–D13, T1 fixed across commits
`style: consolidate design tokens…`, `feat(home): …`, `fix(play): …`, `refactor(about): …`,
`fix(watch): …`, `style(analysis): …`.

## Pass 2 (tag `p02`, 2026-11-22, production build after fix round 1)
20/20 captured, 0 horizontal overflow, 0 console errors. Read via 5 per-page subagent reads
(a single 20-shot subagent read hit the provider request-size cap; image reviews are split per page).

### Verified landed (spot list)
Play two-column layout; About rewrite (zero tarot/ornaments/left-borders); Watch merged stats with
real Gate hint and derived losses ("0W 16D 0L" correct); no chart gradient fill; shared gutters;
no bottom UPDATED strip; FEN ellipsis; no "−0.00"; labeled Flip; no WHITE POV labels; Home flat hero;
footer both texts at 390px.

### Pass-2 defects found
- **P2-1** Uppercase badges everywhere ("TRAINING LIVE", "WHITE TO MOVE", "HELD", "REPLAY · ILLUSTRATIVE", "DEPTH 16", "LIVE") — **Mantine's Badge CSS applies `text-transform: uppercase` by default**, so sentence-case strings rendered uppercase. [theme: Badge `textTransform: 'none'`; intentionally-typed uppercase mono chips like "DEMO · PRACTICE" stay as authored]
- **P2-2** Theme toggle shows a moon icon while the page renders dark — `colorScheme === 'dark'` is false for `'auto'`. [ColorSchemeToggle now uses `useComputedColorScheme`]
- **P2-3** Disabled "Undo" still borderless — Mantine's disabled style sets `border-color: transparent`, my earlier rule didn't override it. [rule now sets `border-color: var(--mantine-color-default-border)`; verified live: bg `rgb(46,46,46)`, border `rgb(66,66,66)`, dimmed text]
- **P2-4** Footer floats mid-page on short desktop pages (~125–190px dead background below it). [AppShell main is now a flex column with `min-height: 100dvh`, routes grow, footer pinned]
- **P2-5** Elo inconsistency: About showed 810 (`bestKnownElo` regime-map peak) vs 801 elsewhere; Play used the same. [About + Play now show the champion's current `metrics.elo`]
- **P2-6** Dark secondary text at the readability floor (footer right text, status labels, blurbs, ledger timestamps ~3.5–4:1). [dark `--mantine-color-dimmed` raised one step to gray-5]
- **P2-7** Board coordinates illegible, worst on dark squares (~8px effective, 0.72 opacity). [coord text 12px/600, fills raised to 0.9/0.95 with brighter tokens]
- **P2-8** Dark board squares nearly uniform (#2f4a3d vs #556459) — black pieces survived only via outlines. [dark palette lightened: `#40584b` / `#5b6a5e`]
- **P2-9** Analysis: board grid inset ~52px vs page header on desktop (grid was centered with a fixed 440px rail). [grid stretches to the container; rail `minmax(300px, 1fr)`]
- **P2-10** Analysis: Stockfish move rows un-jammed but scores sat next to the move, not right-aligned — Mantine `Button justify` doesn't reach the label wrapper. [MoveButton wraps both spans in a `top-move-inner` flex with `space-between`]
- **P2-11** Analysis: green Fool arrow (#2f9e6e) camouflaged against green squares — single-stroke arrows, no casing. [Board now draws white-cased two-layer arrows; brand color kept]
- **P2-12** Watch: journal card retained a ~220px internal void (fixed 240px scroll area with 1 line). [feed height auto, max-height 240]
- **P2-13** Watch: live-games card left-column void. [sample replay board reduced 340→260/220]
- **P2-14** Watch: W/D/L bar renders 100% gray on all-draw matches — degenerate, carries nothing the adjacent mono score doesn't. [WdlBar removed — subtraction]
- **P2-15** Watch: Elo chart micro-labels ~5px effective at 390px (560-unit viewBox scales text down). [tick/x 12px, last 14px, best 12px]
- **P2-16 (accepted)** Row-1 card bottoms unequal on desktop — cards hug content by design (`align-items: start`); stretching would re-introduce internal voids. Data-honest heights.
- **P2-17 (accepted)** Chart champion ring sits on the champion checkpoint's 810 point while current Elo reads 801 — the ring marks the checkpoint, caption is accurate; Elo drifts between gauntlets.
- **P2-18 (accepted)** Progress-bar fill differences between engines (+0.01 → ~58% vs +0.20 → ~52%) are engine-provided `expected_white` values, not a mapping bug. Verified mapping is direct ×100%.
- **P2-19 (logged, deferred)** 390px nav is one tight row (fits, no overflow); a menu/priority-nav is a feature decision — logged to docs/TODO, not silently changed.

**Pass-2 disposition:** P2-1…P2-15 fixed and deployed (round 2); P2-16…P2-19 accepted/deferred with rationale.

## Pass 3 (tag `p03`, 2026-11-22, production build after fix round 2)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page subagent reads (the analysis one died on
request size once — retried in a fresh session).

### Verified landed
Play: Undo chrome, coords legible, squares two-tone, grid flush both edges. Watch: journal hugs,
ledger mono-only, chart legible at 390px, shared gutters, footer flush on watch. About: Elo 801,
dark labels readable. Home/Analysis: sentence-case badges, resolved toggle icon, no gradient.

### Pass-3 defects found
- **P3-1** Footer pin CSS shipped **without its `routes-grow` class attached to the content
  wrapper** — dead background still below the footer on short desktop pages (home/about).
  [Box wrapper added in App.tsx; verified structurally in pass 4]
- **P3-2** Analysis: move rows still centered — `width: 100%` on the inner span resolved against
  Mantine's content-sized label wrapper. [`.mantine-Button-label` forced block/100%; verified live:
  san left 446, meta right 943 within row 437–952]
- **P3-3** Primary-button label contrast in dark (~2.5–3:1 white on `emerald-4`). [`primaryShade`
  7/7 → white on `#1f7250` ≈ 5.3:1 both themes]
- **P3-4** Watch: dead region right of the featured board (info column top-aligned vs tall board).
  [`align-items: center` on `.live-feature`]
- **P3-5** Home hero orphan ("public." alone on line 2 at 390px). [`text-wrap: balance`]
- **P3-6** Watch: under-board coordinate strip near-invisible in dark. [opacity 0.62 → 0.8]
- **P3-7 (user report)** Featured live game **jumped between parallel games and vanished** —
  `slots[0]` re-sorts by last update, and ended games drop after a 4 s TTL. [featured game is now
  pinned until it actually leaves the feed, then re-pins; thumbs exclude the featured game]
- **P3-8 (user request)** No watchable "main game". [pinned game + live **Fool eval bar** under the
  board: debounced `POST /api/analyze` (32 visits, abortable, paused when the game ends), shared
  `ExpectedBar`/`evalScoreText` component now also used by Analysis]

**Pass-3 disposition:** P3-1…P3-8 fixed and deployed (rounds 3–4); verification in pass 4.

## Pass 4 (tag `p04`, 2026-11-22, production build after fix rounds 3–4)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page subagent reads with pixel-sampled
WCAG math. **The live Fool eval bar and the pinned featured game both verified working on the
real feed** ("Fool eval · white −0.08" → 49% bar; pin advanced only after the game ended).

### Verified landed
Footer pinned in both themes on all pages; deep-green primary labels 5.86:1; disabled Undo chrome;
nav pills 9.98–15.85:1; move rows flush both edges (5× crops); cased arrows; H1 balanced at 390px;
About Elo 801; watch featured/info column centered; journal hugs; ledger mono-only.

### Pass-4 defects found
- **P4-1** Latest move rendered twice on Watch (meta-row badge + last "Recent moves" chip; absent
  entirely on desktop-dark — inconsistent). [duplicate badge removed — subtraction]
- **P4-2** Board coordinate ink ~2.1:1 on cream squares in LIGHT theme. [light fills darkened to
  `#1d2f24`/`#14231b` at 0.95]
- **P4-3 (systemic)** Muted secondary text below AA everywhere: `#868E96` light (3.2–3.3:1),
  `#828282` dark (3.5–4.0:1). First fix attempt ([data-mantine-color-scheme] overrides) **lost the
  cascade to Mantine's runtime-injected vars sheet** — root-caused via live computed-style probe.
  [:root-scoped overrides win; verified live: dark dimmed now `#adb5bd` (~8:1), light `#495057` (~7:1)]
- **P4-4 (accepted)** Dark-theme black pieces read via white outlines; squares now two-tone —
  asset-level improvement possible later (custom piece set), logged to TODO if it recurs.
- **P4-5 (watch, pass 5)** Mini-board caption possibly clipped ("e4t") at 390px — low-confidence
  read; re-check in pass 5 zoom.

**Pass-4 disposition:** P4-1…P4-3 fixed and deployed (round 5); P4-4 accepted; P4-5 re-checked next pass.

## Pass 5 (tag `p05`, 2026-11-22, production build after fix round 5)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page subagent reads (pixel-sampled where noted).

### Result: clean sweep — zero defects across all 20 shots
- Dimmed-token fix verified with sampled ratios: light `#495057` on white = **8.18:1**; dark `#ADB5BD`
  on card `#2E2E2E` = **6.54:1** — AA cleared everywhere the token is used.
- Play: light-theme cream-square coordinate ink fixed (near-black-green, legible at 1x); no regressions.
- Watch: latest move renders once; **Fool eval bar live with numeric scores** ("white +0.16" mobile,
  pending-state on one desktop capture — timing, not layout); featured game pinned (Game 532→539
  advance only on game end); mini-board captions zoomed 5× — pass-4 "e4t" was valid UCI `h8f7`.
- About: Elo 801 intact; hierarchy intact after token change.
- Analysis: clean; engine-unavailable state text not exercised this pass (both engines responded) —
  deferred to a later pass.

**Pass-5 disposition:** no fixes required. Loop continues (passes 6–20) as verification cycles; any
regression restarts the fix→re-capture cadence. PR evidence set: pass01 (before) vs pass05 (after).

## Pass 6 (tag `p06`, 2026-11-22, same bundle as pass 5 — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Eval bar confirmed with numeric scores on two more live games (+0.20, −0.57, +0.05); duplicate-move
fix holds; pin, footer, tokens, coords, rows all stable.

## Pass 7 (tag `p07`, 2026-11-22, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Eval bar present in all states across 4 watch shots (pending + scored); latest-move single-render
holds (spot-checked 5 featured boards); Elo 801, footer pin, tokens, rows stable.

## Pass 8 (tag `p08`, 2026-11-22, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Gate/sample panel verified this round (title match state, replay controls, disclaimers) alongside
the usual checklist. Functional prod probe re-run after this pass: `rect.light` fill correct,
1.e4 → engine replied, analysis evals (+0.00/+0.36) + 2 arrows, 0 console errors.

## Pass 9 (tag `p09`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Gate/sample panel verified again (caption, badges, centered board); journal wrap at 390px clean;
all tokens, rows, pins stable. Fifth consecutive clean pass (5–9).

## Pass 10 (tag `p10`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Flagged item investigated and resolved: About/Home "Elo 793" (was 801) is **live data drift, not a
UI defect** — gate 22 concluded and `ckpt-001536-0768` dethroned `ckpt-000128-0064` (8W/8D/0L,
Elo 792.9) — the first promotion after 22 successful defences; UI renders the new champion
faithfully across Home, About, and the ledger. Gate/sample panel verified on all 4 watch shots.

## Pass 11 (tag `p11`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
New champion (`ckpt-001536-0768`, Elo 793) renders correctly on Home/About/ledger. Seventh
consecutive clean pass (5–11).

## Pass 12 (tag `p12` + re-shoot `p12b`, 2026-11-23, same bundle — stability cycle)
20/20 + 8 re-shots captured, 0 overflow, 0 errors. Read via 5 per-page sweeps + 1 targeted
re-verification. **Clean — no regressions.**
Flagged item investigated and resolved: pass-12 home/about dark-mobile shots showed "Elo 801 /
1200 games" while the other six showed "793 / 1152". Root cause: **gate 23 concluded mid-capture**
(`ckpt-001600-0800` lost its challenge; metrics file rewritten: elo 793→801.25, games 1152→1200).
Re-shoot `p12b` confirms all 8 shots agree with `/api/metrics` (Elo 801, 1200 games, champion
ckpt-001536-0768). App renders a live changing feed faithfully — no caching bug, no UI defect.
Also: live eval bar verified mid-self-play ("white +0.97"), latest move single-rendered.

## Pass 13 (tag `p13`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Live data consistent across all shots (Elo 801 / 1200 games / ckpt-001536-0768); eval bar with
numeric scores on 3 of 4 watch shots (−0.42, +0.08, +0.12). Ninth consecutive clean pass (5–13).

## Pass 14 (tag `p14`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Elo chart annotations (816/792) legible at 390px; eval bar scored (+0.01, +0.83) and pending states
both verified. Tenth consecutive clean pass (5–14).

## Pass 15 (tag `p15`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Eval bar scored on all 4 watch shots (+0.48, −0.17, +0.02, +0.14); live data uniform across all
home/about shots. Eleventh consecutive clean pass (5–15).

## Pass 16 (tag `p16`, 2026-11-23, same bundle — stability cycle)
20/20 captured, 0 overflow, 0 errors. Read via 5 per-page sweeps. **Clean — no regressions.**
Eval bar scored on all 4 watch shots (−0.20, +0.04, −0.27, −0.04); live data uniform. Twelfth
consecutive clean pass (5–16).


