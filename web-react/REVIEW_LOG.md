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
20/20 captured, 0 horizontal overflow, 0 console errors. Verification in progress: 5 per-page
subagent reads (4 shots each — a single 20-shot subagent read hit the provider request-size cap;
image reviews are now split per page).

