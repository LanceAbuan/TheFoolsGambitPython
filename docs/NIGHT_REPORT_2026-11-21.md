# Night report — 2026-11-21 (`second-light`)

Snapshot written while the run is still live. The run is the deliverable; this file is the
readable summary. Everything below is measured from `data/train_log.jsonl`,
`data/arena.jsonl`, SQLite, and `/api/metrics` — no projections.

**Live now:** training run `second-light`, iteration 9 self-play in flight after the 20:32
gate. Training PIDs 27076/13764, server PID 30104. Champion: **`ckpt-000128-0064`**
(regime Elo 813.38).

---

## 1. TL;DR

- The bot trained all evening on the RTX 3080, non-stop, no restart: **9 iterations,
  432 games, 9 rated gates**.
- One promotion (13:55) and **seven title defences**. The 19:45 gate was the first
  *decisive* result (`0W 8D 8L`, score 0.250) — the champion held, and the rating series
  finally moved: **793.68 → 801.88** overall, **813.38** for the champion's lineage.
- Three production defects were found and fixed with proof: the **WS deaf window**
  (server fan-out + client watchdog), a **false `IDLE`** copy from `Number(null) === 0`, and
  the **empty About card** under the classical themes.
- The owner-requested redesign shipped: three themes, footer switcher, Apple-grade spacing
  recolored to ivory/ebony/walnut/brass. Verified 24/24 screenshots + 6/6 About shots.
- Two honest misses are on record: **N1** (Rust gate 1.1× vs ≥3×) and **N3**
  (49 min/iteration vs <12 min). Neither is a regression; both are attribution-documented.

## 2. The run

| | |
|---|---|
| Run id | `second-light` (regime `scratch`, profile `night`) |
| Started | ~13:05 local (first iteration logged 13:06:37) |
| Iterations | 9 (0–8), 32 games each, 16 in parallel |
| Search | PUCT @ 400 sims + playout cap, fastpath ON, Gumbel OFF |
| Model | 10 blocks × 128 filters, value 32/512 |
| Games | 432 in the metrics ledger |
| Champion | `ckpt-000128-0064`, regime Elo **813.38** |
| Overall Elo | 800.00 → **801.88** (with a dip to 792.0 at promotion) |
| GPU | ~39 % util, 3.4 / 10.2 GB — starved by the CPU-side tree, not by batch size |

### Iterations

| iter | finished | loss | policy | value | buffer | games/h |
|---|---|---|---|---|---|---|
| 0 | 13:06 | 5.6804 | 5.5441 | 0.1363 | 1 316 | 77.7 |
| 1 | 13:42 | 3.9281 | 3.8360 | 0.0921 | 2 754 | 70.3 |
| 2 | 14:23 | 3.7684 | 3.6942 | 0.0742 | 4 244 | 68.2 |
| 3 | 15:06 | 3.6312 | 3.5211 | 0.1101 | 5 783 | 70.3 |
| 4 | 16:03 | 3.4397 | 3.3244 | 0.1153 | 7 086 | 76.7 |
| 5 | 16:50 | 3.3043 | 3.1896 | 0.1147 | 8 550 | 70.0 |
| 6 | 17:41 | 3.2932 | 3.1848 | 0.1084 | 10 056 | 66.7 |
| 7 | 19:04 | 3.2231 | 3.1110 | 0.1121 | 11 557 | 51.5 |
| 8 | 20:15 | 3.2052 | 3.0868 | 0.1184 | 13 093 | 64.8 |

Policy loss fell 5.54 → 3.09 (−44 %) and the value head stabilised around 0.11. The loss
curve is flattening — expected at this scale, and the reason the gate, not the loss, decides
what ships.

### Gates (candidate vs champion)

| # | finished | candidate | W-D-L | score | result |
|---|---|---|---|---|---|
| 9 | 13:14 | `ckpt-000064-0032` | 0-16-0 | 0.500 | held |
| 10 | 13:55 | `ckpt-000128-0064` | 8-8-0 | 0.750 | **PROMOTED** |
| 11 | 14:39 | `ckpt-000192-0096` | 0-16-0 | 0.500 | held |
| 12 | 15:38 | `ckpt-000256-0128` | 0-16-0 | 0.500 | held |
| 13 | 16:23 | `ckpt-000320-0160` | 0-16-0 | 0.500 | held |
| 14 | 17:12 | `ckpt-000384-0192` | 0-16-0 | 0.500 | held |
| 15 | 18:27 | `ckpt-000448-0224` | 0-16-0 | 0.500 | held |
| 16 | 19:45 | `ckpt-000512-0256` | 0-8-8 | 0.250 | held |
| 17 | 20:32 | `ckpt-000576-0288` | 0-16-0 | 0.500 | held |

Gates 1–8 in the ledger predate the F-1 fix and are self-vs-self fossils (2 games each,
0-2-0); they are kept visible on Watch under the SHOW ALL footnote rather than deleted.

**The interesting one is gate 16.** After five straight 16-draw defences, the newest
candidate lost eight games and drew eight. The champion is genuinely stronger, and the
rating series responded for the first time since promotion. This is the signal the evening
needed: the ladder is not frozen, it is just hard to beat.

### Cost model

Time between gate results (self-play + train + gate): 40.3, 44.5, 58.5, 45.3, 49.0, 74.6,
78.7, 46.4 min — mean ~56. The 74.6/78.7 pair stretched because games now routinely reach
the 200-ply cap; the gate alone is roughly 20–45 min of that. Pre-registered response
**P-U9-4** (32 games × 100 plies, `promote_rd_cap` 150 → 100) is queued for the next run,
not applied mid-run.

## 3. Research predictions, scored honestly

From `docs/RESEARCH_STANDARD.md` (never edited after the fact):

| id | prediction | verdict |
|---|---|---|
| N1 | Rust `gambit_core` yields ≥3× e2e sims/s | **MISS — 1.1×**. Movegen/encoding is only ~4.5 % of GIL demand; ~82 % of per-sim cost is in `_simulate`. Bit-exactness held (perft 4.87 M, 5 990-position fuzz), so it lands as a free 1.1× + the foundation for a search-in-Rust slice. |
| N2 | Gumbel ≥4× games/h with h2h ≥0.35 | **Bench hit (6.14×), strength unproven** — the h2h was draw-saturated (0.500, zero decisive games). Gated OFF for this run; the gate-grade A/B is still queued. |
| N3 | <12 min/iteration | **MISS — ~49 min measured, mean cycle ~56**. Attribution: game lengths doubled as blunders vanished (35–173 → 127–200 plies, most at the cap), and the <12 min path assumed Gumbel. Levers: U6 shards, P-U9-4 gate re-sizing. |
| N4 | ≥1 live event/s end-to-end | **PASS — ~2.4 events/s sustained** (33 226 live lines over ~230 min of self-play). Gap found: gates are not broadcast per-ply, which is what produced the false "IDLE" state — fixed tonight. |

## 4. What broke, and what fixed it

### 4.1 Server crash loop (found 18:0x, fixed 18:49)

`TypeError: float() argument ... not 'NoneType'` inside `elo_by_regime` on every metrics
tick. Honest `elo: null` rows (F-5) were flowing into a `float()` coercion. Caught per-key,
so headline metrics stayed up while the regime map silently degraded to `{}`. Fixed in
`b39c722`; deployed 18:49; zero tracebacks since (server log is now just WS connection
lines from probes).

### 4.2 The WS deaf window (fixed `2aedcc7` + `c0d7e97`)

Symptom: a fresh `/watch` load showed LIVE · CONNECTED but received no training frames for
minutes while the journal streamed.

- **Server half** — the hub had one shared send path, so a single ping-answering,
  non-reading consumer parked the whole fan-out. Reproduced: 11 → 1.8 → 0 frames/s with one
  parked reader, and a fresh mid-wedge reader got **zero** live frames. `server/ws.py` now
  gives every subscriber a bounded queue (512) and its own pump; overflow drops only the
  slow consumer and closes its transport with code **1013** (bounded 5 s) so the client
  reconnects instead of hanging. Ledger replay + journal scans moved to `asyncio.to_thread`.
  7 regression tests in `tests/test_ws_hub.py`.
- **Client half** — `web/src/lib/api/ws.ts` now has a watchdog: 10 s tick, `train` topic,
  25 s stale threshold, 15 s refocus threshold, 2 s half-open grace, and a single reconnect
  owner. Verified in real Chrome: 29.8 s recovery, exactly one reconnect, clean teardown.

### 4.3 False `IDLE` (fixed `c0d7e97`)

`GATE_QUIET_MS` was parsed from a missing query param and `Number(null) === 0`, so the
"quiet" threshold was zero: any gap at all read as idle. It now falls back to 90 000 ms and
the journal state machine has honest states: LIVE FEED / BUFFERED / STREAMING / CONNECTING /
LISTENING / IDLE.

### 4.4 Empty About card under the classical themes (fixed `503a6fb`)

Hiding the tarot scene (correct — it is tarot artwork) left a 280 × 420 px bordered box with
only "0" and "THE FOOL". Classical now renders an engraved piece medallion: brass outer
ring, hairline inner ring, ♞, rule beneath. 164 px desktop / 119 px mobile, inside the card,
zero overflow, zero console errors; Refined Tarot keeps the full scene.

### 4.5 `gate_in_progress` was structurally false — FIXED (`b4a7f6f`)

A `/watch` page opened mid-gate showed the LIVE GAMES chip as **IDLE** while a gate was
actually running. Cause: the server's gate hint required a journal write within 120 s, but
the journal is *silent by definition* during a gate (gates are not broadcast move by move),
and journal writes are 40–80 min apart. The hint was therefore false in exactly the window
it exists for.

Fixed by reading the fact instead of inferring it: **is a `scripts/train.py` process
alive?** (`psutil`, run-id scoped, cached 30 s). The rule is now *trainer alive + an arena
row exists + the journal has been silent > 180 s*. An additive `trainer_alive` key is
exposed, and a trainer-owned key still wins via `setdefault`. The client's quiet-window
inference remains as a second line of defence.

Proof, all on the real process table:

- stale-journal fixture (600 s quiet) → `trainer_alive: true`, `gate_in_progress: true`
- fresh self-play journal → `trainer_alive: true`, `gate_in_progress: false`
- isolated `:8001` harness (stale fixture + real ledger) → `/watch` renders
  **LIVE GAMES · TITLE MATCH** at 1440 px and 390 px, zero overflow, zero console errors,
  with the honest copy "the run's last written journal entry was 7m ago"
- live `:8000` during self-play → **11 IN PLAY**, no title-match claim
- 11 new tests (fake process seams, no dependence on local processes); full suite 490 passed

## 5. The redesign (owner request)

Direction: "supermodern, super clean, maybe like Apple but chess-like or classical."
Delivered as an Apple-grade *design system* recolored to a classical chess identity, not an
Apple palette:

- ivory/ebony surfaces, walnut/oxblood single accent, brass hairlines only
- frosted nav, soft elevation, pill controls, generous whitespace, system sans + Cormorant
  display for headings
- three themes, switchable in the footer (radiogroup, no floating pill, persists via
  `fsg:settings`):
  - **Classical Light** — ivory paper, walnut accent
  - **Classical Dark** — ebony, brass, ember accent
  - **Refined Tarot** — the original sepia/gold identity (still the default)
- per-theme board palettes, `prefers-reduced-motion` respected, disabled states legible
- 210 ms piece glide only on provable single-ply deltas; everything else snaps

Evidence: 24/24 screenshots (home, play, watch, about × 3 themes × 390/1440 px) with zero
overflow and zero console errors, theme switcher measured `static` inside the footer on
every shot, no label clipping at 390 px; About medallion 6/6; `svelte-check` 0 errors /
0 warnings; vite build green.

**Needs your call:** which theme becomes the default. `Refined Tarot` is the safe default
until you pick; flipping it is a one-line change.

## 6. Repo / PR state

- Branch `feature/play-and-local-overhaul` → PR **#130** (description refreshed tonight):
  https://github.com/LanceAbuan/fools-gambit/pull/130
- Tonight's commits: `2aedcc7` (WS hub), `c0d7e97` (themes + watchdog + IDLE),
  `503a6fb` (About medallion), `b4a7f6f` (gate hint from process liveness), plus
  docs/ledger updates.
- Verification: **131 backend/WS-scoped tests pass** (490 in the full suite), `ruff` clean,
  no console errors on any page, deployed bundle verified on :8000 (3 cold loads →
  LIVE · CONNECTED · LIVE FEED).
- Not touched (yours): Cloudflare tunnel config, HF upload credentials, PR merge.

## 7. Open items for the morning

1. **Theme default** — pick one; I flip the default and re-run the matrix.
2. **P-U9-4 gate re-sizing** — 32 games × 100 plies, `promote_rd_cap` 100. Activates on the
   next run; would cut the dominant per-hour cost roughly in half.
3. **Cap adjudication** — a 200-ply auto-draw teaches "shuffle = draw". Needs an
   adjudication rule before the next long run.
4. **U6 multiprocess shards** (3–4×) and the **Gumbel A/B at gate quality** — both need an
   idle GPU, so they are next-run work.
5. **Search-in-Rust slice** — N1 proved movegen is not the bottleneck; `_simulate` is. This
   is the only path to the ≥3× the Rust gate wanted.
6. **Tunnel + hosting** — waiting on your config; the app is local-first and ready.

## 8. How to check it yourself

- Site: http://127.0.0.1:8000 — `/`, `/play`, `/watch`, `/about`; theme switcher is in the
  footer.
- Health: `curl http://127.0.0.1:8000/api/health`
- Metrics: `curl http://127.0.0.1:8000/api/metrics`
- Ledger: `data/arena.jsonl`; journal: `data/train_log.jsonl`; trainer stderr:
  `data/train.err.log`
- Tests: `& .venv\Scripts\python.exe -m pytest tests -q` (490) or
  `-k "api or ws or server"` (131)
