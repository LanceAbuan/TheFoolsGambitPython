<script lang="ts">
  import { Chess } from 'chess.js';
  import Board from '../lib/components/Board.svelte';
  import Tip from '../lib/components/Tip.svelte';
  import {
    metrics as fetchMetrics,
    games as fetchGames,
    type ArenaResult,
    type MetricsSnapshot,
  } from '../lib/api/http';
  import { socket, type ServerMessage, type SocketStatus } from '../lib/api/ws';

  interface Props {
    /** Wired by App.svelte so the primer can send newcomers to Play/About. */
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();

  // FS-8b: the "What am I looking at?" primer — collapsed by default, one click
  // to expand. No persistence: every visit starts collapsed so the page stays calm.
  let primerOpen = $state(false);

  // A real miniature game, replayed frame by frame through the playback controls.
  // Each frame carries the played move's from/to so the board can glide it
  // (stepping BACKWARDS still snaps: the reverse delta never matches the
  // forward lastMove, and the honesty guard prefers a snap over a fake glide).
  const line = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'Nf6', 'O-O', 'Be7', 'd3', 'd6', 'c3', 'a6'];
  const frames: { fen: string; label: string; lm: { from: string; to: string } | null }[] = (() => {
    const c = new Chess();
    const out: { fen: string; label: string; lm: { from: string; to: string } | null }[] = [
      { fen: c.fen(), label: 'START', lm: null },
    ];
    line.forEach((san, i) => {
      const mv = c.move(san);
      out.push({
        fen: c.fen(),
        label: `${Math.floor(i / 2) + 1}. ${i % 2 === 0 ? san : out.at(-1)!.label.split(' ').slice(1).join(' ') + ' ' + san}`,
        lm: { from: mv.from, to: mv.to },
      });
    });
    return out;
  })();
  // Cleaner labels: build "1. e4", "1… e5", "2. Nf3" style.
  for (let i = 1; i < frames.length; i++) {
    const moveNo = Math.floor((i - 1) / 2) + 1;
    frames[i].label = i % 2 === 1 ? `${moveNo}. ${line[i - 1]}` : `${moveNo}… ${line[i - 1]}`;
  }

  let frame = $state(0);
  let playing = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  const current = $derived(frames[frame]);
  const score = $derived(['+0.18', '+0.18', '+0.21', '+0.09', '+0.31', '+0.12', '+0.34', '+0.22', '+0.28', '+0.19', '+0.24', '+0.15', '+0.21'][frame] ?? '+0.18');
  const depth = $derived(String(6 + (frame % 5)).padStart(2, '0') + ' PLY');

  function toggle(): void {
    if (playing) {
      stop();
    } else {
      playing = true;
      timer = setInterval(() => {
        if (frame >= frames.length - 1) {
          frame = 0;
        } else {
          frame += 1;
        }
      }, 1100);
    }
  }
  function stop(): void {
    playing = false;
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }
  function step(delta: number): void {
    stop();
    frame = Math.min(frames.length - 1, Math.max(0, frame + delta));
  }

  // ---- live telemetry: WS `train` topic (§14.0) with REST one-shot fallback ----
  let conn = $state<SocketStatus>('connecting');
  let liveMetrics = $state<MetricsSnapshot | null>(null);

  function clockTime(ts: string): string {
    const d = new Date(ts);
    return Number.isNaN(d.getTime())
      ? new Date().toLocaleTimeString('en-GB', { hour12: false })
      : d.toLocaleTimeString('en-GB', { hour12: false });
  }

  // ---- training journal (FS-6/FS-8): a scrolling mono feed of `log` frames.
  // The frame shape is deliberately parsed defensively: the contract pins
  // {level, message}, FS-6 ships {line}, and a bare string is accepted too.
  interface JournalLine {
    id: string;
    time: string;
    level: string;
    text: string;
  }
  const JOURNAL_CAP = 200;
  let journal = $state<JournalLine[]>([]);
  let journalSeq = 0;
  let feedEl: HTMLDivElement | undefined = $state();
  let follow = $state(true);
  /** Arrival times of recent `live` frames (capped) — the journal panel's pulse. */
  let liveArrivals = $state<number[]>([]);

  function logLineFrom(msg: ServerMessage): { text: string; level: string } | null {
    const p: unknown = msg.payload;
    if (typeof p === 'string' && p.length > 0) return { text: p, level: 'log' };
    if (p && typeof p === 'object') {
      const o = p as Record<string, unknown>;
      const text = [o.line, o.message, o.msg, o.text].find(
        (v): v is string => typeof v === 'string' && v.length > 0,
      );
      if (text) return { text, level: typeof o.level === 'string' ? o.level.toUpperCase() : 'LOG' };
    }
    return null;
  }

  function pushJournal(msg: ServerMessage): void {
    const parsed = logLineFrom(msg);
    if (!parsed) return;
    journalSeq += 1;
    const entry: JournalLine = {
      id: `${typeof msg.seq === 'number' ? msg.seq : 'x'}-${journalSeq}`,
      time: clockTime(msg.ts),
      level: parsed.level,
      text: parsed.text,
    };
    // Oldest → newest reading order; hard cap so a chatty run cannot grow the DOM forever.
    journal = [...journal, entry].slice(-JOURNAL_CAP);
  }

  function onFeedScroll(): void {
    if (!feedEl) return;
    follow = feedEl.scrollHeight - feedEl.scrollTop - feedEl.clientHeight < 40;
  }
  // Auto-follow: pin to the newest line unless the reader scrolled up to browse.
  $effect(() => {
    const count = journal.length;
    if (follow && feedEl && count > 0) feedEl.scrollTop = feedEl.scrollHeight;
  });

  // ---- live games grid (FS-6c): `live` frames on the train topic -------------
  // Contract (server/telemetry.py + FS-6c correction): payload {game, ply,
  // move(uci), fen, run_id, ts} = position AFTER the move; newest per game;
  // NEVER replayed on subscribe (every join is mid-stream). `game` ids are
  // RUN-UNIQUE (iteration i owns [i*16, (i+1)*16); ids never repeat within a
  // run), so cells are addressed by `game % gridSize` and refill only when a
  // NEW game id maps to them; newest-fen-wins per CELL resolves any overlap.
  // game_end marks the occupying cell for ~5s, then its game entry is dropped.
  const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const END_TTL_MS = 5000;
  const MOVE_LIST_CAP = 160;
  const MAX_STORE = 64; // memory safety net; TTL cleanup keeps this far below in practice

  interface LiveMove {
    ply: number;
    uci: string;
    san: string | null;
  }
  interface LiveSlot {
    game: number;
    fen: string;
    ply: number;
    lastUci: string | null;
    lastSan: string | null;
    runId: string | null;
    updated: number; // Date.now() of last frame — per-cell newest-fen-wins
    ended: { result: string } | null;
    moves: LiveMove[];
  }

  // Plain record (not a Map): Svelte 5 deep-proxy tracks property writes
  // reliably; Map iteration in $derived proved not to invalidate here.
  let liveSlots = $state<Record<number, LiveSlot>>({});
  let sawLive = $state(false);
  let featured = $state<number | null>(null);
  let gridSize = $state(16); // desktop 4×4; shrinks on narrow viewports
  const endTimers = new Map<number, ReturnType<typeof setTimeout>>();

  // ---- gate-aware panel states (three-state LIVE GAMES) ----------------------
  // Arena gates (title matches) stream NO per-ply frames, so while a gate runs
  // the grid legitimately goes quiet — which must not read as "training runs
  // offline". Everything here is client-side from RECEIVED frames plus an
  // ARRIVAL clock (Date.now() at receive time; frame ts is never trusted for
  // this clock): any live/game_end/iteration frame proves a run exists this
  // session, and >90s without a `live` frame while connected is gate quiet.
  const GATE_QUIET_MS = (() => {
    // Dev/test override: ?gateQuietMs=<0..600000>. Production pages never pass it.
    // NOTE: an ABSENT param must fall back to 90s — Number(null) is 0, which would
    // silently satisfy the range check and collapse the quiet window to zero.
    const raw = new URLSearchParams(location.search).get('gateQuietMs');
    if (raw === null) return 90_000;
    const v = Number(raw);
    return Number.isFinite(v) && v >= 0 && v <= 600_000 ? v : 90_000;
  })();
  let sawRunFrame = $state(false); // a run announced itself this session
  let lastLiveAt = $state(0); // arrival of the last `live` frame
  let lastRunAt = $state(0); // arrival of the last run frame of any kind
  let lastRunKind = $state<'live' | 'game_end' | 'iteration' | null>(null);
  let lastIter = $state<{ n: number; elo: number | null } | null>(null);

  function markRunFrame(kind: 'live' | 'game_end' | 'iteration'): void {
    sawRunFrame = true;
    lastRunKind = kind;
    lastRunAt = Date.now();
    if (kind === 'live') lastLiveAt = lastRunAt;
  }

  /**
   * `log` frames carry the raw journal JSON as `message` — the backend routes
   * only live/game_end to dedicated frames (server/journal.py classify_line),
   * so kind:"iteration" lines arrive here. They are the gate state's
   * breadcrumb: the iteration number + elo the panel may quote while quiet.
   *
   * History guard: the subscribe backlog replays the last ≤50 journal lines
   * (server/telemetry.py docstring) — including lines from a run that is
   * already over. A line older than STALE_RUN_S at ARRIVAL is backlog replay,
   * not a heartbeat, and must not arm the title-match state. (The 90s quiet
   * clock itself still runs on arrival times only; this check merely dates
   * the line's own content, which is exactly what the backlog falsifies.)
   */
  const STALE_RUN_S = 300;
  function noteLogFrame(p: Record<string, unknown>): void {
    const text = typeof p.message === 'string' ? p.message : '';
    if (!text.startsWith('{')) return;
    try {
      const o = JSON.parse(text) as Record<string, unknown>;
      if (o.kind !== 'iteration') return;
      // Post-deploy servers mark backlog replay authoritatively; the ts check
      // below remains only for pre-deploy servers that do not send the flag.
      if (o.replayed === true) return;
      if (typeof o.ts === 'number' && Date.now() / 1000 - o.ts > STALE_RUN_S) return;
      const n =
        typeof o.iteration === 'number' && Number.isFinite(o.iteration) ? Math.round(o.iteration) : null;
      if (n === null) return;
      lastIter = { n, elo: typeof o.elo === 'number' && Number.isFinite(o.elo) ? o.elo : null };
      markRunFrame('iteration');
    } catch {
      /* not JSON — an ordinary journal line */
    }
  }

  $effect(() => {
    const onResize = (): void => {
      gridSize = window.innerWidth >= 900 ? 16 : window.innerWidth >= 560 ? 9 : 6;
    };
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  });

  /** UCI → SAN via the position the move was played from. The first move seen
   *  for a late-joined game has no known pre-move position — it stays UCI. */
  function sanFor(prevFen: string | null, ply: number, uci: string): string | null {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promo = uci.slice(4, 5);
    if (from.length !== 2 || to.length !== 2) return null;
    const base = prevFen ?? (ply <= 1 ? START_FEN : null);
    if (!base) return null;
    try {
      const c = new Chess(base);
      const m = c.move({ from, to, ...(promo ? { promotion: promo } : {}) });
      return m.san;
    } catch {
      return null; // defensive: a bad fen/uci pair degrades to UCI display
    }
  }

  function onLiveFrame(p: Record<string, unknown>): void {
    const game = p.game;
    const fen = p.fen;
    if (typeof game !== 'number' || !Number.isInteger(game) || typeof fen !== 'string' || fen.length === 0) return;
    const ply = typeof p.ply === 'number' ? p.ply : 0;
    const uci = typeof p.move === 'string' ? p.move : '';
    const prev = liveSlots[game];
    if (prev && prev.fen === fen && prev.ply === ply) return; // duplicate frame — skip re-render
    const san = sanFor(prev ? prev.fen : null, ply, uci);
    liveSlots[game] = {
      game,
      fen,
      ply,
      lastUci: uci || null,
      lastSan: san,
      runId: typeof p.run_id === 'string' && p.run_id ? p.run_id : (prev?.runId ?? null),
      updated: Date.now(),
      ended: null,
      moves: [...(prev?.moves ?? []), { ply, uci, san }].slice(-MOVE_LIST_CAP),
    };
    sawLive = true;
    markRunFrame('live');
    // Journal heartbeat (production reality): `log` frames only carry
    // iteration/arena/aggregate lines — roughly one per ~45 min of self-play —
    // so the written diary is quiet for long stretches while practice moves
    // stream. The journal panel treats these arrivals as its pulse.
    liveArrivals = [...liveArrivals, Date.now()].slice(-240);
    // Memory safety net: drop the stalest game when the store grows past MAX_STORE.
    const ids = Object.keys(liveSlots).map(Number);
    if (ids.length > MAX_STORE) {
      let stalestId = ids[0];
      for (const id of ids) {
        if (liveSlots[id].updated < liveSlots[stalestId].updated) stalestId = id;
      }
      delete liveSlots[stalestId];
      if (featured === stalestId) featured = null;
    }
  }

  function onGameEnd(p: Record<string, unknown>): void {
    // FS-6d: every terminal frame feeds the finishes ticker — even for a game
    // whose live frames we never saw (the end event itself is authoritative).
    pushFinish(p);
    markRunFrame('game_end'); // a game_end proves the run exists even when the slot is unknown
    const game = p.game;
    if (typeof game !== 'number' || !Number.isInteger(game)) return;
    const slot = liveSlots[game];
    if (!slot) return; // never saw this game — nothing to mark
    const result = typeof p.result === 'string' && p.result ? p.result : '—';
    slot.ended = { result };
    const prior = endTimers.get(game);
    if (prior) clearTimeout(prior);
    endTimers.set(
      game,
      setTimeout(() => {
        endTimers.delete(game);
        // ids are run-unique: the entry is simply dropped; its cell refills
        // only when a NEW game id maps to it via modulo.
        if (liveSlots[game]) {
          delete liveSlots[game];
          if (featured === game) featured = null;
        }
      }, END_TTL_MS),
    );
  }

  /** The server never replays live frames: every (re)connect is a fresh join. */
  function resetLiveSlots(): void {
    liveSlots = {};
    sawLive = false;
    featured = null;
    liveArrivals = []; // fresh join: the pulse re-anchors (no replay exists)
    for (const t of endTimers.values()) clearTimeout(t);
    endTimers.clear();
    // A fresh join re-anchors the quiet clocks: the stream has GATE_QUIET_MS to
    // prove itself again before the title-match state may speak (frames are
    // never replayed, so pre-disconnect arrivals must not count for this join).
    lastLiveAt = Date.now();
    lastRunAt = Date.now();
    // NOTE: the finishes ticker is deliberately NOT reset here. Its entries are
    // real received game_end frames with their own arrival time; they age out
    // on their own TTL, and a reconnect does not make them unhappened.
  }

  // ---- recent-finishes ticker (FS-6d) ----------------------------------------
  // CRITICAL HONESTY: `game_end` frames are TRAINING SELF-PLAY — practice reps
  // The Fool plays against itself while it learns. They are NOT arena ledger
  // rows: the ledger (separate section below) records only champion title
  // matches, the gauntlet a new version must win to take over. The label and
  // tip keep that separation explicit; nothing here may borrow ledger wording.
  const FINISHES_CAP = 5; // newest five, newest first
  const FINISHES_TTL_MS = 180_000; // pruned three minutes after arrival
  const FINISHES_FADE_MS = 40_000; // fade begins for the final 40s

  interface FinishEntry {
    game: number;
    result: string;
    plies: number;
    at: number; // Date.now() when the frame arrived (drives relative time)
  }

  let recentEnds = $state<FinishEntry[]>([]);
  let nowTick = $state(Date.now());

  function pushFinish(p: Record<string, unknown>): void {
    const game = p.game;
    if (typeof game !== 'number' || !Number.isInteger(game)) return;
    const entry: FinishEntry = {
      game,
      result: typeof p.result === 'string' && p.result.length > 0 ? p.result : '—',
      plies: typeof p.plies === 'number' && Number.isFinite(p.plies) ? Math.max(0, Math.round(p.plies)) : 0,
      at: Date.now(),
    };
    // Duplicate-frame guard: the same game ending twice within seconds renders once.
    if (recentEnds.some((e) => e.game === game && entry.at - e.at < 5000)) return;
    recentEnds = [entry, ...recentEnds].slice(0, FINISHES_CAP);
  }

  /** Relative arrival time, refreshed by the one-second ticker heartbeat. */
  function relAge(at: number): string {
    const s = Math.max(0, Math.floor((nowTick - at) / 1000));
    if (s < 10) return 'just now';
    if (s < 60) return `${s}s ago`;
    return `${Math.floor(s / 60)}m ago`;
  }
  // Server-declared journal age (additive metrics key, post-deploy): used in the
  // gate line when the gate was armed by the metrics flag and no frame timing
  // exists to quote. Hours-scale ages read better in h than 300m.
  const journalAgeS = $derived.by(() => {
    const ts = liveMetrics?.last_journal_ts;
    return typeof ts === 'number' ? Math.max(0, nowTick / 1000 - ts) : null;
  });
  const journalAgeTxt = $derived.by(() => {
    const s = journalAgeS;
    if (s === null) return '';
    if (s < 10) return 'just now';
    if (s < 60) return `${Math.floor(s)}s ago`;
    if (s < 5400) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  });

  // Heartbeat while anything time-based is on screen: refreshes the finishes'
  // relative times, flags the fade window, prunes expired entries, and drives
  // the gate-quiet clock. Stops itself only when none of that can matter.
  $effect(() => {
    if (recentEnds.length === 0 && !sawRunFrame) return;
    const t = setInterval(() => {
      nowTick = Date.now();
      if (recentEnds.some((e) => nowTick - e.at >= FINISHES_TTL_MS)) {
        recentEnds = recentEnds.filter((e) => nowTick - e.at < FINISHES_TTL_MS);
      }
    }, 1000);
    return () => clearInterval(t);
  });

  const TIP_FINISHES =
    'These are practice games — The Fool playing reps against itself while it trains, thousands of which are thrown away after the lesson. They are not the arena ledger: a title match is the gauntlet a newly trained version must win to take over, and only those matches are recorded there.';

  function toggleFeatured(game: number): void {
    featured = featured === game ? null : game;
  }

  /** Escape anywhere collapses the featured view (Tip's own Escape stops propagation). */
  function onGlobalKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape' && featured !== null) featured = null;
  }

  function slotLabel(s: LiveSlot): string {
    if (s.ended) return `game ${s.game} finished, result ${s.ended.result}`;
    return `game ${s.game}, move ${s.ply}, position after ${s.lastSan ?? s.lastUci ?? 'start'}`;
  }

  /** UCI → the {from,to} shape Board's last-move highlight expects. */
  function fromTo(uci: string | null): { from: string; to: string } | null {
    if (!uci || uci.length < 4) return null;
    return { from: uci.slice(0, 2), to: uci.slice(2, 4) };
  }

  // Cell model (FS-6c correction): cell c is occupied by the game with the
  // most recent frame among those with `game % gridSize === c` — newest-fen-
  // wins per CELL resolves any overlap between iteration blocks. Cells render
  // in cell-index order; only occupied cells render.
  const liveList = $derived.by(() => {
    const byCell = new Map<number, LiveSlot>();
    for (const s of Object.values(liveSlots)) {
      const cell = ((s.game % gridSize) + gridSize) % gridSize;
      const cur = byCell.get(cell);
      if (!cur || s.updated > cur.updated) byCell.set(cell, s);
    }
    return [...byCell.entries()].sort((a, b) => a[0] - b[0]).map(([, s]) => s);
  });
  // Honest chip: real count of games in progress (may exceed visible cells
  // during a rare iteration overlap — the tooltip explains the count).
  const liveInPlay = $derived(Object.values(liveSlots).filter((s) => s.ended === null).length);
  /**
   * Three-state LIVE GAMES panel (gate-aware):
   * - 'live'  — boards on the wall, or live frames flowed within the quiet window;
   * - 'gate'  — a run announced itself this session, the socket is healthy, and
   *             no `live` frame has arrived for GATE_QUIET_MS: a title match
   *             (arena gate) is the only thing that silences a live run;
   * - 'empty' — no run has announced itself on this visit.
   */
  const panelMode = $derived.by(() => {
    if (liveList.length > 0 || (sawLive && nowTick - lastLiveAt < GATE_QUIET_MS)) return 'live';
    // PRIMARY (post-deploy servers): the metrics payload's own gate flag —
    // ground truth that also covers a page opened MID-GATE, where metrics
    // arrive within 10s but no journal/log frame ever will.
    if (liveMetrics?.gate_in_progress === true) return 'gate';
    // FALLBACK (pre-deploy servers): quiet-window inference from received frames.
    if (conn === 'open' && sawRunFrame && nowTick - lastRunAt >= GATE_QUIET_MS) return 'gate';
    return 'empty';
  });
  const liveChip = $derived(
    panelMode === 'live' ? `${liveInPlay} IN PLAY` : panelMode === 'gate' ? 'TITLE MATCH' : 'IDLE',
  );
  const liveChipTip = $derived.by(() => {
    if (panelMode !== 'gate') {
      return 'How many of the training games are still in progress at this moment. A finished game shows its result for a few seconds, then clears its board.';
    }
    return liveMetrics?.gate_in_progress === true
      ? 'The server reports a title match in progress — the current version is playing the champion for the crown, and that match is not broadcast move by move. The boards return when it ends.'
      : 'No practice games are streaming at the moment — the machine is likely playing a title match, which is not broadcast move by move. The boards return when it ends.';
  });
  const featuredSlot = $derived(featured !== null ? (liveSlots[featured] ?? null) : null);

  let movesEl: HTMLDivElement | undefined = $state();
  // Featured move list auto-follows the newest move.
  $effect(() => {
    const n = featuredSlot?.moves.length ?? 0;
    if (movesEl && n > 0) movesEl.scrollTop = movesEl.scrollHeight;
  });

  // Tooltip copy (FS-8b voice: plain clarity).
  const TIP_LIVE_GAMES =
    'The Fool is playing itself right now — every board is one real game running on the training machine, updating the moment a move is made. Boards advance every few seconds, not every second: between moves the bot is thinking, weighing hundreds of candidate positions before it commits.';
  const TIP_MOVES =
    'Each line is one move, newest at the bottom. A few lines show coordinates (like e2e4) instead of standard notation — that happens when we join a game mid-way and never saw the position the move was played from.';


  // ---- metrics parsing: base fields are pinned, FS-1/FS-2 additions are optional ----
  function mapOfNumbers(v: unknown): Record<string, number> | undefined {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return undefined;
    const out: Record<string, number> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      if (typeof val === 'number' && Number.isFinite(val)) out[k] = val;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  /** regime → { ckpt-tag → elo } (FS-2 nested map); non-numeric leaves dropped. */
  function mapOfRegimeElo(v: unknown): Record<string, Record<string, number>> | undefined {
    if (!v || typeof v !== 'object' || Array.isArray(v)) return undefined;
    const out: Record<string, Record<string, number>> = {};
    for (const [regime, ckpts] of Object.entries(v as Record<string, unknown>)) {
      if (!ckpts || typeof ckpts !== 'object' || Array.isArray(ckpts)) continue;
      const inner: Record<string, number> = {};
      for (const [tag, val] of Object.entries(ckpts as Record<string, unknown>)) {
        if (typeof val === 'number' && Number.isFinite(val)) inner[tag] = val;
      }
      if (Object.keys(inner).length > 0) out[regime] = inner;
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  function normalizeMetrics(m: unknown): MetricsSnapshot | null {
    if (!m || typeof m !== 'object') return null;
    const o = m as Record<string, unknown>;
    const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    return {
      elo: num(o.elo),
      loss: num(o.loss),
      win_rate: num(o.win_rate),
      games_played: typeof o.games_played === 'number' ? o.games_played : 0,
      regime: typeof o.regime === 'string' ? o.regime : null,
      champion: typeof o.champion === 'string' ? o.champion : null,
      arena_games: num(o.arena_games) ?? undefined,
      best_streak: num(o.best_streak) ?? undefined,
      elo_by_regime: mapOfRegimeElo(o.elo_by_regime),
      regime_elo: mapOfNumbers(o.regime_elo),
      checkpoints: mapOfNumbers(o.checkpoints),
      updated_at: num(o.updated_at) ?? undefined,
      // Gate visibility (additive, post-deploy servers): the server's own
      // title-match flag is the PRIMARY panel signal; absent → false and the
      // quiet-window fallback decides.
      gate_in_progress: o.gate_in_progress === true,
      last_arena_ts: num(o.last_arena_ts) ?? undefined,
      last_journal_ts: num(o.last_journal_ts) ?? undefined,
    };
  }

  // Per-regime Elo leaderboard (FS-2): the map is regime → { ckpt-tag → elo };
  // each regime is represented by its best checkpoint. Fall back to the flat
  // `regime_elo` map, then to a single row for the current regime; else nothing.
  const regimeElo = $derived.by(() => {
    const m = liveMetrics;
    if (!m) return [];
    const map = m.elo_by_regime;
    if (map && typeof map === 'object' && !Array.isArray(map)) {
      const rows = Object.entries(map)
        .map(([regime, ckpts]) => {
          let best = -Infinity;
          if (ckpts && typeof ckpts === 'object') {
            for (const v of Object.values(ckpts)) {
              if (typeof v === 'number' && Number.isFinite(v) && v > best) best = v;
            }
          }
          return { regime, elo: best === -Infinity ? null : best };
        })
        .filter((r): r is { regime: string; elo: number } => r.elo !== null);
      if (rows.length > 0) return rows.sort((a, b) => b.elo - a.elo);
    }
    const flat = m.regime_elo;
    if (flat && Object.keys(flat).length > 0) {
      return Object.entries(flat).map(([regime, elo]) => ({ regime, elo }));
    }
    if (m.regime && typeof m.elo === 'number') return [{ regime: m.regime, elo: m.elo }];
    return [];
  });

  /** Ledger `ts` (epoch seconds or ISO) → short local stamp; '' when unusable. */
  function fmtWhen(ts: number | string | undefined): string {
    if (ts === undefined) return '';
    const d = typeof ts === 'number' ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
  }

  // ---- arena ledger (FS-2 adds /api/games?kind=arena; absent → graceful note) ----
  let arena = $state<ArenaResult[] | null>(null);
  let arenaCountSeen = $state<number | null>(null);

  async function refreshArena(): Promise<void> {
    try {
      const r = await fetchGames('arena');
      arena = r.games;
    } catch {
      arena = null; // endpoint absent/unreachable — the fallback copy stays honest
    }
  }

  // F-A: newest-first ledger, rendered newest on top (defensive — the pinned
  // contract does not pin row order). The full history stays reachable via the
  // SHOW ALL toggle: fossil rows are honest history and are never filtered out.
  const ARENA_VISIBLE = 8;
  let arenaOpen = $state(false);

  /** Ledger `ts` (epoch seconds, epoch ms, or ISO) → epoch ms; NaN when unusable. */
  function tsMs(v: number | string | undefined): number {
    if (v === undefined) return Number.NaN;
    const d = typeof v === 'number' ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
    const t = d.getTime();
    return Number.isNaN(t) ? Number.NaN : t;
  }

  const arenaSorted = $derived.by(() => {
    const rows = arena ?? [];
    return [...rows].sort((a, b) => {
      const ta = tsMs(a.ts);
      const tb = tsMs(b.ts);
      if (Number.isNaN(ta) && Number.isNaN(tb)) return 0;
      if (Number.isNaN(ta)) return 1; // undated rows keep their order at the end
      if (Number.isNaN(tb)) return -1;
      return tb - ta;
    });
  });
  const arenaVisible = $derived(arenaOpen ? arenaSorted : arenaSorted.slice(0, ARENA_VISIBLE));
  /** Early rows where both sides were the same model (pre-fix recordings). */
  const arenaSelfMatch = $derived(
    arenaVisible.some((r) => typeof r.ckpt === 'string' && r.ckpt.length > 0 && r.opponent === r.ckpt),
  );
  /** Rows carrying no rating claim (non-promoted challengers) — explained when present. */
  const arenaEloNone = $derived(arenaSorted.some((r) => typeof r.elo !== 'number'));

  // ---- FS-8b: honest rating chart -------------------------------------------
  // The points are REAL when the arena ledger has rows: each approved checkpoint
  // contributes the Elo recorded right after its gauntlet. Secondary real source
  // is the metrics checkpoint ladder. With neither, the decorative curve stays
  // but is explicitly labelled SAMPLE — it never masquerades as data.
  type ChartPoint = { label: string; elo: number };

  const chartData = $derived.by(() => {
    const rows = (arena ?? []).filter((r) => typeof r.elo === 'number');
    if (rows.length > 0) {
      const byTs = [...rows].sort(
        (a, b) => (typeof a.ts === 'number' ? a.ts : 0) - (typeof b.ts === 'number' ? b.ts : 0),
      );
      const pts: ChartPoint[] = [];
      for (const r of byTs) {
        const label = typeof r.ckpt === 'string' && r.ckpt.length > 0 ? r.ckpt : `#${pts.length + 1}`;
        // Same checkpoint re-gauntleted later → keep the latest rating.
        const existing = pts.findIndex((p) => p.label === label);
        if (existing >= 0) pts[existing] = { label, elo: r.elo as number };
        else pts.push({ label, elo: r.elo as number });
      }
      return { real: true as const, pts };
    }
    const ladder = liveMetrics?.checkpoints;
    if (ladder && Object.keys(ladder).length > 0) {
      return { real: true as const, pts: Object.entries(ladder).map(([label, elo]) => ({ label, elo })) };
    }
    return { real: false as const, pts: [] as ChartPoint[] };
  });

  /** Real data → scaled geometry; no data → null (sample SVG renders instead). */
  const chartView = $derived.by(() => {
    if (!chartData.real || chartData.pts.length === 0) return null;
    const elos = chartData.pts.map((p) => p.elo);
    const lo = Math.min(...elos);
    const hi = Math.max(...elos);
    const span = Math.max(hi - lo, 40);
    const min = lo - span * 0.15;
    const max = hi + span * 0.15;
    const W = 520;
    const H = 190;
    // Inset the x domain so the first/last 7px dots are never half-clipped by
    // the plot edge (they used to sit exactly on x=0 and x=520).
    const PAD_X = 10;
    const plotW = W - PAD_X * 2;
    const n = chartData.pts.length;
    const xy = chartData.pts.map((p, i) => ({
      ...p,
      x: PAD_X + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW),
      y: H - ((p.elo - min) / (max - min)) * (H - 12) - 6,
    }));
    const d = xy.length > 1 ? `M${xy.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' L')}` : null;
    // Whole-number Elo, no locale truncation (labels render outside the SVG).
    const ticks = [max, (max + min) / 2, min].map((v) => String(Math.round(v)));
    // Two or fewer distinct ratings would read as a crash. Say so instead of
    // fabricating points; the caption only appears in that case.
    const sparse = new Set(chartData.pts.map((p) => Math.round(p.elo))).size < 3;
    const last = xy[xy.length - 1]!;
    const xPct = (last.x / W) * 100;
    const anchor = xPct > 72 ? 'end' : xPct < 28 ? 'start' : 'middle';
    const lastLabel = {
      value: Math.round(last.elo).toLocaleString('en-US'),
      xPct,
      yPct: (last.y / H) * 100,
      anchor,
    };
    return { xy, d, ticks, sparse, lastLabel };
  });

  const yTicks = $derived(chartView ? chartView.ticks : ['1800', '1600', '1400', '1200', '1000']);

  // ---- FS-8b: plain-English tooltip copy (bodies are pure clarity) ----------
  const TIP_ELO =
    'A chess rating — higher means a stronger player. A casual club player is around 1200 and world champions are near 2800. The Fool starts at 800 (a beginner) and climbs by beating versions of itself.';
  const TIP_GAMES =
    'How many games The Fool has played against itself. Like a musician practicing — every game is a repetition, and the bot studies each one afterwards.';
  const TIP_WIN_RATE =
    "How often the newest version beats the previous version in a head-to-head match. A version only becomes 'champion' — and takes over — if it wins.";
  const TIP_LOSS =
    "A score for how wrong The Fool's guesses were during its latest study session. High = still confused; falling over time = it's learning. Some noise is normal.";
  const TIP_BOARD =
    "This board replays a sample game so you can see what The Fool's practice looks like: it plays both sides, one move at a time. It is an illustration — the real training games run on the machine doing the training. The score and depth figures below come from the sample, too.";
  const TIP_JOURNAL =
    "The bot's training diary, newest at the bottom. Each entry is one round of practice: how many games it played, how wrong its guesses were, and its current rating. Some lines show raw JSON — that is the diary in its rawest form; you never need to read it.";
  const TIP_ARENA =
    'Every time The Fool learns something new, the new version must beat the old one in a short match before it takes over. This is the record of those matches — one row per challenge, newest first. A row shows the rating that match recorded: challengers that were not promoted claim no rating ("elo none"), while the chart above shows the recomputed values.';
  const TIP_LEADERBOARD =
    "The Fool can train in different 'schools' — different practice methods with their own schedules. This compares how each school's champion is rated.";
  // F-C: the per-regime table is only as fresh as the newest match that fed it.
  // Stamping that date on the panel keeps a stale feed visibly dated instead of
  // silently authoritative; the metrics file's own updated_at is the fallback.
  const TIP_LB_ASOF =
    'The date of the newest recorded match feeding this table. Per-regime ratings only change when a match is scored — if this date looks old, the numbers are exactly that old too.';
  const lbAsOf = $derived.by(() => {
    const row = arenaSorted.find((r) => !Number.isNaN(tsMs(r.ts)));
    if (row) return fmtWhen(row.ts);
    const u = liveMetrics?.updated_at;
    return typeof u === 'number' ? fmtWhen(u) : null;
  });
  const chipTip = $derived(
    conn === 'open'
      ? "You're watching the training in real time — this page refreshes itself every couple of seconds while the bot learns."
      : 'The training feed is not connected right now, so this page shows sample numbers. When a run is live, this chip lights up and the page updates itself every couple of seconds.',
  );
  const telemetryTip = $derived(
    liveMetrics !== null
      ? 'These numbers come straight from the live training run — they update on their own.'
      : 'The bot is not connected right now, so these are sample numbers showing what the panel looks like. Real figures appear as soon as training telemetry arrives.',
  );
  const chartTip = $derived(
    chartData.real
      ? "When The Fool finishes training a new version, that version must beat the old one in a short match (a 'gauntlet') before it takes over. Each point is the official rating recorded right after one of those matches — newest on the right. Up means the challenger won the title. These points are real: they come from the arena ledger below."
      : "When The Fool finishes training a new version, that version must beat the old one in a short match (a 'gauntlet') before it takes over. This panel will plot its rating after each of those matches. No matches are recorded yet, so the curve shown is a labelled sample — not real data.",
  );

  function onMessage(msg: ServerMessage): void {
    if (msg.topic !== 'train') return;
    if (msg.type === 'metrics') {
      const m = normalizeMetrics(msg.payload);
      if (m) {
        liveMetrics = m;
        // Refetch the ledger only when the arena counter actually moves.
        if (typeof m.arena_games === 'number' && m.arena_games !== arenaCountSeen) {
          arenaCountSeen = m.arena_games;
          void refreshArena();
        }
      }
    } else if (msg.type === 'log') {
      pushJournal(msg);
      noteLogFrame(msg.payload); // sniff kind:"iteration" lines for the gate state
    } else if (msg.type === 'live') {
      onLiveFrame(msg.payload);
    } else if (msg.type === 'game_end') {
      onGameEnd(msg.payload);
    }
  }

  $effect(() => {
    const offMessage = socket.onMessage(onMessage);
    const offStatus = socket.onStatus((s) => {
      conn = s;
      if (s === 'open') resetLiveSlots(); // (re)connect = fresh join: no replay exists
    });
    socket.start();
    socket.subscribe('train');
    void fetchMetrics()
      .then((m) => {
        if (liveMetrics === null) liveMetrics = normalizeMetrics(m);
      })
      .catch(() => {});
    void refreshArena();
    return () => {
      offMessage();
      offStatus();
      for (const t of endTimers.values()) clearTimeout(t);
      endTimers.clear();
      socket.stop();
    };
  });

  // Live values when the backend reports them; illustrative numbers otherwise.
  const eloText = $derived(liveMetrics?.elo != null ? Math.round(liveMetrics.elo).toLocaleString('en-US') : '1,625');
  const lossText = $derived(liveMetrics?.loss != null ? liveMetrics.loss.toFixed(3) : '0.384');
  const winRateText = $derived(
    liveMetrics?.win_rate != null
      ? `${((liveMetrics.win_rate <= 1 ? liveMetrics.win_rate * 100 : liveMetrics.win_rate)).toFixed(1)}%`
      : '54.7%',
  );
  const gamesText = $derived(
    liveMetrics?.games_played != null ? liveMetrics.games_played.toLocaleString('en-US') : '12,482',
  );
  // Honesty (F-B): the BEST STREAK stat was removed — no server code path ever
  // writes `best_streak` (verified: absent from data/metrics.json and unwritten
  // in training/), so the tile could only ever show a dead "—". The field stays
  // in the typed metrics contract below so a backend that adds it later lights
  // the stat back up without a client change.
  const telemetryLabel = $derived(liveMetrics !== null ? 'LIVE TELEMETRY' : 'ILLUSTRATIVE DATA');
  const regimeText = $derived(liveMetrics?.regime ?? null);
  // ---- journal heartbeat (production-reality fix) ---------------------------
  // During self-play the wire is ~all live/game_end frames; `log` frames
  // (iteration/arena/aggregates) arrive only every ~45 min, and the subscribe
  // backlog excludes live/game_end — so a mid-run page legitimately has an
  // empty diary. The practice-move stream IS the journal's pulse until the
  // diary fills; showing "0 LINES" then reads as broken.
  const liveRate10s = $derived.by(() => {
    const cutoff = nowTick - 10_000;
    return liveArrivals.filter((t) => t >= cutoff).length / 10;
  });
  /** Practice moves are flowing right now (frames this join, last within the quiet window). */
  const journalStreaming = $derived(liveArrivals.length > 0 && nowTick - lastLiveAt < GATE_QUIET_MS);
  const rateLabel = $derived(
    liveRate10s >= 0.95 ? String(Math.round(liveRate10s)) : liveRate10s.toFixed(1),
  );
  // ---- journal display state (label only; no thresholds touched) -------------
  // The heartbeat above only fires once a `live` frame has arrived this join.
  // Between self-play batches the practice stream legitimately pauses a second
  // or two, so a fresh page could render "IDLE" before its first frame — which
  // falsely implies training is off. Distinguish:
  // - 'streaming'  — practice moves flowing (the unchanged heartbeat rule);
  // - 'listening'  — connected, but zero frames since this join: the feed is
  //                  waiting for the next practice move (not a training idle);
  // - 'connecting' — the socket has not opened yet;
  // - 'idle'       — only the honest quiet cases: frames flowed this join and
  //                  then went silent past the gate-quiet window (gate/idle),
  //                  or the feed is offline.
  type JournalState = 'feed' | 'buffered' | 'streaming' | 'connecting' | 'listening' | 'idle';
  const JOURNAL_LABEL: Record<JournalState, string> = {
    feed: 'LIVE FEED',
    buffered: 'BUFFERED',
    streaming: 'STREAMING',
    connecting: 'CONNECTING',
    listening: 'LISTENING',
    idle: 'IDLE',
  };
  const journalState = $derived.by<JournalState>(() => {
    if (journal.length > 0) return conn === 'open' ? 'feed' : 'buffered';
    // A closed socket cannot have frames flowing, even if one landed moments ago.
    if (journalStreaming && conn === 'open') return 'streaming';
    if (conn === 'connecting') return 'connecting';
    if (conn === 'open' && liveArrivals.length === 0) return 'listening';
    return 'idle';
  });
  const journalLabel = $derived(JOURNAL_LABEL[journalState]);
  const chip = $derived(
    conn === 'open' ? 'LIVE · CONNECTED' : conn === 'connecting' ? 'LIVE · CONNECTING' : 'DEMO · OFFLINE',
  );
</script>

<svelte:window onkeydown={onGlobalKeydown} />

<main class="page-wrap">
  <div class="page-heading">
    <div>
      <div class="eyebrow">AN OBSERVATION OF GROWTH</div>
      <h1>Watch</h1>
      <p>Observe The Fool as it learns the game.</p>
    </div>
    <div class="training-state" class:off={conn !== 'open'}>
      <i class="live-dot"></i>
      <span>{chip}</span>
      <Tip text={chipTip} label="What does this connection chip mean?" />
    </div>
  </div>

  <!-- FS-8b: the newcomer primer — the whole training loop in story form. -->
  <section class="panel primer">
    <button
      class="primer-toggle"
      aria-expanded={primerOpen}
      aria-controls="primer-body"
      onclick={() => (primerOpen = !primerOpen)}
    >
      <span class="primer-mark">✦</span>
      <span class="primer-title">New here? <b>What am I looking at?</b></span>
      <span class="primer-hint">{primerOpen ? 'HIDE −' : 'SHOW +'}</span>
    </button>
    {#if primerOpen}
      <div class="primer-body" id="primer-body">
        <p>
          The Fool is a chess bot that is teaching itself, and this page is its training diary. The whole loop, in
          order:
        </p>
        <ol>
          <li><b>It plays itself.</b> Thousands of fast games, both sides, no teacher — just the rules and curiosity.</li>
          <li><b>It studies.</b> After each round it reviews every game and adjusts its guesses about good moves.</li>
          <li><b>It must defend the title.</b> A newly trained version only becomes the champion if it beats the previous champion in a short match.</li>
          <li><b>Its rating climbs.</b> Every approved champion adds a point on the Elo scale — higher means stronger — recorded in the arena ledger.</li>
          <li><b>You can play the champion.</b> The current version is always ready for a game on the Play page.</li>
        </ol>
        <p class="primer-close">
          No chess knowledge needed to watch — and you can play it yourself on the
          <button class="text-link primer-link" onclick={() => onnav?.('play')}>Play page</button>.
          The full story lives on the
          <button class="text-link primer-link" onclick={() => onnav?.('about')}>About page</button>.
        </p>
      </div>
    {/if}
  </section>

  <div class="watch-grid">
    <section class="panel training-board live-panel">
      <div class="section-title">
        <span><i class="live-dot"></i> LIVE GAMES <Tip text={TIP_LIVE_GAMES} label="What is this wall of boards?" /></span>
        <span class="tiny-label">{liveChip} <Tip text={liveChipTip} label="What does the count mean?" /></span>
      </div>

      {#if panelMode === 'live'}
        {#if featuredSlot}
          <div class="featured" role="region" aria-label="featured game {featuredSlot.game}">
            <div class="featured-head">
              <span class="live-tag">#{String(featuredSlot.game).padStart(2, '0')}</span>
              <span class="tiny-label">
                {featuredSlot.ended ? `RESULT ${featuredSlot.ended.result}` : `MOVE ${featuredSlot.ply}`}
                {#if featuredSlot.runId}<em class="featured-run"> · {featuredSlot.runId}</em>{/if}
              </span>
              <button class="featured-close" onclick={() => (featured = null)} aria-label="close featured game">✕</button>
            </div>
            <div class="featured-body">
              <div class="featured-board">
                <Board fen={featuredSlot.fen} lastMove={fromTo(featuredSlot.lastUci)} interactive={false} />
              </div>
              <div class="featured-side">
                <div class="featured-moves-title">
                  <span class="tiny-label">MOVES</span>
                  <Tip text={TIP_MOVES} label="About the move list" />
                </div>
                <div class="featured-moves" bind:this={movesEl}>
                  {#each featuredSlot.moves as m, i (`${m.ply}-${i}`)}
                    <span class="fm-row">
                      <b>{Math.floor((m.ply - 1) / 2) + 1}{m.ply % 2 === 1 ? '.' : '…'}</b>
                      <span>{m.san ?? m.uci}</span>
                    </span>
                  {/each}
                </div>
              </div>
            </div>
          </div>
        {/if}
        <div class="live-grid">
          {#each liveList as s (s.game)}
            <button
              type="button"
              class="live-cell"
              class:ended={s.ended !== null}
              class:active={featured === s.game}
              aria-label={slotLabel(s)}
              aria-pressed={featured === s.game}
              onclick={() => toggleFeatured(s.game)}
            >
              <span class="live-tag">#{String(s.game).padStart(2, '0')}</span>
              {#if s.ended}<span class="live-result">{s.ended.result}</span>{/if}
              <span class="live-mini"><Board mini interactive={false} fen={s.fen} lastMove={fromTo(s.lastUci)} /></span>
              <span class="live-meta">MOVE {s.ply}{s.lastSan ? ` · ${s.lastSan}` : ''}</span>
            </button>
          {/each}
        </div>
      {:else}
        {#if panelMode === 'gate'}
          <!-- Gate quiet: a run exists (it announced itself with live/game_end/
               iteration frames this session) but no practice game has streamed
               for a while — that is what an arena gate (title match) looks like
               from the outside. Grounded in what was actually received. -->
          <p class="live-empty" role="status">
            <b>Title match in progress.</b> The current version is playing the champion for the crown —
            {#if lastRunKind === 'iteration' && lastIter}
              the run's last word was iteration {lastIter.n}{lastIter.elo !== null ? ` · elo ${Math.round(lastIter.elo)}` : ''}, {relAge(lastRunAt)};
            {:else if sawRunFrame}
              the last practice frame arrived {relAge(lastRunAt)};
            {:else if journalAgeTxt}
              the run's last written journal entry was {journalAgeTxt};
            {/if}
            these matches are not streamed move by move, and the practice boards resume when it ends.
          </p>
        {:else}
          <p class="live-empty">
            No games yet. When a training run starts, its practice games appear here move by move.
          </p>
        {/if}
        <div class="sample-block">
          <div class="sample-head">
            <span class="tiny-label">SAMPLE GAME · REPLAY</span>
            <Tip text={TIP_BOARD} label="About the sample game" />
          </div>
          <div class="board-shell small-board">
            <div class="coordinates top-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
            <Board mini fen={current.fen} lastMove={current.lm} />
            <div class="coordinates bottom-coords">a　 b　 c　 d　 e　 f　 g　 h</div>
          </div>
          <div class="playback">
            <button class="round-button" onclick={() => step(-1)} aria-label="previous position">←</button>
            <button class="round-button play-toggle" onclick={toggle} aria-label={playing ? 'pause' : 'play'}>
              {playing ? '❙❙' : '▶'}
            </button>
            <button class="round-button" onclick={() => step(1)} aria-label="next position">→</button>
            <span>POSITION {String(frame + 1).padStart(2, '0')} / {String(frames.length).padStart(2, '0')}</span>
            <input type="range" min="0" max={frames.length - 1} bind:value={frame} oninput={stop} aria-label="timeline" />
          </div>
          <div class="analysis-row">
            <div><small>CURRENT MOVE</small><strong>{current.label}</strong></div>
            <div><small>POSITION SCORE</small><strong>{score}</strong></div>
            <div><small>SEARCH DEPTH</small><strong>{depth}</strong></div>
          </div>
          {#if regimeText}
            <p class="regime-line">REGIME <b>{regimeText}</b>{#if liveMetrics?.champion} · CHAMPION <b>{liveMetrics.champion}</b>{/if}</p>
          {/if}
        </div>
      {/if}

      <!-- FS-6d: recent-finishes ticker. PRACTICE self-play results, newest
           first — explicitly not the arena ledger (title matches only).
           role="group", not a live region: the relative times mutate every
           second and must not chatter at screen-reader users. -->
      {#if recentEnds.length > 0}
        <div class="finishes" role="group" aria-label="Recently finished practice games">
          <span class="finishes-label">
            PRACTICE GAMES · FINISHED
            <Tip text={TIP_FINISHES} label="Practice games vs title matches — what is the difference?" />
          </span>
          <ul class="finishes-row">
            {#each recentEnds as e (e.game + '-' + e.at)}
              <li class="finishes-item" class:expiring={nowTick - e.at >= FINISHES_TTL_MS - FINISHES_FADE_MS}>
                <b>#{String(e.game).padStart(2, '0')}</b> finished {e.result} · {e.plies} plies
                <time>{relAge(e.at)}</time>
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    </section>

    <section class="panel progress-panel">
      <div class="section-title">
        <span>Training progress <Tip text={chartTip} label="What does this chart show?" /></span>
        <span class="tiny-label">{telemetryLabel} <Tip text={telemetryTip} label="Is this data real?" /></span>
      </div>
      <div class="metric-main">
        <strong>{eloText}</strong>
        <span>ESTIMATED ELO <Tip text={TIP_ELO} label="What is Estimated Elo?" /></span>
        <em>{liveMetrics !== null ? 'LIVE · WS FEED' : '↗ +42 THIS WEEK'}</em>
      </div>
      <div class="chart-wrap">
        <div class="chart-y" aria-hidden="true">{#each yTicks as t (t)}<span>{t}</span>{/each}</div>
        <div class="chart-plot">
          {#if chartView}
            <svg
              class="line-chart"
              viewBox="0 0 520 190"
              preserveAspectRatio="none"
              role="img"
              aria-label="Real Elo rating after each arena-approved checkpoint"
            >
              <path class="chart-grid" d="M0 20H520 M0 60H520 M0 100H520 M0 140H520 M0 180H520" />
              {#if chartView.d}
                <path class="chart-line" d={chartView.d} />
              {/if}
              {#each chartView.xy as p, i (`${i}-${p.label}`)}
                <rect class="chart-pt" x={p.x - 3.5} y={p.y - 3.5} width="7" height="7" />
              {/each}
            </svg>
            <span
              class="chart-last"
              class:at-end={chartView.lastLabel.anchor === 'end'}
              class:at-start={chartView.lastLabel.anchor === 'start'}
              style:left="{chartView.lastLabel.xPct}%"
              style:top="{chartView.lastLabel.yPct}%"
              aria-hidden="true">{chartView.lastLabel.value}</span
            >
          {:else}
            <svg class="line-chart" viewBox="0 0 520 190" preserveAspectRatio="none" role="img" aria-label="Sample rating curve, rising trend — an illustration until real arena results arrive">
              <path class="chart-grid" d="M0 20H520 M0 60H520 M0 100H520 M0 140H520 M0 180H520" />
              <path
                class="chart-area"
                d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18 L520 180 L0 180Z"
              />
              <path
                class="chart-line chart-line-sample"
                d="M0 164 L35 151 L70 157 L105 130 L140 137 L175 112 L210 118 L245 92 L280 103 L315 76 L350 83 L385 57 L420 62 L455 35 L490 42 L520 18"
              />
            </svg>
          {/if}
        </div>
      </div>
      <div class="chart-legend">
        {#if chartView}
          <span><i></i> RATING AFTER EACH GAUNTLET · REAL</span>
          <span>SOURCE · ARENA LEDGER</span>
        {:else}
          <span><i></i> SAMPLE CURVE</span>
          <span>REAL PLOTS APPEAR AFTER THE FIRST ARENA MATCH</span>
        {/if}
      </div>
      {#if chartView?.sparse}
        <p class="chart-note">
          Fewer than three rated checkpoints so far — the curve fills in as champions defend.
        </p>
      {/if}
      <div class="stat-grid">
        <div>
          <small>GAMES PLAYED <Tip text={TIP_GAMES} label="What counts as games played?" /></small>
          <strong>{gamesText}</strong>
        </div>
        <div>
          <small>WIN RATE <Tip text={TIP_WIN_RATE} label="What is win rate here?" /></small>
          <strong>{winRateText}</strong>
        </div>
        <div>
          <small>TRAINING LOSS <Tip text={TIP_LOSS} label="What is training loss?" /></small>
          <strong>{lossText}</strong>
        </div>
      </div>
    </section>

    <section class="panel log-panel">
      <div class="section-title">
        <span>Training journal <Tip text={TIP_JOURNAL} label="What is the training journal?" /></span>
        <span class="tiny-label">{journalLabel}</span>
      </div>
      <div class="journal" bind:this={feedEl} onscroll={onFeedScroll}>
        {#if journal.length === 0}
          {#if journalState === 'streaming'}
            <!-- Production reality: during self-play the diary is quiet for
                 ~45 min stretches (log frames carry only iteration/arena lines
                 and the backlog excludes live/game_end). The practice-move
                 stream is the journal's pulse until the diary fills. -->
            <p class="journal-empty journal-heartbeat" role="status">
              <b>Practice moves are streaming</b> — {liveRate10s >= 0.3
                ? `about ${rateLabel} a second`
                : 'a move every few seconds'}, the last was {relAge(lastLiveAt)}. The written journal
              (training iterations, arena gates) is appended only occasionally and appears here when it is.
            </p>
          {:else if journalState === 'listening'}
            <!-- Connected but no frame yet: between self-play batches. Say what is
                 true (waiting for a move) — never that training is idle. -->
            <p class="journal-empty" role="status">
              The live feed is connected — waiting for the next practice move.
            </p>
          {:else if journalState === 'connecting'}
            <p class="journal-empty" role="status">Connecting to the live feed…</p>
          {:else}
            <p class="journal-empty">Awaiting training telemetry — the journal fills as the pipeline runs.</p>
          {/if}
        {:else}
          {#each journal as j (j.id)}
            <p class="journal-line">
              <time>{j.time}</time>
              <b data-level={j.level}>{j.level}</b>
              <span>{j.text}</span>
            </p>
          {/each}
        {/if}
      </div>
      <p class="journal-foot">
        {#if journal.length > 0}
          {journal.length} LINE{journal.length === 1 ? '' : 'S'} · FOLLOW {follow ? 'ON' : 'PAUSED'}
        {:else if journalStreaming}
          PRACTICE MOVES · ~{rateLabel}/S · LAST MOVE {relAge(lastLiveAt)}
        {:else}
          0 LINES · FOLLOW {follow ? 'ON' : 'PAUSED'}
        {/if}
      </p>
    </section>

    <section class="panel lb-panel">
      <div class="section-title">
        <span>Regime leaderboard <Tip text={TIP_LEADERBOARD} label="What is the regime leaderboard?" /></span>
        <span class="tiny-label">
          PER-REGIME ELO{lbAsOf ? ` · AS OF ${lbAsOf}` : ''}
          <Tip text={TIP_LB_ASOF} label="How fresh is this table?" />
        </span>
      </div>
      {#if regimeElo.length === 0}
        <p class="panel-empty">No per-regime Elo reported yet — the ledger appears once arenas are scored.</p>
      {:else}
        <div class="lb-rows">
          {#each regimeElo as row, i (`${row.regime}-${i}`)}
            <div class="lb-row">
              <span class="lb-rank">{String(i + 1).padStart(2, '0')}</span>
              <span class="lb-name">{row.regime}</span>
              <span class="lb-elo">{Math.round(row.elo).toLocaleString('en-US')}</span>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <section class="panel arena-panel">
      <div class="section-title">
        <span>Arena results <Tip text={TIP_ARENA} label="What is the arena ledger?" /></span>
        <span class="tiny-label">LEDGER</span>
      </div>
      {#if arena === null}
        <p class="panel-empty">Arena ledger not available yet.</p>
      {:else if arena.length === 0}
        <p class="panel-empty">No arena games recorded yet — checkpoints are gauntleted as training runs.</p>
      {:else}
        <div class="arena-rows">
          {#each arenaVisible as a, i (`${i}-${a.ckpt ?? a.run_id ?? 'run'}`)}
            <div class="arena-row">
              <span class="arena-title">
                {[a.regime, a.ckpt].filter((v) => typeof v === 'string' && v.length > 0).join(' / ')
                  || a.run_id
                  || 'Arena run'}
                <em>vs {a.opponent ?? 'sparring partner'}</em>
              </span>
              <span class="arena-sub">
                {[
                  typeof a.games === 'number' ? `${a.games} games` : null,
                  typeof a.wins === 'number' && typeof a.draws === 'number'
                    ? `${a.wins}W ${a.draws}D${typeof a.losses === 'number' ? ` ${a.losses}L` : ''}`
                    : (a.result ?? null),
                  typeof a.elo === 'number' ? `elo ${Math.round(a.elo)}` : 'elo none',
                  fmtWhen(a.ts),
                ]
                  .filter((v) => typeof v === 'string' && v.length > 0)
                  .join(' · ') || '—'}
              </span>
            </div>
          {/each}
        </div>
        {#if arenaSorted.length > ARENA_VISIBLE}
          <button class="text-link arena-toggle" onclick={() => (arenaOpen = !arenaOpen)}>
            {arenaOpen ? 'SHOW LATEST ONLY' : `SHOW ALL ${arenaSorted.length} MATCHES`}
          </button>
        {/if}
        {#if arenaSelfMatch}
          <!-- F-A: the aliased-champion era recorded model-vs-itself matches.
               Honest history — explained, never hidden. -->
          <p class="arena-note">
            The self-vs-self rows are early history: they were recorded before a scoring bug was fixed, when
            both sides were the same model. They stay in the ledger.
          </p>
        {/if}
        {#if arenaEloNone}
          <p class="arena-note">
            "ELO NONE" marks a challenger that was not promoted — that match carries no rating claim. The chart
            above shows the recomputed values.
          </p>
        {/if}
      {/if}
    </section>

    <section class="panel quote-panel">
      <span class="quote-star">✧</span>
      <p>“It is not the destination,<br />but the journey that teaches.”</p>
      <small>THE FOOL'S JOURNAL · ENTRY 001</small>
    </section>
  </div>
</main>

<style>
  /* ---- FS-8b: newcomer primer (spans the full grid width) ---- */
  .primer {
    grid-column: 1 / -1;
    padding: 14px 20px;
  }
  .primer-toggle {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    background: none;
    border: 0;
    padding: 0;
    cursor: pointer;
    color: var(--paper);
    text-align: left;
    font-family: inherit;
  }
  .primer-toggle:focus-visible {
    outline: 1px solid var(--gold-bright);
    outline-offset: 3px;
  }
  .primer-mark {
    color: var(--gold);
    font-size: 13px;
  }
  .primer-title {
    font: 15px var(--serif);
  }
  .primer-title b {
    color: var(--gold-bright);
    font-weight: 600;
  }
  .primer-hint {
    margin-left: auto;
    font: 9px var(--mono);
    letter-spacing: 0.14em;
    color: var(--faint);
    white-space: nowrap;
  }
  .primer-body {
    margin-top: 12px;
    border-top: 1px solid var(--hairline);
    padding-top: 12px;
  }
  .primer-body > p {
    margin: 0 0 10px;
    font: 14px/1.6 var(--serif);
    color: var(--paper-dim);
  }
  .primer-body ol {
    margin: 0 0 10px;
    padding-left: 20px;
    display: grid;
    gap: 7px;
  }
  .primer-body li {
    font: 13.5px/1.6 var(--serif);
    color: var(--paper-dim);
  }
  .primer-body li b {
    color: var(--paper);
    font-weight: 600;
  }
  .primer-close {
    margin: 0;
  }
  .primer-close .primer-link {
    margin: 0 2px;
    padding-bottom: 1px;
    font-size: 13px;
  }
  /* sample curve reads as an illustration: dashed stroke, no marker squares */
  .chart-line-sample {
    stroke-dasharray: 6 5;
    opacity: 0.85;
  }
  .chart-pt {
    fill: var(--gold-bright);
    stroke: var(--ink);
    stroke-width: 1;
  }
  /* value of the newest checkpoint, anchored inside the plot */
  .chart-last {
    position: absolute;
    transform: translate(-50%, calc(-100% - 6px));
    font: 10px/1 var(--mono);
    color: var(--paper);
    background: var(--panel);
    border: 1px solid var(--line-soft);
    border-radius: 4px;
    padding: 2px 4px;
    white-space: nowrap;
    pointer-events: none;
  }
  .chart-last.at-end {
    transform: translate(calc(-100% - 6px), calc(-100% - 6px));
  }
  .chart-last.at-start {
    transform: translate(6px, calc(-100% - 6px));
  }
  .chart-note {
    font: 10px/1.5 var(--mono);
    color: var(--faint);
    margin: -8px 0 16px;
  }
  /* content height: the leaderboard must not stretch to the journal's height */
  .lb-panel {
    align-self: start;
  }
  @media (max-width: 480px) {
    /* stack the header so the long title never races the timestamp */
    .lb-panel .section-title {
      flex-direction: column;
      align-items: flex-start;
      gap: 4px;
    }
    .lb-panel .section-title .tiny-label {
      font-size: 8px;
      letter-spacing: 0.1em;
    }
  }

  .training-state.off .live-dot {
    background: var(--red);
    box-shadow: 0 0 9px var(--red);
  }
  .regime-line {
    font: 9px var(--mono);
    letter-spacing: 0.12em;
    color: var(--faint);
    border-top: 1px solid var(--hairline);
    padding-top: 10px;
    margin: 12px 0 0;
  }
  .regime-line b {
    color: var(--gold-bright);
    font-weight: 500;
  }
  /* journal: a terminal-style mono feed, auto-following, capped at 200 lines */
  .journal {
    max-height: 240px;
    overflow-y: auto;
    background: var(--field-bg);
    border: 1px solid var(--hairline);
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .journal-line {
    display: flex;
    gap: 9px;
    align-items: baseline;
    font: 11px var(--mono);
    line-height: 1.5;
    margin: 0;
    color: var(--paper-dim);
  }
  .journal-line time {
    color: var(--faint);
    flex: none;
  }
  .journal-line b {
    font-weight: 400;
    flex: none;
    min-width: 44px;
    color: var(--green);
  }
  .journal-line b[data-level='ERROR'],
  .journal-line b[data-level='WARN'] {
    color: var(--red);
  }
  .journal-line span {
    word-break: break-word;
    overflow-wrap: anywhere;
  }
  .journal-empty {
    font: 11px var(--mono);
    color: var(--faint);
    margin: 0;
  }
  .journal-foot {
    font: 8px var(--mono);
    letter-spacing: 0.12em;
    color: var(--faint);
    margin: 8px 0 0;
  }
  .panel-empty {
    font: 11px var(--mono);
    color: var(--faint);
    line-height: 1.6;
    margin: 6px 0;
  }
  .lb-rows {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .lb-row {
    display: grid;
    grid-template-columns: 30px 1fr auto;
    align-items: baseline;
    gap: 10px;
    padding: 9px 10px;
    background: var(--panel2);
    border: 1px solid var(--hairline);
    font: 12px var(--mono);
  }
  .lb-rank {
    color: var(--faint);
  }
  .lb-name {
    color: var(--paper-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .lb-elo {
    color: var(--gold-bright);
  }
  .arena-rows {
    display: flex;
    flex-direction: column;
  }
  .arena-row {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 0;
    border-bottom: 1px solid var(--hairline);
  }
  .arena-title {
    font: 12px var(--mono);
    color: var(--paper-dim);
  }
  .arena-title em {
    font-style: normal;
    color: var(--muted);
  }
  .arena-sub {
    font: 10px var(--mono);
    color: var(--muted);
  }
  /* F-A: history toggle + honesty footnotes (mono, quiet, on-token) */
  .arena-toggle {
    margin-top: 10px;
  }
  .arena-note {
    font: 10px/1.7 var(--mono);
    letter-spacing: 0.02em;
    color: var(--faint);
    margin: 10px 0 0;
  }

  /* ---- FS-6c: live games grid + featured view ---- */
  .live-panel {
    grid-column: 1 / -1;
  }
  .live-empty {
    font: 14px/1.6 var(--serif);
    color: var(--paper-dim);
    margin: 4px 0 14px;
  }
  .live-empty b {
    color: var(--gold);
    font-weight: 600;
  }
  .journal-heartbeat b {
    color: var(--gold);
    font-weight: 600;
  }
  /* honest empty state keeps the sample replay, clearly labelled */
  .sample-block {
    max-width: 440px;
  }
  .sample-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }
  .live-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 4px;
  }
  .live-cell {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 6px;
    background: var(--field-bg);
    border: 1px solid var(--hairline);
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    color: inherit;
    min-width: 0;
  }
  .live-cell:hover {
    border-color: var(--line);
  }
  .live-cell:focus-visible {
    outline: 2px solid var(--gold);
    outline-offset: 2px;
  }
  .live-cell.active {
    border-color: var(--gold-bright);
  }
  .live-cell.ended {
    border-color: var(--red);
  }
  .live-tag {
    position: absolute;
    top: 6px;
    left: 6px;
    z-index: 2;
    font: 9px var(--mono);
    letter-spacing: 0.08em;
    color: var(--on-accent);
    background: var(--accent-solid);
    padding: 1px 4px;
  }
  .live-result {
    position: absolute;
    inset: 0;
    z-index: 2;
    display: grid;
    place-items: center;
    font: 20px var(--mono);
    color: var(--scrim-text);
    background: var(--scrim);
    pointer-events: none;
  }
  .live-mini {
    display: block;
    min-width: 0;
  }
  .live-meta {
    font: 9px var(--mono);
    letter-spacing: 0.08em;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .featured {
    margin: 2px 0 14px;
    border: 1px solid var(--hairline);
    background: var(--panel2);
    padding: 12px 14px;
  }
  .featured-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .featured-run {
    font-style: normal;
    color: var(--muted);
  }
  .featured-close {
    margin-left: auto;
    background: none;
    border: 1px solid var(--hairline);
    color: var(--paper-dim);
    cursor: pointer;
    font: 11px var(--mono);
    padding: 3px 9px;
  }
  .featured-close:hover {
    color: var(--gold-bright);
    border-color: var(--line);
  }
  .featured-close:focus-visible {
    outline: 2px solid var(--gold);
    outline-offset: 2px;
  }
  .featured-body {
    display: grid;
    grid-template-columns: minmax(0, 380px) minmax(0, 1fr);
    gap: 14px;
  }
  .featured-board {
    min-width: 0;
  }
  .featured-side {
    min-width: 0;
  }
  .featured-moves-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .featured-moves {
    max-height: 320px;
    overflow-y: auto;
    background: var(--field-bg);
    border: 1px solid var(--hairline);
    padding: 8px 10px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
    gap: 2px 12px;
    align-content: start;
  }
  .fm-row {
    display: flex;
    gap: 6px;
    font: 11px var(--mono);
    color: var(--paper-dim);
    white-space: nowrap;
  }
  .fm-row b {
    font-weight: 400;
    color: var(--faint);
  }
  @media (max-width: 900px) {
    .featured-body {
      grid-template-columns: 1fr;
    }
    .featured-board {
      max-width: 340px;
    }
  }
  @media (max-width: 820px) {
    .live-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
  @media (max-width: 560px) {
    .live-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  /* ---- FS-6d: recent-finishes ticker ----
     Deliberately quiet mono styling under the wall of boards: practice results,
     not ledger rows — no gold-row arena language, no shared borders with the
     arena panel. */
  .finishes {
    display: flex;
    align-items: baseline;
    gap: 6px 14px;
    flex-wrap: wrap;
    border-top: 1px solid var(--hairline);
    margin-top: 12px;
    padding-top: 10px;
  }
  .finishes-label {
    font: 9px var(--mono);
    letter-spacing: 0.12em;
    color: var(--faint);
    white-space: nowrap;
  }
  .finishes-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 18px;
    list-style: none;
    margin: 0;
    padding: 0;
    min-width: 0;
  }
  .finishes-item {
    font: 10px var(--mono);
    letter-spacing: 0.06em;
    color: var(--paper-dim);
    white-space: nowrap;
    /* fade matches FINISHES_FADE_MS: the expiring class flips 40s before the prune */
    transition: opacity 40s linear;
  }
  .finishes-item.expiring {
    opacity: 0.25;
  }
  .finishes-item b {
    font-weight: 500;
    color: var(--gold);
  }
  .finishes-item time {
    color: var(--faint);
    margin-left: 6px;
  }
  @media (max-width: 540px) {
    /* one result is wider than a 320px content column: let it wrap instead of
       pushing the page sideways */
    .finishes-item {
      white-space: normal;
    }
  }
</style>
