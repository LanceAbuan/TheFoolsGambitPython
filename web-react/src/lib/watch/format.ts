/** Formatting helpers for the Watch dashboard. Pure, defensive, no React. */

export function clock(ts: string | number | undefined): string {
  if (ts === undefined) return '--:--:--';
  const d = typeof ts === 'number' ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
  if (Number.isNaN(d.getTime())) return '--:--:--';
  return d.toLocaleTimeString('en-GB', { hour12: false });
}

export function fmtWhen(ts: string | number | undefined): string {
  if (ts === undefined) return '';
  const d = typeof ts === 'number' ? new Date(ts < 1e12 ? ts * 1000 : ts) : new Date(ts);
  if (Number.isNaN(d.getTime())) return '';
  return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
}

export function tsMs(v: string | number | undefined): number {
  if (v === undefined) return Number.NaN;
  const d = typeof v === 'number' ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
  return d.getTime();
}

/** `ckpt-000704-0352` → `704·352`, the form used in dense ledgers. */
export function shortCkpt(tag: string | null | undefined): string {
  if (!tag) return '—';
  const m = /^ckpt-0*(\d+)-0*(\d+)$/.exec(tag);
  return m ? `${m[1]}·${m[2]}` : tag;
}

/** Checkpoint sequence number for sorting (`ckpt-000704-0352` → 704). */
export function ckptOrdinal(tag: string): number {
  const m = /(\d+)/.exec(tag);
  return m ? Number(m[1]) : 0;
}

export function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Elo after each gauntlet, sorted by checkpoint order, deduped. */
export interface ChartPoint {
  tag: string;
  short: string;
  elo: number;
  champion: boolean;
}

export function chartPoints(
  checkpoints: Record<string, number | null> | undefined,
  champion: string | null | undefined,
): ChartPoint[] {
  if (!checkpoints) return [];
  const byTag = new Map<string, number>();
  for (const [tag, elo] of Object.entries(checkpoints)) {
    if (typeof elo === 'number' && Number.isFinite(elo)) byTag.set(tag, elo);
  }
  return [...byTag.entries()]
    .sort((a, b) => ckptOrdinal(a[0]) - ckptOrdinal(b[0]))
    .map(([tag, elo]) => ({ tag, short: shortCkpt(tag), elo, champion: tag === champion }));
}

export interface JournalView {
  kind: string;
  level: string;
  text: string;
}

/**
 * Turn one journal line into a human sentence. The training journal is
 * structured JSON; rendering it raw is what made the old feed unreadable.
 */
export function describeJournal(raw: string, fallbackLevel = 'INFO'): JournalView {
  const text = raw.trim();
  if (!text.startsWith('{')) {
    return { kind: 'log', level: fallbackLevel.toUpperCase(), text };
  }
  let o: Record<string, unknown>;
  try {
    o = JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { kind: 'log', level: fallbackLevel.toUpperCase(), text };
  }
  const kind = typeof o.kind === 'string' ? o.kind : 'event';
  const level = typeof o.level === 'string' ? o.level.toUpperCase() : fallbackLevel.toUpperCase();
  const n = asNumber;

  if (kind === 'iteration') {
    const iter = n(o.iteration);
    const loss = n(o.loss);
    const gph = n(o.games_per_hour);
    const elo = n(o.elo);
    const games = n(o.games);
    const parts = [
      iter !== null ? `Iteration ${Math.round(iter)}` : 'Iteration complete',
      loss !== null ? `loss ${loss.toFixed(3)}` : null,
      games !== null ? `${Math.round(games)} games` : null,
      gph !== null ? `${gph.toFixed(1)} games/h` : null,
      elo !== null ? `Elo ${Math.round(elo)}` : null,
    ].filter(Boolean);
    return { kind, level, text: parts.join(' · ') };
  }

  if (kind === 'game_end') {
    const game = n(o.game);
    const plies = n(o.plies);
    const result = typeof o.result === 'string' ? o.result : 'finished';
    return {
      kind,
      level,
      text: `Game ${game !== null ? Math.round(game) : '?'} ended ${result}${plies !== null ? ` after ${Math.round(plies)} plies` : ''}`,
    };
  }

  if (kind === 'arena' || kind === 'arena_result' || kind === 'gate') {
    const ckpt = typeof o.ckpt === 'string' ? o.ckpt : typeof o.checkpoint === 'string' ? o.checkpoint : null;
    const opp = typeof o.opponent === 'string' ? o.opponent : null;
    const wins = n(o.wins);
    const draws = n(o.draws);
    const losses = n(o.losses);
    const promoted = o.promoted === true;
    const score = wins !== null && draws !== null && losses !== null && wins + draws + losses > 0
      ? (wins + draws * 0.5) / (wins + draws + losses)
      : null;
    const head = ckpt && opp ? `${shortCkpt(ckpt)} vs ${shortCkpt(opp)}` : 'Gate result';
    const wdl = wins !== null && draws !== null && losses !== null ? `${wins}W ${draws}D ${losses}L` : null;
    const verdict = promoted ? 'promoted' : 'champion held';
    return {
      kind,
      level,
      text: [head, wdl, score !== null ? score.toFixed(3) : null, verdict].filter(Boolean).join(' · '),
    };
  }

  if (kind === 'aggregate') {
    const games = n(o.games);
    const elo = n(o.elo);
    return {
      kind,
      level,
      text: ['Aggregate', games !== null ? `${Math.round(games)} games` : null, elo !== null ? `Elo ${Math.round(elo)}` : null]
        .filter(Boolean)
        .join(' · '),
    };
  }

  if (kind === 'live') {
    const game = n(o.game);
    return { kind, level, text: `Live · game ${game !== null ? Math.round(game) : '?'}` };
  }

  return { kind, level, text: text.length > 220 ? `${text.slice(0, 217)}…` : text };
}

/** `0.500` → `50.0%`; null-safe. */
export function pct(v: number | null | undefined, digits = 1): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}

/** Score fraction → `0.500` (three decimals, the ledger's convention). */
export function scoreText(wins: number, draws: number, losses: number): string {
  const total = wins + draws + losses;
  if (total === 0) return '—';
  return ((wins + draws * 0.5) / total).toFixed(3);
}
