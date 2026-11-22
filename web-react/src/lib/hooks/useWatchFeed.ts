import { useEffect, useRef, useState } from 'react';
import { Chess } from 'chess.js';
import { useSocket } from '../hooks/useSocket';
import { useMetrics } from '../hooks/useMetrics';
import type { MetricsSnapshot } from '../api/http';
import type { SocketStatus } from '../api/ws';
import { describeJournal, type JournalView } from '../watch/format';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const JOURNAL_CAP = 200;
const STORE_CAP = 64;
const SLOT_TTL_MS = 120_000;
const END_TTL_MS = 4_000;

export interface LiveMove {
  ply: number;
  uci: string;
  san: string | null;
}

export interface LiveSlot {
  game: number;
  fen: string;
  ply: number;
  lastUci: string | null;
  lastSan: string | null;
  runId: string | null;
  updated: number;
  ended: string | null;
  moves: LiveMove[];
}

export interface JournalEntry extends JournalView {
  id: string;
  at: number;
}

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
    return null;
  }
}

function logText(payload: Record<string, unknown>): { raw: string; level: string } | null {
  const direct = [payload.line, payload.message, payload.msg, payload.text].find(
    (v): v is string => typeof v === 'string' && v.length > 0,
  );
  if (!direct) return null;
  return {
    raw: direct,
    level: typeof payload.level === 'string' ? payload.level : 'INFO',
  };
}

/**
 * Watch feed: one WS subscription (`train`) fanned into live boards, the
 * structured journal and fresh metrics. REST polling remains the fallback when
 * the socket is down, so the page is never blank.
 */
export function useWatchFeed(): {
  status: SocketStatus;
  metrics: MetricsSnapshot | null;
  metricsError: string | null;
  journal: JournalEntry[];
  slots: LiveSlot[];
  lastLiveAt: number | null;
  movesPerSec: number;
} {
  const { metrics: polled, error: metricsError } = useMetrics();
  const [wsMetrics, setWsMetrics] = useState<MetricsSnapshot | null>(null);
  const [wsMetricsAt, setWsMetricsAt] = useState(0);
  const [journal, setJournal] = useState<JournalEntry[]>([]);
  const [slotMap, setSlotMap] = useState<Record<number, LiveSlot>>({});
  const [lastLiveAt, setLastLiveAt] = useState<number | null>(null);
  const [movesPerSec, setMovesPerSec] = useState(0);

  const arrivals = useRef<number[]>([]);
  const seen = useRef(new Set<string>());
  const endTimers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const journalSeq = useRef(0);

  const { status } = useSocket(['train'], (msg) => {
    if (msg.topic !== 'train') return;

    if (msg.type === 'metrics') {
      setWsMetrics(msg.payload as unknown as MetricsSnapshot);
      setWsMetricsAt(Date.now());
      return;
    }

    if (msg.type === 'live') {
      const p = msg.payload;
      const game = Number(p.game);
      const fen = typeof p.fen === 'string' ? p.fen : '';
      if (!Number.isInteger(game) || !fen) return;
      const ply = typeof p.ply === 'number' ? p.ply : 0;
      const uci = typeof p.move === 'string' ? p.move : '';
      setSlotMap((prev) => {
        const before = prev[game];
        if (before && before.fen === fen && before.ply === ply) return prev;
        const san = sanFor(before?.fen ?? null, ply, uci);
        const next: LiveSlot = {
          game,
          fen,
          ply,
          lastUci: uci || null,
          lastSan: san,
          runId: typeof p.run_id === 'string' && p.run_id ? p.run_id : (before?.runId ?? null),
          updated: Date.now(),
          ended: null,
          moves: [...(before?.moves ?? []), { ply, uci, san }].slice(-160),
        };
        const out = { ...prev, [game]: next };
        const ids = Object.keys(out).map(Number);
        if (ids.length > STORE_CAP) {
          let stalest = ids[0];
          for (const id of ids) if (out[id].updated < out[stalest].updated) stalest = id;
          delete out[stalest];
        }
        return out;
      });
      const now = Date.now();
      arrivals.current = [...arrivals.current, now].slice(-240);
      const recent = arrivals.current.filter((t) => now - t <= 5_000);
      setMovesPerSec(Math.round((recent.length / 5) * 10) / 10);
      setLastLiveAt(now);
      return;
    }

    if (msg.type === 'game_end') {
      const p = msg.payload;
      const game = Number(p.game);
      if (!Number.isInteger(game)) return;
      const result = typeof p.result === 'string' ? p.result : 'finished';
      setSlotMap((prev) => (prev[game] ? { ...prev, [game]: { ...prev[game], ended: result } } : prev));
      const existing = endTimers.current.get(game);
      if (existing) clearTimeout(existing);
      endTimers.current.set(
        game,
        setTimeout(() => {
          endTimers.current.delete(game);
          setSlotMap((prev) => {
            if (!prev[game]) return prev;
            const next = { ...prev };
            delete next[game];
            return next;
          });
        }, END_TTL_MS),
      );
      return;
    }

    if (msg.type === 'log') {
      const parsed = logText(msg.payload);
      if (!parsed) return;
      const key = `${msg.ts}|${parsed.raw.slice(0, 48)}`;
      if (seen.current.has(key)) return;
      seen.current.add(key);
      if (seen.current.size > 500) {
        const first = seen.current.values().next().value;
        if (first) seen.current.delete(first);
      }
      const at = Date.parse(msg.ts);
      const when = Number.isNaN(at) ? Date.now() : at;
      const view = describeJournal(parsed.raw, parsed.level);
      journalSeq.current += 1;
      const entry: JournalEntry = {
        ...view,
        id: `${typeof msg.seq === 'number' ? msg.seq : 'x'}-${journalSeq.current}`,
        at: when,
      };
      setJournal((prev) => [...prev, entry].slice(-JOURNAL_CAP));
    }
  });

  // Age out boards that stopped streaming (finished, gate started, or a
  // dropped connection) so stale positions never pose as live.
  useEffect(() => {
    const timer = setInterval(() => {
      const cutoff = Date.now() - SLOT_TTL_MS;
      setSlotMap((prev) => {
        const next: Record<number, LiveSlot> = {};
        let changed = false;
        for (const [key, slot] of Object.entries(prev)) {
          if (slot.updated < cutoff) changed = true;
          else next[Number(key)] = slot;
        }
        return changed ? next : prev;
      });
    }, 10_000);
    return () => clearInterval(timer);
  }, []);

  useEffect(
    () => () => {
      for (const timer of endTimers.current.values()) clearTimeout(timer);
      endTimers.current.clear();
    },
    [],
  );

  const metrics = wsMetrics && Date.now() - wsMetricsAt < 35_000 ? wsMetrics : polled;

  return {
    status,
    metrics,
    metricsError,
    journal,
    slots: Object.values(slotMap).sort((a, b) => b.updated - a.updated),
    lastLiveAt,
    movesPerSec,
  };
}
