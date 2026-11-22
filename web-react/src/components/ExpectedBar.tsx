import { useEffect, useState } from 'react';
import { analyze } from '../lib/api/http';

/**
 * White-expected-score bar shared by Analysis and the Watch featured game.
 * `value` is White's expected score in [0, 1]; null renders the 50% mark.
 */
export function ExpectedBar({ value }: { value: number | null }) {
  const w = value === null ? 50 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="expected" role="img" aria-label={`White expected score ${Math.round(w)} percent`}>
      <div className="expected-fill" style={{ width: `${w}%` }} />
      <div className="expected-mid" aria-hidden />
    </div>
  );
}

/** White-POV score label: `+0.36`, `−M4`, `—`. Mate 0 renders via the caller. */
export function evalScoreText(cp: number | null, mate: number | null): string {
  if (mate !== null && mate !== 0) return mate > 0 ? `+M${mate}` : `−M${Math.abs(mate)}`;
  if (cp === null || !Number.isFinite(cp)) return '—';
  const v = cp / 100;
  const s = Math.abs(v).toFixed(2);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

/** Debounced, abortable Fool evaluation of one FEN (used by the Watch eval bar). */
export function useFoolEval(fen: string | null, visits = 32, debounceMs = 1200): {
  cp: number | null;
  mate: number | null;
  expectedWhite: number | null;
  state: 'idle' | 'thinking' | 'ready' | 'unavailable';
} {
  const [result, setResult] = useState<{
    cp: number | null;
    mate: number | null;
    expectedWhite: number | null;
    state: 'idle' | 'thinking' | 'ready' | 'unavailable';
  }>({ cp: null, mate: null, expectedWhite: null, state: 'idle' });

  useEffect(() => {
    if (!fen) {
      setResult({ cp: null, mate: null, expectedWhite: null, state: 'idle' });
      return;
    }
    let alive = true;
    setResult((r) => ({ ...r, state: 'thinking' }));
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      analyze({ fen, engines: ['fool'], visits, multipv: 1 }, ctrl.signal)
        .then((r) => {
          if (!alive) return;
          const fool = r.engines.fool;
          if (!fool?.available) {
            setResult({ cp: null, mate: null, expectedWhite: null, state: 'unavailable' });
            return;
          }
          setResult({
            cp: fool.score_cp,
            mate: fool.mate,
            expectedWhite: fool.expected_white,
            state: 'ready',
          });
        })
        .catch(() => {
          if (alive) setResult({ cp: null, mate: null, expectedWhite: null, state: 'unavailable' });
        });
    }, debounceMs);
    return () => {
      alive = false;
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [fen, visits, debounceMs]);

  return result;
}
