import { useMemo } from 'react';
import type { ChartPoint } from '../lib/watch/format';

interface EloChartProps {
  points: ChartPoint[];
  height?: number;
}

/**
 * Checkpoint Elo ladder. Real data only: points come from the trainer's
 * checkpoint map; a one-point series says so instead of drawing a fake curve.
 */
export function EloChart({ points, height = 210 }: EloChartProps) {
  const view = useMemo(() => {
    if (points.length === 0) return null;
    const W = 560;
    const H = 200;
    const PAD = { l: 46, r: 18, t: 16, b: 24 };
    const plotW = W - PAD.l - PAD.r;
    const plotH = H - PAD.t - PAD.b;
    const elos = points.map((p) => p.elo);
    const lo = Math.min(...elos);
    const hi = Math.max(...elos);
    const span = Math.max(hi - lo, 40);
    const min = lo - span * 0.18;
    const max = hi + span * 0.18;
    const x = (i: number) => PAD.l + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
    const y = (elo: number) => PAD.t + plotH - ((elo - min) / (max - min)) * plotH;
    const pts = points.map((p, i) => ({ ...p, x: x(i), y: y(p.elo) }));
    const line = pts.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' L');
    const ticks = [max, (max + min) / 2, min].map((v) => Math.round(v));
    const last = pts[pts.length - 1];
    const first = pts[0];
    const best = pts.reduce((a, b) => (b.elo > a.elo ? b : a), pts[0]);
    return { W, H, PAD, pts, line, ticks, last, first, best, sparse: new Set(points.map((p) => Math.round(p.elo))).size < 3 };
  }, [points]);

  if (!view) {
    return (
      <div className="chart-empty">
        <span>No rated checkpoints yet</span>
        <p>The curve appears after the first gauntlet records an Elo.</p>
      </div>
    );
  }

  const { W, H, PAD, pts, line, ticks, last, first, best, sparse } = view;
  const lastAnchor = last.x > W * 0.72 ? 'end' : last.x < W * 0.28 ? 'start' : 'middle';

  return (
    <div className="elo-chart" style={{ minHeight: height }}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Estimated Elo after each checkpoint gauntlet">
        {ticks.map((tick, i) => {
          const gy = PAD.t + (i / 2) * (H - PAD.t - PAD.b);
          return (
            <g key={tick}>
              <line x1={PAD.l} x2={W - PAD.r} y1={gy} y2={gy} className="chart-grid" />
              <text x={PAD.l - 9} y={gy + 3.5} textAnchor="end" className="chart-tick">
                {tick}
              </text>
            </g>
          );
        })}

        <path d={`M${line}`} className="chart-line" pathLength={1} />

        {pts.map((p, i) => (
          <g key={p.tag}>
            {p.champion ? <circle cx={p.x} cy={p.y} r={7} className="chart-champion-ring" /> : null}
            <circle
              cx={p.x}
              cy={p.y}
              r={p.champion ? 3.6 : i === pts.length - 1 ? 3.2 : 2.4}
              className={p.champion ? 'chart-dot champion' : 'chart-dot'}
            >
              <title>
                {p.tag} · {Math.round(p.elo)}
              </title>
            </circle>
          </g>
        ))}

        <text x={first.x} y={H - 6} textAnchor={first.x < W * 0.1 ? 'start' : 'middle'} className="chart-x">
          {first.short}
        </text>
        <text x={last.x} y={H - 6} textAnchor={lastAnchor} className="chart-x">
          {last.short}
        </text>
        <text x={last.x} y={last.y - 10} textAnchor={lastAnchor} className="chart-last">
          {Math.round(last.elo)}
        </text>
        {best.tag !== last.tag && best.tag !== first.tag ? (
          <text x={best.x} y={best.y - 10} textAnchor="middle" className="chart-best">
            {Math.round(best.elo)}
          </text>
        ) : null}
      </svg>
      {sparse ? <p className="chart-note">Too few distinct ratings to read a trend yet — each point is a real gauntlet result.</p> : null}
    </div>
  );
}
