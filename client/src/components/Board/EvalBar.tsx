import { useGame } from '../../GameContext';

function evalColor(cp: number): string {
  // White advantage → light (white-ish), Black advantage → dark (black-ish)
  const t = Math.max(-1, Math.min(1, cp / 2000)); // normalize to -1..1
  // Mix between black (bad for white) and white (good for white)
  const r = Math.round(128 + t * 127);
  const g = Math.round(128 - Math.abs(t) * 80);
  const b = Math.round(128 - t * 127);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function EvalBar() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const evalNorm = analysis?.evaluation_normalized ?? 0;
  const cp = analysis?.evaluation ?? 0;

  // evalNorm is -1..1. White positive = white winning.
  // Fill from bottom: 50% = equal. White advantage pushes fill up, black pushes down.
  const pct = Math.max(2, Math.min(98, 50 - evalNorm * 48));

  return (
    <div className="eval-bar-container">
      <div className="eval-bar-track">
        <div
          className="eval-bar-fill"
          style={{
            height: `${100 - pct}%`,
            top: 0,
            background: `linear-gradient(to bottom, ${evalColor(Math.max(0, cp))}, ${evalColor(cp)})`,
          }}
        />
        <div
          className="eval-bar-fill-bottom"
          style={{
            height: `${pct}%`,
            bottom: 0,
            background: `linear-gradient(to top, ${evalColor(Math.min(0, cp))}, ${evalColor(cp)})`,
          }}
        />
        <div className="eval-bar-divider" />
      </div>
      <div className="eval-bar-value">{cp > 0 ? '+' : ''}{cp.toFixed(1)}</div>
    </div>
  );
}
