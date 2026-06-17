import { useGame } from '../../GameContext';

export default function EvalBar() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const evalNorm = analysis?.evaluation_normalized ?? 0;
  const cp = analysis?.evaluation ?? 0;
  const pct = Math.max(0, Math.min(100, 50 + evalNorm * 200));

  return (
    <div className="eval-bar-container">
      <div className="eval-bar-track">
        <div
          className="eval-bar-fill"
          style={{
            height: `${pct}%`,
            background: cp >= 0 ? '#fff' : '#000',
          }}
        />
      </div>
      <div className="eval-bar-value">{cp > 0 ? '+' : ''}{cp.toFixed(1)}</div>
    </div>
  );
}
