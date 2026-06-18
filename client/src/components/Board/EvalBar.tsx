import { Box, Text } from '@mantine/core';
import { useGame } from '../../GameContext';

export default function EvalBar() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const cp = analysis?.evaluation ?? 0;

  // cp in centipawns, typically -500..+500 for normal positions
  // normalize to -1..1, then map to percentage from bottom
  const t = Math.max(-1, Math.min(1, cp / 2000));
  // White advantage pushes white fill up from center
  const pct = Math.max(2, Math.min(98, 50 - t * 48));

  // White at top (white's advantage), black at bottom (black's advantage)
  // like chess.com / lichess style
  const whiteFill = '#ffffff';
  const blackFill = '#000000';
  const neutralGray = '#30363D';
  const midGray = '#484F58';

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 36, flexShrink: 0, alignSelf: 'stretch' }}>
      <Box
        style={{
          position: 'relative',
          width: 28,
          flex: 1,
          background: neutralGray,
          borderRadius: 6,
          overflow: 'hidden',
          boxShadow: 'inset 0 0 6px rgba(0,0,0,0.6)',
        }}
      >
        {/* Subtle background hint */}
        <Box
          style={{
            position: 'absolute',
            inset: 0,
            background: `linear-gradient(to bottom, ${whiteFill}88 0%, ${whiteFill}22 30%, transparent 50%, #00000022 70%, #00000088 100%)`,
            opacity: 0.3,
          }}
        />

        {/* Top portion — White's advantage */}
        <Box
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: `${100 - pct}%`,
            background: `linear-gradient(to bottom, ${whiteFill}, ${midGray})`,
            transition: 'height 0.3s ease',
            opacity: 0.85,
          }}
        />

        {/* Bottom portion — Black's advantage */}
        <Box
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: '100%',
            height: `${pct}%`,
            background: `linear-gradient(to top, ${blackFill}, ${midGray})`,
            transition: 'height 0.3s ease',
            opacity: 0.85,
          }}
        />

        {/* Center divider line */}
        <Box
          style={{
            position: 'absolute',
            top: '50%',
            left: 2,
            width: 'calc(100% - 4px)',
            height: 2,
            background: midGray,
            zIndex: 1,
            borderRadius: 1,
          }}
        />

        {/* Tick marks */}
        <Box
          style={{
            position: 'absolute',
            top: '25%',
            left: 4,
            width: 'calc(100% - 8px)',
            height: 1,
            background: midGray,
            opacity: 0.3,
          }}
        />
        <Box
          style={{
            position: 'absolute',
            top: '75%',
            left: 4,
            width: 'calc(100% - 8px)',
            height: 1,
            background: midGray,
            opacity: 0.3,
          }}
        />
      </Box>

      <Text
        size="xs"
        fw={700}
        mt={6}
        c={cp > 0 ? '#C9D1D9' : cp < 0 ? '#8B949E' : '#8B949E'}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {cp > 0 ? '+' : ''}{cp.toFixed(1)}
      </Text>
    </Box>
  );
}
