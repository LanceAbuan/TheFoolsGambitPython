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

  // Green for white advantage, red for black advantage
  const greenTop = '#3fb950';
  const redBottom = '#f85149';
  const neutralGray = '#30363D';

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
        {/* Background gradient — full-bar gradient from green → gray → red */}
        <Box
          style={{
            position: 'absolute',
            inset: 0,
            background: `linear-gradient(to bottom, ${greenTop} 0%, ${greenTop} 30%, ${neutralGray} 50%, ${redBottom} 70%, ${redBottom} 100%)`,
            opacity: 0.15,
          }}
        />

        {/* Top fill (white advantage zone) */}
        <Box
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: `${100 - pct}%`,
            background: `linear-gradient(to bottom, ${greenTop}, color-mix(in srgb, ${greenTop} 40%, ${neutralGray} 60%))`,
            transition: 'height 0.3s ease',
          }}
        />

        {/* Bottom fill (black advantage zone) */}
        <Box
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: '100%',
            height: `${pct}%`,
            background: `linear-gradient(to top, ${redBottom}, color-mix(in srgb, ${redBottom} 40%, ${neutralGray} 60%))`,
            transition: 'height 0.3s ease',
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
            background: '#484F58',
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
            background: '#484F58',
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
            background: '#484F58',
            opacity: 0.3,
          }}
        />
      </Box>

      <Text
        size="xs"
        fw={700}
        mt={6}
        c={cp > 0 ? '#3fb950' : cp < 0 ? '#f85149' : '#8B949E'}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {cp > 0 ? '+' : ''}{cp.toFixed(1)}
      </Text>
    </Box>
  );
}
