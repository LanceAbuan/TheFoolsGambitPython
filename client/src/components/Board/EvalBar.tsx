import { Box, Text } from '@mantine/core';
import { useGame } from '../../GameContext';

export default function EvalBar() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const cp = analysis?.evaluation ?? 0;

  // Normalize cp (-2000..+2000) to -1..1
  const t = Math.max(-1, Math.min(1, cp / 2000));

  // Divider position from bottom (0–100%)
  // cp > 0 (white winning) pushes divider up → more bottom fill
  const pct = Math.max(2, Math.min(98, 50 + t * 48));

  // Flip colors when board orientation is flipped
  const isNormal = state.boardOrientation === 'white';
  const bottomColor = isNormal ? '#ffffff' : '#000000';
  const topColor = isNormal ? '#000000' : '#ffffff';

  // When flipped, invert which side grows with advantage
  const bottomHeight = isNormal ? pct : 100 - pct;

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 36, flexShrink: 0, alignSelf: 'stretch' }}>
      <Box
        style={{
          position: 'relative',
          width: 28,
          flex: 1,
          background: '#30363D',
          borderRadius: 6,
          overflow: 'hidden',
          boxShadow: 'inset 0 0 6px rgba(0,0,0,0.6)',
        }}
      >
        {/* Bottom portion */}
        <Box
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: '100%',
            height: `${bottomHeight}%`,
            background: bottomColor,
            transition: 'height 0.3s ease',
          }}
        />

        {/* Top portion */}
        <Box
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: `${100 - bottomHeight}%`,
            background: topColor,
            transition: 'height 0.3s ease',
          }}
        />

        {/* Center divider */}
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
        c="#8B949E"
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {cp > 0 ? '+' : ''}{cp.toFixed(1)}
      </Text>
    </Box>
  );
}
