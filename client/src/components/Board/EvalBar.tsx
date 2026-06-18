import { Box, Text, useMantineTheme } from '@mantine/core';
import { useGame } from '../../GameContext';

export default function EvalBar() {
  const { state } = useGame();
  const theme = useMantineTheme();
  const analysis = state.analysis as any;
  const cp = analysis?.evaluation ?? 0;

  // cp in centipawns, typically -500..+500 for normal positions
  // normalize to -1..1, then map to percentage from bottom
  const t = Math.max(-1, Math.min(1, cp / 2000));
  // White advantage pushes white fill up from center
  const pct = Math.max(2, Math.min(98, 50 - t * 48));

  // Colors using Mantine theme palette
  const whiteFill = theme.colors.gray[0];   // light for white's advantage
  const blackFill = theme.colors.dark[8];    // dark for black's advantage
  const trackBg = theme.colors.dark[6];
  const dividerColor = theme.colors.dark[3];
  const textColor = theme.colors.gray[5];

  // For the top portion (white advantage) — gradient from light to medium
  const topColor = cp > 0 ? whiteFill : trackBg;
  // For the bottom portion (black advantage) — gradient from dark to medium
  const bottomColor = cp < 0 ? blackFill : trackBg;

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 28, flexShrink: 0 }}>
      <Box
        style={{
          position: 'relative',
          width: 20,
          height: 400,
          background: trackBg,
          borderRadius: theme.radius.sm,
          overflow: 'hidden',
          boxShadow: 'inset 0 0 4px rgba(0,0,0,0.5)',
        }}
      >
        {/* Top fill (white advantage zone) — extends downward from top */}
        <Box
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: `${100 - pct}%`,
            background: `linear-gradient(to bottom, ${topColor}, ${theme.colors.dark[5]})`,
            transition: 'height 0.3s ease, background 0.3s ease',
          }}
        />
        {/* Bottom fill (black advantage zone) — extends upward from bottom */}
        <Box
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: '100%',
            height: `${pct}%`,
            background: `linear-gradient(to top, ${bottomColor}, ${theme.colors.dark[5]})`,
            transition: 'height 0.3s ease, background 0.3s ease',
          }}
        />
        {/* Center divider line */}
        <Box
          style={{
            position: 'absolute',
            top: '50%',
            left: 0,
            width: '100%',
            height: 2,
            background: dividerColor,
            zIndex: 1,
          }}
        />
      </Box>
      <Text
        size="xs"
        c={textColor}
        fw={600}
        mt={6}
        style={{ fontVariantNumeric: 'tabular-nums' }}
      >
        {cp > 0 ? '+' : ''}{cp.toFixed(1)}
      </Text>
    </Box>
  );
}
