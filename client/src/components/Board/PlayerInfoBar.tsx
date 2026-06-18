import { Group, Text, Paper, Indicator } from '@mantine/core';

interface Props {
  name: string;
  detail: string;
  isTurn: boolean;
  color: 'top' | 'bottom';
}

export default function PlayerInfoBar({ name, detail, isTurn, color }: Props) {
  const turnColor = '#58a6ff';
  const idleColor = '#30363D';

  return (
    <Paper
      p="xs"
      radius="sm"
      style={{
        maxWidth: 672,
        width: '100%',
        background: isTurn ? '#1c2333' : '#0D1117',
        border: isTurn ? `1px solid ${turnColor}` : '1px solid #21262D',
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      <Group gap="sm" justify={color === 'bottom' ? 'flex-start' : 'flex-end'} style={{ width: '100%' }}>
        {color === 'bottom' && (
          <>
            <Indicator
              size={10}
              color={isTurn ? turnColor : idleColor}
              withBorder={false}
              processing={isTurn}
            />
            <div>
              <Text size="sm" fw={700} c={isTurn ? turnColor : '#C9D1D9'}>
                ♙ {name}
              </Text>
              <Text size="xs" c="#8B949E">{detail}</Text>
            </div>
          </>
        )}
        {color === 'top' && (
          <>
            <div style={{ textAlign: 'right' }}>
              <Text size="sm" fw={700} c={isTurn ? turnColor : '#C9D1D9'}>
                {name} ♟
              </Text>
              <Text size="xs" c="#8B949E">{detail}</Text>
            </div>
            <Indicator
              size={10}
              color={isTurn ? turnColor : idleColor}
              withBorder={false}
              processing={isTurn}
            />
          </>
        )}
      </Group>
    </Paper>
  );
}
