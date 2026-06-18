import { Group, Text, Paper } from '@mantine/core';

interface Props {
  name: string;
  detail: string;
  isTurn: boolean;
  color: 'top' | 'bottom';
}

export default function PlayerInfoBar({ name, detail, isTurn, color }: Props) {
  return (
    <Paper
      p="xs"
      radius="sm"
      style={{
        maxWidth: 672,
        width: '100%',
        background: isTurn ? '#1c2333' : '#0D1117',
        border: isTurn ? '1px solid #58a6ff' : '1px solid #21262D',
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      <Group gap="sm" justify={color === 'bottom' ? 'flex-start' : 'flex-end'} style={{ width: '100%' }}>
        {color === 'bottom' && (
          <>
            {/* Glowing turn indicator */}
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: isTurn ? '#58a6ff' : '#30363D',
                boxShadow: isTurn ? '0 0 8px rgba(88, 166, 255, 0.6)' : 'none',
                transition: 'all 0.3s',
                flexShrink: 0,
              }}
            />
            <div>
              <Text size="sm" fw={700} c={isTurn ? '#58a6ff' : '#C9D1D9'}>
                ♙ {name}
              </Text>
              <Text size="xs" c="#8B949E">{detail}</Text>
            </div>
          </>
        )}
        {color === 'top' && (
          <>
            <div style={{ textAlign: 'right' }}>
              <Text size="sm" fw={700} c={isTurn ? '#58a6ff' : '#C9D1D9'}>
                {name} ♟
              </Text>
              <Text size="xs" c="#8B949E">{detail}</Text>
            </div>
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: isTurn ? '#58a6ff' : '#30363D',
                boxShadow: isTurn ? '0 0 8px rgba(88, 166, 255, 0.6)' : 'none',
                transition: 'all 0.3s',
                flexShrink: 0,
              }}
            />
          </>
        )}
      </Group>
    </Paper>
  );
}
