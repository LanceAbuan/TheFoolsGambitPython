import { Group, Text } from '@mantine/core';

interface Props {
  name: string;
  detail: string;
  isTurn: boolean;
  color: 'top' | 'bottom';
}

export default function PlayerInfoBar({ name, detail, isTurn, color }: Props) {
  return (
    <div className={`player-info-bar${isTurn ? ' on-turn' : ''}`} style={{ maxWidth: 672, width: '100%', padding: '6px 10px' }}>
      <Group gap={6} justify={color === 'bottom' ? 'flex-start' : 'flex-end'} style={{ width: '100%' }}>
        {color === 'bottom' ? (
          <>
            <div className="turn-dot" />
            <div>
              <Text size="sm" fw={700}>{name}</Text>
              <Text size="xs" c="dimmed">{detail}</Text>
            </div>
          </>
        ) : (
          <>
            <div style={{ textAlign: 'right' }}>
              <Text size="sm" fw={700}>{name}</Text>
              <Text size="xs" c="dimmed">{detail}</Text>
            </div>
            <div className="turn-dot" />
          </>
        )}
      </Group>
    </div>
  );
}
