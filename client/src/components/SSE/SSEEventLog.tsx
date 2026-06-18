import { ScrollArea, Text, Group, Paper, Badge } from '@mantine/core';
import { IconBroadcast } from '@tabler/icons-react';
import { useGame } from '../../GameContext';

function formatSSETime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString();
}

function formatSSEData(data: unknown): string {
  if (!data) return '';
  if (typeof data === 'object') {
    try {
      return JSON.stringify(data).slice(0, 200);
    } catch {
      return String(data);
    }
  }
  return String(data).slice(0, 200);
}

export default function SSEEventLog() {
  const { state } = useGame();

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #30363D' }}>
      <Group gap="xs" mb="xs">
        <IconBroadcast size={16} color="#8B949E" />
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
          Event Log
        </Text>
        <Badge size="sm" color="gray" variant="filled" ml="auto">
          {state.sseEvents.length}
        </Badge>
      </Group>

      <ScrollArea h={250}>
        {state.sseEvents.length === 0 ? (
          <Text size="xs" c="dimmed">No events yet</Text>
        ) : (
          [...state.sseEvents].reverse().map((ev) => (
            <div key={ev.id} style={{ padding: '3px 6px', borderBottom: '1px solid #21262D', fontSize: 11, lineHeight: 1.4 }}>
              <span style={{ color: '#6E7681' }}>{formatSSETime(ev.timestamp)}</span>
              {' '}
              <Badge size="xs" color="green" variant="filled" style={{ textTransform: 'none' }}>
                {ev.type}
              </Badge>
              {' '}
              <span style={{ color: '#8B949E' }}>{formatSSEData(ev.data)}</span>
            </div>
          ))
        )}
      </ScrollArea>
    </Paper>
  );
}
