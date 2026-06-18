import { ScrollArea, Text, Badge, Group } from '@mantine/core';
import { IconBroadcast } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import SectionCard from '../Layout/SectionCard';

function formatSSETime(ts: number): string {
  return new Date(ts).toLocaleTimeString();
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
    <SectionCard
      icon={<IconBroadcast size={16} color="#8B949E" />}
      title="Event Log"
      rightSection={
        <Badge size="sm" color="gray" variant="filled">
          {state.sseEvents.length}
        </Badge>
      }
    >
      <ScrollArea h={250} scrollbarSize={5}>
        {state.sseEvents.length === 0 ? (
          <Text size="xs" c="dimmed">No events yet</Text>
        ) : (
          [...state.sseEvents].reverse().map((ev) => (
            <Group
              key={ev.id}
              gap={6}
              p={3}
              wrap="nowrap"
              style={{
                borderBottom: '1px solid #21262D',
                fontSize: 11,
                lineHeight: 1.4,
              }}
            >
              <Text size="xs" c="#6E7681" component="span" style={{ whiteSpace: 'nowrap' }}>
                {formatSSETime(ev.timestamp)}
              </Text>
              <Badge size="xs" color="green" variant="filled" style={{ textTransform: 'none' }}>
                {ev.type}
              </Badge>
              <Text size="xs" c="#8B949E" component="span" truncate="end" style={{ flex: 1 }}>
                {formatSSEData(ev.data)}
              </Text>
            </Group>
          ))
        )}
      </ScrollArea>
    </SectionCard>
  );
}
