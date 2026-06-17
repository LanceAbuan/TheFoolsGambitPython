import { ScrollArea, Text } from '@mantine/core';
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
    <div>
      <div className="section-header">
        Event Log ({state.sseEvents.length})
      </div>
      <ScrollArea h={250}>
        {state.sseEvents.length === 0 ? (
          <Text size="xs" c="dimmed">No events yet</Text>
        ) : (
          state.sseEvents.map((ev) => (
            <div key={ev.id} className="sse-entry">
              <span className="sse-time">{formatSSETime(ev.timestamp)}</span>
              {' '}
              <span className="sse-type">{ev.type}</span>
              {' '}
              <span className="sse-data">{formatSSEData(ev.data)}</span>
            </div>
          ))
        )}
      </ScrollArea>
    </div>
  );
}
