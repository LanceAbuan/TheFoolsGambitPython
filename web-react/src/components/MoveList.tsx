import { ScrollArea, Stack, Text } from '@mantine/core';

interface MoveListProps {
  sans: string[];
  height?: number;
}

/** Two-column move ledger in the classical score-sheet shape (1. e4 e5). */
export function MoveList({ sans, height = 220 }: MoveListProps) {
  const rows: { no: number; w?: string; b?: string; cur: 'w' | 'b' | null }[] = [];
  for (let i = 0; i < sans.length; i += 2) {
    rows.push({
      no: i / 2 + 1,
      w: sans[i],
      b: sans[i + 1],
      cur: sans.length - 1 === i ? 'w' : sans.length - 1 === i + 1 ? 'b' : null,
    });
  }

  if (rows.length === 0) {
    return (
      <Stack gap={2}>
        <Text fz="sm" c="dimmed" className="mono">
          01. — —
        </Text>
        <Text fz="xs" c="dimmed">
          Your story has yet to unfold.
        </Text>
      </Stack>
    );
  }

  return (
    <ScrollArea h={height} type="auto" offsetScrollbars>
      <Stack gap={2}>
        {rows.map((r) => (
          <Text key={r.no} fz="sm" className="mono move-row" component="div">
            <span className="move-no">{String(r.no).padStart(2, '0')}.</span>
            <b className={r.cur === 'w' ? 'move-cur' : undefined}>{r.w}</b>
            <b className={r.cur === 'b' ? 'move-cur' : undefined}>{r.b ?? ''}</b>
          </Text>
        ))}
      </Stack>
    </ScrollArea>
  );
}
