import { Table, Text, Group, Paper, Badge } from '@mantine/core';
import { IconChartBar } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { MOVE_COLORS, MOVE_LABELS } from '../../types';

export default function AnalysisTable() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const rows = analysis?.move_analysis;

  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #30363D' }}>
      <Group gap="xs" mb="xs">
        <IconChartBar size={16} color="#8B949E" />
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
          Analysis
        </Text>
      </Group>

      {!rows || !rows.length ? (
        <Text size="sm" c="dimmed">
          {state.isAnalyzing ? 'Analyzing...' : 'Waiting for position...'}
        </Text>
      ) : (
        <Table>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>#</Table.Th>
              <Table.Th>Move</Table.Th>
              <Table.Th>Eval</Table.Th>
              <Table.Th>Type</Table.Th>
              <Table.Th>Quality</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((row: any, i: number) => (
              <Table.Tr key={i}>
                <Table.Td>{i + 1}</Table.Td>
                <Table.Td style={{ fontFamily: 'monospace' }}>{row.san || row.move}</Table.Td>
                <Table.Td>{row.evaluation != null ? row.evaluation.toFixed(1) : '-'}</Table.Td>
                <Table.Td>{row.type || '-'}</Table.Td>
                <Table.Td>
                  {row.quality && (
                    <Badge
                      size="sm"
                      color={MOVE_COLORS[row.quality] || 'gray'}
                      variant="filled"
                      autoContrast
                    >
                      {MOVE_LABELS[row.quality] || row.quality}
                    </Badge>
                  )}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}
    </Paper>
  );
}
