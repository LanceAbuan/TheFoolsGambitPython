import { Table, Text, Badge } from '@mantine/core';
import { IconChartBar } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { MOVE_COLORS, MOVE_LABELS } from '../../types';
import SectionCard from '../Layout/SectionCard';

export default function AnalysisTable() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const rows = analysis?.move_analysis;

  return (
    <SectionCard icon={<IconChartBar size={16} color="#8B949E" />} title="Analysis">
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
              <Table.Th>Quality</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((row: any, i: number) => (
              <Table.Tr key={i}>
                <Table.Td>{i + 1}</Table.Td>
                <Table.Td style={{ fontFamily: 'monospace' }}>{row.san || row.move}</Table.Td>
                <Table.Td>{row.evaluation != null ? (row.evaluation / 100).toFixed(2) : '-'}</Table.Td>
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
    </SectionCard>
  );
}
