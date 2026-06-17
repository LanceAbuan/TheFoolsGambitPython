import { Table, Text } from '@mantine/core';
import { useGame } from '../../GameContext';
import { MOVE_COLORS, MOVE_LABELS } from '../../types';

export default function AnalysisTable() {
  const { state } = useGame();
  const analysis = state.analysis as any;
  const rows = analysis?.move_analysis;

  if (!rows || !rows.length) {
    return (
      <div>
        <div className="section-header">Analysis</div>
        <Text size="sm" c="dimmed">
          {state.isAnalyzing ? 'Analyzing...' : 'Waiting for position...'}
        </Text>
      </div>
    );
  }

  return (
    <div>
      <div className="section-header">Analysis</div>
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
              <Table.Td>{row.eval || '-'}</Table.Td>
              <Table.Td>{row.type || '-'}</Table.Td>
              <Table.Td>
                {row.quality && (
                  <Text size="xs" style={{ color: MOVE_COLORS[row.quality] || '#fff' }}>
                    {MOVE_LABELS[row.quality] || row.quality}
                  </Text>
                )}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </div>
  );
}
