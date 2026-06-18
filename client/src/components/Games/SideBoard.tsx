import { Chessboard } from 'react-chessboard';
import { Paper, Text, Badge, Group } from '@mantine/core';
import { useGame } from '../../GameContext';
import { BOARD_COLORS, DEFAULT_FEN } from '../../utils/board';

interface Props {
  gameId: number;
}

export default function SideBoard({ gameId }: Props) {
  const { state } = useGame();
  const fen = state.sideFens[gameId];
  const moveCount = state.sideMoveCounts[gameId] || 0;

  return (
    <Paper
      p="sm"
      radius="sm"
      style={{
        background: '#161B22',
        border: '1px solid #21262D',
        textAlign: 'center',
        position: 'relative',
      }}
    >
      <Group gap="xs" mb={6} justify="space-between" wrap="nowrap">
        <Text size="xs" fw={600} c="#6E7681" tt="uppercase" style={{ letterSpacing: '0.3px' }}>
          Game {gameId}
        </Text>
        {moveCount > 0 && (
          <Badge size="sm" color="green" variant="filled">
            Move {moveCount}
          </Badge>
        )}
      </Group>

      <div>
        <Chessboard
          options={{
            position: fen || DEFAULT_FEN,
            animationDurationInMs: 200,
            showAnimations: true,
            allowDragging: false,
            boardStyle: { borderRadius: '2px' },
            darkSquareStyle: { backgroundColor: BOARD_COLORS.dark },
            lightSquareStyle: { backgroundColor: BOARD_COLORS.light },
          }}
        />
      </div>

      {moveCount === 0 && (
        <Text size="xs" c="#6E7681" mt={4}>Waiting...</Text>
      )}
    </Paper>
  );
}
