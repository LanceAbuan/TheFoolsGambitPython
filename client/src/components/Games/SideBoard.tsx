import { Chessboard } from 'react-chessboard';
import { Paper, Text, Badge } from '@mantine/core';
import { useGame } from '../../GameContext';

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
      <Text size="xs" fw={600} c="#6E7681" tt="uppercase" mb={6} style={{ letterSpacing: '0.3px' }}>
        Game {gameId}
      </Text>

      <div style={{ position: 'relative' }}>
        <Chessboard
          options={{
            position: fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
            animationDurationInMs: 200,
            showAnimations: true,
            allowDragging: false,
            boardStyle: { borderRadius: '2px' },
            darkSquareStyle: { backgroundColor: '#625b4d' },
            lightSquareStyle: { backgroundColor: '#b7b09c' },
          }}
        />
        {moveCount > 0 && (
          <Badge
            size="sm"
            color="green"
            variant="filled"
            style={{
              position: 'absolute',
              top: 4,
              right: 4,
              pointerEvents: 'none',
              zIndex: 10,
              backdropFilter: 'blur(2px)',
            }}
          >
            {moveCount}
          </Badge>
        )}
      </div>

      {moveCount === 0 && (
        <Text size="xs" c="#6E7681" mt={4}>Waiting...</Text>
      )}
    </Paper>
  );
}
