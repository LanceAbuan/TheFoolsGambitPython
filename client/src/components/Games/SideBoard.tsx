import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';

interface Props {
  gameId: number;
}

export default function SideBoard({ gameId }: Props) {
  const { state } = useGame();
  const fen = state.sideFens[gameId];
  const moveCount = state.sideMoveCounts[gameId] || 0;
  const boardKey = `${gameId}-${moveCount}-${(fen || 'start').slice(0, 8)}`;

  return (
    <div className="side-board-item">
      <div className="board-label">Game {gameId}</div>
      <Chessboard
        key={boardKey}
        options={{
          position: fen || 'start',
          animationDurationInMs: 200,
          showAnimations: true,
          allowDragging: false,
          boardStyle: { borderRadius: '2px' },
          darkSquareStyle: { backgroundColor: '#625b4d' },
          lightSquareStyle: { backgroundColor: '#b7b09c' },
        }}
      />
      <div className="board-status">
        {moveCount > 0 ? `${moveCount} moves` : 'Waiting...'}
      </div>
    </div>
  );
}
