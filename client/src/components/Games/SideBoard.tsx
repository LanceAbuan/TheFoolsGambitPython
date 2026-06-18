import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';

interface Props {
  gameId: number;
}

export default function SideBoard({ gameId }: Props) {
  const { state } = useGame();
  const fen = state.sideFens[gameId];
  const moveCount = state.sideMoveCounts[gameId] || 0;

  return (
    <div className="side-board-item">
      <div className="board-label">Game {gameId}</div>
      <div className="side-board-wrapper">
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
          <div className="side-board-badge">{moveCount}</div>
        )}
      </div>
      {moveCount === 0 && (
        <div className="board-status">Waiting...</div>
      )}
    </div>
  );
}
