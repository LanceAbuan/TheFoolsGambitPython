import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';
import { BOARD_COLORS } from '../../utils/board';

export default function LiveBoard() {
  const { state, getCurrentFen } = useGame();

  return (
    <div id="live-board">
      <Chessboard
        options={{
          position: getCurrentFen(),
          boardOrientation: state.boardOrientation,
          animationDurationInMs: 300,
          showAnimations: true,
          allowDragging: false,
          boardStyle: {
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
          },
          darkSquareStyle: { backgroundColor: BOARD_COLORS.dark },
          lightSquareStyle: { backgroundColor: BOARD_COLORS.light },
        }}
      />
    </div>
  );
}
