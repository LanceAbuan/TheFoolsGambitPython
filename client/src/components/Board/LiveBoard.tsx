import { useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';
import { BOARD_COLORS, DEFAULT_FEN } from '../../utils/board';

export default function LiveBoard() {
  const { state } = useGame();

  const getPosition = useCallback((): string => {
    if (state.fenCache.length > 0) {
      const idx = Math.min(state.currentViewIndex, state.fenCache.length - 1);
      if (state.fenCache[idx] && state.fenCache[idx] !== '') {
        return state.fenCache[idx];
      }
    }
    return DEFAULT_FEN;
  }, [state.fenCache, state.currentViewIndex]);

  return (
    <div id="live-board">
      <Chessboard
        options={{
          position: getPosition(),
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
