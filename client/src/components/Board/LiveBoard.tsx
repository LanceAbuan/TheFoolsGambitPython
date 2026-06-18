import { useCallback } from 'react';
import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';

const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';

export default function LiveBoard() {
  const { state } = useGame();

  const getPosition = useCallback((): string => {
    if (state.fenCache.length > 0) {
      const idx = Math.min(state.currentViewIndex, state.fenCache.length - 1);
      if (state.fenCache[idx] && state.fenCache[idx] !== '') {
        return state.fenCache[idx];
      }
    }
    return STARTING_FEN;
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
          darkSquareStyle: { backgroundColor: '#625b4d' },
          lightSquareStyle: { backgroundColor: '#b7b09c' },
        }}
      />
    </div>
  );
}
