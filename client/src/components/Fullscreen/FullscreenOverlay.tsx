import { Modal, ActionIcon, Group } from '@mantine/core';
import { IconMinimize } from '@tabler/icons-react';
import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';
import PlayerInfoBar from '../Board/PlayerInfoBar';
import BoardNav from '../Board/BoardNav';
import EvalBar from '../Board/EvalBar';

export default function FullscreenOverlay() {
  const { state, dispatch } = useGame();
  const close = () => dispatch({ type: 'CLOSE_FULLSCREEN' });

  const pos = state.fenCache.length > 0
    ? state.fenCache[Math.min(state.currentViewIndex, state.fenCache.length - 1)]
    : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';

  return (
    <Modal
      opened={state.isFullscreen}
      onClose={close}
      fullScreen
      withCloseButton={false}
      styles={{
        body: {
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: 'calc(100vh - 60px)',
          background: '#312e22',
        },
      }}
    >
      <Group justify="flex-end" style={{ width: '100%', maxWidth: 672 }}>
        <ActionIcon onClick={close} variant="subtle" size="lg">
          <IconMinimize size={20} />
        </ActionIcon>
      </Group>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <EvalBar />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <PlayerInfoBar
            name="Self-Play (NN)"
            detail="vs Self"
            isTurn={state.allMoves.length % 2 !== 0}
            color="top"
          />
          <Chessboard
            options={{
              position: pos,
              boardOrientation: state.boardOrientation,
              animationDurationInMs: 300,
              showAnimations: true,
              allowDragging: false,
              boardStyle: { borderRadius: '4px', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' },
              darkSquareStyle: { backgroundColor: '#625b4d' },
              lightSquareStyle: { backgroundColor: '#b7b09c' },
            }}
          />
          <PlayerInfoBar
            name="Fool's Gambit AI"
            detail="Training"
            isTurn={state.allMoves.length % 2 === 0}
            color="bottom"
          />
          <div className="fullscreen-bar">
            <BoardNav />
          </div>
        </div>
      </div>
    </Modal>
  );
}
