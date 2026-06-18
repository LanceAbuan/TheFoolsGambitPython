import { Modal, ActionIcon, Group, Paper } from '@mantine/core';
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
          background: '#0D1117',
        },
      }}
    >
      <Group justify="flex-end" style={{ width: '100%', maxWidth: 672 }}>
        <ActionIcon onClick={close} variant="subtle" color="gray" size="lg">
          <IconMinimize size={20} />
        </ActionIcon>
      </Group>

      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, maxWidth: 672, width: '100%' }}>
        <EvalBar />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, minWidth: 0 }}>
          <PlayerInfoBar
            name="Self-Play (NN)"
            detail="vs Self"
            isTurn={state.allMoves.length % 2 !== 0}
            color="top"
          />
          <Paper
            p="sm"
            radius="md"
            mt={4}
            mb={4}
            style={{ background: '#161B22', border: '1px solid #30363D' }}
          >
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
          </Paper>
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
