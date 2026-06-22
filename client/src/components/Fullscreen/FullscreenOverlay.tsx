import { Modal, ActionIcon, Group, Paper } from '@mantine/core';
import { IconMinimize } from '@tabler/icons-react';
import { Chessboard } from 'react-chessboard';
import { useGame } from '../../GameContext';
import { BOARD_COLORS } from '../../utils/board';
import PlayerInfoBar from '../Board/PlayerInfoBar';
import BoardNav from '../Board/BoardNav';
import EvalBar from '../Board/EvalBar';

export default function FullscreenOverlay() {
  const { state, dispatch, getCurrentFen } = useGame();
  const close = () => dispatch({ type: 'CLOSE_FULLSCREEN' });
  const s = state.trainingStatus;

  const isMainGame = state.selectedGameId === 0;
  const whiteToMove = isMainGame
    ? state.allMoves.length % 2 === 0
    : (state.sideMoveCounts[state.selectedGameId] || 0) % 2 === 0;

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
            name="Neural Network"
            detail={isMainGame && s?.status === 'critic' ? 'vs Stockfish' : 'vs Self'}
            isTurn={whiteToMove}
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
                position: getCurrentFen(),
                boardOrientation: state.boardOrientation,
                animationDurationInMs: 300,
                showAnimations: true,
                allowDragging: false,
                boardStyle: { borderRadius: '4px', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' },
                darkSquareStyle: { backgroundColor: BOARD_COLORS.dark },
                lightSquareStyle: { backgroundColor: BOARD_COLORS.light },
              }}
            />
          </Paper>
          <PlayerInfoBar
            name={isMainGame && s?.status === 'critic' ? 'Stockfish' : 'Neural Network'}
            detail={isMainGame && s?.status === 'critic' ? 'Engine' : `Step ${(s?.step ?? 0).toLocaleString()}`}
            isTurn={!whiteToMove}
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
