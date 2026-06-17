import { Group, Title, Badge, ActionIcon, Tooltip } from '@mantine/core';
import { IconPlayerPlay, IconPlayerStop, IconRefresh, IconRotate, IconMaximize } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { api } from '../../api';

export default function TopBar() {
  const { state, dispatch, flipBoard } = useGame();

  const handleStart = async () => {
    try {
      await api.startTraining();
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Training started' });
    } catch {
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Failed to start training' });
    }
  };

  const handleStop = async () => {
    try {
      await api.stopTraining();
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Training stopped' });
    } catch {
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Failed to stop training' });
    }
  };

  const handleReset = async () => {
    try {
      await api.resetTraining();
      dispatch({ type: 'RESET_MOVES' });
      dispatch({ type: 'SET_FEN_CACHE', cache: [] });
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Training reset' });
    } catch {
      dispatch({ type: 'SET_WHAT_HAPPENING', text: 'Failed to reset training' });
    }
  };

  const toggleFullscreen = () => dispatch({ type: 'TOGGLE_FULLSCREEN' });

  const sseColor =
    state.sseStatus === 'Connected'
      ? 'accent-green'
      : state.sseStatus === 'Connecting...'
      ? 'accent-yellow'
      : 'accent-red';

  return (
    <Group h="100%" px="md" justify="space-between" wrap="nowrap">
      <Group gap="xs">
        <Title order={4} style={{ fontFamily: '"Inter", sans-serif', fontWeight: 800 }}>
          Fool's Gambit
        </Title>
        <Badge color={sseColor} variant="filled" size="sm">
          {state.sseStatus}
        </Badge>
      </Group>

      <Group gap={4} wrap="nowrap">
        <Tooltip label="Start training">
          <ActionIcon onClick={handleStart} color="accent-green" variant="subtle">
            <IconPlayerPlay size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Stop training">
          <ActionIcon onClick={handleStop} color="accent-red" variant="subtle">
            <IconPlayerStop size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Reset training">
          <ActionIcon onClick={handleReset} variant="subtle">
            <IconRefresh size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Flip board">
          <ActionIcon onClick={flipBoard} variant="subtle">
            <IconRotate size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Fullscreen">
          <ActionIcon onClick={toggleFullscreen} variant="subtle">
            <IconMaximize size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}
