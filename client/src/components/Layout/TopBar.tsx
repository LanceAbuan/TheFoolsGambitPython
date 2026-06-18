import { Group, Title, Badge, ActionIcon, Tooltip, Text } from '@mantine/core';
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
      ? 'green'
      : state.sseStatus === 'Connecting...'
      ? 'yellow'
      : 'red';

  return (
    <Group
      h="100%"
      px="md"
      justify="space-between"
      wrap="nowrap"
      style={{
        background: '#161B22',
        borderBottom: '1px solid #30363D',
      }}
    >
      <Group gap="sm">
        <Title
          order={4}
          style={{
            fontFamily: '"Inter", sans-serif',
            fontWeight: 800,
            background: 'linear-gradient(135deg, #58a6ff, #3fb950)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Fool's Gambit
        </Title>
        <Badge color={sseColor} variant="filled" size="sm" style={{ textTransform: 'none' }}>
          {state.sseStatus}
        </Badge>
        {state.whatHappening && (
          <Text size="xs" c="#6E7681" truncate="end" style={{ maxWidth: 300 }}>
            {state.whatHappening}
          </Text>
        )}
      </Group>

      <Group gap={2} wrap="nowrap">
        <Tooltip label="Start training">
          <ActionIcon onClick={handleStart} color="green" variant="subtle" size="md">
            <IconPlayerPlay size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Stop training">
          <ActionIcon onClick={handleStop} color="red" variant="subtle" size="md">
            <IconPlayerStop size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Reset training">
          <ActionIcon onClick={handleReset} variant="subtle" color="gray" size="md">
            <IconRefresh size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Flip board">
          <ActionIcon onClick={flipBoard} variant="subtle" color="gray" size="md">
            <IconRotate size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Fullscreen">
          <ActionIcon onClick={toggleFullscreen} variant="subtle" color="gray" size="md">
            <IconMaximize size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}
