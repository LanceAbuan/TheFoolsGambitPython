import { Group, Title, Badge, ActionIcon, Tooltip, Text } from '@mantine/core';
import { IconRotate, IconMaximize } from '@tabler/icons-react';
import { useGame } from '../../GameContext';

export default function TopBar() {
  const { state, flipBoard, dispatch } = useGame();
  const handleFullscreen = () => dispatch({ type: 'TOGGLE_FULLSCREEN' });

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
        <Tooltip label="Flip board">
          <ActionIcon onClick={flipBoard} variant="subtle" color="gray" size="md">
            <IconRotate size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Fullscreen">
          <ActionIcon onClick={handleFullscreen} variant="subtle" color="gray" size="md">
            <IconMaximize size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}
