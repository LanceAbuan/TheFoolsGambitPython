import { Group, Title, Badge, ActionIcon, Tooltip, Text } from '@mantine/core';
import { IconRotate, IconMaximize, IconBook, IconLayoutSidebarRight } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { Link } from 'react-router-dom';

interface TopBarProps {
  onToggleSidebar?: () => void;
}

export default function TopBar({ onToggleSidebar }: TopBarProps) {
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
        <Link to="/docs" style={{ textDecoration: 'none' }}>
          <Group gap={2} style={{ color: '#c9d1d9' }}>
            <ActionIcon variant="subtle" color="gray" size="md">
              <IconBook size={18} />
            </ActionIcon>
            <Text size="sm" fw={500}>Docs</Text>
          </Group>
        </Link>
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
        <Tooltip label="Toggle stats panel">
          <ActionIcon onClick={onToggleSidebar} variant="subtle" color="gray" size="md">
            <IconLayoutSidebarRight size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}
