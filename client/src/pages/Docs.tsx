import { Container, Group, Text, ActionIcon } from '@mantine/core';
import { IconArrowLeft } from '@tabler/icons-react';
import { Link } from 'react-router-dom';
import Readme from '../components/Docs/Readme';

export default function Docs() {
  return (
    <div style={{ background: '#0D1117', minHeight: '100vh' }}>
      {/* Header bar */}
      <Group
        px="md"
        h={50}
        style={{
          background: '#161B22',
          borderBottom: '1px solid #30363D',
        }}
      >
        <Link to="/" style={{ textDecoration: 'none' }}>
          <Group gap={4} style={{ color: '#c9d1d9' }}>
            <ActionIcon variant="subtle" color="gray" size="md">
              <IconArrowLeft size={18} />
            </ActionIcon>
            <Text size="sm" fw={500}>Dashboard</Text>
          </Group>
        </Link>
      </Group>

      {/* Content */}
      <Readme />
    </div>
  );
}
