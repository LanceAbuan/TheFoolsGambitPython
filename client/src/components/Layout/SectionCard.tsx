import { Paper, Group, Text } from '@mantine/core';
import type { ReactNode } from 'react';

interface SectionCardProps {
  icon?: ReactNode;
  title: string;
  children: ReactNode;
  rightSection?: ReactNode;
}

/** Reusable wrapper for consistent sidebar panel styling. */
export default function SectionCard({ icon, title, children, rightSection }: SectionCardProps) {
  return (
    <Paper
      p="md"
      radius="md"
      style={{ background: '#161B22', border: '1px solid #30363D' }}
    >
      <Group gap="xs" mb="xs">
        {icon}
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
          {title}
        </Text>
        {rightSection && <div style={{ marginLeft: 'auto' }}>{rightSection}</div>}
      </Group>
      {children}
    </Paper>
  );
}
