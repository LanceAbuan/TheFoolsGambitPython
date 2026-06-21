import { useState } from 'react';
import { ActionIcon, Tooltip, Text } from '@mantine/core';
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';

interface CollapsiblePanelProps {
  title?: string;
  defaultCollapsed?: boolean;
  width?: number;
  collapsedWidth?: number;
  children: React.ReactNode;
}

export default function CollapsiblePanel({
  title,
  defaultCollapsed = false,
  width = 240,
  collapsedWidth = 0,
  children,
}: CollapsiblePanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div
      style={{
        position: 'relative',
        width: collapsed ? collapsedWidth : width,
        flexShrink: 0,
        transition: 'width 0.2s ease',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        background: '#161B22',
        borderRight: '1px solid #30363D',
      }}
    >
      {!collapsed && (
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid #30363D',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          {title && (
            <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
              {title}
            </Text>
          )}
        </div>
      )}

      <div
        style={{
          flex: 1,
          overflow: 'auto',
          padding: collapsed ? 0 : undefined,
        }}
      >
        {!collapsed && children}
      </div>

      {/* Collapse toggle button */}
      <Tooltip label={collapsed ? 'Expand' : 'Collapse'}>
        <ActionIcon
          variant="subtle"
          color="gray"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          style={{
            position: 'absolute',
            top: 8,
            right: collapsed ? 4 : 8,
            zIndex: 10,
            background: '#21262D',
            border: '1px solid #30363D',
          }}
        >
          {collapsed ? <IconChevronRight size={14} /> : <IconChevronLeft size={14} />}
        </ActionIcon>
      </Tooltip>
    </div>
  );
}
