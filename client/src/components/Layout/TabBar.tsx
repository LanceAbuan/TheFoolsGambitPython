import { Group, Text, ActionIcon, Tooltip } from '@mantine/core';
import { IconSettings, IconRefresh } from '@tabler/icons-react';

export type TabId = 'games' | 'analysis' | 'training' | 'logs';

interface Tab {
  id: TabId;
  label: string;
}

const tabs: Tab[] = [
  { id: 'games', label: 'Games' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'training', label: 'Training' },
  { id: 'logs', label: 'Logs' },
];

interface TabBarProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export default function TabBar({ activeTab, onTabChange }: TabBarProps) {
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
      {/* Left: title */}
      <Text
        fw={800}
        size="lg"
        style={{
          fontFamily: '"Inter", sans-serif',
          background: 'linear-gradient(135deg, #58a6ff, #3fb950)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          whiteSpace: 'nowrap',
          minWidth: 140,
        }}
      >
        Fool's Gambit
      </Text>

      {/* Center: tabs */}
      <Group gap={0} justify="center" style={{ flex: 1 }}>
        {tabs.map((tab) => (
          <Group
            key={tab.id}
            px="lg"
            py="xs"
            style={{
              cursor: 'pointer',
              borderBottom: activeTab === tab.id ? '2px solid #58a6ff' : '2px solid transparent',
              transition: 'all 0.15s ease',
              background: activeTab === tab.id ? 'rgba(88, 166, 255, 0.08)' : 'transparent',
              borderRadius: '4px 4px 0 0',
            }}
            onClick={() => onTabChange(tab.id)}
          >
            <Text
              size="sm"
              fw={activeTab === tab.id ? 600 : 400}
              c={activeTab === tab.id ? '#58a6ff' : '#8B949E'}
              style={{ transition: 'color 0.15s ease' }}
            >
              {tab.label}
            </Text>
          </Group>
        ))}
      </Group>

      {/* Right: controls */}
      <Group gap={4} wrap="nowrap" style={{ minWidth: 140 }} justify="flex-end">
        <Tooltip label="Refresh">
          <ActionIcon variant="subtle" color="gray" size="md">
            <IconRefresh size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Settings">
          <ActionIcon variant="subtle" color="gray" size="md">
            <IconSettings size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Group>
  );
}
