import { Group, Text, Paper, Badge, TextInput, ActionIcon, Table, Tooltip } from '@mantine/core';
import { IconSearch, IconPlayerPause, IconFilter } from '@tabler/icons-react';
import { useGame } from '../GameContext';
import { useState, useMemo } from 'react';

type LogLevel = 'info' | 'warning' | 'error';
type LogCategory = 'all' | 'system' | 'training' | 'games' | 'network' | 'checkpoints' | 'errors' | 'warnings';

interface LogEntry {
  id: number;
  time: string;
  level: LogLevel;
  category: LogCategory;
  message: string;
  details: string;
}

const categories: { id: LogCategory; label: string }[] = [
  { id: 'all', label: 'All Logs' },
  { id: 'system', label: 'System' },
  { id: 'training', label: 'Training' },
  { id: 'games', label: 'Games' },
  { id: 'network', label: 'Network' },
  { id: 'checkpoints', label: 'Checkpoints' },
  { id: 'errors', label: 'Errors' },
  { id: 'warnings', label: 'Warnings' },
];

function getCategoryFromType(type: string): LogCategory {
  if (type.includes('error')) return 'errors';
  if (type.includes('warning')) return 'warnings';
  if (type.includes('game')) return 'games';
  if (type.includes('train')) return 'training';
  if (type.includes('connect') || type.includes('sse')) return 'network';
  if (type.includes('checkpoint')) return 'checkpoints';
  return 'system';
}

function getLevelFromType(type: string): LogLevel {
  if (type.includes('error')) return 'error';
  if (type.includes('warning')) return 'warning';
  return 'info';
}

function LogCategories({ active, onSelect }: { active: LogCategory; onSelect: (cat: LogCategory) => void }) {
  return (
    <Paper p="md" radius="md" style={{ background: '#161B22', border: '1px solid #21262D' }}>
      <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 12 }}>
        Log Categories
      </Text>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {categories.map((cat) => (
          <Group
            key={cat.id}
            px="sm"
            py={6}
            gap="xs"
            style={{
              borderRadius: 6,
              cursor: 'pointer',
              background: active === cat.id ? 'rgba(88, 166, 255, 0.1)' : 'transparent',
              border: active === cat.id ? '1px solid rgba(88, 166, 255, 0.3)' : '1px solid transparent',
              transition: 'all 0.15s ease',
            }}
            onClick={() => onSelect(cat.id)}
          >
            <IconFilter size={14} color={active === cat.id ? '#58a6ff' : '#6E7681'} />
            <Text size="sm" c={active === cat.id ? '#58a6ff' : '#C9D1D9'}>
              {cat.label}
            </Text>
          </Group>
        ))}
      </div>
    </Paper>
  );
}

export default function LogsView() {
  const { state } = useGame();
  const [activeCategory, setActiveCategory] = useState<LogCategory>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isPaused, setIsPaused] = useState(false);

  // Convert SSE events to log entries
  const logs: LogEntry[] = useMemo(() => {
    return state.sseEvents.map((ev: { id: number; type: string; data: unknown; timestamp: number }) => ({
      id: ev.id,
      time: new Date(ev.timestamp).toLocaleTimeString(),
      level: getLevelFromType(ev.type),
      category: getCategoryFromType(ev.type),
      message: ev.type.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
      details: typeof ev.data === 'object' ? JSON.stringify(ev.data).slice(0, 150) : String(ev.data).slice(0, 150),
    }));
  }, [state.sseEvents]);

  // Filter logs
  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      if (activeCategory !== 'all' && log.category !== activeCategory) return false;
      if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !log.details.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    });
  }, [logs, activeCategory, searchQuery]);

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Left: Categories */}
      <div
        style={{
          width: 200,
          flexShrink: 0,
          padding: 16,
          borderRight: '1px solid #30363D',
          background: '#161B22',
        }}
      >
        <LogCategories active={activeCategory} onSelect={setActiveCategory} />
      </div>

      {/* Right: Log table */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header with search and controls */}
        <Group px="md" py="sm" justify="space-between" style={{ borderBottom: '1px solid #30363D', background: '#161B22' }}>
          <Group gap="sm" style={{ flex: 1 }}>
            <TextInput
              placeholder="Search logs..."
              size="sm"
              leftSection={<IconSearch size={14} />}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.currentTarget.value)}
              style={{ flex: 1, maxWidth: 300 }}
              styles={{
                input: {
                  background: '#0D1117',
                  border: '1px solid #30363D',
                  color: '#C9D1D9',
                },
              }}
            />
            <Text size="xs" c="#6E7681">
              {filteredLogs.length} entries
            </Text>
          </Group>

          <Group gap="xs">
            <Tooltip label={isPaused ? 'Resume' : 'Pause'}>
              <ActionIcon
                variant={isPaused ? 'filled' : 'subtle'}
                color={isPaused ? 'yellow' : 'gray'}
                size="md"
                onClick={() => setIsPaused(!isPaused)}
              >
                <IconPlayerPause size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>

        {/* Table */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          <Table>
            <Table.Thead>
              <Table.Tr style={{ background: '#0D1117' }}>
                <Table.Th style={{ color: '#8B949E', fontWeight: 600, fontSize: 11 }}>TIME</Table.Th>
                <Table.Th style={{ color: '#8B949E', fontWeight: 600, fontSize: 11 }}>LEVEL</Table.Th>
                <Table.Th style={{ color: '#8B949E', fontWeight: 600, fontSize: 11 }}>MESSAGE</Table.Th>
                <Table.Th style={{ color: '#8B949E', fontWeight: 600, fontSize: 11 }}>DETAILS</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {filteredLogs.length === 0 ? (
                <Table.Tr>
                  <Table.Td colSpan={4}>
                    <Text size="sm" c="#6E7681" ta="center" py="md">
                      No logs found
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ) : (
                filteredLogs.map((log) => (
                  <Table.Tr
                    key={log.id}
                    style={{
                      borderBottom: '1px solid #21262D',
                      transition: 'background 0.1s',
                    }}
                  >
                    <Table.Td style={{ color: '#6E7681', fontSize: 12, fontFamily: 'ui-monospace, monospace', whiteSpace: 'nowrap' }}>
                      {log.time}
                    </Table.Td>
                    <Table.Td>
                      <Badge
                        size="sm"
                        color={log.level === 'info' ? 'blue' : log.level === 'warning' ? 'yellow' : 'red'}
                        variant="filled"
                        style={{ textTransform: 'none', minWidth: 50, justifyContent: 'center' }}
                      >
                        {log.level.toUpperCase()}
                      </Badge>
                    </Table.Td>
                    <Table.Td style={{ color: '#C9D1D9', fontSize: 13 }}>
                      {log.message}
                    </Table.Td>
                    <Table.Td style={{ color: '#8B949E', fontSize: 12, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {log.details}
                    </Table.Td>
                  </Table.Tr>
                ))
              )}
            </Table.Tbody>
          </Table>
        </div>

        {/* Footer */}
        <Group px="md" py="sm" justify="space-between" style={{ borderTop: '1px solid #30363D', background: '#161B22' }}>
          <Group gap="xs">
            <ActionIcon variant="subtle" color="gray" size="sm">
              <IconFilter size={14} />
            </ActionIcon>
            <Text size="xs" c="#6E7681">Clear Filters</Text>
          </Group>
          <Group gap="xs">
            <Text size="xs" c="#6E7681">Auto-scroll</Text>
            <div
              style={{
                width: 32,
                height: 18,
                borderRadius: 9,
                background: '#58a6ff',
                position: 'relative',
                cursor: 'pointer',
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: '#fff',
                  position: 'absolute',
                  top: 2,
                  right: 2,
                }}
              />
            </div>
          </Group>
        </Group>
      </div>
    </div>
  );
}
