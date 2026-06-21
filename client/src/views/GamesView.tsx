import { Group, Text, Badge, Paper, ScrollArea, ActionIcon, Tooltip } from '@mantine/core';
import { IconEye } from '@tabler/icons-react';
import { useGame } from '../GameContext';
import LiveBoard from '../components/Board/LiveBoard';
import EvalBar from '../components/Board/EvalBar';
import PlayerInfoBar from '../components/Board/PlayerInfoBar';
import BoardNav from '../components/Board/BoardNav';
import { detectOpening } from '../utils/board';
import type { AnalysisResult } from '../types';

interface GameEntry {
  id: number;
  name: string;
  subtitle: string;
  status: 'running' | 'thinking' | 'finished' | 'waiting';
  eval?: number;
  isMain?: boolean;
}

function getStatusColor(status: GameEntry['status']): string {
  switch (status) {
    case 'running': return '#3fb950';
    case 'thinking': return '#d6b81e';
    case 'finished': return '#8B949E';
    case 'waiting': return '#58a6ff';
    default: return '#8B949E';
  }
}

function getStatusLabel(status: GameEntry['status']): string {
  switch (status) {
    case 'running': return 'Running';
    case 'thinking': return 'Thinking...';
    case 'finished': return 'Finished';
    case 'waiting': return 'Waiting...';
    default: return '';
  }
}

function GameListItem({ game, isSelected, onClick }: { game: GameEntry; isSelected: boolean; onClick: () => void }) {
  return (
    <Paper
      p="sm"
      radius="sm"
      onClick={onClick}
      style={{
        background: isSelected ? 'rgba(88, 166, 255, 0.1)' : '#0D1117',
        border: `1px solid ${isSelected ? '#58a6ff' : '#21262D'}`,
        cursor: 'pointer',
        transition: 'all 0.15s ease',
      }}
    >
      <Group justify="space-between" wrap="nowrap">
        <Group gap="xs" wrap="nowrap">
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: getStatusColor(game.status),
              flexShrink: 0,
            }}
          />
          <div>
            <Text size="sm" fw={600} c="#C9D1D9">
              {game.name}
            </Text>
            <Text size="xs" c="#6E7681">{game.subtitle}</Text>
          </div>
        </Group>
        {game.eval != null && (
          <Badge
            size="sm"
            color={game.eval > 0 ? 'green' : game.eval < 0 ? 'red' : 'gray'}
            variant="filled"
          >
            {game.eval > 0 ? '+' : ''}{game.eval.toFixed(2)}
          </Badge>
        )}
        {game.eval == null && (
          <Text size="xs" c="#6E7681">{getStatusLabel(game.status)}</Text>
        )}
      </Group>
    </Paper>
  );
}

function InfoPanel() {
  const { state } = useGame();
  const analysis = state.analysis as AnalysisResult | null;
  const currentMove = state.allMoves[state.currentViewIndex];
  const moveNum = Math.floor(state.currentViewIndex / 2) + 1;
  const s = state.trainingStatus;

  // Determine opening from move history
  const openingName = (s as any)?.last_game_result ? detectOpening(state.allMoves) : 'Unknown';

  return (
    <div
      style={{
        width: 240,
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        padding: '0 16px',
      }}
    >
      {/* Moves */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
          Moves
        </Text>
        {currentMove ? (
          <Group gap="xs">
            <Text size="sm" c="#6E7681">{moveNum}.</Text>
            <Text size="sm" fw={600} c="#C9D1D9" style={{ fontFamily: 'ui-monospace, monospace' }}>
              {currentMove}
            </Text>
          </Group>
        ) : (
          <Text size="sm" c="#6E7681">No moves yet</Text>
        )}
        {state.isAnalyzing && (
          <Text size="xs" c="#58a6ff" mt={4}>Analyzing...</Text>
        )}
      </Paper>

      {/* Evaluation */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
          Evaluation
        </Text>
        {analysis ? (
          <Group justify="space-between">
            <Text size="xl" fw={700} c={analysis.evaluation > 0 ? '#3fb950' : analysis.evaluation < 0 ? '#f85149' : '#C9D1D9'}>
              {analysis.evaluation > 0 ? '+' : ''}{(analysis.evaluation / 100).toFixed(2)}
            </Text>
            <Text size="xs" c="#6E7681">Depth {analysis.depth ?? '—'}</Text>
          </Group>
        ) : (
          <Text size="sm" c="#6E7681">—</Text>
        )}
      </Paper>

      {/* Players */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
          Players
        </Text>
        <Group gap="xs" mb={4}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#C9D1D9' }} />
          <Text size="sm" c="#C9D1D9">White</Text>
          <Text size="xs" c="#6E7681">Neural Network</Text>
        </Group>
        <Group gap="xs">
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#484F58' }} />
          <Text size="sm" c="#C9D1D9">Black</Text>
          <Text size="xs" c="#6E7681">
            {s?.status === 'critic' ? 'Stockfish' : 'Neural Network'}
          </Text>
        </Group>
      </Paper>

      {/* Opening */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 8 }}>
          Opening
        </Text>
        <Text size="sm" c="#C9D1D9">{openingName}</Text>
        <Text size="xs" c="#6E7681">Move {moveNum}</Text>
      </Paper>
    </div>
  );
}

export default function GamesView() {
  const { state, navigateToMove, setAutoFollow } = useGame();
  const s = state.trainingStatus;
  const whiteToMove = state.allMoves.length % 2 === 0;
  const statusLabel = s?.status || 'idle';

  // Build game list from state
  const games: GameEntry[] = [
    {
      id: 0,
      name: 'Main Training Game',
      subtitle: s?.status === 'critic' ? 'vs Stockfish' : 'NN vs NN',
      status: state.allMoves.length > 0 ? (statusLabel === 'training' ? 'thinking' : 'running') : 'waiting',
      eval: (state.analysis as AnalysisResult | null)?.evaluation != null
        ? ((state.analysis as AnalysisResult).evaluation / 100)
        : undefined,
      isMain: true,
    },
  ];

  // Add side games
  for (let i = 1; i <= 9; i++) {
    const moveCount = state.sideMoveCounts[i] || 0;
    if (moveCount > 0 || state.sideFens[i]) {
      games.push({
        id: i,
        name: `Self-Play #${i}`,
        subtitle: 'NN vs NN',
        status: 'running',
      });
    }
  }

  // Add recent finished games
  const recentGames = s?.recent_games || [];
  recentGames.slice(0, 5).forEach((g: { game_id?: number; length?: number; moves?: string[]; result?: string }, i: number) => {
    games.push({
      id: 100 + i,
      name: g.result ? `Game ${g.game_id ?? i + 1}` : `Game ${i + 1}`,
      subtitle: `${g.moves?.length ?? g.length ?? 0} moves${g.result ? ` • ${g.result}` : ''}`,
      status: 'finished',
    });
  });

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Game List Sidebar */}
      <div
        style={{
          width: 240,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          background: '#161B22',
          borderRight: '1px solid #30363D',
        }}
      >
        <Group px="md" py="sm" justify="space-between" style={{ borderBottom: '1px solid #21262D' }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px' }}>
            Games
          </Text>
        </Group>

        <ScrollArea flex={1} scrollbarSize={5}>
          <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {games.map((game) => (
              <GameListItem
                key={game.id}
                game={game}
                isSelected={game.id === 0}
                onClick={() => {
                  if (game.id === 0) {
                    navigateToMove(state.allMoves.length);
                    setAutoFollow(true);
                  }
                }}
              />
            ))}
          </div>
        </ScrollArea>

        <Group px="md" py="sm" justify="space-between" style={{ borderTop: '1px solid #21262D' }}>
          <Text size="xs" c="#6E7681">
            Show Finished ({recentGames.length})
          </Text>
          <Tooltip label="Settings">
            <ActionIcon variant="subtle" color="gray" size="sm">
              <IconEye size={14} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </div>

      {/* Main Board Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', overflow: 'auto' }}>
        {/* Game Header */}
        <Group gap="xs" mb={8}>
          <Text fw={700} size="lg" c="#C9D1D9">
            Main Training Game
          </Text>
          <Badge color={statusLabel === 'running' || statusLabel === 'playing' || statusLabel === 'critic' || statusLabel === 'self-play' ? 'green' : statusLabel === 'training' ? 'blue' : 'gray'} variant="filled" size="sm">
            {statusLabel}
          </Badge>
        </Group>
        <Text size="xs" c="#6E7681" mb={12}>
          {s?.status === 'critic' ? 'vs Stockfish' : 'NN vs NN'}
        </Text>

        <PlayerInfoBar
          name="Neural Network"
          detail={s?.status === 'critic' ? 'vs Stockfish' : 'vs Self'}
          isTurn={!whiteToMove}
          color="top"
        />

        <div style={{ display: 'flex', alignItems: 'stretch', gap: 8, width: '100%', maxWidth: 672 }}>
          <EvalBar />
          <LiveBoard />
        </div>

        <PlayerInfoBar
          name={s?.status === 'critic' ? 'Stockfish' : 'Neural Network'}
          detail={s?.status === 'critic' ? 'Engine' : `Step ${(s?.step ?? 0).toLocaleString()}`}
          isTurn={whiteToMove}
          color="bottom"
        />

        <BoardNav />
      </div>

      {/* Info Panel */}
      <InfoPanel />
    </div>
  );
}
