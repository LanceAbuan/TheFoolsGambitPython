import { Group, Text, Badge, Paper, ScrollArea, ActionIcon, Tooltip, Divider } from '@mantine/core';
import { IconEye, IconPlayerTrackNext } from '@tabler/icons-react';
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
  moveCount?: number;
  result?: string;
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

function GameListItem({ game, isSelected, onClick }: { game: GameEntry; isSelected: boolean; onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'stretch',
        borderRadius: 6,
        cursor: 'pointer',
        background: isSelected ? 'rgba(88, 166, 255, 0.08)' : 'transparent',
        border: `1px solid ${isSelected ? 'rgba(88, 166, 255, 0.3)' : 'transparent'}`,
        transition: 'all 0.12s ease',
        overflow: 'hidden',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) e.currentTarget.style.background = 'rgba(139, 148, 158, 0.06)';
      }}
      onMouseLeave={(e) => {
        if (!isSelected) e.currentTarget.style.background = 'transparent';
      }}
    >
      {/* Left accent bar */}
      <div style={{
        width: 3,
        flexShrink: 0,
        background: isSelected ? '#58a6ff' : 'transparent',
        borderRadius: '3px 0 0 3px',
      }} />
      <div style={{ flex: 1, padding: '8px 10px', display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
        {/* Status dot */}
        <div style={{
          width: 7,
          height: 7,
          borderRadius: '50%',
          background: getStatusColor(game.status),
          flexShrink: 0,
          boxShadow: game.status === 'running' ? `0 0 6px ${getStatusColor(game.status)}40` : 'none',
        }} />
        {/* Name + subtitle */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Text size="sm" fw={isSelected ? 600 : 500} c={isSelected ? '#C9D1D9' : '#8B949E'} truncate>
              {game.name}
            </Text>
          </div>
          <Text size="xs" c="#484F58" truncate style={{ marginTop: 1 }}>
            {game.subtitle}
            {game.moveCount != null && game.moveCount > 0 && ` \u00b7 Move ${game.moveCount}`}
          </Text>
        </div>
        {/* Right side: badge or status */}
        {game.eval != null ? (
          <Badge
            size="sm"
            color={game.eval > 0 ? 'green' : game.eval < 0 ? 'red' : 'gray'}
            variant="filled"
            style={{ flexShrink: 0, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}
          >
            {game.eval > 0 ? '+' : ''}{game.eval.toFixed(2)}
          </Badge>
        ) : game.status === 'finished' && game.result ? (
          <Badge size="sm" color="gray" variant="light" style={{ flexShrink: 0 }}>
            {game.result}
          </Badge>
        ) : game.status === 'running' ? (
          <IconPlayerTrackNext size={12} color="#3fb950" style={{ flexShrink: 0 }} />
        ) : null}
      </div>
    </div>
  );
}

function InfoPanel({ selectedGameId }: { selectedGameId: number }) {
  const { state } = useGame();
  const analysis = state.analysis as AnalysisResult | null;
  const s = state.trainingStatus;
  const isMainGame = selectedGameId === 0;

  // Compute data based on selected game
  const currentMoves = isMainGame ? state.allMoves : [];
  const currentMove = currentMoves[state.currentViewIndex];
  const moveNum = Math.floor(state.currentViewIndex / 2) + 1;
  const openingName = currentMoves.length > 0 ? detectOpening(currentMoves) : 'Unknown';

  // Side game move count
  const sideGameMoveCount = state.sideMoveCounts[selectedGameId] || 0;
  const totalMoves = isMainGame ? state.allMoves.length : sideGameMoveCount;

  return (
    <div
      style={{
        width: 240,
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        padding: '0 12px',
        overflow: 'auto',
      }}
    >
      {/* Moves */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 6 }}>
          Moves
        </Text>
        {isMainGame && currentMove ? (
          <Group gap="xs">
            <Text size="sm" c="#6E7681">{moveNum}.</Text>
            <Text size="sm" fw={600} c="#C9D1D9" style={{ fontFamily: 'ui-monospace, monospace' }}>
              {currentMove}
            </Text>
          </Group>
        ) : totalMoves > 0 ? (
          <Group gap="xs">
            <Text size="sm" c="#6E7681">{Math.floor(totalMoves / 2)}.</Text>
            <Text size="sm" fw={500} c="#8B949E" style={{ fontFamily: 'ui-monospace, monospace' }}>
              {totalMoves} half-moves
            </Text>
          </Group>
        ) : (
          <Text size="sm" c="#6E7681">No moves yet</Text>
        )}
        {isMainGame && state.isAnalyzing && (
          <Text size="xs" c="#58a6ff" mt={4}>Analyzing...</Text>
        )}
      </Paper>

      {/* Evaluation — only for main game */}
      {isMainGame && (
        <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 6 }}>
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
      )}

      {/* Players */}
      <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
        <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 6 }}>
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
            {isMainGame && s?.status === 'critic' ? 'Stockfish' : 'Neural Network'}
          </Text>
        </Group>
      </Paper>

      {/* Opening — only for main game */}
      {isMainGame && (
        <Paper p="sm" radius="sm" style={{ background: '#161B22', border: '1px solid #21262D' }}>
          <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.5px', marginBottom: 6 }}>
            Opening
          </Text>
          <Text size="sm" c="#C9D1D9">{openingName}</Text>
          <Text size="xs" c="#6E7681">Move {moveNum}</Text>
        </Paper>
      )}
    </div>
  );
}

export default function GamesView() {
  const { state, selectGame } = useGame();
  const s = state.trainingStatus;
  const statusLabel = s?.status || 'idle';
  const selectedGameId = state.selectedGameId;

  // Build game list from state
  const runningGames: GameEntry[] = [];
  const finishedGames: GameEntry[] = [];

  // Main game
  runningGames.push({
    id: 0,
    name: 'Main Training Game',
    subtitle: s?.status === 'critic' ? 'vs Stockfish' : 'NN vs NN',
    status: state.allMoves.length > 0 ? (statusLabel === 'training' ? 'thinking' : 'running') : 'waiting',
    eval: (state.analysis as AnalysisResult | null)?.evaluation != null
      ? ((state.analysis as AnalysisResult).evaluation / 100)
      : undefined,
    moveCount: state.allMoves.length,
    isMain: true,
  });

  // Side games
  for (let i = 1; i <= 9; i++) {
    const moveCount = state.sideMoveCounts[i] || 0;
    if (moveCount > 0 || state.sideFens[i]) {
      runningGames.push({
        id: i,
        name: `Self-Play #${i}`,
        subtitle: 'NN vs NN',
        status: 'running',
        moveCount,
      });
    }
  }

  // Recent finished games
  const recentGames = s?.recent_games || [];
  recentGames.slice(0, 5).forEach((g: { game_id?: number; length?: number; moves?: string[]; result?: string }, i: number) => {
    finishedGames.push({
      id: 100 + i,
      name: g.result ? `Game ${g.game_id ?? i + 1}` : `Game ${i + 1}`,
      subtitle: `${g.moves?.length ?? g.length ?? 0} moves`,
      status: 'finished',
      result: g.result,
    });
  });

  const selectedGame = [...runningGames, ...finishedGames].find(g => g.id === selectedGameId) || runningGames[0];

  // Determine if we're viewing the main game or a side game
  const isMainGame = selectedGameId === 0;
  const whiteToMove = isMainGame
    ? state.allMoves.length % 2 === 0
    : (state.sideMoveCounts[selectedGameId] || 0) % 2 === 0;

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Game List Sidebar */}
      <div
        style={{
          width: 260,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          background: '#0D1117',
          borderRight: '1px solid #21262D',
        }}
      >
        {/* Header */}
        <div style={{ padding: '14px 14px 10px', borderBottom: '1px solid #161B22' }}>
          <Group justify="space-between">
            <Text size="xs" fw={700} c="#8B949E" tt="uppercase" style={{ letterSpacing: '0.8px' }}>
              Games
            </Text>
            <Text size="xs" c="#484F58">
              {runningGames.length + finishedGames.length}
            </Text>
          </Group>
        </div>

        {/* Scrollable game list */}
        <ScrollArea flex={1} scrollbarSize={4} styles={{ thumb: { background: '#21262D' } }}>
          <div style={{ padding: '6px 6px' }}>
            {/* Running games */}
            {runningGames.map((game) => (
              <GameListItem
                key={game.id}
                game={game}
                isSelected={game.id === selectedGameId}
                onClick={() => selectGame(game.id)}
              />
            ))}

            {/* Finished games divider */}
            {finishedGames.length > 0 && runningGames.length > 0 && (
              <div style={{ padding: '6px 10px' }}>
                <Divider size="xs" color="#21262D" />
                <Text size="xs" c="#484F58" mt={4}>Finished</Text>
              </div>
            )}

            {/* Finished games */}
            {finishedGames.map((game) => (
              <GameListItem
                key={game.id}
                game={game}
                isSelected={game.id === selectedGameId}
                onClick={() => selectGame(game.id)}
              />
            ))}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div style={{ padding: '10px 14px', borderTop: '1px solid #161B22' }}>
          <Group justify="space-between">
            <Text size="xs" c="#484F58">
              {finishedGames.length} finished
            </Text>
            <Tooltip label="Settings">
              <ActionIcon variant="subtle" color="gray" size="sm">
                <IconEye size={14} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </div>
      </div>

      {/* Main Board Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', overflow: 'auto' }}>
        {/* Game Header */}
        <Group gap="xs" mb={8}>
          <Text fw={700} size="lg" c="#C9D1D9">
            {selectedGame?.name || 'Main Training Game'}
          </Text>
          <Badge
            color={selectedGame?.status === 'running' ? 'green' : selectedGame?.status === 'thinking' ? 'yellow' : selectedGame?.status === 'waiting' ? 'blue' : 'gray'}
            variant="filled"
            size="sm"
          >
            {selectedGame?.status || statusLabel}
          </Badge>
        </Group>
        <Text size="xs" c="#6E7681" mb={12}>
          {isMainGame ? (s?.status === 'critic' ? 'vs Stockfish' : 'NN vs NN') : 'NN vs NN'}
        </Text>

        <PlayerInfoBar
          name="Neural Network"
          detail={isMainGame && s?.status === 'critic' ? 'vs Stockfish' : 'vs Self'}
          isTurn={whiteToMove}
          color="top"
        />

        <div style={{ display: 'flex', alignItems: 'stretch', gap: 8, width: '100%', maxWidth: 672 }}>
          <EvalBar />
          <LiveBoard />
        </div>

        <PlayerInfoBar
          name={isMainGame && s?.status === 'critic' ? 'Stockfish' : 'Neural Network'}
          detail={isMainGame && s?.status === 'critic' ? 'Engine' : `Move ${(isMainGame ? state.allMoves.length : (state.sideMoveCounts[selectedGameId] || 0))}`}
          isTurn={!whiteToMove}
          color="bottom"
        />

        <BoardNav />
      </div>

      {/* Info Panel */}
      <InfoPanel selectedGameId={selectedGameId} />
    </div>
  );
}
