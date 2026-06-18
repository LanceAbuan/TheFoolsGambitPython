import { ScrollArea, Text, Group, Paper, Badge } from '@mantine/core';
import { IconHistory } from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { useCallback } from 'react';
import SectionCard from '../Layout/SectionCard';

export default function RecentGames() {
  const { state, navigateToMove, setAutoFollow } = useGame();
  const games = state.trainingStatus?.recent_games || [];

  const loadGame = useCallback(
    (moves: string[]) => {
      navigateToMove(moves.length);
      setAutoFollow(false);
    },
    [navigateToMove, setAutoFollow]
  );

  return (
    <SectionCard icon={<IconHistory size={16} color="#8B949E" />} title="Recent Games">
      <ScrollArea h={200} scrollbarSize={5}>
        {games.length === 0 ? (
          <Text size="sm" c="dimmed">No finished games yet</Text>
        ) : (
          games.map((g: any, i: number) => (
            <Paper
              key={i}
              p="sm"
              radius="sm"
              mb="xs"
              style={{
                background: '#0D1117',
                border: '1px solid #30363D',
                cursor: 'pointer',
              }}
              onClick={() => g.moves && loadGame(g.moves)}
            >
              <Group justify="space-between" wrap="nowrap">
                <Text size="sm" fw={600} c="#C9D1D9">
                  {g.result || '?'}
                </Text>
                <Badge size="sm" color="gray" variant="filled">
                  {g.length || g.moves?.length || 0} moves
                  {g.game_id != null ? ` • #${g.game_id}` : ''}
                </Badge>
              </Group>
            </Paper>
          ))
        )}
      </ScrollArea>
    </SectionCard>
  );
}
