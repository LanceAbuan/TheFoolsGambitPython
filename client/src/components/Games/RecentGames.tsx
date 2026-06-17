import { ScrollArea, Text } from '@mantine/core';
import { useGame } from '../../GameContext';
import { useCallback } from 'react';

export default function RecentGames() {
  const { state, navigateToMove, setAutoFollow } = useGame();
  const games = state.trainingStatus?.recent_games || [];

  const loadGame = useCallback(
    (moves: string[]) => {
      // Rebuild the game from moves
      // In a real implementation, we'd store the full game data
      // For now, just navigate to the end
      navigateToMove(moves.length);
      setAutoFollow(false);
    },
    [navigateToMove, setAutoFollow]
  );

  return (
    <div>
      <div className="section-header">Recent Games</div>
      <ScrollArea h={200}>
        {games.length === 0 ? (
          <Text size="sm" c="dimmed">No finished games yet</Text>
        ) : (
          games.map((g: any, i: number) => (
            <div
              key={i}
              className="game-card"
              onClick={() => g.moves && loadGame(g.moves)}
            >
              <div className="game-result">{g.result || '?'}</div>
              <div className="game-meta">
                {g.length || g.moves?.length || 0} moves
                {g.game_id != null ? ` • #${g.game_id}` : ''}
              </div>
            </div>
          ))
        )}
      </ScrollArea>
    </div>
  );
}
