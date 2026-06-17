import { ScrollArea, Text } from '@mantine/core';
import { useGame } from '../../GameContext';
import { useCallback } from 'react';

export default function MoveList() {
  const { state, navigateToMove, setAutoFollow } = useGame();

  const handleMoveClick = useCallback(
    (index: number) => {
      navigateToMove(index);
      setAutoFollow(index === state.allMoves.length);
    },
    [navigateToMove, setAutoFollow, state.allMoves.length]
  );

  // Group moves into pairs for display
  const pairs: { num: number; white?: string; black?: string }[] = [];
  for (let i = 0; i < state.allMoves.length; i += 2) {
    const pair: { num: number; white?: string; black?: string } = {
      num: Math.floor(i / 2) + 1,
      white: state.allMoves[i],
    };
    if (i + 1 < state.allMoves.length) {
      pair.black = state.allMoves[i + 1];
    }
    pairs.push(pair);
  }

  return (
    <div>
      <div className="section-header">Moves</div>
      <ScrollArea h={200}>
        <div className="move-list">
          {pairs.length === 0 && (
            <Text size="sm" c="dimmed">No moves yet</Text>
          )}
          {pairs.map((pair, pi) => (
            <div key={pi} style={{ display: 'flex', gap: 4, alignItems: 'center', width: '100%', padding: '2px 0' }}>
              <Text size="xs" c="dimmed" style={{ width: 24, textAlign: 'right' }}>{pair.num}.</Text>
              <div
                className={`move-entry${state.currentViewIndex === pi * 2 ? ' active' : ''}`}
                onClick={() => handleMoveClick(pi * 2)}
                style={{ flex: 1, cursor: 'pointer' }}
              >
                {pair.white || ''}
              </div>
              <div
                className={`move-entry${state.currentViewIndex === pi * 2 + 1 ? ' active' : ''}`}
                onClick={() => pair.black ? handleMoveClick(pi * 2 + 1) : undefined}
                style={{ flex: 1, cursor: pair.black ? 'pointer' : 'default' }}
              >
                {pair.black || ''}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
