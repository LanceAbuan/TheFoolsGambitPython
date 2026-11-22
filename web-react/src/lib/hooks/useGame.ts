import { useSyncExternalStore } from 'react';
import { game, type GameSnapshot } from '../state/game';

/** Subscribe to the game store's immutable snapshot. */
export function useGame(): GameSnapshot {
  return useSyncExternalStore(game.subscribe, game.getSnapshot, game.getSnapshot);
}
