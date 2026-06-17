import { useEffect, useRef, useCallback } from 'react';
import { useGame } from '../GameContext';
import { Chess } from 'chess.js';
import { api } from '../api';

/** Rebuild a single side game from its move list (no-op if moves match). */
function rebuildOne(g: any | null, existing: Chess | null): Chess | null {
  if (!g?.moves?.length) return null;
  const existingCount = existing ? existing.moveNumber() : 0;
  if (existingCount === g.moves.length) return existing; // already in sync
  const game = new Chess();
  for (const m of g.moves) {
    try { game.move(m); } catch { break; }
  }
  return game;
}

export function useStatus() {
  const { dispatch, sideGameRefs } = useGame();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const refsRef = useRef(sideGameRefs);
  refsRef.current = sideGameRefs;

  const processStatus = useCallback((status: any) => {
    dispatch({ type: 'SET_TRAINING_STATUS', status });
    // Seed side games if status carries the data
    const games = status.side_games;
    if (!games) return;
    for (let gid = 1; gid < games.length; gid++) {
      const gd = games[gid];
      if (!gd) continue;
      const rebuilt = rebuildOne(gd, refsRef.current.current[gid]);
      if (rebuilt) {
        refsRef.current.current[gid] = rebuilt;
        dispatch({
          type: 'SET_SIDE_FEN',
          gameId: gid,
          fen: rebuilt.fen(),
          moveCount: rebuilt.moveNumber(),
        });
      } else if (!refsRef.current.current[gid]) {
        const fresh = new Chess();
        refsRef.current.current[gid] = fresh;
        dispatch({
          type: 'SET_SIDE_FEN',
          gameId: gid,
          fen: fresh.fen(),
          moveCount: 0,
        });
      }
    }
  }, [dispatch]);

  useEffect(() => {
    api.getStatus().then(processStatus).catch(() => {});

    intervalRef.current = setInterval(() => {
      api.getStatus().then(processStatus).catch(() => {});
    }, 5000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [processStatus]);
}
