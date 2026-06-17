import { useEffect, useRef, useCallback } from 'react';
import { useGame } from '../GameContext';
import { Chess } from 'chess.js';
import { api } from '../api';
import type { SSEEvent } from '../types';

const SSE_URL = (() => {
  const loc = window.location.hostname;
  if (loc === 'localhost' || loc === '127.0.0.1' || loc === '') {
    return 'http://localhost:5001/api/train/stream';
  }
  return 'https://api.lanceabuan.tech/api/train/stream';
})();

let eventCounter = 0;

export function useSSE() {
  const { state, dispatch, mainGameRef, sideGameRefs } = useGame();
  const sourceRef = useRef<EventSource | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  const logEvent = useCallback(
    (type: string, data: unknown) => {
      const ev: SSEEvent = { id: ++eventCounter, type, data, timestamp: Date.now() };
      dispatch({ type: 'ADD_SSE_EVENT', event: ev });
    },
    [dispatch]
  );

  const catchUpMoves = useCallback(
    (moves: string[]) => {
      if (!moves.length) return;
      const game = new Chess();
      const cache: string[] = [];
      for (const m of moves) {
        try {
          game.move(m);
          cache.push(game.fen());
        } catch {
          break;
        }
      }
      mainGameRef.current = game;
      dispatch({ type: 'SET_ALL_MOVES', moves });
      dispatch({ type: 'SET_FEN_CACHE', cache });
      const s = stateRef.current;
      if (s.autoFollow) {
        dispatch({ type: 'SET_VIEW_INDEX', index: moves.length });
      }
    },
    [dispatch, mainGameRef]
  );

  // Update side game position from a single SAN move

  const updateSideGame = useCallback(
    (gid: number, san: string) => {
      let g = sideGameRefs.current[gid];
      if (!g) {
        // Lazy init — handles the case where game_start was missed
        // (e.g. reconnect mid-game or late page load)
        g = new Chess();
        sideGameRefs.current[gid] = g;
      }
      try {
        g.move(san);
        dispatch({ type: 'SET_SIDE_FEN', gameId: gid, fen: g.fen(), moveCount: g.moveNumber() });
      } catch {
        /* invalid move — skip silently */
      }
    },
    [dispatch, sideGameRefs]
  );

  const initSideGame = useCallback(
    (gid: number) => {
      const g = new Chess();
      sideGameRefs.current[gid] = g;
      dispatch({ type: 'SET_SIDE_FEN', gameId: gid, fen: g.fen(), moveCount: 0 });
    },
    [dispatch, sideGameRefs]
  );

  const rebuildSideGame = useCallback(
    (gid: number, moves: string[]) => {
      const g = new Chess();
      for (const m of moves) {
        try { g.move(m); } catch { break; }
      }
      sideGameRefs.current[gid] = g;
      dispatch({ type: 'SET_SIDE_FEN', gameId: gid, fen: g.fen(), moveCount: g.moveNumber() });
    },
    [dispatch, sideGameRefs]
  );

  /** Seed (or refresh) side games from a full status response.
   *  Skips games that already have the same FEN to avoid churn. */
  const seedSideGames = useCallback(
    (status: any) => {
      const games = status.side_games;
      if (!games) return;
      for (let gid = 1; gid < games.length; gid++) {
        const gd = games[gid];
        if (!gd) continue;
        if (gd.moves?.length) {
          // Only rebuild if the number of moves differs (avoids flicker)
          const existing = sideGameRefs.current[gid];
          const existingCount = existing ? existing.moveNumber() : 0;
          if (existingCount !== gd.moves.length) {
            rebuildSideGame(gid, gd.moves);
          }
        } else if (!sideGameRefs.current[gid]) {
          initSideGame(gid);
        }
      }
    },
    [rebuildSideGame, initSideGame, sideGameRefs]
  );

  useEffect(() => {
    const connect = () => {
      if (sourceRef.current) sourceRef.current.close();
      dispatch({ type: 'SET_SSE_STATUS', status: 'Connecting...' });
      const source = new EventSource(SSE_URL);
      sourceRef.current = source;

      source.onopen = () => {
        dispatch({ type: 'SET_SSE_STATUS', status: 'Connected' });
        logEvent('connected', {});
        api.getStatus().then((status) => {
          dispatch({ type: 'SET_TRAINING_STATUS', status });
          // Catch up main game if we're behind
          if (status.current_game?.moves?.length) {
            const s = stateRef.current;
            if (s.allMoves.length < status.current_game.moves.length) {
              catchUpMoves(status.current_game.moves);
            }
          }
          // Seed side games (skips ones already in sync)
          seedSideGames(status);
        }).catch(() => {});
      };

      source.onerror = () => {
        dispatch({ type: 'SET_SSE_STATUS', status: 'Error — reconnecting...' });
      };

      source.addEventListener('game_start', (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        logEvent('game_start', data);
        if (data.game_id && parseInt(data.game_id, 10) > 0) {
          initSideGame(parseInt(data.game_id, 10));
          return;
        }
        dispatch({ type: 'RESET_MOVES' });
        mainGameRef.current = new Chess();
        dispatch({ type: 'SET_WHAT_HAPPENING', text: 'New game started' });
      });

      source.addEventListener('game_progress', (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        logEvent('game_progress', data);
        if (data.game_id && parseInt(data.game_id, 10) > 0) {
          const gid = parseInt(data.game_id, 10);
          if (data.move) {
            updateSideGame(gid, data.move);
          } else if (data.moves) {
            rebuildSideGame(gid, data.moves);
          }
          return;
        }
        const game = mainGameRef.current;
        if (!game) return;

        if (data.move) {
          dispatch({ type: 'ADD_MOVE', move: data.move });
          try {
            const mResult = game.move(data.move);
            if (mResult) {
              dispatch({ type: 'ADD_FEN', fen: game.fen() });
            }
          } catch { /* */ }
          const s = stateRef.current;
          if (s.autoFollow) {
            dispatch({ type: 'SET_VIEW_INDEX', index: s.allMoves.length + 1 });
          }
        } else if (data.moves) {
          catchUpMoves(data.moves);
          if (data.result) {
            dispatch({ type: 'SET_WHAT_HAPPENING', text: `Game ended: ${data.result}` });
          }
        }
      });

      source.addEventListener('status_update', (e: MessageEvent) => {
        const data = JSON.parse(e.data);
        logEvent('status_update', data);
        if (data.data) {
          dispatch({ type: 'SET_TRAINING_STATUS', status: data.data });
          // status_update may carry side_games from the server broadcast
          if (data.data.side_games) {
            seedSideGames(data.data);
          }
        }
      });

      source.addEventListener('mcts_progress', (e: MessageEvent) => {
        logEvent('mcts_progress', JSON.parse(e.data));
      });
    };

    connect();

    return () => {
      if (sourceRef.current) sourceRef.current.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { logEvent };
}
