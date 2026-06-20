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
      } catch (err) {
        console.warn(`[useSSE] invalid move for side game ${gid}: "${san}"`, err);
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
   *  Skips games that already have the same FEN to avoid churn.
   *  Uses gd.game_id rather than array index to handle sparse arrays correctly. */
  const seedSideGames = useCallback(
    (status: any) => {
      const games: any[] | undefined = status.side_games;
      if (!games) return;
      for (const gd of games) {
        if (!gd) continue;
        const gid = gd.game_id;
        if (gid == null) continue;
        if (gd.moves?.length) {
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
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (sourceRef.current) sourceRef.current.close();
      dispatch({ type: 'SET_SSE_STATUS', status: 'Connecting...' });
      let source: EventSource;
      try {
        source = new EventSource(SSE_URL, { withCredentials: true });
      } catch (err) {
        console.warn('[useSSE] Failed to create EventSource, retrying in 3s', err);
        reconnectTimer = setTimeout(connect, 3000);
        return;
      }
      sourceRef.current = source;

      source.addEventListener('open', () => {
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

        // Load persistent historical events
        api.getEvents(500).then((data: { events: Array<{ type: string; data: unknown; timestamp: number }> }) => {
          if (data.events) {
            const historical: SSEEvent[] = data.events.map((ev, i) => ({
              id: -(data.events.length - i),  // negative IDs so they don't clash with live events
              type: ev.type || 'unknown',
              data: ev.data || {},
              timestamp: ev.timestamp ? ev.timestamp * 1000 : Date.now(),
            }));
            dispatch({ type: 'SET_HISTORICAL_EVENTS', events: historical });
          }
        }).catch(() => {});

        // Load metric history for sparklines
        api.getMetrics().then((data: Record<string, Array<{ t: number; v: number }>>) => {
          dispatch({ type: 'SET_METRIC_HISTORY', history: data });
        }).catch(() => {});
      });

      source.addEventListener('error', () => {
        dispatch({ type: 'SET_SSE_STATUS', status: 'Error — reconnecting...' });
        // As a safety net, schedule an explicit reconnect
        // (EventSource auto-reconnects but sometimes the 'open' event
        //  doesn't fire on the new connection, leaving us stuck)
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
          if (sourceRef.current) sourceRef.current.close();
          connect();
        }, 5000);
      });

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
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (sourceRef.current) sourceRef.current.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { logEvent };
}
