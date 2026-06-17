import { useEffect, useRef, useCallback } from 'react';
import { useGame } from '../GameContext';
import { api } from '../api';

export function useAnalysis() {
  const { state, dispatch, getCurrentFen } = useGame();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFenRef = useRef<string | null>(null);

  const doEval = useCallback(
    async (fen: string) => {
      if (!fen || fen === 'start' || fen === lastFenRef.current) return;
      lastFenRef.current = fen;
      dispatch({ type: 'SET_IS_ANALYZING', analyzing: true });
      try {
        const result = await api.analyze(fen);
        dispatch({ type: 'SET_ANALYSIS', fen, analysis: result });
      } catch {
        dispatch({ type: 'SET_IS_ANALYZING', analyzing: false });
      }
    },
    [dispatch]
  );

  // Debounce analysis when FEN changes
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const fen = getCurrentFen();
    if (!fen || fen === 'start') return;
    timerRef.current = setTimeout(() => doEval(fen), 500);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [state.currentViewIndex, state.allMoves.length, getCurrentFen, doEval]);
}
