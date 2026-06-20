import { useEffect, useRef, useCallback } from 'react';
import { useGame } from '../GameContext';
import { api } from '../api';
import type { AnalysisResult } from '../types';

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

        // Record eval data point for the chart
        const analysis = result as AnalysisResult;
        const moveIndex = state.currentViewIndex;
        const san = state.allMoves[moveIndex] || '';
        if (san && analysis.evaluation != null) {
          // Determine quality from best move in analysis
          const bestMove = analysis.move_analysis?.[0];
          const quality = bestMove?.quality || 'ok';
          dispatch({
            type: 'ADD_EVAL_POINT',
            point: {
              move_num: Math.floor(moveIndex / 2) + 1,
              san,
              eval_cp: analysis.evaluation,
              eval_norm: analysis.evaluation_normalized,
              quality,
            },
          });
        }
      } catch {
        dispatch({ type: 'SET_IS_ANALYZING', analyzing: false });
      }
    },
    [dispatch, state.currentViewIndex, state.allMoves]
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
