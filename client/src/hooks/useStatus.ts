import { useEffect, useRef } from 'react';
import { useGame } from '../GameContext';
import { api } from '../api';

export function useStatus() {
  const { dispatch } = useGame();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    api.getStatus().then((s) => dispatch({ type: 'SET_TRAINING_STATUS', status: s })).catch(() => {});

    intervalRef.current = setInterval(() => {
      api.getStatus().then((s) => dispatch({ type: 'SET_TRAINING_STATUS', status: s })).catch(() => {});
    }, 5000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [dispatch]);
}
