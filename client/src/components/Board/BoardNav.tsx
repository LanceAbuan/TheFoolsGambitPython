import { ActionIcon, Text } from '@mantine/core';
import {
  IconPlayerTrackPrev,
  IconPlayerSkipBack,
  IconPlayerPlay,
  IconPlayerStop,
  IconPlayerSkipForward,
  IconPlayerTrackNext,
} from '@tabler/icons-react';
import { useGame } from '../../GameContext';
import { useRef, useEffect, useCallback, useState } from 'react';

export default function BoardNav() {
  const { state, navigateToMove, setAutoFollow } = useGame();
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;
  const navRef = useRef(navigateToMove);
  navRef.current = navigateToMove;

  const goToStart = useCallback(() => {
    navigateToMove(0);
    setAutoFollow(false);
  }, [navigateToMove, setAutoFollow]);

  const goBack = useCallback(() => {
    navigateToMove(Math.max(0, state.currentViewIndex - 1));
    setAutoFollow(false);
  }, [navigateToMove, state.currentViewIndex, setAutoFollow]);

  const goForward = useCallback(() => {
    navigateToMove(Math.min(state.allMoves.length, state.currentViewIndex + 1));
    setAutoFollow(false);
  }, [navigateToMove, state.allMoves.length, state.currentViewIndex, setAutoFollow]);

  const goToEnd = useCallback(() => {
    navigateToMove(state.allMoves.length);
    setAutoFollow(true);
  }, [navigateToMove, state.allMoves.length, setAutoFollow]);

  const togglePlay = useCallback(() => {
    if (playRef.current !== null) {
      clearInterval(playRef.current);
      playRef.current = null;
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playRef.current = setInterval(() => {
        const s = stateRef.current;
        if (s.currentViewIndex >= s.allMoves.length) {
          if (playRef.current !== null) clearInterval(playRef.current);
          playRef.current = null;
          setIsPlaying(false);
          return;
        }
        navRef.current(s.currentViewIndex + 1);
      }, 400);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (playRef.current !== null) clearInterval(playRef.current);
    };
  }, []);

  return (
    <div className="board-nav">
      <ActionIcon onClick={goToStart} variant="subtle"><IconPlayerTrackPrev size={16} /></ActionIcon>
      <ActionIcon onClick={goBack} variant="subtle"><IconPlayerSkipBack size={16} /></ActionIcon>
      <ActionIcon onClick={togglePlay} variant="subtle" color={isPlaying ? 'accent-orange' : undefined}>
        {isPlaying ? <IconPlayerStop size={16} /> : <IconPlayerPlay size={16} />}
      </ActionIcon>
      <ActionIcon onClick={goForward} variant="subtle"><IconPlayerSkipForward size={16} /></ActionIcon>
      <ActionIcon onClick={goToEnd} variant="subtle"><IconPlayerTrackNext size={16} /></ActionIcon>
      <Text size="sm" c="dimmed" style={{ minWidth: 80, textAlign: 'center' }}>
        Move {state.currentViewIndex}/{state.allMoves.length}
      </Text>
    </div>
  );
}
