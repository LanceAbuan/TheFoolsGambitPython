import { ActionIcon, Text, Group, Paper } from '@mantine/core';
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

  // Compute total moves based on selected game
  const selectedGameId = state.selectedGameId;
  const isMainGame = selectedGameId === 0;
  const totalMoves = isMainGame
    ? state.allMoves.length
    : (state.sideFenCaches[selectedGameId]?.length ?? 0) - 1; // -1 because cache includes start position

  const goToStart = useCallback(() => {
    navigateToMove(0);
    setAutoFollow(false);
  }, [navigateToMove, setAutoFollow]);

  const goBack = useCallback(() => {
    navigateToMove(Math.max(0, state.currentViewIndex - 1));
    setAutoFollow(false);
  }, [navigateToMove, state.currentViewIndex, setAutoFollow]);

  const goForward = useCallback(() => {
    navigateToMove(Math.min(totalMoves, state.currentViewIndex + 1));
    setAutoFollow(false);
  }, [navigateToMove, state.currentViewIndex, totalMoves, setAutoFollow]);

  const goToEnd = useCallback(() => {
    navigateToMove(totalMoves);
    setAutoFollow(true);
  }, [navigateToMove, totalMoves, setAutoFollow]);

  const togglePlay = useCallback(() => {
    if (playRef.current !== null) {
      clearInterval(playRef.current);
      playRef.current = null;
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playRef.current = setInterval(() => {
        const s = stateRef.current;
        const sid = s.selectedGameId;
        const isMain = sid === 0;
        const tm = isMain ? s.allMoves.length : (s.sideFenCaches[sid]?.length ?? 0) - 1;
        if (s.currentViewIndex >= tm) {
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

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigateToMove(Math.max(0, stateRef.current.currentViewIndex - 1));
        setAutoFollow(false);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const s = stateRef.current;
        const sid = s.selectedGameId;
        const isMain = sid === 0;
        const tm = isMain ? s.allMoves.length : (s.sideFenCaches[sid]?.length ?? 0) - 1;
        navigateToMove(Math.min(tm, s.currentViewIndex + 1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigateToMove, setAutoFollow]);

  return (
    <Paper
      p="xs"
      radius="sm"
      mt={8}
      style={{
        maxWidth: 672,
        width: '100%',
        background: '#161B22',
        border: '1px solid #21262D',
      }}
    >
      <Group justify="center" gap="xs" wrap="nowrap">
        <ActionIcon onClick={goToStart} variant="subtle" color="gray" size="md">
          <IconPlayerTrackPrev size={16} />
        </ActionIcon>
        <ActionIcon onClick={goBack} variant="subtle" color="gray" size="md">
          <IconPlayerSkipBack size={16} />
        </ActionIcon>
        <ActionIcon
          onClick={togglePlay}
          variant="filled"
          color={isPlaying ? 'orange' : 'blue'}
          size="md"
        >
          {isPlaying ? <IconPlayerStop size={16} /> : <IconPlayerPlay size={16} />}
        </ActionIcon>
        <ActionIcon onClick={goForward} variant="subtle" color="gray" size="md">
          <IconPlayerSkipForward size={16} />
        </ActionIcon>
        <ActionIcon onClick={goToEnd} variant="subtle" color="gray" size="md">
          <IconPlayerTrackNext size={16} />
        </ActionIcon>
        <Text size="sm" c="#6E7681" ml="xs" style={{ fontVariantNumeric: 'tabular-nums', minWidth: 80, textAlign: 'center' }}>
          Move {state.currentViewIndex}/{totalMoves}
        </Text>
      </Group>
    </Paper>
  );
}
