import { createContext, useContext, useReducer, type ReactNode, type Dispatch, useRef, useCallback } from 'react';
import { Chess } from 'chess.js';
import type { SSEEvent, TrainingStatus } from './types';

/* ── State shape ── */
export interface GameState {
  allMoves: string[];
  currentViewIndex: number;
  autoFollow: boolean;
  boardOrientation: 'white' | 'black';
  fenCache: string[];
  sideFens: Record<number, string>;
  sideMoveCounts: Record<number, number>;
  sseStatus: string;
  sseEvents: SSEEvent[];
  trainingStatus: TrainingStatus | null;
  analysisFen: string | null;
  analysis: unknown | null;
  isAnalyzing: boolean;
  isFullscreen: boolean;
  whatHappening: string;
}

const initialState: GameState = {
  allMoves: [],
  currentViewIndex: 0,
  autoFollow: true,
  boardOrientation: 'white',
  fenCache: [],
  sideFens: {},
  sideMoveCounts: {},
  sseStatus: 'Disconnected',
  sseEvents: [],
  trainingStatus: null,
  analysisFen: null,
  analysis: null,
  isAnalyzing: false,
  isFullscreen: false,
  whatHappening: 'Waiting for data...',
};

/* ── Actions ── */
export type GameAction =
  | { type: 'SET_ALL_MOVES'; moves: string[] }
  | { type: 'ADD_MOVE'; move: string }
  | { type: 'RESET_MOVES' }
  | { type: 'SET_VIEW_INDEX'; index: number }
  | { type: 'SET_AUTO_FOLLOW'; follow: boolean }
  | { type: 'FLIP_BOARD' }
  | { type: 'SET_FEN_CACHE'; cache: string[] }
  | { type: 'ADD_FEN'; fen: string }
  | { type: 'SET_SIDE_FEN'; gameId: number; fen: string; moveCount: number }
  | { type: 'SET_SSE_STATUS'; status: string }
  | { type: 'ADD_SSE_EVENT'; event: SSEEvent }
  | { type: 'SET_TRAINING_STATUS'; status: TrainingStatus }
  | { type: 'SET_ANALYSIS'; fen: string | null; analysis: unknown | null }
  | { type: 'SET_IS_ANALYZING'; analyzing: boolean }
  | { type: 'TOGGLE_FULLSCREEN' }
  | { type: 'CLOSE_FULLSCREEN' }
  | { type: 'SET_WHAT_HAPPENING'; text: string };

function reducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'SET_ALL_MOVES': return { ...state, allMoves: action.moves };
    case 'ADD_MOVE': return { ...state, allMoves: [...state.allMoves, action.move] };
    case 'RESET_MOVES': return { ...state, allMoves: [], fenCache: [], currentViewIndex: 0, autoFollow: true };
    case 'SET_VIEW_INDEX': return { ...state, currentViewIndex: action.index, autoFollow: false };
    case 'SET_AUTO_FOLLOW': return { ...state, autoFollow: action.follow };
    case 'FLIP_BOARD': return { ...state, boardOrientation: state.boardOrientation === 'white' ? 'black' : 'white' };
    case 'SET_FEN_CACHE': return { ...state, fenCache: action.cache };
    case 'ADD_FEN': return { ...state, fenCache: [...state.fenCache, action.fen] };
    case 'SET_SIDE_FEN':
      return {
        ...state,
        sideFens: { ...state.sideFens, [action.gameId]: action.fen },
        sideMoveCounts: { ...state.sideMoveCounts, [action.gameId]: action.moveCount },
      };
    case 'SET_SSE_STATUS': return { ...state, sseStatus: action.status };
    case 'ADD_SSE_EVENT': return { ...state, sseEvents: [...state.sseEvents.slice(-499), action.event] };
    case 'SET_TRAINING_STATUS': return { ...state, trainingStatus: action.status };
    case 'SET_ANALYSIS': return { ...state, analysisFen: action.fen, analysis: action.analysis, isAnalyzing: false };
    case 'SET_IS_ANALYZING': return { ...state, isAnalyzing: action.analyzing };
    case 'TOGGLE_FULLSCREEN': return { ...state, isFullscreen: !state.isFullscreen };
    case 'CLOSE_FULLSCREEN': return { ...state, isFullscreen: false };
    case 'SET_WHAT_HAPPENING': return { ...state, whatHappening: action.text };
    default: return state;
  }
}

/* ── Context ── */
interface GameContextType {
  state: GameState;
  dispatch: Dispatch<GameAction>;
  mainGameRef: React.MutableRefObject<Chess | null>;
  sideGameRefs: React.MutableRefObject<Record<number, Chess | null>>;
  getCurrentFen: () => string;
  navigateToMove: (index: number) => void;
  setAutoFollow: (follow: boolean) => void;
  flipBoard: () => void;
}

const GameContext = createContext<GameContextType | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const mainGameRef = useRef<Chess | null>(null);
  const sideGameRefs = useRef<Record<number, Chess | null>>({});

  const getCurrentFen = useCallback((): string => {
    if (state.fenCache.length > 0 && state.currentViewIndex < state.fenCache.length) {
      return state.fenCache[state.currentViewIndex] || 'start';
    }
    if (mainGameRef.current) return mainGameRef.current.fen();
    return 'start';
  }, [state.fenCache, state.currentViewIndex]);

  const navigateToMove = useCallback((index: number) => {
    dispatch({ type: 'SET_VIEW_INDEX', index });
  }, []);

  const setAutoFollow = useCallback((follow: boolean) => {
    dispatch({ type: 'SET_AUTO_FOLLOW', follow });
  }, []);

  const flipBoard = useCallback(() => {
    dispatch({ type: 'FLIP_BOARD' });
  }, []);

  return (
    <GameContext.Provider value={{ state, dispatch, mainGameRef, sideGameRefs, getCurrentFen, navigateToMove, setAutoFollow, flipBoard }}>
      {children}
    </GameContext.Provider>
  );
}

export function useGame(): GameContextType {
  const ctx = useContext(GameContext);
  if (!ctx) throw new Error('useGame must be used within a GameProvider');
  return ctx;
}
