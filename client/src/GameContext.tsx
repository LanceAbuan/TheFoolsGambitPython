import { createContext, useContext, useReducer, useState, type ReactNode, type Dispatch, useRef, useCallback } from 'react';
import { Chess } from 'chess.js';
import type { SSEEvent, TrainingStatus } from './types';
import type { EvalDataPoint } from './types';
import type { TabId } from './components/Layout/TabBar';

/* ── Metric history data point ── */
export interface MetricDataPoint {
  t: number;  // timestamp
  v: number;  // value
}

/* ── State shape ── */
export interface GameState {
  allMoves: string[];
  currentViewIndex: number;
  autoFollow: boolean;
  boardOrientation: 'white' | 'black';
  fenCache: string[];
  sideFens: Record<number, string>;
  sideMoveCounts: Record<number, number>;
  selectedGameId: number;                   // 0 = main game, 1-9 = side games
  sideFenCaches: Record<number, string[]>;  // Full FEN history per side game
  sseStatus: string;
  sseEvents: SSEEvent[];
  trainingStatus: TrainingStatus | null;
  analysisFen: string | null;
  analysis: unknown | null;
  isAnalyzing: boolean;
  isFullscreen: boolean;
  whatHappening: string;
  // Persistent data
  historicalEvents: SSEEvent[];
  metricHistory: Record<string, MetricDataPoint[]>;
  evalHistory: EvalDataPoint[];
}

const initialState: GameState = {
  allMoves: [],
  currentViewIndex: 0,
  autoFollow: true,
  boardOrientation: 'white',
  fenCache: [],
  sideFens: {},
  sideMoveCounts: {},
  selectedGameId: 0,
  sideFenCaches: {},
  sseStatus: 'Disconnected',
  sseEvents: [],
  trainingStatus: null,
  analysisFen: null,
  analysis: null,
  isAnalyzing: false,
  isFullscreen: false,
  whatHappening: 'Waiting for data...',
  historicalEvents: [],
  metricHistory: {},
  evalHistory: [],
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
  | { type: 'SELECT_GAME'; gameId: number }
  | { type: 'ADD_SIDE_FEN'; gameId: number; fen: string }
  | { type: 'SET_SIDE_FEN_CACHE'; gameId: number; cache: string[] }
  | { type: 'SET_SSE_STATUS'; status: string }
  | { type: 'ADD_SSE_EVENT'; event: SSEEvent }
  | { type: 'SET_TRAINING_STATUS'; status: TrainingStatus }
  | { type: 'SET_ANALYSIS'; fen: string | null; analysis: unknown | null }
  | { type: 'SET_IS_ANALYZING'; analyzing: boolean }
  | { type: 'TOGGLE_FULLSCREEN' }
  | { type: 'CLOSE_FULLSCREEN' }
  | { type: 'SET_WHAT_HAPPENING'; text: string }
  | { type: 'SET_HISTORICAL_EVENTS'; events: SSEEvent[] }
  | { type: 'SET_METRIC_HISTORY'; history: Record<string, MetricDataPoint[]> }
  | { type: 'SET_EVAL_HISTORY'; history: EvalDataPoint[] }
  | { type: 'ADD_EVAL_POINT'; point: EvalDataPoint };

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
    case 'SELECT_GAME':
      return {
        ...state,
        selectedGameId: action.gameId,
        currentViewIndex: action.gameId === 0
          ? state.allMoves.length  // Jump to latest move for main game
          : (state.sideFenCaches[action.gameId]?.length ?? 0),  // Jump to latest for side game
        autoFollow: true,
      };
    case 'ADD_SIDE_FEN': {
      const existing = state.sideFenCaches[action.gameId] || [];
      return {
        ...state,
        sideFenCaches: { ...state.sideFenCaches, [action.gameId]: [...existing, action.fen] },
      };
    }
    case 'SET_SIDE_FEN_CACHE':
      return {
        ...state,
        sideFenCaches: { ...state.sideFenCaches, [action.gameId]: action.cache },
      };
    case 'SET_SSE_STATUS': return { ...state, sseStatus: action.status };
    case 'ADD_SSE_EVENT': return { ...state, sseEvents: [...state.sseEvents.slice(-499), action.event] };
    case 'SET_TRAINING_STATUS': return { ...state, trainingStatus: action.status };
    case 'SET_ANALYSIS': return { ...state, analysisFen: action.fen, analysis: action.analysis, isAnalyzing: false };
    case 'SET_IS_ANALYZING': return { ...state, isAnalyzing: action.analyzing };
    case 'TOGGLE_FULLSCREEN': return { ...state, isFullscreen: !state.isFullscreen };
    case 'CLOSE_FULLSCREEN': return { ...state, isFullscreen: false };
    case 'SET_WHAT_HAPPENING': return { ...state, whatHappening: action.text };
    case 'SET_HISTORICAL_EVENTS': return { ...state, historicalEvents: action.events };
    case 'SET_METRIC_HISTORY': return { ...state, metricHistory: action.history };
    case 'SET_EVAL_HISTORY': return { ...state, evalHistory: action.history };
    case 'ADD_EVAL_POINT': return { ...state, evalHistory: [...state.evalHistory, action.point] };
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
  selectGame: (gameId: number) => void;
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
}

const GameContext = createContext<GameContextType | null>(null);

export function GameProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [activeTab, setActiveTab] = useState<TabId>('games');
  const mainGameRef = useRef<Chess | null>(null);
  const sideGameRefs = useRef<Record<number, Chess | null>>({});

  const getCurrentFen = useCallback((): string => {
    const gid = state.selectedGameId;
    if (gid === 0) {
      // Main game — use fenCache
      if (state.fenCache.length > 0 && state.currentViewIndex < state.fenCache.length) {
        return state.fenCache[state.currentViewIndex] || 'start';
      }
      if (mainGameRef.current) return mainGameRef.current.fen();
      return 'start';
    }
    // Side game — use sideFenCaches
    const cache = state.sideFenCaches[gid];
    if (cache && cache.length > 0) {
      const idx = Math.min(state.currentViewIndex, cache.length - 1);
      return cache[idx];
    }
    return state.sideFens[gid] || 'start';
  }, [state.selectedGameId, state.fenCache, state.currentViewIndex, state.sideFenCaches, state.sideFens]);

  const navigateToMove = useCallback((index: number) => {
    dispatch({ type: 'SET_VIEW_INDEX', index });
  }, []);

  const setAutoFollow = useCallback((follow: boolean) => {
    dispatch({ type: 'SET_AUTO_FOLLOW', follow });
  }, []);

  const flipBoard = useCallback(() => {
    dispatch({ type: 'FLIP_BOARD' });
  }, []);

  const selectGame = useCallback((gameId: number) => {
    dispatch({ type: 'SELECT_GAME', gameId });
  }, []);

  return (
    <GameContext.Provider value={{ state, dispatch, mainGameRef, sideGameRefs, getCurrentFen, navigateToMove, setAutoFollow, flipBoard, selectGame, activeTab, setActiveTab }}>
      {children}
    </GameContext.Provider>
  );
}

export function useGame(): GameContextType {
  const ctx = useContext(GameContext);
  if (!ctx) throw new Error('useGame must be used within a GameProvider');
  return ctx;
}
