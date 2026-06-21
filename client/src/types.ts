/** A single SSE event stored in the log */
export interface SSEEvent {
  id: number;
  type: string;
  data: unknown;
  timestamp: number;
}

export interface CycleInfo {
  games_this_cycle: number;
  games_per_cycle: number;
  steps_this_cycle: number;
  steps_per_cycle: number;
  min_buffer_for_train: number;
}

/** Training status from /api/train/status */
export interface TrainingStatus {
  status: string;
  step: number;
  games_played: number;
  loss: number | null;
  policy_loss?: number | null;
  value_loss?: number | null;
  estimated_elo: number | null;
  buffer_size: number;
  learning_rate?: number;
  side_games_completed?: number;
  cycle?: CycleInfo;
  current_game?: {
    id?: number;
    moves: string[];
    result?: string;
    fen?: string;
  };
  recent_games?: RecentGame[];
  resources?: SystemResources | null;
  throughput?: ThroughputData | null;
  top_openings?: OpeningEntry[];
  recent_improvements?: ImprovementEntry[];
}

export interface SystemResources {
  cpu_percent: number;
  cpu_count: number;
  ram: {
    total_gb: number;
    used_gb: number;
    percent: number;
  };
  gpu?: {
    name: string;
    memory_total_gb: number;
    memory_used_gb: number;
    memory_reserved_gb: number;
    memory_percent: number;
    utilization_percent?: number;
  };
}

export interface ThroughputData {
  games_per_hour: number;
  positions_per_sec: number;
  train_steps_per_sec: number;
}

export interface OpeningEntry {
  name: string;
  count: number;
  nn_win_rate: number;
  white_wins: number;
  draws: number;
  black_wins: number;
}

export interface ImprovementEntry {
  text: string;
  type: string;
  delta: number;
  timestamp: number;
}

export interface EvalDataPoint {
  move_num: number;
  san: string;
  eval_cp: number;
  eval_norm: number;
  quality: string;
}

/** A finished game entry */
export interface RecentGame {
  result: string;
  moves: string[];
  length: number;
  game_id?: number;
}

/** Stockfish evaluation result */
export interface EvalResult {
  evaluation: number;        // centipawns
  evaluation_normalized: number;  // -1 .. 1
  top_moves?: string[];
  cached?: boolean;
}

/** Full Stockfish analysis for a position */
export interface AnalysisRow {
  move: string;
  san: string;
  eval: string;
  type: string;
  quality: string;
}

export interface AnalysisResult {
  fen: string;
  evaluation: number;
  evaluation_normalized: number;
  move_analysis: AnalysisRow[];
  depth?: number;
  cached?: boolean;
}

/** Move quality classification colors (Mantine theme color names) */
export const MOVE_COLORS: Record<string, string> = {
  best: 'move-best',
  good: 'move-good',
  ok: 'move-ok',
  bad: 'move-bad',
  blunder: 'move-blunder',
  book: 'move-book',
};

/** Move quality label */
export const MOVE_LABELS: Record<string, string> = {
  best: 'Best',
  good: 'Good',
  ok: 'OK',
  bad: 'Bad',
  blunder: 'Blunder',
  book: 'Book',
};
