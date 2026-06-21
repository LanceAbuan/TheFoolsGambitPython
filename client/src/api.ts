import type { TrainingStatus, EvalResult, AnalysisResult, EvalDataPoint } from './types';

const API_BASE = (() => {
  const loc = window.location.hostname;
  if (loc === 'localhost' || loc === '127.0.0.1' || loc === '') {
    return 'http://localhost:5001';
  }
  return 'https://api.lanceabuan.tech';
})();

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const opts: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== undefined) {
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    throw new Error(`API ${method} ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  /** Poll training status */
  getStatus: (signal?: AbortSignal) =>
    fetch(`${API_BASE}/api/train/status`, { signal }).then((r) => {
      if (!r.ok) throw new Error(`status ${r.status}`);
      return r.json() as Promise<TrainingStatus>;
    }),

  /** Start training */
  startTraining: (config?: { mcts_simulations?: number; use_stockfish?: boolean }) =>
    request<TrainingStatus>('POST', '/api/train/start', config || {}),

  /** Stop training */
  stopTraining: () => request<TrainingStatus>('POST', '/api/train/stop'),

  /** Full training reset */
  resetTraining: () => request<{ status: string }>('POST', '/api/train/reset'),

  /** Play a single self-play game */
  playGame: () => request<TrainingStatus>('POST', '/api/train/play'),

  /** Run a single training step */
  stepTraining: () => request<TrainingStatus>('POST', '/api/train/step'),

  /** Get Stockfish evaluation for a FEN */
  evaluate: (fen: string) =>
    request<EvalResult>('POST', '/api/train/evaluate', { fen }),

  /** Get full Stockfish analysis (top 5 moves) */
  analyze: (fen: string) =>
    request<AnalysisResult>('POST', '/api/train/analyze', { fen }),

  /** Push model to Hugging Face */
  pushModel: () => request<{ status: string }>('POST', '/api/train/push'),

  /** Get recent games list */
  getGames: () =>
    fetch(`${API_BASE}/api/train/games`).then((r) => r.json()),

  /** Get persistent historical events */
  getEvents: (limit = 500) =>
    fetch(`${API_BASE}/api/train/events?limit=${limit}`).then((r) => r.json()),

  /** Get metric history for sparkline charts */
  getMetrics: (key?: string, limit = 100) => {
    const params = new URLSearchParams();
    if (key) params.set('key', key);
    params.set('limit', String(limit));
    return fetch(`${API_BASE}/api/train/metrics?${params}`).then((r) => r.json());
  },

  /** Get system resource usage */
  getResources: () =>
    fetch(`${API_BASE}/api/train/resources`).then((r) => r.json()),

  /** Get per-move evaluation history for charts */
  getEvalHistory: (which: 'current' | 'last' = 'current') =>
    fetch(`${API_BASE}/api/train/eval-history?which=${which}`).then((r) => r.json()) as Promise<{ history: EvalDataPoint[] }>,

  /** Get opening frequency statistics */
  getOpenings: () =>
    fetch(`${API_BASE}/api/train/openings`).then((r) => r.json()),

  /** Get recent improvement records */
  getImprovements: () =>
    fetch(`${API_BASE}/api/train/improvements`).then((r) => r.json()),
};
