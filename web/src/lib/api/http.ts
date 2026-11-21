/**
 * REST client for the §14.0 pinned API contract (all JSON under /api).
 *
 * Errors: `ApiError` when the server answered with a non-2xx (status + detail
 * preserved so 422 illegal-move reasons reach the UI); `NetworkError` when the
 * backend is unreachable (drives the OFFLINE · DEMO fallback).
 */
import { API_BASE } from './config';

export type Color = 'white' | 'black';
export type ColorOrRandom = Color | 'random';
export type Opponent = 'fool' | 'stockfish';

/** GET /api/play/{id} and the body of every play mutation — engine.Session.snapshot(). */
export interface GameState {
  game_id: string;
  fen: string;
  moves: string[];
  status: 'active' | 'finished';
  winner: 'white' | 'black' | 'draw' | null;
  in_check: boolean;
  last_move: string | null;
}

/** POST /api/play/new response. */
export interface NewGameResponse {
  game_id: string;
  fen: string;
  orientation: Color;
}

/** GET /api/metrics — mirrors the WS `train` metrics payload. */
export interface MetricsSnapshot {
  elo: number | null;
  loss: number | null;
  win_rate: number | null;
  games_played: number;
}

export interface GameArchiveEntry {
  game_id: string;
  kind: string;
  white_name?: string;
  black_name?: string;
  result?: string | null;
  termination?: string | null;
  ply_count?: number;
  started_at?: number;
  ended_at?: number | null;
  [key: string]: unknown;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class NetworkError extends Error {
  constructor(message = 'backend unreachable') {
    super(message);
    this.name = 'NetworkError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    });
  } catch {
    throw new NetworkError(`${API_BASE} unreachable`);
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body?.detail === 'string') detail = body.detail;
      else if (body?.detail !== undefined) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });
}

/** Liveness probe used by the Play view to decide LIVE vs OFFLINE · DEMO. */
export async function health(timeoutMs = 2500): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function newGame(color: ColorOrRandom, opponent: Opponent, strength: string): Promise<NewGameResponse> {
  return post<NewGameResponse>('/play/new', { color, opponent, strength });
}

export async function getState(gameId: string): Promise<GameState> {
  return request<GameState>(`/play/${encodeURIComponent(gameId)}`);
}

export async function move(gameId: string, from: string, to: string, promotion?: string): Promise<GameState> {
  return post<GameState>(`/play/${encodeURIComponent(gameId)}/move`, {
    from,
    to,
    ...(promotion ? { promotion } : {}),
  });
}

export async function undo(gameId: string): Promise<GameState> {
  return post<GameState>(`/play/${encodeURIComponent(gameId)}/undo`);
}

export async function resign(gameId: string): Promise<GameState> {
  return post<GameState>(`/play/${encodeURIComponent(gameId)}/resign`);
}

export async function games(kind?: 'play' | 'selfplay', limit = 50): Promise<{ games: GameArchiveEntry[] }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  if (kind) qs.set('kind', kind);
  return request<{ games: GameArchiveEntry[] }>(`/games?${qs.toString()}`);
}

export async function metrics(): Promise<MetricsSnapshot> {
  return request<MetricsSnapshot>('/metrics');
}

export async function checkpoints(): Promise<{ checkpoints: string[] }> {
  return request<{ checkpoints: string[] }>('/checkpoints');
}

export async function gamePgn(gameId: string): Promise<string> {
  const res = await fetch(`${API_BASE}/api/games/${encodeURIComponent(gameId)}/pgn`);
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return res.text();
}
