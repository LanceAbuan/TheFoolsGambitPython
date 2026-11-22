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
/** `nn` is FS-1 (backend): the option ships disabled until a checkpoint exists. */
export type Opponent = 'fool' | 'stockfish' | 'nn';

/** GET /api/play/{id} and the body of every play mutation — engine.Session.snapshot(). */
export interface GameState {
  game_id: string;
  fen: string;
  moves: string[];
  status: 'active' | 'finished';
  winner: 'white' | 'black' | 'draw' | null;
  in_check: boolean;
  last_move: string | null;
  /** FS-9: 'abandoned' when the TTL reaper auto-resigned an idle game. */
  reason?: string | null;
}

/** POST /api/play/new response. */
export interface NewGameResponse {
  game_id: string;
  fen: string;
  orientation: Color;
}

/** GET /api/metrics — mirrors the WS `train` metrics payload.
 *
 * The four base fields are pinned by §14.0; FS-1/FS-2 grow the payload with a
 * per-regime Elo map and arena counters. Everything beyond the pinned four is
 * optional and parsed defensively, so old and new backends both render.
 */
export interface MetricsSnapshot {
  elo: number | null;
  loss: number | null;
  win_rate: number | null;
  games_played: number;
  /** Active regime label (e.g. "scratch") when the backend reports one. */
  regime?: string | null;
  /** Champion checkpoint id when present. */
  champion?: string | null;
  /** Number of arena games played (FS-2). */
  arena_games?: number;
  /** Longest run of consecutive wins vs earlier selves (optional; absent → '—'). */
  best_streak?: number;
  /** Per-regime Elo map (FS-2): regime name → { checkpoint tag → elo }. */
  elo_by_regime?: Record<string, Record<string, number | null>>;
  regime_elo?: Record<string, number>;
  /** Checkpoint → Elo ladder. */
  checkpoints?: Record<string, number>;
  /** Epoch seconds the pipeline last rewrote its metrics file (F-C dating). */
  updated_at?: number;
  /** Gate visibility: server-declared title-match state (process-liveness based). */
  gate_in_progress?: boolean;
  /** Additive: is a `scripts/train.py` process alive right now? */
  trainer_alive?: boolean;
  /** Gate visibility (additive): epoch seconds of the last arena row / journal line. */
  last_arena_ts?: number;
  last_journal_ts?: number;
  [key: string]: unknown;
}

/**
 * Best known Elo for "the trained opponent": prefers the per-regime map
 * (regime → ckpt-tag → elo, best entry wins), falls back to the flat `elo`
 * field, else null (renders as "—").
 */
export function bestKnownElo(m: MetricsSnapshot | null | undefined): number | null {
  if (!m) return null;
  const map = m.elo_by_regime;
  if (map && typeof map === 'object' && !Array.isArray(map)) {
    let best: number | null = null;
    for (const ckpts of Object.values(map)) {
      if (!ckpts || typeof ckpts !== 'object') continue;
      for (const v of Object.values(ckpts)) {
        if (typeof v === 'number' && Number.isFinite(v) && (best === null || v > best)) best = v;
      }
    }
    if (best !== null) return best;
  }
  return typeof m.elo === 'number' && Number.isFinite(m.elo) ? m.elo : null;
}

/**
 * One archive-row shape, ALIGNED ON `game_id`: the WS `lobby` payload and this
 * client speak `game_id`; the REST `/api/games` row field is `id`. The backend
 * carries BOTH (additive aliases, equal); `games()` additionally adopts a
 * legacy `id`-only row's value as the `game_id` fallback.
 */
export interface GameArchiveEntry {
  game_id: string;
  id?: string;
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

function normalizeArchiveRow(row: GameArchiveEntry): GameArchiveEntry {
  if (typeof row.game_id === 'string' && row.game_id.length > 0) return row;
  const legacy = row.id;
  if (typeof legacy === 'string' && legacy.length > 0) return { ...row, game_id: legacy };
  return row;
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
  } catch (e) {
    // A caller-requested abort is not a network failure — re-raise it so
    // debounced callers can distinguish "superseded" from "backend down".
    if ((e as Error)?.name === 'AbortError') throw e;
    throw new NetworkError(`${API_BASE || 'backend'} unreachable`);
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
  return request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** Liveness probe used by Play to decide LIVE vs OFFLINE · DEMO. */
export async function health(timeoutMs = 2500): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(timeoutMs) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function newGame(
  color: ColorOrRandom,
  opponent: Opponent,
  strength: string,
): Promise<NewGameResponse> {
  return post<NewGameResponse>('/play/new', { color, opponent, strength });
}

export async function getState(gameId: string): Promise<GameState> {
  return request<GameState>(`/play/${encodeURIComponent(gameId)}`);
}

export async function move(
  gameId: string,
  from: string,
  to: string,
  promotion?: string,
): Promise<GameState> {
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

export async function games(
  kind?: 'play' | 'selfplay' | 'arena',
  limit = 50,
): Promise<{ games: GameArchiveEntry[] }> {
  const qs = new URLSearchParams({ limit: String(limit) });
  if (kind) qs.set('kind', kind);
  const data = await request<{ games: GameArchiveEntry[] }>(`/games?${qs.toString()}`);
  return { games: Array.isArray(data.games) ? data.games.map(normalizeArchiveRow) : [] };
}

/** Arena ledger row (FS-2): a ledger entry, NOT a play game — no id/status/moves. */
export interface ArenaResult {
  run_id?: string;
  kind?: string;
  regime?: string;
  ckpt?: string;
  checkpoint?: string;
  opponent?: string;
  games?: number;
  wins?: number;
  draws?: number;
  losses?: number;
  elo?: number | null;
  promoted?: boolean;
  ts?: number | string;
  [key: string]: unknown;
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

/* ---- POST /api/analyze (pinned contract) --------------------------------- */

export type AnalysisEngineName = 'fool' | 'stockfish';
export type TerminalKind = 'checkmate' | 'stalemate' | 'insufficient' | 'seventyfive' | 'fivefold';

export interface FoolTopMove {
  uci: string;
  san: string;
  visits: number;
  /** Mover-POV expected value after this move, [-1, 1]. */
  q: number;
}

export interface FoolAnalysis {
  available: boolean;
  label: string;
  checkpoint: string | null;
  elo: number | null;
  visits: number | null;
  root_value: number | null;
  /** White-POV centipawns. */
  score_cp: number | null;
  mate: number | null;
  /** White expected score, [0, 1]. */
  expected_white: number | null;
  best_move: string | null;
  best_san: string | null;
  pv: string[];
  top: FoolTopMove[];
  elapsed_ms: number | null;
  error: string | null;
}

export interface StockfishTopMove {
  uci: string;
  san: string | null;
  score_cp: number | null;
  mate: number | null;
  pv: string[];
}

export interface StockfishAnalysis {
  available: boolean;
  label: string;
  depth: number | null;
  score_cp: number | null;
  mate: number | null;
  expected_white: number | null;
  best_move: string | null;
  best_san: string | null;
  pv: string[];
  top: StockfishTopMove[];
  elapsed_ms: number | null;
  error: string | null;
}

export interface AnalyzeResponse {
  fen: string;
  turn: 'white' | 'black';
  terminal: TerminalKind | null;
  engines: {
    fool?: FoolAnalysis;
    stockfish?: StockfishAnalysis;
  };
}

export interface AnalyzeRequest {
  fen: string;
  engines?: AnalysisEngineName[];
  visits?: number;
  depth?: number;
  multipv?: number;
}

/**
 * Evaluate one FEN with the Fool and/or Stockfish. Pass an AbortSignal to
 * cancel a superseded request (debounced re-analysis while the user moves).
 */
export async function analyze(body: AnalyzeRequest, signal?: AbortSignal): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify(body),
    ...(signal ? { signal } : {}),
  });
}
