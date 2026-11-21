import { Chess, type Move, type Square } from 'chess.js';
import * as api from './api/http';
import type { GameState, Opponent } from './api/http';

export type Status = 'playing' | 'checkmate' | 'stalemate' | 'draw';
export type Mode = 'demo' | 'live';

const VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

/**
 * Client-side game state: a local demo engine AND a live backend mode
 * (§14.0 pinned contract — the server is the source of truth when live).
 *
 * Reactivity note: everything the UI reads MUST be a $state field.
 * chess.js internals are plain objects — reading them in getters would be
 * invisible to Svelte's dependency tracking (and short-circuits make it
 * worse: an untracked read guarded by an earlier condition never registers).
 * sync()/applySnapshot() copy engine state into $state after every mutation.
 */
export class GameStore {
  private engine: Chess = new Chess();
  private demoTimer: ReturnType<typeof setTimeout> | null = null;
  private noticeTimer: ReturnType<typeof setTimeout> | null = null;
  private lastLive: { color: 'white' | 'black'; opponent: Opponent; strength: string } | null = null;

  fen = $state(new Chess().fen());
  lastMove = $state<{ from: string; to: string } | null>(null);
  sans = $state<string[]>([]);
  thinking = $state(false);
  orientation = $state<'white' | 'black'>('white');
  resigned = $state(false);
  turn = $state<'w' | 'b'>('w');
  status = $state<Status>('playing');
  inCheck = $state(false);
  checkSquare = $state<string | null>(null);
  material = $state(0);

  // Live-mode fields: which engine owns the position, backend reachability,
  // the server session id, and a transient mono notice (illegal moves, 4xx…).
  mode = $state<Mode>('demo');
  backendUp = $state<boolean | null>(null);
  gameId = $state<string | null>(null);
  notice = $state<string | null>(null);
  /** The color the human plays in this game (flip() may change `orientation` only). */
  playColor = $state<'white' | 'black'>('white');
  /** FS-9: server-provided end reason ('abandoned' from the TTL reaper, …). */
  endReason = $state<string | null>(null);

  get over(): boolean {
    return this.status !== 'playing' || this.resigned;
  }
  get winner(): 'w' | 'b' | null {
    if (this.resigned) return this.playColor === 'white' ? 'b' : 'w';
    if (this.status === 'checkmate') return this.turn === 'w' ? 'b' : 'w';
    return null;
  }
  /** Undo is only meaningful when the human has at least one move of their own to take back. */
  get canUndo(): boolean {
    if (this.over || this.thinking) return false;
    return this.playColor === 'white' ? this.sans.length >= 2 : this.sans.length >= 3;
  }

  legalTargets(square: string): Move[] {
    try {
      return this.engine.moves({ square: square as Square, verbose: true }) as Move[];
    } catch {
      return [];
    }
  }

  /** Entry point from the board: routes to the live API or the demo engine. */
  playerMove(from: string, to: string, promotion?: string): void {
    if (this.over) return;
    if (this.mode === 'live') void this.liveMove(from, to, promotion);
    else this.tryMove(from, to, promotion);
  }

  tryMove(from: string, to: string, promotion?: string): boolean {
    if (this.over || this.thinking || this.turn !== 'w') return false;
    try {
      const mv = this.engine.move({ from, to, promotion });
      if (!mv) return false;
      this.afterMove();
      this.scheduleDemo();
      return true;
    } catch {
      return false;
    }
  }

  undo(): void {
    if (this.mode === 'live') {
      void this.liveUndo();
      return;
    }
    this.cancelDemo();
    if (this.engine.history().length === 0) return;
    this.engine.undo();
    // Undo the demo spirit's reply too, so it is the human's turn again.
    if (this.engine.turn() !== 'w' && this.engine.history().length > 0) this.engine.undo();
    this.resigned = false;
    this.afterMove();
  }

  /** Start a game: live backend when reachable, demo fallback when not. */
  async startGame(color: 'white' | 'black', opponent: Opponent, strength: string): Promise<void> {
    this.cancelDemo();
    this.lastLive = { color, opponent, strength };
    try {
      const created = await api.newGame(color, opponent, strength);
      const snap = await api.getState(created.game_id);
      this.mode = 'live';
      this.backendUp = true;
      this.gameId = created.game_id;
      this.playColor = created.orientation;
      this.orientation = created.orientation;
      this.resigned = false;
      this.endReason = null;
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      this.backendUp = err instanceof api.ApiError;
      this.mode = 'demo';
      this.gameId = null;
      this.newGame(color);
      this.setNotice(err instanceof api.ApiError ? `BACKEND ERROR · ${err.message}` : 'OFFLINE · DEMO MODE');
    }
  }

  newGame(color: 'white' | 'black' = this.playColor): void {
    if (this.mode === 'live' && this.lastLive) {
      void this.startGame(this.lastLive.color, this.lastLive.opponent, this.lastLive.strength);
      return;
    }
    this.cancelDemo();
    this.engine = new Chess();
    this.resigned = false;
    this.endReason = null;
    this.orientation = color;
    this.playColor = color;
    this.afterMove();
    if (color === 'black') this.scheduleDemo();
  }

  flip(): void {
    this.orientation = this.orientation === 'white' ? 'black' : 'white';
  }

  resign(): void {
    if (this.over) return;
    if (this.mode === 'live') {
      void this.liveResign();
      return;
    }
    this.cancelDemo();
    this.resigned = true;
    this.thinking = false;
  }

  /** One-shot reachability probe so the mode label is honest before playing. */
  async probeBackend(): Promise<void> {
    this.backendUp = await api.health();
  }

  setNotice(message: string | null): void {
    if (this.noticeTimer !== null) {
      clearTimeout(this.noticeTimer);
      this.noticeTimer = null;
    }
    this.notice = message;
    if (message !== null) {
      this.noticeTimer = setTimeout(() => {
        this.noticeTimer = null;
        this.notice = null;
      }, 4000);
    }
  }

  /** Reconcile the whole UI from a server snapshot: fen + moves + last_move + status. */
  private applySnapshot(snap: GameState): void {
    // A response from a superseded game (user started a new one while a move
    // was in flight) must never clobber the current game's state.
    if (this.mode === 'live' && this.gameId !== null && snap.game_id !== this.gameId) return;
    try {
      this.engine.load(snap.fen);
    } catch {
      this.setNotice('STATE SYNC FAILED');
      return;
    }
    this.fen = this.engine.fen();
    this.sans = [...snap.moves];
    const lm = snap.last_move;
    this.lastMove = lm && lm.length >= 4 ? { from: lm.slice(0, 2), to: lm.slice(2, 4) } : null;
    this.endReason = typeof snap.reason === 'string' ? snap.reason : null;
    this.deriveFromBoard();
  }

  private async liveMove(from: string, to: string, promotion?: string): Promise<boolean> {
    if (!this.gameId || this.thinking) return false;
    if (this.turn !== (this.playColor === 'white' ? 'w' : 'b')) return false;
    this.thinking = true;
    try {
      const snap = await api.move(this.gameId, from, to, promotion);
      this.applySnapshot(snap);
      this.setNotice(null);
      return true;
    } catch (err) {
      if (err instanceof api.ApiError) {
        const detail = err.message.replace(/^illegal move /i, '');
        if (/abandon/i.test(detail)) {
          this.setNotice('GAME ABANDONED · TTL REAPER CLAIMED IT');
        } else {
          this.setNotice(err.status === 422 ? `ILLEGAL MOVE · ${detail}` : `MOVE REJECTED · ${detail}`);
        }
      } else {
        this.setNotice('CONNECTION LOST · MOVE NOT SENT');
      }
      return false;
    } finally {
      this.thinking = false;
    }
  }

  private async liveUndo(): Promise<void> {
    if (!this.gameId || this.thinking) return;
    try {
      const snap = await api.undo(this.gameId);
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      this.setNotice(err instanceof api.ApiError ? `UNDO REJECTED · ${err.message}` : 'CONNECTION LOST · UNDO NOT SENT');
    }
  }

  private async liveResign(): Promise<void> {
    if (!this.gameId || this.thinking) return;
    this.resigned = true;
    this.thinking = false;
    try {
      const snap = await api.resign(this.gameId);
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      this.resigned = false;
      this.setNotice(err instanceof api.ApiError ? `RESIGN REJECTED · ${err.message}` : 'CONNECTION LOST · RESIGN NOT SENT');
    }
  }

  /** Copy everything the UI renders out of the engine into $state (demo path). */
  private sync(): void {
    this.fen = this.engine.fen();
    const hist = this.engine.history({ verbose: true });
    this.sans = hist.map((m) => m.san);
    const last = hist.at(-1);
    this.lastMove = last ? { from: last.from, to: last.to } : null;
    this.deriveFromBoard();
  }

  /** Status/check/material are board-derived in both modes. */
  private deriveFromBoard(): void {
    this.turn = this.engine.turn();
    this.status = this.engine.isCheckmate()
      ? 'checkmate'
      : this.engine.isStalemate()
        ? 'stalemate'
        : this.engine.isDraw()
          ? 'draw'
          : 'playing';
    this.inCheck = !this.over && this.engine.inCheck();
    this.checkSquare = null;
    if (this.engine.inCheck()) {
      const b = this.engine.board();
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = b[r][c];
          if (sq && sq.type === 'k' && sq.color === this.engine.turn()) this.checkSquare = sq.square;
        }
      }
    }
    let diff = 0;
    for (const row of this.engine.board()) {
      for (const sq of row) {
        if (sq && sq.type !== 'k') diff += sq.color === 'w' ? VALUES[sq.type] : -VALUES[sq.type];
      }
    }
    this.material = diff;
  }

  private afterMove(): void {
    this.sync();
  }

  private scheduleDemo(): void {
    this.cancelDemo();
    if (this.over || this.turn !== 'b') return;
    this.thinking = true;
    this.demoTimer = setTimeout(() => {
      this.demoTimer = null;
      const mv = this.pickDemoMove();
      this.thinking = false;
      if (mv && !this.over) {
        try {
          this.engine.move(mv);
          this.afterMove();
        } catch {
          /* ignore illegal demo move */
        }
      }
    }, 750 + Math.random() * 950);
  }

  private cancelDemo(): void {
    if (this.demoTimer !== null) {
      clearTimeout(this.demoTimer);
      this.demoTimer = null;
    }
    this.thinking = false;
  }

  private pickDemoMove(): { from: string; to: string; promotion?: string } | null {
    const moves = this.engine.moves({ verbose: true }) as Move[];
    if (moves.length === 0) return null;
    const mate = moves.find((m) => m.san.endsWith('#'));
    if (mate) return { from: mate.from, to: mate.to, promotion: mate.promotion };
    const captures = moves
      .filter((m) => m.captured)
      .sort((a, b) => VALUES[b.captured!] - VALUES[a.captured!]);
    const chosen =
      captures.length > 0 && Math.random() < 0.75 ? captures[0] : moves[Math.floor(Math.random() * moves.length)];
    return { from: chosen.from, to: chosen.to, promotion: chosen.promotion };
  }
}

export const game = new GameStore();
