import { Chess, type Move, type Square } from 'chess.js';
import * as api from '../api/http';
import type { Color, GameState, Opponent } from '../api/http';

/**
 * `finished` is a server-decided ending the FEN alone cannot express: the TTL
 * reaper's abandonment, or a server-arbitrated draw (threefold — the live
 * client loads bare FENs, so chess.js has no repetition history to derive it
 * from). Board-visible endings keep their specific labels.
 */
export type Status = 'playing' | 'checkmate' | 'stalemate' | 'draw' | 'finished';
export type Mode = 'demo' | 'live';

const VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

export interface GameSnapshot {
  fen: string;
  lastMove: { from: string; to: string } | null;
  sans: string[];
  thinking: boolean;
  orientation: Color;
  resigned: boolean;
  turn: 'w' | 'b';
  status: Status;
  inCheck: boolean;
  checkSquare: string | null;
  material: number;
  mode: Mode;
  backendUp: boolean | null;
  gameId: string | null;
  notice: string | null;
  /** The color the human plays in this game (`flip()` changes orientation only). */
  playColor: Color;
  /** Server-provided end reason ('abandoned' from the TTL reaper, …). */
  endReason: string | null;
  /** Server-authoritative winner, normalized. Legacy snapshots leave it null. */
  snapWinner: 'w' | 'b' | 'draw' | null;
}

/**
 * Client-side game state: a local demo engine AND a live backend mode (§14.0 —
 * the server is the source of truth when live).
 *
 * Reactivity: chess.js internals are plain mutable objects, invisible to React.
 * Every mutation copies the fields the UI reads into an immutable snapshot and
 * notifies subscribers; components consume it with `useGame()`.
 */
class GameStore {
  private engine: Chess = new Chess();
  private demoTimer: ReturnType<typeof setTimeout> | null = null;
  private noticeTimer: ReturnType<typeof setTimeout> | null = null;
  private lastLive: { color: Color; opponent: Opponent; strength: string } | null = null;

  private listeners = new Set<() => void>();

  private state: GameSnapshot = {
    fen: this.engine.fen(),
    lastMove: null,
    sans: [],
    thinking: false,
    orientation: 'white',
    resigned: false,
    turn: 'w',
    status: 'playing',
    inCheck: false,
    checkSquare: null,
    material: 0,
    mode: 'demo',
    backendUp: null,
    gameId: null,
    notice: null,
    playColor: 'white',
    endReason: null,
    snapWinner: null,
  };

  // ---- external store plumbing -------------------------------------------
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): GameSnapshot => this.state;

  private set(patch: Partial<GameSnapshot>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }

  // ---- derived ------------------------------------------------------------
  get over(): boolean {
    return this.state.status !== 'playing' || this.state.resigned;
  }

  get winner(): 'w' | 'b' | null {
    const s = this.state;
    if (s.resigned) return s.playColor === 'white' ? 'b' : 'w';
    if (s.snapWinner === 'w' || s.snapWinner === 'b') return s.snapWinner;
    if (s.status === 'checkmate') return s.turn === 'w' ? 'b' : 'w';
    return null;
  }

  /** The color the human owns this game — demo mode respects it too. */
  private get humanColor(): 'w' | 'b' {
    return this.state.playColor === 'white' ? 'w' : 'b';
  }

  /** The demo spirit owns the other side. */
  private get engineColor(): 'w' | 'b' {
    return this.state.playColor === 'white' ? 'b' : 'w';
  }

  /**
   * Whether taking back is offered. Mid-game the human needs at least one of
   * their own plies to pop (the reply goes with it — the server's parity rule).
   *
   * LIVE: the backend 200s undo on finished games (mate / stalemate /
   * resignation — abandoned games still 409), so `over` no longer blocks the
   * button; only the reaper's verdict does.
   */
  get canUndo(): boolean {
    const s = this.state;
    if (s.thinking) return false;
    if (this.over) {
      if (s.endReason === 'abandoned') return false;
      if (s.mode === 'live') return s.gameId !== null;
      return s.resigned || s.sans.length > 0;
    }
    return s.playColor === 'white' ? s.sans.length >= 2 : s.sans.length >= 3;
  }

  legalTargets(square: string): Move[] {
    // Only the human's side is ever selectable, in both modes.
    if (this.over || this.state.thinking || this.state.turn !== this.humanColor) return [];
    try {
      return this.engine.moves({ square: square as Square, verbose: true }) as Move[];
    } catch {
      return [];
    }
  }

  /** Entry point from the board: routes to the live API or the demo engine. */
  playerMove(from: string, to: string, promotion?: string): void {
    if (this.over) return;
    if (this.state.mode === 'live') void this.liveMove(from, to, promotion);
    else this.tryMove(from, to, promotion);
  }

  tryMove(from: string, to: string, promotion?: string): boolean {
    if (this.over || this.state.thinking || this.state.turn !== this.humanColor) return false;
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
    if (this.state.mode === 'live') {
      void this.liveUndo();
      return;
    }
    this.cancelDemo();
    if (this.state.resigned) {
      this.set({ resigned: false });
      this.afterMove();
    } else {
      if (this.engine.history().length === 0) return;
      this.engine.undo();
      // Undo the demo spirit's reply too, so it is the human's turn again.
      if (this.engine.history().length > 0 && this.engine.turn() !== this.humanColor) this.engine.undo();
      this.set({ resigned: false });
      this.afterMove();
    }
    // A take-back can land on the spirit's turn — keep its clock running.
    if (!this.over && this.state.turn === this.engineColor) this.scheduleDemo();
  }

  /** Start a game: live backend when reachable, demo fallback when not. */
  async startGame(color: Color, opponent: Opponent, strength: string): Promise<void> {
    this.cancelDemo();
    this.lastLive = { color, opponent, strength };
    try {
      const created = await api.newGame(color, opponent, strength);
      const snap = await api.getState(created.game_id);
      this.set({
        mode: 'live',
        backendUp: true,
        gameId: created.game_id,
        playColor: created.orientation,
        orientation: created.orientation,
        resigned: false,
        endReason: null,
        snapWinner: null,
      });
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      this.set({
        backendUp: err instanceof api.ApiError,
        mode: 'demo',
        gameId: null,
      });
      this.newGame(color);
      this.setNotice(err instanceof api.ApiError ? `BACKEND ERROR · ${err.message}` : 'OFFLINE · DEMO MODE');
    }
  }

  newGame(color: Color = this.state.playColor): void {
    if (this.state.mode === 'live' && this.lastLive) {
      void this.startGame(this.lastLive.color, this.lastLive.opponent, this.lastLive.strength);
      return;
    }
    this.cancelDemo();
    this.engine = new Chess();
    this.set({
      resigned: false,
      endReason: null,
      snapWinner: null,
      orientation: color,
      playColor: color,
    });
    this.afterMove();
    if (color === 'black') this.scheduleDemo();
  }

  flip(): void {
    this.set({ orientation: this.state.orientation === 'white' ? 'black' : 'white' });
  }

  resign(): void {
    if (this.over) return;
    if (this.state.mode === 'live') {
      void this.liveResign();
      return;
    }
    this.cancelDemo();
    this.set({ resigned: true, thinking: false });
  }

  /** One-shot reachability probe so the mode label is honest before playing. */
  async probeBackend(): Promise<void> {
    this.set({ backendUp: await api.health() });
  }

  setNotice(message: string | null): void {
    if (this.noticeTimer !== null) {
      clearTimeout(this.noticeTimer);
      this.noticeTimer = null;
    }
    this.set({ notice: message });
    if (message !== null) {
      this.noticeTimer = setTimeout(() => {
        this.noticeTimer = null;
        this.set({ notice: null });
      }, 4000);
    }
  }

  /**
   * Reconcile the whole UI from a server snapshot: fen + moves + last_move +
   * status. The snapshot is authoritative for the game's outcome (threefold /
   * 50-move and reaper-finished games are invisible to a FEN-only read).
   */
  private applySnapshot(snap: GameState, fallbackReason?: string): void {
    // A response from a superseded game must never clobber the current one.
    if (this.state.mode === 'live' && this.state.gameId !== null && snap.game_id !== this.state.gameId) return;
    try {
      this.engine.load(snap.fen);
    } catch {
      this.setNotice('STATE SYNC FAILED');
      return;
    }
    this.set({
      fen: this.engine.fen(),
      sans: [...snap.moves],
      lastMove: snap.last_move && snap.last_move.length >= 4
        ? { from: snap.last_move.slice(0, 2), to: snap.last_move.slice(2, 4) }
        : null,
      endReason:
        typeof snap.reason === 'string' && snap.reason.length > 0 ? snap.reason : (fallbackReason ?? null),
      snapWinner:
        snap.winner === 'white' ? 'w' : snap.winner === 'black' ? 'b' : snap.winner === 'draw' ? 'draw' : null,
      ...(snap.status === 'active' ? { resigned: false } : {}),
    });
    this.deriveFromBoard();
    // A server-finished game must never render as movable — but keep the
    // specific board-derived label (checkmate/stalemate/draw) when it exists.
    if (snap.status === 'finished' && this.state.status === 'playing') this.set({ status: 'finished' });
  }

  private async liveMove(from: string, to: string, promotion?: string): Promise<boolean> {
    if (!this.state.gameId || this.state.thinking) return false;
    if (this.state.turn !== (this.state.playColor === 'white' ? 'w' : 'b')) return false;
    this.set({ thinking: true });
    try {
      const snap = await api.move(this.state.gameId, from, to, promotion);
      this.applySnapshot(snap);
      this.setNotice(null);
      return true;
    } catch (err) {
      if (err instanceof api.ApiError) {
        const detail = err.message.replace(/^illegal move /i, '');
        if (/abandon/i.test(detail)) {
          this.setNotice('GAME ABANDONED · TTL REAPER CLAIMED IT');
          await this.refreshState('abandoned');
        } else {
          this.setNotice(err.status === 422 ? `ILLEGAL MOVE · ${detail}` : `MOVE REJECTED · ${detail}`);
          if (err.status === 409) {
            // A conflict means the server's state contradicts ours — resync.
            await this.refreshState();
          }
        }
      } else {
        this.setNotice('CONNECTION LOST · MOVE NOT SENT');
      }
      return false;
    } finally {
      this.set({ thinking: false });
    }
  }

  /** Re-pull the authoritative snapshot (409 recovery). */
  private async refreshState(fallbackReason?: string): Promise<void> {
    if (!this.state.gameId) return;
    try {
      const snap = await api.getState(this.state.gameId);
      this.applySnapshot(snap, fallbackReason);
    } catch {
      /* unreachable/404 — the notice already tells the user what happened */
    }
  }

  /**
   * Undo, coordinated purely by contract: a 200 means the take-back was
   * accepted (re-render from the response); the one meaningful 409 is the
   * reaper's verdict ("abandoned games cannot take back").
   */
  private async liveUndo(): Promise<void> {
    if (!this.state.gameId || this.state.thinking) return;
    try {
      const snap = await api.undo(this.state.gameId);
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      if (err instanceof api.ApiError && err.status === 409) {
        if (/abandon/i.test(err.message)) {
          this.setNotice('GAME ABANDONED · TTL REAPER CLAIMED IT');
          await this.refreshState('abandoned');
        }
        return;
      }
      this.setNotice(
        err instanceof api.ApiError ? `UNDO REJECTED · ${err.message}` : 'CONNECTION LOST · UNDO NOT SENT',
      );
    }
  }

  private async liveResign(): Promise<void> {
    if (!this.state.gameId || this.state.thinking) return;
    this.set({ resigned: true, thinking: false });
    try {
      const snap = await api.resign(this.state.gameId);
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      this.set({ resigned: false });
      this.setNotice(
        err instanceof api.ApiError ? `RESIGN REJECTED · ${err.message}` : 'CONNECTION LOST · RESIGN NOT SENT',
      );
    }
  }

  /** Copy everything the UI renders out of the engine into the snapshot (demo path). */
  private sync(): void {
    const hist = this.engine.history({ verbose: true });
    const last = hist.at(-1);
    this.set({
      fen: this.engine.fen(),
      sans: hist.map((m) => m.san),
      lastMove: last ? { from: last.from, to: last.to } : null,
    });
    this.deriveFromBoard();
  }

  /** Status/check/material are board-derived in both modes. */
  private deriveFromBoard(): void {
    const status: Status = this.engine.isCheckmate()
      ? 'checkmate'
      : this.engine.isStalemate()
        ? 'stalemate'
        : this.engine.isDraw()
          ? 'draw'
          : 'playing';
    // `over` must reflect the freshly derived status, not last render's.
    const inCheck = status === 'playing' && !this.state.resigned && this.engine.inCheck();
    let checkSquare: string | null = null;
    if (this.engine.inCheck()) {
      const b = this.engine.board();
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const sq = b[r][c];
          if (sq && sq.type === 'k' && sq.color === this.engine.turn()) checkSquare = sq.square;
        }
      }
    }
    let diff = 0;
    for (const row of this.engine.board()) {
      for (const sq of row) {
        if (sq && sq.type !== 'k') diff += sq.color === 'w' ? VALUES[sq.type] : -VALUES[sq.type];
      }
    }
    this.set({
      turn: this.engine.turn(),
      status,
      inCheck,
      checkSquare,
      material: diff,
    });
  }

  private afterMove(): void {
    this.sync();
  }

  private scheduleDemo(): void {
    this.cancelDemo();
    if (this.over || this.state.turn !== this.engineColor) return;
    this.set({ thinking: true });
    this.demoTimer = setTimeout(() => {
      this.demoTimer = null;
      const mv = this.pickDemoMove();
      this.set({ thinking: false });
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
    if (this.state.thinking) this.set({ thinking: false });
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
