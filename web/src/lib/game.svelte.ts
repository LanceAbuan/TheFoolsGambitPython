import { Chess, type Move, type Square } from 'chess.js';
import * as api from './api/http';
import type { GameState, Opponent } from './api/http';

/**
 * 'finished' is a server-decided ending the FEN alone cannot express (F-11):
 * the TTL reaper's abandonment, or a server-arbitrated draw (threefold —
 * the live client loads bare FENs, so chess.js has no repetition history
 * to derive it from). Board-visible endings keep their specific labels.
 */
export type Status = 'playing' | 'checkmate' | 'stalemate' | 'draw' | 'finished';
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
  /**
   * F-11: the winner from the server snapshot, normalized ('w' | 'b' | 'draw').
   * Server-authoritative — a reaper-finished or server-drawn game has a winner
   * the board cannot derive. Legacy snapshots without `winner` leave it null.
   */
  snapWinner = $state<'w' | 'b' | 'draw' | null>(null);

  get over(): boolean {
    return this.status !== 'playing' || this.resigned;
  }
  get winner(): 'w' | 'b' | null {
    if (this.resigned) return this.playColor === 'white' ? 'b' : 'w';
    if (this.snapWinner === 'w' || this.snapWinner === 'b') return this.snapWinner;
    if (this.status === 'checkmate') return this.turn === 'w' ? 'b' : 'w';
    return null;
  }
  /** F-10: the color the human owns this game — demo mode must respect it too. */
  private get humanColor(): 'w' | 'b' {
    return this.playColor === 'white' ? 'w' : 'b';
  }
  /** The demo spirit owns the other side. */
  private get engineColor(): 'w' | 'b' {
    return this.playColor === 'white' ? 'b' : 'w';
  }
  /**
   * Whether taking back is offered. Mid-game the human needs at least one of
   * their own plies to pop (the reply goes with it — the server's parity rule).
   *
   * LIVE since 2026-11-21: the backend 200s undo on finished games (mate /
   * stalemate / resignation — abandoned games still 409), so `over` no longer
   * blocks the button; only the reaper's verdict does (and in demo mode the
   * local engine mirrors the same semantics). liveUndo() still coordinates by
   * contract: a 200 re-renders from the response snapshot, an unexpected 409
   * keeps the rendered state — no crash, no false-alarm notice.
   */
  get canUndo(): boolean {
    if (this.thinking) return false;
    if (this.over) {
      if (this.endReason === 'abandoned') return false; // reaper verdicts are final (server 409s)
      if (this.mode === 'live') return this.gameId !== null;
      return this.resigned || this.sans.length > 0; // demo: un-resign, or pop a finished position
    }
    return this.playColor === 'white' ? this.sans.length >= 2 : this.sans.length >= 3;
  }

  legalTargets(square: string): Move[] {
    // F-10: only the human's side is ever selectable, in both modes — before
    // this guard the demo board exposed the engine's pieces as clickable.
    if (this.over || this.thinking || this.turn !== this.humanColor) return [];
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
    // F-10: gate on the human's color, not a hardcoded 'w' — the demo spirit
    // owns the other side, so as-black the human moves when it is black's turn.
    if (this.over || this.thinking || this.turn !== this.humanColor) return false;
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
    if (this.resigned) {
      // Mirror the server's resignation take-back: un-resign, history untouched.
      this.resigned = false;
      this.afterMove();
    } else {
      if (this.engine.history().length === 0) return;
      this.engine.undo();
      // Undo the demo spirit's reply too, so it is the human's turn again
      // (color-aware — F-10: the human is not always white).
      if (this.engine.history().length > 0 && this.engine.turn() !== this.humanColor) this.engine.undo();
      this.resigned = false;
      this.afterMove();
    }
    // A take-back can land on the spirit's turn (e.g. un-resigning while it was
    // mid-pair) — keep its clock running so the board is never stranded.
    if (!this.over && this.turn === this.engineColor) this.scheduleDemo();
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
      this.snapWinner = null;
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
    this.snapWinner = null;
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

  /**
   * Reconcile the whole UI from a server snapshot: fen + moves + last_move +
   * status. F-11: the snapshot is authoritative for the game's outcome —
   * threefold/50-move and reaper-finished games are invisible to a FEN-only
   * read (the live client loads bare FENs with no repetition history, and the
   * reaper decides outside the board), so trust `status`/`winner` when present
   * and fall back to FEN derivation only for legacy/missing fields.
   *
   * `fallbackReason` carries an end reason the server only told us
   * out-of-band (a 409 detail): snapshots do not emit `reason` yet (backend
   * gap, flagged), so the reaper's verdict reaches us via the error path.
   */
  private applySnapshot(snap: GameState, fallbackReason?: string): void {
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
    this.endReason =
      typeof snap.reason === 'string' && snap.reason.length > 0 ? snap.reason : (fallbackReason ?? null);
    this.snapWinner =
      snap.winner === 'white' ? 'w' : snap.winner === 'black' ? 'b' : snap.winner === 'draw' ? 'draw' : null;
    // Undo-after-finish readiness: the server snapshot is authoritative — when
    // it says the game is active again (a 200 take-back after a finish or
    // resignation), drop the stale local resignation so the board genuinely
    // re-opens instead of staying locked.
    if (snap.status === 'active') this.resigned = false;
    this.deriveFromBoard();
    // F-11: a server-finished game must never render as movable — but keep the
    // specific board-derived label (checkmate/stalemate/draw) when it exists.
    if (snap.status === 'finished' && this.status === 'playing') this.status = 'finished';
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
          await this.refreshState('abandoned');
        } else {
          this.setNotice(err.status === 422 ? `ILLEGAL MOVE · ${detail}` : `MOVE REJECTED · ${detail}`);
          if (err.status === 409) {
            // A conflict means the server's state contradicts ours (the reaper
            // finished the game, another tab moved, …) — resync from the
            // source of truth instead of rendering a stale board (F-11).
            await this.refreshState();
          }
        }
      } else {
        this.setNotice('CONNECTION LOST · MOVE NOT SENT');
      }
      return false;
    } finally {
      this.thinking = false;
    }
  }

  /** Re-pull the authoritative snapshot (409 recovery): render what the server says. */
  private async refreshState(fallbackReason?: string): Promise<void> {
    if (!this.gameId) return;
    try {
      const snap = await api.getState(this.gameId);
      this.applySnapshot(snap, fallbackReason);
    } catch {
      /* unreachable/404 — the notice already tells the user what happened */
    }
  }

  /**
   * Undo, coordinated purely by contract. Since 2026-11-21 the backend 200s
   * take-backs on finished games (mate/stalemate/resignation): a 200 means the
   * take-back was accepted — re-render the active board from the response
   * (applySnapshot restores fen/moves/status and clears a stale resignation).
   * The one remaining 409 is the reaper's verdict ("abandoned games cannot
   * take back"): resync so the finished state renders with the reaper copy.
   * Any other 409 keeps the rendered state — no crash, no false-alarm notice.
   */
  private async liveUndo(): Promise<void> {
    if (!this.gameId || this.thinking) return;
    try {
      const snap = await api.undo(this.gameId); // 200 = take-back accepted
      this.applySnapshot(snap);
      this.setNotice(null);
    } catch (err) {
      if (err instanceof api.ApiError && err.status === 409) {
        if (/abandon/i.test(err.message)) {
          this.setNotice('GAME ABANDONED · TTL REAPER CLAIMED IT');
          await this.refreshState('abandoned');
        }
        return; // 409 = keep the rendered state
      }
      this.setNotice(
        err instanceof api.ApiError ? `UNDO REJECTED · ${err.message}` : 'CONNECTION LOST · UNDO NOT SENT',
      );
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
    // F-10: the spirit plays the side the human does NOT own — hardcoding
    // black made the live-fallback-as-black board self-play.
    if (this.over || this.turn !== this.engineColor) return;
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
