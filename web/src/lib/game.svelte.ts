import { Chess, type Move, type Square } from 'chess.js';

export type Status = 'playing' | 'checkmate' | 'stalemate' | 'draw';

const VALUES: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

/**
 * Client-side game state for the demo (no backend yet).
 * The server will remain the source of truth once it exists; this store
 * is written so the same call sites can be swapped to API/WS calls later.
 *
 * Reactivity note: everything the UI reads MUST be a $state field.
 * chess.js internals are plain objects — reading them in getters would be
 * invisible to Svelte's dependency tracking (and short-circuits make it
 * worse: an untracked read guarded by an earlier condition never registers).
 * sync() copies engine state into $state after every mutation.
 */
export class GameStore {
  private engine: Chess = new Chess();
  private demoTimer: ReturnType<typeof setTimeout> | null = null;

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

  get over(): boolean {
    return this.status !== 'playing' || this.resigned;
  }
  get winner(): 'w' | 'b' | null {
    if (this.resigned) return 'b';
    if (this.status === 'checkmate') return this.turn === 'w' ? 'b' : 'w';
    return null;
  }

  legalTargets(square: string): Move[] {
    try {
      return this.engine.moves({ square: square as Square, verbose: true }) as Move[];
    } catch {
      return [];
    }
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
    this.cancelDemo();
    if (this.engine.history().length === 0) return;
    this.engine.undo();
    // Undo the demo spirit's reply too, so it is the human's turn again.
    if (this.engine.turn() !== 'w' && this.engine.history().length > 0) this.engine.undo();
    this.resigned = false;
    this.afterMove();
  }

  newGame(color: 'white' | 'black' = 'white'): void {
    this.cancelDemo();
    this.engine = new Chess();
    this.resigned = false;
    this.orientation = color;
    this.afterMove();
    if (color === 'black') this.scheduleDemo();
  }

  flip(): void {
    this.orientation = this.orientation === 'white' ? 'black' : 'white';
  }

  resign(): void {
    if (this.over) return;
    this.cancelDemo();
    this.resigned = true;
    this.thinking = false;
  }

  /** Copy everything the UI renders out of the engine into $state. */
  private sync(): void {
    this.fen = this.engine.fen();
    const hist = this.engine.history({ verbose: true });
    this.sans = hist.map((m) => m.san);
    const last = hist.at(-1);
    this.lastMove = last ? { from: last.from, to: last.to } : null;
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
