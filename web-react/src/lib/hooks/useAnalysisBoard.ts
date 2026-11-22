/**
 * Free-position state for the Analysis page.
 *
 * Two modes share one `fen` source of truth:
 *  - `play`  — chess.js owns the position: legal moves, undo, reset, check.
 *  - `setup` — a raw BoardMap + FenMeta owns it (chess.js would reject
 *              king-less drafts). Analysis still runs when both kings exist.
 *
 * Leaving setup re-validates with chess.js: the position must be a legal
 * game state (two kings, side not-to-move not already in check, …) before
 * moves can be played on it.
 */
import { useCallback, useMemo, useRef, useState } from 'react';
import { Chess, type Square } from 'chess.js';
import {
  boardMap,
  fenFromBoardMap,
  hasBothKings,
  sanitizeCastling,
  setPiece,
  START_FEN,
  type BoardMap,
  type FenMeta,
  type PlacedPiece,
} from '../analysis/fen';

export type AnalysisMode = 'play' | 'setup';

export interface EditorState {
  map: BoardMap;
  meta: FenMeta;
}

export function useAnalysisBoard(startFen: string = START_FEN) {
  const gameRef = useRef<Chess>(new Chess(startFen));
  const [mode, setMode] = useState<AnalysisMode>('play');
  const [fen, setFen] = useState(startFen);
  const [lastMove, setLastMove] = useState<{ from: string; to: string } | null>(null);
  const [sanHistory, setSanHistory] = useState<string[]>([]);
  const [checkSquare, setCheckSquare] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState>(() => ({ map: {}, meta: boardMap(startFen).meta }));

  const game = gameRef.current;

  const computeCheck = useCallback((): string | null => {
    if (!game.inCheck()) return null;
    for (const row of game.board()) {
      for (const sq of row) {
        if (sq && sq.type === 'k' && sq.color === game.turn()) return sq.square;
      }
    }
    return null;
  }, [game]);

  const syncFromGame = useCallback((): void => {
    setFen(game.fen());
    setSanHistory(game.history());
    const hist = game.history({ verbose: true });
    const prev = hist[hist.length - 1];
    setLastMove(prev ? { from: prev.from, to: prev.to } : null);
    setCheckSquare(computeCheck());
  }, [game, computeCheck]);

  /** legalTargets for Board: every legal origin move for one square. */
  const legalTargets = useCallback(
    (sq: string) => {
      if (mode !== 'play') return [];
      try {
        return game.moves({ square: sq as Square, verbose: true });
      } catch {
        return [];
      }
    },
    [game, mode],
  );

  const tryMove = useCallback(
    (from: string, to: string, promotion?: string): boolean => {
      if (mode !== 'play') return false;
      try {
        game.move({ from, to, ...(promotion ? { promotion } : {}) });
        setNotice(null);
        syncFromGame();
        return true;
      } catch {
        return false;
      }
    },
    [game, mode, syncFromGame],
  );

  /** Play an engine suggestion ("e2e4" / "e7e8q"). */
  const applyUci = useCallback(
    (uci: string): boolean => {
      const from = uci.slice(0, 2);
      const to = uci.slice(2, 4);
      const promo = uci.slice(4, 5);
      return tryMove(from, to, promo || undefined);
    },
    [tryMove],
  );

  const undo = useCallback((): void => {
    if (mode !== 'play') return;
    if (game.undo()) syncFromGame();
  }, [game, mode, syncFromGame]);

  const reset = useCallback((): void => {
    if (mode !== 'play') return;
    game.reset();
    setNotice(null);
    syncFromGame();
  }, [game, mode, syncFromGame]);

  /** Replace the position (play mode) with a chess.js-loadable FEN. */
  const loadFen = useCallback(
    (value: string): boolean => {
      try {
        const next = new Chess(value);
        gameRef.current = next;
        setNotice(null);
        setFen(next.fen());
        setSanHistory(next.history());
        setLastMove(null);
        setCheckSquare(null);
        return true;
      } catch {
        return false;
      }
    },
    [],
  );

  // ---- setup mode ----------------------------------------------------------

  const enterSetup = useCallback((): void => {
    const { map, meta } = boardMap(gameRef.current.fen());
    setEditor({ map, meta });
    setMode('setup');
    setNotice(null);
  }, []);

  /** Leave setup only when chess.js accepts the draft; else surface why. */
  const leaveSetup = useCallback((): void => {
    const candidate = fenFromBoardMap(editor.map, editor.meta);
    try {
      new Chess(candidate); // throws when the draft is not a legal game state
    } catch {
      setNotice('This position cannot be played yet — both kings are required and the side not to move must not be in check.');
      return;
    }
    loadFen(candidate);
    setMode('play');
  }, [editor, loadFen]);

  const editSquare = useCallback(
    (square: string, piece: PlacedPiece | null): void => {
      setEditor((prev) => {
        const map = setPiece(prev.map, square, piece);
        return { ...prev, map, meta: { ...prev.meta, castling: sanitizeCastling(map) } };
      });
    },
    [],
  );

  const editMeta = useCallback((patch: Partial<FenMeta>): void => {
    setEditor((prev) => ({ ...prev, meta: { ...prev.meta, ...patch } }));
  }, []);

  const clearBoard = useCallback((): void => {
    setEditor((prev) => ({ map: {}, meta: { ...prev.meta, castling: '-', ep: '-' } }));
  }, []);

  const loadEditorFen = useCallback((value: string): boolean => {
    try {
      setEditor(boardMap(value));
      setNotice(null);
      return true;
    } catch {
      return false;
    }
  }, []);

  // The live draft FEN (setup) — rebuilt on every editor change.
  const editorFen = useMemo(
    () => (mode === 'setup' ? fenFromBoardMap(editor.map, editor.meta) : fen),
    [mode, editor, fen],
  );

  const editorEvaluable = useMemo(() => hasBothKings(editor.map), [editor.map]);

  return {
    mode,
    fen: mode === 'setup' ? editorFen : fen,
    lastMove,
    sanHistory,
    checkSquare,
    notice,
    setNotice,
    editor,
    editorEvaluable,
    legalTargets,
    tryMove,
    applyUci,
    undo,
    reset,
    loadFen,
    enterSetup,
    leaveSetup,
    editSquare,
    editMeta,
    clearBoard,
    loadEditorFen,
  };
}
