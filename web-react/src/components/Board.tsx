import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import type { Move } from 'chess.js';
import type { Color } from '../lib/api/http';
import './Board.css';

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;
const EMPTY_IDS: Set<number> = new Set();
const EMPTY_FROM: Map<number, { fx: number; fy: number }> = new Map();

const GLYPHS: Record<string, string> = {
  k: '\u265A',
  q: '\u265B',
  r: '\u265C',
  b: '\u265D',
  n: '\u265E',
  p: '\u265F',
};

interface PieceInfo {
  type: string;
  color: 'w' | 'b';
}

interface Pc extends PieceInfo {
  id: number;
  square: string;
}

type SquareMap = Map<string, PieceInfo>;

export interface BoardHint {
  from: string;
  to: string;
  color?: string;
}

export interface BoardProps {
  fen: string;
  lastMove?: { from: string; to: string } | null;
  checkSquare?: string | null;
  orientation?: Color;
  mini?: boolean;
  /** When false (game over, spectator view) the board renders but ignores input. */
  interactive?: boolean;
  legalTargets?: (square: string) => Move[];
  onMove?: (from: string, to: string, promotion?: string) => void;
  /** Render pieces as unicode glyphs instead of the SVG set (settings toggle). */
  unicode?: boolean;
  /** Engine suggestion arrows (analysis): drawn over the pieces, never interactive. */
  hints?: BoardHint[];
  /**
   * Editor mode (Analysis → Set up): every click — square or piece — is
   * reported here; selection, drag and legal-move logic are disabled.
   */
  onSquareClick?: (square: string) => void;
}

function parseFen(f: string): SquareMap {
  const out: SquareMap = new Map();
  const rows = f.split(' ')[0].split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= '1' && ch <= '8') {
        file += Number(ch);
        continue;
      }
      out.set(FILES[file] + (8 - r), {
        type: ch.toLowerCase(),
        color: ch === ch.toLowerCase() ? 'b' : 'w',
      });
      file += 1;
    }
  }
  return out;
}

/**
 * Diff a new square map against the existing pieces, preserving ids so moves
 * can glide. A vanished pawn is reused as the arriving promoted piece.
 */
function reconcile(oldP: Pc[], next: SquareMap, nextId: { v: number }): Pc[] {
  const oldBySq = new Map(oldP.map((p) => [p.square, p]));
  const vanished = oldP.filter((p) => !next.has(p.square));
  const used = new Set<number>();
  const out: Pc[] = [];
  for (const [sq, info] of next) {
    const same = oldBySq.get(sq);
    if (same && same.type === info.type && same.color === info.color) {
      out.push({ ...same, square: sq });
      continue;
    }
    const vi = vanished.find(
      (v) =>
        !used.has(v.id) &&
        v.color === info.color &&
        (v.type === info.type || (v.type === 'p' && ['q', 'r', 'b', 'n'].includes(info.type))),
    );
    if (vi) {
      used.add(vi.id);
      out.push({ ...vi, square: sq, type: info.type });
    } else {
      out.push({ id: nextId.v++, square: sq, type: info.type, color: info.color });
    }
  }
  return out;
}

function pieceName(t: string): string {
  return ({ k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' } as Record<string, string>)[t] ?? t;
}

/**
 * The board is an SVG composite keyboard widget. Moves glide only when the
 * position delta is provably one clean ply:
 *  - the piece now on `lastMove.to` is the SAME piece painted on `.from`;
 *  - its type is unchanged (promotions snap);
 *  - coalesced multi-ply jumps fail the from-square test â†’ snap;
 *  - the human's own move snaps (it already traveled under the pointer);
 *  - a board flip mid-update snaps everything;
 *  - castling glides the rook too, only when both deltas prove the pattern.
 */
export function Board({
  fen,
  lastMove = null,
  checkSquare = null,
  orientation = 'white',
  mini = false,
  interactive = true,
  legalTargets = () => [],
  onMove = () => {},
  unicode = false,
  hints = [],
  onSquareClick = undefined,
}: BoardProps) {
  const editor = onSquareClick !== undefined;
  const active = !mini && interactive && !editor;
  const gradientId = `check-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;

  const boardRef = useRef<HTMLDivElement | null>(null);
  const promoRef = useRef<HTMLDivElement | null>(null);
  /** Monotonic piece-id source; a plain object so `reconcile` can bump it. */
  const idBox = useRef({ v: 1 });
  const piecesRef = useRef<Pc[]>([]);

  const [pieces, setPieces] = useState<Pc[]>(() => reconcile([], parseFen(fen), idBox.current));
  piecesRef.current = pieces;

  const [selected, setSelected] = useState<string | null>(null);
  const [targets, setTargets] = useState<Move[]>([]);
  const [dragId, setDragId] = useState<number | null>(null);
  const [dragX, setDragX] = useState(0);
  const [dragY, setDragY] = useState(0);
  const [hoverSquare, setHoverSquare] = useState<string | null>(null);
  const [awaiting, setAwaiting] = useState<{ from: string; to: string } | null>(null);
  const [kcursor, setKcursor] = useState<string | null>(null);
  const [glideIds, setGlideIds] = useState<Set<number>>(EMPTY_IDS);
  const [glideFrom, setGlideFrom] = useState<Map<number, { fx: number; fy: number }>>(EMPTY_FROM);

  const dragIdRef = useRef<number | null>(null);
  dragIdRef.current = dragId;
  const localAttempt = useRef<{ from: string; to: string; at: number } | null>(null);
  const lastOrientation = useRef<Color | null>(null);

  const xy = useCallback(
    (sq: string): { x: number; y: number } => {
      let fi = FILES.indexOf(sq[0] as (typeof FILES)[number]);
      let ri = Number(sq[1]) - 1;
      if (orientation === 'black') {
        fi = 7 - fi;
        ri = 7 - ri;
      }
      return { x: fi * 100, y: (7 - ri) * 100 };
    },
    [orientation],
  );

  const computeGlide = useCallback(
    (prev: Pc[], curr: Pc[], flipped: boolean): { ids: Set<number>; from: Map<number, { fx: number; fy: number }> } => {
      const ids = new Set<number>();
      const from = new Map<number, { fx: number; fy: number }>();
      const lm = lastMove;
      if (flipped || !lm || lm.from === lm.to) return { ids, from };
      if (dragIdRef.current !== null) return { ids, from };
      const la = localAttempt.current;
      if (la && Date.now() - la.at < 1500 && la.from === lm.from && la.to === lm.to) {
        localAttempt.current = null;
        return { ids, from };
      }
      const prevById = new Map(prev.map((p) => [p.id, p]));
      const mover = curr.find((p) => p.square === lm.to);
      if (!mover) return { ids, from };
      const was = prevById.get(mover.id);
      if (!was || was.square !== lm.from || was.type !== mover.type) return { ids, from };
      ids.add(mover.id);
      const m0 = xy(was.square);
      from.set(mover.id, { fx: m0.x, fy: m0.y });
      const fFrom = FILES.indexOf(lm.from[0] as (typeof FILES)[number]);
      const fTo = FILES.indexOf(lm.to[0] as (typeof FILES)[number]);
      if (mover.type === 'k' && lm.from[1] === lm.to[1] && Math.abs(fTo - fFrom) === 2) {
        const kingside = fTo > fFrom;
        const corner = (kingside ? 'h' : 'a') + lm.from[1];
        const landing = (kingside ? 'f' : 'd') + lm.from[1];
        const rook = curr.find((p) => p.square === landing && p.type === 'r' && p.color === mover.color);
        const rookWas = rook ? prevById.get(rook.id) : undefined;
        if (rook && rookWas && rookWas.square === corner) {
          ids.add(rook.id);
          const r0 = xy(rookWas.square);
          from.set(rook.id, { fx: r0.x, fy: r0.y });
        }
      }
      return { ids, from };
    },
    [lastMove, xy],
  );

  // Reconcile before paint so the new position never flickers behind a frame.
  useLayoutEffect(() => {
    const flipped = lastOrientation.current !== null && lastOrientation.current !== orientation;
    lastOrientation.current = orientation;
    const prev = piecesRef.current;
    const curr = reconcile(prev, parseFen(fen), idBox.current);
    const g = computeGlide(prev, curr, flipped);
    piecesRef.current = curr;
    setPieces(curr);
    setGlideIds(g.ids);
    setGlideFrom(g.from);
  }, [fen, orientation, computeGlide]);

  // Release the pin two frames later: by then the glide transition is active
  // in the before-state and the fallback to the attribute position animates.
  useEffect(() => {
    if (glideFrom.size === 0) return;
    let r1 = 0;
    let r2 = 0;
    r1 = requestAnimationFrame(() => {
      r2 = requestAnimationFrame(() => setGlideFrom(EMPTY_FROM));
    });
    return () => {
      cancelAnimationFrame(r1);
      cancelAnimationFrame(r2);
    };
  }, [glideFrom]);

  // Any position change invalidates in-progress selection / promotion state.
  useEffect(() => {
    setAwaiting(null);
    setSelected(null);
    setTargets([]);
    setDragId(null);
    setHoverSquare(null);
  }, [fen]);

  // Safety net: lost pointer capture / alt-tab must never brick the board.
  useEffect(() => {
    const release = () => {
      setDragId(null);
      setHoverSquare(null);
    };
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
    window.addEventListener('blur', release);
    return () => {
      window.removeEventListener('pointerup', release);
      window.removeEventListener('pointercancel', release);
      window.removeEventListener('blur', release);
    };
  }, []);

  // Focus follows an open promotion picker so Enter/Tab can choose.
  useEffect(() => {
    if (awaiting && promoRef.current) promoRef.current.focus();
  }, [awaiting]);

  const displayOrder = useMemo(() => {
    const ranks = orientation === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
    const files = orientation === 'white' ? FILES : [...FILES].reverse();
    const arr: string[] = [];
    for (const r of ranks) for (const f of files) arr.push(f + r);
    return arr;
  }, [orientation]);

  /** Promotion generates up to four moves per destination; render one dot. */
  const targetDots = useMemo(() => {
    const seen = new Set<string>();
    const out: Move[] = [];
    for (const t of targets) {
      if (!seen.has(t.to)) {
        seen.add(t.to);
        out.push(t);
      }
    }
    return out;
  }, [targets]);

  const isDark = (sq: string): boolean => (FILES.indexOf(sq[0] as (typeof FILES)[number]) + Number(sq[1])) % 2 === 1;

  const sqAt = (pt: { x: number; y: number }): string | null => {
    const fi = Math.floor(pt.x / 100);
    const ri = Math.floor(pt.y / 100);
    if (fi < 0 || fi > 7 || ri < 0 || ri > 7) return null;
    let file = fi;
    let rank = 7 - ri;
    if (orientation === 'black') {
      file = 7 - file;
      rank = 7 - rank;
    }
    return FILES[file] + (rank + 1);
  };

  const toSvgPt = (e: ReactPointerEvent): { x: number; y: number } => {
    const r = boardRef.current!.getBoundingClientRect();
    const s = 800 / r.width;
    return { x: (e.clientX - r.left) * s, y: (e.clientY - r.top) * s };
  };

  const clearSel = (): void => {
    setSelected(null);
    setTargets([]);
  };

  const select = (sq: string): void => {
    setSelected(sq);
    setTargets(legalTargets(sq));
  };

  const attempt = (from: string, to: string): void => {
    const cands = legalTargets(from).filter((m) => m.to === to);
    if (cands.length === 0) {
      clearSel();
      return;
    }
    if (cands.some((m) => m.promotion)) {
      setAwaiting({ from, to });
      return;
    }
    // The human's own move snaps when the position updates.
    localAttempt.current = { from, to, at: Date.now() };
    onMove(from, to);
    clearSel();
  };

  const onSquareDown = (sq: string, e: ReactPointerEvent): void => {
    if (editor) {
      e.preventDefault();
      onSquareClick(sq);
      return;
    }
    if (!active || awaiting || dragId !== null) return;
    if (selected && selected !== sq && targets.some((t) => t.to === sq)) {
      attempt(selected, sq);
      return;
    }
    if (legalTargets(sq).length > 0) select(sq);
    else clearSel();
    e.preventDefault();
  };

  const onPieceDown = (p: Pc, e: ReactPointerEvent): void => {
    if (editor) {
      e.preventDefault();
      onSquareClick(p.square);
      return;
    }
    if (!active || awaiting) return;
    e.preventDefault();
    if (selected && selected !== p.square && targets.some((t) => t.to === p.square)) {
      attempt(selected, p.square);
      return;
    }
    // A piece only responds when the human may actually move it.
    if (legalTargets(p.square).length === 0) {
      clearSel();
      return;
    }
    select(p.square);
    setDragId(p.id);
    const { x, y } = xy(p.square);
    setDragX(x);
    setDragY(y);
    setHoverSquare(p.square);
    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers cannot be captured; drag still works via move/up */
    }
  };

  const onPieceMove = (e: ReactPointerEvent): void => {
    if (dragId === null) return;
    const pt = toSvgPt(e);
    setDragX(pt.x - 50);
    setDragY(pt.y - 50);
    setHoverSquare(sqAt(pt));
  };

  const onPieceUp = (e: ReactPointerEvent): void => {
    if (dragId === null) return;
    const pt = toSvgPt(e);
    const target = sqAt(pt);
    const dragged = pieces.find((p) => p.id === dragId);
    setDragId(null);
    setHoverSquare(null);
    if (!dragged || !target || target === dragged.square) return;
    attempt(dragged.square, target);
  };

  const pick = (ptype: string): void => {
    if (!awaiting) return;
    localAttempt.current = { from: awaiting.from, to: awaiting.to, at: Date.now() };
    onMove(awaiting.from, awaiting.to, ptype);
    setAwaiting(null);
    clearSel();
  };

  const cancelPromo = (): void => {
    setAwaiting(null);
    clearSel();
  };

  const stepCursor = (sq: string, df: number, dr: number): string | null => {
    const fi = FILES.indexOf(sq[0] as (typeof FILES)[number]) + df;
    const ri = Number(sq[1]) - 1 + dr;
    if (fi < 0 || fi > 7 || ri < 0 || ri > 7) return null;
    return FILES[fi] + (ri + 1);
  };

  const onBoardKeydown = (e: ReactKeyboardEvent): void => {
    if (!active) return;
    if (e.key === 'Escape') {
      if (awaiting) cancelPromo();
      else clearSel();
      return;
    }
    if (awaiting) return;
    const deltas: Record<string, [number, number]> = {
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
    };
    if (e.key in deltas) {
      e.preventDefault();
      let [df, dr] = deltas[e.key];
      if (orientation === 'black') {
        df = -df;
        dr = -dr;
      }
      if (kcursor === null) {
        setKcursor(lastMove ? lastMove.to : orientation === 'white' ? 'e2' : 'e7');
        return;
      }
      const next = stepCursor(kcursor, df, dr);
      if (next) setKcursor(next);
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (kcursor === null) return;
      if (selected && selected !== kcursor && targets.some((t) => t.to === kcursor)) {
        const dest = kcursor;
        attempt(selected, dest);
        setKcursor(dest);
        return;
      }
      if (legalTargets(kcursor).length > 0) select(kcursor);
      else clearSel();
    }
  };

  const dragPiece = dragId === null ? null : pieces.find((p) => p.id === dragId) ?? null;

  const promoColor: 'w' | 'b' = (() => {
    if (!awaiting) return 'w';
    const p = pieces.find((pp) => pp.square === awaiting.from);
    return p?.color ?? 'w';
  })();

  const promoStyle = (() => {
    if (!awaiting) return undefined;
    let fi = FILES.indexOf(awaiting.to[0] as (typeof FILES)[number]);
    let top = awaiting.to[1] === '8';
    if (orientation === 'black') {
      fi = 7 - fi;
      top = !top;
    }
    return { left: `${fi * 12.5}%`, ...(top ? { top: '2.5%' } : { bottom: '2.5%' }) };
  })();

  const pieceGraphic = (p: PieceInfo) =>
    unicode ? (
      <text className={`uglyph ${p.color}`} x="50" y="54" textAnchor="middle">
        {GLYPHS[p.type] ?? p.type}
      </text>
    ) : (
      <image href={`/pieces/${p.color}${p.type.toUpperCase()}.svg`} x="3" y="3" width="94" height="94" />
    );

  return (
    <div
      ref={boardRef}
      className={mini ? 'board-wrap mini' : 'board-wrap'}
      tabIndex={active ? 0 : -1}
      role={active ? 'application' : undefined}
      aria-roledescription={active ? 'chess board' : undefined}
      aria-label={
        active ? 'Chess board. Arrow keys move the cursor, Enter selects or moves, Escape cancels.' : 'chess board'
      }
      onKeyDown={onBoardKeydown}
    >
      <svg viewBox="0 0 800 800" role="presentation" className={active ? undefined : editor ? 'editor' : 'inert'}>
        {!mini && (
          <defs>
            <radialGradient id={gradientId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="var(--check-glow, #c05a4a)" stopOpacity="0.55" />
              <stop offset="100%" stopColor="var(--check-glow, #c05a4a)" stopOpacity="0" />
            </radialGradient>
          </defs>
        )}

        {displayOrder.map((sq) => {
          const { x, y } = xy(sq);
          const dark = isDark(sq);
          return (
            <g key={sq}>
              <rect
                x={x}
                y={y}
                width={100}
                height={100}
                className={dark ? 'dark' : 'light'}
                role={active ? 'button' : undefined}
                aria-label={`square ${sq}`}
                onPointerDown={(e) => onSquareDown(sq, e)}
              />
            </g>
          );
        })}

        {lastMove &&
          [lastMove.from, lastMove.to].map((sq) => {
            const { x, y } = xy(sq);
            return <rect key={`last-${sq}`} x={x} y={y} width={100} height={100} className="last" pointerEvents="none" />;
          })}

        {hoverSquare && dragId !== null && (
          <rect {...xy(hoverSquare)} width={100} height={100} className="hover" pointerEvents="none" />
        )}

        {selected && (
          <rect
            x={xy(selected).x + 2}
            y={xy(selected).y + 2}
            width={96}
            height={96}
            className="sel"
            pointerEvents="none"
          />
        )}

        {active && kcursor && (
          <rect
            x={xy(kcursor).x + 3}
            y={xy(kcursor).y + 3}
            width={94}
            height={94}
            className="kcursor"
            pointerEvents="none"
          />
        )}

        {active &&
          targetDots.map((t) => {
            const { x, y } = xy(t.to);
            return t.captured ? (
              <circle key={t.to} cx={x + 50} cy={y + 50} r={40} className="cap" pointerEvents="none" />
            ) : (
              <circle key={t.to} cx={x + 50} cy={y + 50} r={13} className="dot" pointerEvents="none" />
            );
          })}

        {checkSquare && (
          <circle cx={xy(checkSquare).x + 50} cy={xy(checkSquare).y + 50} r={55} fill={`url(#${gradientId})`} pointerEvents="none" />
        )}

        {!mini &&
          displayOrder.map((sq, i) => {
            const { x, y } = xy(sq);
            const dark = isDark(sq);
            return (
              <g key={`coord-${sq}`} pointerEvents="none">
                {i >= 56 && (
                  <text className={`coord ${dark ? 'on-dark' : 'on-light'}`} x={x + 93} y={y + 94} textAnchor="end">
                    {sq[0]}
                  </text>
                )}
                {i % 8 === 0 && (
                  <text className={`coord ${dark ? 'on-dark' : 'on-light'}`} x={x + 6} y={y + 16} textAnchor="start">
                    {sq[1]}
                  </text>
                )}
              </g>
            );
          })}

        {pieces.map((p) => {
          const pos = xy(p.square);
          const pin = glideFrom.get(p.id);
          return (
            <g
              key={p.id}
              className={[
                'piece',
                p.id === dragId ? 'lifted' : '',
                glideIds.has(p.id) ? 'glide' : '',
              ]
                .filter(Boolean)
                .join(' ')}
              style={pin ? { transform: `translate(${pin.fx}px, ${pin.fy}px)`, transition: 'none' } : undefined}
              transform={`translate(${pos.x} ${pos.y})`}
              role={active ? 'button' : undefined}
              aria-label={`${p.color === 'w' ? 'white' : 'black'} ${pieceName(p.type)} on ${p.square}`}
              onPointerDown={(e) => onPieceDown(p, e)}
              onPointerMove={onPieceMove}
              onPointerUp={onPieceUp}
            >
              {pieceGraphic(p)}
            </g>
          );
        })}

        {hints.map((h, idx) => {
          const a = xy(h.from);
          const b = xy(h.to);
          const x1 = a.x + 50;
          const y1 = a.y + 50;
          const x2 = b.x + 50;
          const y2 = b.y + 50;
          const len = Math.hypot(x2 - x1, y2 - y1) || 1;
          const ux = (x2 - x1) / len;
          const uy = (y2 - y1) / len;
          const sx = x1 + ux * 24;
          const sy = y1 + uy * 24;
          const ex = x2 - ux * 28;
          const ey = y2 - uy * 28;
          const color = h.color ?? '#2a8c61';
          // White casing keeps engine arrows readable on both square colors.
          return (
            <g key={`hint-${idx}`} pointerEvents="none" opacity={0.9}>
              <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="#f8faf8" strokeWidth={22} strokeLinecap="round" />
              <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={color} strokeWidth={13} strokeLinecap="round" />
              <polygon
                points={`${x2 + ux * 6},${y2 + uy * 6} ${ex - uy * 21},${ey + ux * 21} ${ex + uy * 21},${ey - ux * 21}`}
                fill="#f8faf8"
              />
              <polygon
                points={`${x2},${y2} ${ex - uy * 14},${ey + ux * 14} ${ex + uy * 14},${ey - ux * 14}`}
                fill={color}
              />
            </g>
          );
        })}

        {dragPiece && (
          <g className="piece dragging" transform={`translate(${dragX} ${dragY})`}>
            {pieceGraphic(dragPiece)}
          </g>
        )}
      </svg>

      {awaiting && (
        <div className="promo" style={promoStyle} ref={promoRef} tabIndex={-1}>
          {['q', 'r', 'b', 'n'].map((pt) => (
            <button key={pt} className="promo-piece" onClick={() => pick(pt)} aria-label={`promote to ${pieceName(pt)}`}>
              {unicode ? (
                <span className={`promo-glyph ${promoColor}`}>{GLYPHS[pt]}</span>
              ) : (
                <img src={`/pieces/${promoColor}${pt.toUpperCase()}.svg`} alt="" />
              )}
            </button>
          ))}
          <button className="promo-x" onClick={cancelPromo} aria-label="cancel promotion">
            âœ•
          </button>
        </div>
      )}
    </div>
  );
}
