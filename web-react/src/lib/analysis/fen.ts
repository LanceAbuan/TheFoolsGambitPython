/**
 * FEN utilities for the free-position editor (Analysis → Set up).
 *
 * Deliberately pure and chess.js-free: the editor must build and render FENs
 * for positions a strict validator would reject (no kings yet, pawns on the
 * first rank…), while the *loadable* check decides whether the position can
 * leave the editor and be played/evaluated. Round-trips through `boardMap`
 * preserve everything the editor exposes (turn, castling, ep, counters).
 */

export type PieceType = 'k' | 'q' | 'r' | 'b' | 'n' | 'p';
export type PieceColor = 'w' | 'b';

export interface PlacedPiece {
  type: PieceType;
  color: PieceColor;
}

/** square (`a1`…`h8`) → piece. Absent keys are empty. */
export type BoardMap = Record<string, PlacedPiece>;

export interface FenMeta {
  turn: PieceColor;
  /** Subset of `KQkq`, or `-`. */
  castling: string;
  /** En-passant target square (`a3`…`h6`) or `-`. */
  ep: string;
  halfmove: number;
  fullmove: number;
}

export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

const FILES = 'abcdefgh';

function metaFromParts(parts: string[]): FenMeta {
  const halfmove = Number(parts[4]);
  const fullmove = Number(parts[5]);
  return {
    turn: parts[1] === 'b' ? 'b' : 'w',
    castling: /^-?[KQkq]{1,4}$/.test(parts[2] ?? '-') ? (parts[2] || '-') : '-',
    ep: /^[a-h][36]$/.test(parts[3] ?? '') ? parts[3] : '-',
    halfmove: Number.isFinite(halfmove) && halfmove >= 0 ? Math.floor(halfmove) : 0,
    fullmove: Number.isFinite(fullmove) && fullmove >= 1 ? Math.floor(fullmove) : 1,
  };
}

/** Split a FEN into a square map + metadata. Tolerant: garbage falls back to the start board. */
export function boardMap(fen: string): { map: BoardMap; meta: FenMeta } {
  const parts = fen.trim().split(/\s+/);
  const rows = parts[0]?.split('/') ?? [];
  if (rows.length === 8) {
    const map: BoardMap = {};
    for (let r = 0; r < 8; r++) {
      let file = 0;
      for (const ch of rows[r] ?? '8') {
        if (ch >= '1' && ch <= '8') {
          file += Number(ch);
          continue;
        }
        const type = ch.toLowerCase();
        if (!'kqrbnp'.includes(type) || file > 7) return boardMap(START_FEN);
        map[FILES[file] + (8 - r)] = { type: type as PieceType, color: ch === ch.toLowerCase() ? 'b' : 'w' };
        file += 1;
      }
      if (file !== 8) return boardMap(START_FEN);
    }
    return { map, meta: metaFromParts(parts) };
  }
  return boardMap(START_FEN);
}

/** Assemble a FEN from a square map + metadata. Always yields a syntactically valid FEN. */
export function fenFromBoardMap(map: BoardMap, meta: FenMeta): string {
  const rows: string[] = [];
  for (let r = 8; r >= 1; r--) {
    let row = '';
    let empty = 0;
    for (const f of FILES) {
      const p = map[f + r];
      if (!p) {
        empty += 1;
        continue;
      }
      if (empty > 0) {
        row += String(empty);
        empty = 0;
      }
      row += p.color === 'w' ? p.type.toUpperCase() : p.type;
    }
    if (empty > 0) row += String(empty);
    rows.push(row);
  }
  const castling = meta.castling && meta.castling !== '-' ? meta.castling.replace(/[^KQkq]/g, '') || '-' : '-';
  const ep = /^[a-h][36]$/.test(meta.ep) ? meta.ep : '-';
  const halfmove = Number.isFinite(meta.halfmove) ? Math.max(0, Math.floor(meta.halfmove)) : 0;
  const fullmove = Number.isFinite(meta.fullmove) ? Math.max(1, Math.floor(meta.fullmove)) : 1;
  return `${rows.join('/')} ${meta.turn} ${castling} ${ep} ${halfmove} ${fullmove}`;
}

/** Immutable put/erase: `piece === null` clears the square. */
export function setPiece(map: BoardMap, square: string, piece: PlacedPiece | null): BoardMap {
  const next = { ...map };
  if (piece === null) delete next[square];
  else next[square] = piece;
  return next;
}

/** Drop castling rights whose king/rook are no longer on their origin squares. */
export function sanitizeCastling(map: BoardMap): string {
  const king = (color: PieceColor): boolean =>
    map[(color === 'w' ? 'e1' : 'e8')]?.type === 'k' && map[(color === 'w' ? 'e1' : 'e8')]?.color === color;
  const rook = (color: PieceColor, file: string): boolean =>
    map[file + (color === 'w' ? '1' : '8')]?.type === 'r' &&
    map[file + (color === 'w' ? '1' : '8')]?.color === color;
  let out = '';
  if (king('w') && rook('w', 'h')) out += 'K';
  if (king('w') && rook('w', 'a')) out += 'Q';
  if (king('b') && rook('b', 'h')) out += 'k';
  if (king('b') && rook('b', 'a')) out += 'q';
  return out || '-';
}

/** True when both kings are on the board — the minimum for any meaningful evaluation. */
export function hasBothKings(map: BoardMap): boolean {
  let w = false;
  let b = false;
  for (const p of Object.values(map)) {
    if (p.type !== 'k') continue;
    if (p.color === 'w') w = true;
    else b = true;
  }
  return w && b;
}

/** Cheap evaluable check on a raw FEN: both kings present in the placement field. */
export function fenEvaluable(fen: string): boolean {
  const placement = fen.trim().split(/\s+/)[0] ?? '';
  return placement.includes('K') && placement.includes('k');
}
