/** Shared chess board square colors — single source of truth. */
export const BOARD_COLORS = {
  dark: '#625b4d',
  light: '#b7b09c',
} as const;

export const DEFAULT_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';

/** Simple client-side opening detection from SAN moves */
export function detectOpening(moves: string[]): string {
  if (moves.length === 0) return 'Unknown';
  const m = moves.slice(0, 4);
  if (m[0] === 'e4') {
    if (m[1] === 'c5') return 'Sicilian Defense';
    if (m[1] === 'e5') {
      if (m[2] === 'Nf3') return "King's Knight Opening";
      if (m[2] === 'f4') return "King's Gambit";
      return 'Open Game';
    }
    if (m[1] === 'e6') return 'French Defense';
    if (m[1] === 'c6') return 'Caro-Kann Defense';
    if (m[1] === 'd5') return 'Scandinavian Defense';
    if (m[1] === 'Nf6') return "Alekhine's Defense";
    if (m[1] === 'd6') return 'Pirc Defense';
    if (m[1] === 'g6') return 'Modern Defense';
    return "King's Pawn Opening";
  }
  if (m[0] === 'd4') {
    if (m[1] === 'd5') {
      if (m[2] === 'c4') return "Queen's Gambit";
      return "Queen's Pawn Opening";
    }
    if (m[1] === 'Nf6') {
      if (m[2] === 'c4' && m[3] === 'g6') return "King's Indian Defense";
      if (m[2] === 'c4' && m[3] === 'e6') return "Nimzo/Queen's Indian";
      if (m[2] === 'c4' && m[3] === 'c5') return 'Benoni Defense';
      return 'Indian Defense';
    }
    if (m[1] === 'f5') return 'Dutch Defense';
    return "Queen's Pawn Opening";
  }
  if (m[0] === 'c4') return 'English Opening';
  if (m[0] === 'Nf3') return 'Reti Opening';
  if (m[0] === 'b3') return "Larsen's Opening";
  if (m[0] === 'f4') return "Bird's Opening";
  return 'Unknown Opening';
}
