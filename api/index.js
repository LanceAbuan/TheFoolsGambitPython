// Simple Node.js handler — avoids Vercel Python runtime issues
// We'll do the chess logic in JavaScript instead of Python

import { Chess } from 'chess.js';

export default async function handler(req) {
  const { __path = '', fen, uci, depth = 3 } = await req.json() || {};

  const headers = {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-store',
  };

  try {
    let data;
    if (__path === '/new-game' || !__path) {
      data = newGame();
    } else if (__path === '/make-move') {
      data = playMove(fen, uci);
    } else if (__path === '/ai-move') {
      data = aiMove(fen, depth);
    } else {
      data = newGame();
    }
    return new Response(JSON.stringify(data), { status: 200, headers });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), { status: 500, headers });
  }
}

// ============================================================
// PIECE-SQUARE TABLES
// ============================================================
const PAWN_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [5, 5, 5, 5, 5, 5, 5, 5],
  [1, 1, 2, 3, 3, 2, 1, 1],
  [0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5],
  [0, 0, 0, 2, 2, 0, 0, 0],
  [0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
  [0.5, 1, 1, -2, -2, 1, 1, 0.5],
  [0, 0, 0, 0, 0, 0, 0, 0],
];

const KNIGHT_TABLE = [
  [-5, -4, -3, -3, -3, -3, -4, -5],
  [-4, -2, 0, 0, 0, 0, -2, -4],
  [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
  [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
  [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
  [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
  [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
  [-5, -4, -3, -3, -3, -3, -4, -5],
];

const BISHOP_TABLE = [
  [-2, -1, -1, -1, -1, -1, -1, -2],
  [-1, 0, 0, 0, 0, 0, 0, -1],
  [-1, 0, 1, 1, 1, 1, 0, -1],
  [-1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1],
  [-1, 0, 0.5, 1, 1, 0.5, 0, -1],
  [-1, 1, 1, 1, 1, 1, 1, -1],
  [-1, 0.5, 0, 0, 0, 0, 0.5, -1],
  [-2, -1, -1, -1, -1, -1, -1, -2],
];

const ROOK_TABLE = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0.5, 1, 1, 1, 1, 1, 1, 0.5],
  [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
  [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
  [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
  [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
  [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
  [0, 0, 0, 0.5, 0.5, 0, 0, 0],
];

const QUEEN_TABLE = [
  [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
  [-1, 0, 0, 0, 0, 0, 0, -1],
  [-1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1],
  [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
  [0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
  [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
  [-1, 0, 0.5, 0, 0, 0, 0, -1],
  [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
];

const KING_MIDDLE_TABLE = [
  [-5, -4, -3, -3, -3, -3, -4, -5],
  [-4, -2, 0, 0, 0, 0, -2, -4],
  [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
  [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
  [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
  [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
  [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
  [-5, -4, -3, -3, -3, -3, -4, -5],
];

const PSQT = {
  p: PAWN_TABLE,
  n: KNIGHT_TABLE,
  b: BISHOP_TABLE,
  r: ROOK_TABLE,
  q: QUEEN_TABLE,
  k: KING_MIDDLE_TABLE,
};

const PIECE_VALUES = {
  p: 100,
  n: 320,
  b: 330,
  r: 500,
  q: 900,
  k: 20000,
};

// ============================================================
// GAME ENGINE
// ============================================================
function stateResponse(game) {
  const board = game.board();
  const status = getGameStatus(game);
  return {
    fen: game.fen(),
    legal: game.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion || '')),
    turn: game.turn() === 'w' ? 'white' : 'black',
    pgn: '',
    status,
    result: status === 'active' ? null : getGameResult(game),
    in_check: game.in_check(),
  };
}

function getGameStatus(game) {
  if (game.game_over()) {
    if (game.in_checkmate()) return 'checkmate';
    if (game.in_stalemate()) return 'stalemate';
    if (game.in_draw()) return 'draw';
    if (game.in_threefold_repetition()) return 'draw';
    if (game.insufficient_material()) return 'draw';
    if (game.half_moves() >= 100) return 'draw';
  }
  return 'active';
}

function getGameResult(game) {
  if (game.in_checkmate()) {
    const winner = game.turn() === 'w' ? 'Black' : 'White';
    return `Checkmate! ${winner} wins.`;
  }
  if (game.in_stalemate()) return 'Stalemate. Draw.';
  if (game.insufficient_material()) return 'Draw — insufficient material.';
  if (game.in_threefold_repetition()) return 'Draw — repetition.';
  if (game.half_moves() >= 100) return 'Draw — 50-move rule.';
  return null;
}

function newGame() {
  const game = new Chess();
  return stateResponse(game);
}

function playMove(fen, uci) {
  const game = new Chess(fen);
  try {
    const move = game.move({ from: uci.substring(0, 2), to: uci.substring(2, 4), promotion: uci.length > 4 ? uci[4] : undefined });
    if (!move) return { error: 'Illegal move', status: 'active' };
    const res = stateResponse(game);
    res.pgn = move.san;
    return res;
  } catch (e) {
    return { error: 'Illegal move', status: 'active' };
  }
}

// ============================================================
// AI ENGINE (minimax + alpha-beta)
// ============================================================
function evaluate(game) {
  let total = 0;
  for (const [row, rankRow] of game.board().entries()) {
    for (const [col, piece] of rankRow.entries()) {
      if (!piece) continue;
      const pst = PSQT[piece.type] || [];
      const pstVal = pst[row]?.[col] || 0;
      const val = (PIECE_VALUES[piece.type] || 0) + pstVal;
      total += piece.color === 'w' ? val : -val;
    }
  }
  return total;
}

function orderMoves(game) {
  const moves = game.moves({ verbose: true });
  moves.sort((a, b) => {
    let scoreA = 0, scoreB = 0;
    const capA = game.get(a.from).at(a.to);
    const capB = game.get(b.from).at(b.to);
    if (capA) scoreA += (PIECE_VALUES[capA.type] || 0) * 10;
    if (capB) scoreB += (PIECE_VALUES[capB.type] || 0) * 10;
    if (a.flags.includes('c')) scoreA += 10;
    if (b.flags.includes('c')) scoreB += 10;
    return scoreB - scoreA;
  });
  return moves;
}

function minimax(game, depth, alpha, beta, maximizing) {
  if (depth === 0) return { score: evaluate(game) };
  if (game.game_over()) {
    if (game.in_checkmate()) return { score: maximizing ? -9999999 + (100 - depth) : 9999999 - (100 - depth) };
    return { score: 0 };
  }

  const moves = orderMoves(game);
  if (moves.length === 0) return { score: evaluate(game) };

  let bestMove = null;
  if (maximizing) {
    let maxEval = -Infinity;
    for (const move of moves) {
      const moveObj = game.move(move);
      const eval_ = minimax(game, depth - 1, alpha, beta, false).score;
      game.undo();
      if (eval_ > maxEval) { maxEval = eval_; bestMove = move; }
      alpha = Math.max(alpha, eval_);
      if (beta <= alpha) break;
    }
    return { score: maxEval, move: bestMove };
  } else {
    let minEval = Infinity;
    for (const move of moves) {
      const moveObj = game.move(move);
      const eval_ = minimax(game, depth - 1, alpha, beta, true).score;
      game.undo();
      if (eval_ < minEval) { minEval = eval_; bestMove = move; }
      beta = Math.min(beta, eval_);
      if (beta <= alpha) break;
    }
    return { score: minEval, move: bestMove };
  }
}

function aiMove(fen, depth = 3) {
  const game = new Chess(fen);
  const moves = game.moves({ verbose: true });
  if (moves.length === 0) return stateResponse(game);

  // Check for immediate checkmate
  for (const move of moves) {
    game.move(move);
    if (game.in_checkmate()) {
      game.undo();
      const res = stateResponse(game);
      res.pgn = move.san;
      return res;
    }
    game.undo();
  }

  const maximizing = game.turn() === 'w';
  const result = minimax(game, depth, -Infinity, Infinity, maximizing);
  if (!result.move) return stateResponse(game);

  const moveResult = game.move(result.move);
  const res = stateResponse(game);
  res.pgn = moveResult?.san || '';
  return res;
}

export const config = { runtime: 'edge' };
