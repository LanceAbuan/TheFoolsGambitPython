const { Chess } = require('chess.js');

const PST = {
  p: [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
  ],
  n: [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
  ],
  b: [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
  ],
  r: [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
  ],
  q: [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
  ],
  k: [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
  ]
};

const PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 20000 };

function evaluate(fen) {
  const game = new Chess();
  game.load(fen);
  const board = game.board();
  let score = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (!piece) continue;
      const row = piece.color === 'w' ? r : 7 - r;
      const val = PIECE_VALUES[piece.type] + PST[piece.type][row][c];
      score += piece.color === 'w' ? val : -val;
    }
  }
  const legalMoves = game.moves();
  const mobility = legalMoves.length * 2;
  score += game.turn() === 'w' ? mobility : -mobility;
  return score;
}

function minimax(game, depth, alpha, beta, maximizing) {
  if (depth === 0 || game.isGameOver()) {
    return evaluate(game.fen());
  }
  const moves = game.moves({ verbose: true }).sort((a, b) => {
    const aCapture = a.flags.includes('c');
    const bCapture = b.flags.includes('c');
    if (aCapture && !bCapture) return -1;
    if (!aCapture && bCapture) return 1;
    return 0;
  });
  if (maximizing) {
    let maxEval = -Infinity;
    for (const move of moves) {
      game.move(move.san);
      const eval_ = minimax(game, depth - 1, alpha, beta, false);
      game.undo();
      maxEval = Math.max(maxEval, eval_);
      alpha = Math.max(alpha, eval_);
      if (beta <= alpha) break;
    }
    return maxEval;
  } else {
    let minEval = Infinity;
    for (const move of moves) {
      game.move(move.san);
      const eval_ = minimax(game, depth - 1, alpha, beta, true);
      game.undo();
      minEval = Math.min(minEval, eval_);
      beta = Math.min(beta, eval_);
      if (beta <= alpha) break;
    }
    return minEval;
  }
}

function getAIMoveMinimax(fen, aiDepth = 3) {
  const game = new Chess();
  game.load(fen);
  const moves = game.moves({ verbose: true }).sort((a, b) => {
    const aCapture = a.flags.includes('c');
    const bCapture = b.flags.includes('c');
    if (aCapture && !bCapture) return -1;
    if (!aCapture && bCapture) return 1;
    return 0;
  });
  if (moves.length === 0) return null;
  const isMaximizing = game.turn() === 'w';
  let bestMove = moves[0].lan;
  let bestEval = isMaximizing ? -Infinity : Infinity;
  for (const move of moves) {
    game.move(move.san);
    const eval_ = minimax(game, aiDepth - 1, -Infinity, Infinity, !isMaximizing);
    game.undo();
    if (isMaximizing && eval_ > bestEval) {
      bestEval = eval_;
      bestMove = move.lan;
    } else if (!isMaximizing && eval_ < bestEval) {
      bestEval = eval_;
      bestMove = move.lan;
    }
  }
  return bestMove;
}

function getStatus(game) {
  if (game.isCheckmate()) return 'checkmate';
  if (game.isDraw() || game.isStalemate() || game.isThreefoldRepetition()) return 'draw';
  return 'active';
}

function getResult(game) {
  if (game.isCheckmate()) return `${game.turn() === 'w' ? 'Black' : 'White'} wins by checkmate`;
  if (game.isStalemate()) return 'Draw by stalemate';
  if (game.isDraw()) return 'Draw';
  return null;
}

function newState(game) {
  const history = game.history();
  return {
    fen: game.fen(),
    legal: game.moves({ verbose: true }).map(m => m.lan),
    turn: game.turn() === 'w' ? 'white' : 'black',
    pgn: history[history.length - 1] || '',
    status: getStatus(game),
    result: getResult(game),
    in_check: game.inCheck()
  };
}

const TRAINING_URL = process.env.TRAINING_URL || 'http://localhost:5001';

async function getNNMove(fen) {
  try {
    const url = `${TRAINING_URL}/api/train/evaluate`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen })
    });
    if (response.ok) {
      const data = await response.json();
      if (data.top_moves && data.top_moves.length > 0) {
        const best = data.top_moves[0];
        return best.uci;
      }
    }
  } catch (e) {
    console.log('NN move failed, falling back to minimax:', e.message);
  }
  return null;
}

module.exports = async (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(204).end();
  }

  if (req.method !== 'POST' && req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  let body = {};
  if (req.body) {
    body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
  }

  const path = req.url;

  try {
    if (path.includes('new-game')) {
      const game = new Chess();
      res.status(200).json(newState(game));
    } else if (path.includes('make-move')) {
      const { uci, fen } = body;
      if (!uci || !fen) {
        return res.status(400).json({ error: 'Missing uci or fen' });
      }
      const game = new Chess();
      game.load(fen);
      const from = uci.slice(0, 2);
      const to = uci.slice(2, 4);
      const promotion = uci.length > 4 ? uci[4] : undefined;
      let move;
      try {
        move = game.move({ from, to, promotion });
      } catch {
        move = null;
      }
      if (!move) {
        return res.status(200).json({ error: 'Illegal move', fen, legal: game.moves({ verbose: true }).map(m => m.lan), turn: game.turn() === 'w' ? 'white' : 'black', status: 'active' });
      }
      res.status(200).json(newState(game));
    } else if (path.includes('ai-move')) {
      const { fen, aiDepth = 3 } = body;
      if (!fen) {
        return res.status(400).json({ error: 'Missing fen' });
      }
      const game = new Chess();
      game.load(fen);
      let lan;
      const nnMove = await getNNMove(fen);
      if (nnMove) {
        lan = nnMove.slice(0, 2) + nnMove.slice(2, 4);
        if (nnMove.length > 4) lan += nnMove[4];
      } else {
        lan = getAIMoveMinimax(fen, aiDepth);
      }
      if (lan) {
        game.move({ from: lan.slice(0, 2), to: lan.slice(2, 4), promotion: lan.length > 4 ? lan[4] : undefined });
      }
      res.status(200).json(newState(game));
    } else if (path.includes('train')) {
      const trainingPath = path.replace('/api/', '');
      const url = `${TRAINING_URL}/api/${trainingPath}`;
      const options = {
        method: req.method,
        headers: { 'Content-Type': 'application/json' }
      };
      if (req.method === 'POST' && req.body) {
        options.body = req.body;
      }
      try {
        const response = await fetch(url, options);
        if (trainingPath.endsWith('/stream')) {
          res.setHeader('Content-Type', 'text/event-stream');
          res.setHeader('Cache-Control', 'no-cache');
          res.setHeader('Connection', 'keep-alive');
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(decoder.decode(value));
          }
          res.end();
          return;
        }
        const data = await response.json();
        res.status(response.status).json(data);
      } catch (e) {
        res.status(502).json({ error: 'Training server unavailable', detail: e.message });
      }
    } else if (path.includes('undo')) {
      const { fen } = body;
      if (!fen) {
        return res.status(400).json({ error: 'Missing fen' });
      }
      const game = new Chess();
      game.load(fen);
      game.undo();
      game.undo();
      res.status(200).json(newState(game));
    } else {
      const game = new Chess();
      res.status(200).json(newState(game));
    }
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
