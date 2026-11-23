//! PUCT MCTS (WP-F C2) — a bit-faithful port of the python `MCTSSearch`
//! (`training/selfplay.py`, the WP-C search v2) over the C1 arena tree, with
//! virtual loss for in-search parallelism.
//!
//! Contract (must match python `MCTSSearch.search` exactly):
//! - PUCT selection: `score = -child.value_mean + cpuct * child.prior *
//!   sqrt(max(parent.visits, 1)) / (1 + child.visits)`; children iterate in
//!   python-chess `board.legal_moves` order and the strict `>` keeps the
//!   FIRST child on ties.
//! - Root expansion evaluates the root once (softmax over legal logits in
//!   f64), mixes Dirichlet noise `(1-eps)*p + eps*noise` only when there are
//!   2+ legal moves, and never stores the root eval as a visit.
//! - Terminal positions (`is_game_over(claim_draw=True)`, i.e. including
//!   fifty-move and threefold-repetition CLAIMS) get value -1.0 when the
//!   side to move is checkmated, else 0.0 — `terminal_value`.
//! - Negamax backup: the leaf value is credited to the leaf and sign-flipped
//!   at every ply up to and including the root.
//! - Forced-move shortcut: exactly one legal move → `visits[move] = sims`,
//!   root value = `root.value_mean()` (0.0 — python never credits the root
//!   eval; replicated as a documented quirk).
//! - Output: `(4672,) f32` visit counts indexed by the 73x64 policy encoding
//!   + root value + argmax-visit move, identical shapes to `MCTSResult`.
//!
//! Parallelism: simulations run in WAVES of `batch_size` descents. Selections
//! inside a wave apply virtual loss (pending visits + pessimistic effective
//! Q) so concurrent paths diverge; leaves of a wave are evaluated in ONE
//! evaluator call per wave — the pattern C3's python callback will drive
//! (mirroring how `BatchNNEvaluator` aggregates). `batch_size = 1` reduces to
//! python-sequential semantics exactly, which is the C6 deterministic parity
//! mode. Threaded execution of a wave (the tree work itself is GIL-free and
//! arena-partitioned) lands with C3/WP-H.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma};

use crate::encoder::{policy_index, POLICY_MOVES};
use crate::position::{Position, TranspositionKey};
use crate::tree::Tree;

/// Knobs for one search — mirrors python `SearchConfig` (puct fields) plus
/// the C2 parallelism/seed knobs python does not have yet.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// PUCT exploration constant (python `SearchConfig.cpuct`).
    pub cpuct: f64,
    /// Root Dirichlet noise alpha (python `dirichlet_alpha`).
    pub dirichlet_alpha: f64,
    /// Root noise mixing weight (python `dirichlet_epsilon`).
    pub dirichlet_epsilon: f64,
    /// Apply root noise at all (python `root_noise`).
    pub root_noise: bool,
    /// Simulations per search (python `num_simulations`).
    pub num_simulations: u32,
    /// Virtual-loss magnitude for in-wave selections.
    pub virtual_loss: f64,
    /// Selections per wave (1 = python-sequential parity mode).
    pub batch_size: usize,
    /// Seed for the Dirichlet draw (python seeds via numpy's global rng).
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            cpuct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            root_noise: true,
            num_simulations: 64,
            virtual_loss: 1.0,
            batch_size: 1,
            seed: 0,
        }
    }
}

/// Batched position evaluator — the Rust mirror of python's `Evaluator`
/// protocol: raw UNMASKED policy logits over the 4672 encoding (row-major,
/// one row per position) and a scalar value per position from the side to
/// move's perspective.
pub trait Evaluator {
    /// Evaluate `batch` (never empty). Returns `(logits, values)` with
    /// `logits.len() == batch.len() * POLICY_MOVES` and
    /// `values.len() == batch.len()`.
    fn evaluate(&mut self, batch: &[Position]) -> (Vec<f32>, Vec<f32>);

    /// History-aware evaluation (WP-H H3). `paths[i]` is the chronological
    /// position sequence from the game ROOT through `batch[i]` inclusive
    /// (`paths[i].last() == &batch[i]`). The 104-plane v2 encoder derives its
    /// deep-history and repetition planes from it; python truncates that
    /// history to the LAST 7 positions (`v2_history_from_stack`), and the
    /// same truncation is the evaluator's job. Default: history-free, so
    /// v1-style evaluators are untouched.
    fn evaluate_history(
        &mut self,
        batch: &[Position],
        _paths: &[Vec<Position>],
    ) -> (Vec<f32>, Vec<f32>) {
        self.evaluate(batch)
    }
}

/// Outcome of one root search — the python `MCTSResult` contract.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// (4672,) f32 visit counts over the 73x64 policy encoding.
    pub visit_counts: Vec<f32>,
    /// Root value from the root side-to-move's perspective
    /// (mean of backed-up values; 0.0 in the forced-move/terminal shortcuts,
    /// exactly like python).
    pub root_value: f64,
    /// Most-visited root move (`None` when the root is terminal / has no
    /// legal moves — python `visited_root=False`).
    pub chosen_move: Option<chess::ChessMove>,
    /// False when the search returned early (terminal root / no legal moves).
    pub visited_root: bool,
}

/// Game history strictly before the search root, for repetition state and
/// v2 history planes (WP-H H3).
///
/// Contract: `positions` is chronological (oldest first) and STRICTLY before
/// the search root — for a game played from `P0` with the root at `Pk`, it is
/// `[P0 .. P(k-1)]`. `moves[i]` is the move that produced `positions[i]`
/// (`None` for the game root `P0`, which is where python-chess's pop-walk
/// stops when the move stack empties). `root_move` is the move that produced
/// the search root itself (`None` when the root IS the game start) — without
/// it the repetition walk would stop at the root and never see positions
/// before it, under-detecting threefold claims against python (whose search
/// board carries the full move stack).
#[derive(Clone, Debug, Default)]
pub struct GameHistory {
    pub positions: Vec<Position>,
    pub moves: Vec<Option<chess::ChessMove>>,
    /// The move that produced the search root (`None` at the game start).
    pub root_move: Option<chess::ChessMove>,
}

impl GameHistory {
    /// An empty history (fresh game: the search root IS the game start).
    pub fn empty() -> GameHistory {
        GameHistory::default()
    }

    /// Replay `start` through `ucis` (validated legal, python-chess state
    /// semantics) into (history, root) — the test harness helper.
    pub fn replay(start: &Position, ucis: &[&str]) -> Result<(GameHistory, Position), String> {
        let mut history = GameHistory::empty();
        let mut pos = *start;
        history.positions.push(pos);
        history.moves.push(None);
        for uci in ucis {
            let m = find_legal_uci(&pos, uci)?;
            history.root_move = Some(m);
            pos = pos.make(m);
            history.positions.push(pos);
            history.moves.push(Some(m));
        }
        // Drop the last entry: `positions` must hold only positions BEFORE
        // the returned root, and the root's own move lives in `root_move`.
        history.positions.pop();
        history.moves.pop();
        Ok((history, pos))
    }
}

/// Find the legal move matching a UCI string (python-chess `Move.from_uci` +
/// legality).
pub fn find_legal_uci(pos: &Position, uci: &str) -> Result<chess::ChessMove, String> {
    let uci = uci.trim().to_ascii_lowercase();
    if uci.len() != 4 && uci.len() != 5 {
        return Err(format!("bad UCI move: {uci:?}"));
    }
    let src = crate::position::parse_square(&uci[0..2])?;
    let dst = crate::position::parse_square(&uci[2..4])?;
    let promo = if uci.len() == 5 {
        Some(crate::position::parse_promo(&uci[4..5])?)
    } else {
        None
    };
    for m in pos.legal_moves() {
        if m.get_source() == src && m.get_dest() == dst && m.get_promotion() == promo {
            return Ok(m);
        }
    }
    Err(format!("illegal move {uci} in position {}", pos.to_fen()))
}

/// python `terminal_value`: a TERMINAL position's value from the side to
/// move's perspective (§10 sign fix: the mated side gets -1.0).
fn terminal_value(pos: &Position) -> f64 {
    if pos.is_checkmate() {
        -1.0
    } else {
        0.0
    }
}

/// The repetition-aware position sequence: game history + the current
/// simulation's path (root ..= current node). Provides python-chess
/// `is_repetition`-style suffix counting over `_transposition_key`s.
struct Seq<'a> {
    base_pos: &'a [Position],
    base_keys: Vec<TranspositionKey>,
    base_moves: &'a [Option<chess::ChessMove>],
    path_pos: Vec<Position>,
    path_keys: Vec<TranspositionKey>,
    path_moves: Vec<Option<chess::ChessMove>>,
}

impl<'a> Seq<'a> {
    fn new(history: &'a GameHistory) -> Seq<'a> {
        debug_assert_eq!(history.positions.len(), history.moves.len());
        Seq {
            base_pos: &history.positions,
            base_keys: history.positions.iter().map(|p| p.transposition_key()).collect(),
            base_moves: &history.moves,
            path_pos: Vec::new(),
            path_keys: Vec::new(),
            path_moves: Vec::new(),
        }
    }

    fn clear_path(&mut self) {
        self.path_pos.clear();
        self.path_keys.clear();
        self.path_moves.clear();
    }

    fn push_path(&mut self, pos: &Position, mv: Option<chess::ChessMove>) {
        self.path_pos.push(*pos);
        self.path_keys.push(pos.transposition_key());
        self.path_moves.push(mv);
    }

    fn len(&self) -> usize {
        self.base_pos.len() + self.path_pos.len()
    }

    /// Chronological snapshot of the whole sequence (base history + path).
    /// Captured per wave-leaf BEFORE the next selection clears the path, so
    /// the batched history-aware evaluator sees each leaf's own ancestry.
    fn full_path(&self) -> Vec<Position> {
        let mut v = Vec::with_capacity(self.base_pos.len() + self.path_pos.len());
        v.extend_from_slice(self.base_pos);
        v.extend_from_slice(&self.path_pos);
        v
    }

    fn pos(&self, i: usize) -> &Position {
        if i < self.base_pos.len() {
            &self.base_pos[i]
        } else {
            &self.path_pos[i - self.base_pos.len()]
        }
    }

    fn key(&self, i: usize) -> &TranspositionKey {
        if i < self.base_keys.len() {
            &self.base_keys[i]
        } else {
            &self.path_keys[i - self.base_keys.len()]
        }
    }

    /// The move that produced position `i` (`None` when unknown/oldest).
    fn move_into(&self, i: usize) -> Option<chess::ChessMove> {
        if i == 0 {
            return None;
        }
        if i < self.base_moves.len() {
            self.base_moves[i]
        } else {
            let j = i - self.base_moves.len();
            self.path_moves.get(j).copied().flatten()
        }
    }

    /// Occurrences of `key` in the reversible suffix of positions ending at
    /// `from` (inclusive) — python-chess pops moves off the stack and stops
    /// at `is_irreversible`; this is the same walk without the pops.
    fn suffix_count(&self, from: usize, key: &TranspositionKey) -> u32 {
        let mut i = from;
        let mut count = 0u32;
        loop {
            if self.key(i) == key {
                count += 1;
            }
            if i == 0 {
                break;
            }
            let mv = match self.move_into(i) {
                Some(mv) => mv,
                None => break,
            };
            if self.pos(i - 1).is_irreversible(&mv) {
                break;
            }
            i -= 1;
        }
        count
    }
}

/// python `board.is_game_over(claim_draw=True)` for the LAST position of
/// `seq` (which must be `leaf`), with `legal` its legal moves:
/// mate/stalemate, insufficient material, fifty-move (clock >= 100, or the
/// clock >= 99 variant with a legal non-zeroing move reaching 100 — python
/// `can_claim_fifty_moves`), and threefold-repetition claims (now, or reached
/// by one legal move — python `can_claim_threefold_repetition`).
/// Seventyfive-move / fivefold are subsumed by their claimable counterparts.
fn is_game_over_claim(seq: &Seq, leaf: &Position, legal: &[chess::ChessMove]) -> bool {
    if legal.is_empty() {
        return true; // checkmate or stalemate
    }
    if leaf.is_insufficient_material() {
        return true;
    }
    let clock = leaf.halfmove_clock();
    if clock >= 100 {
        return true; // is_fifty_moves (legal moves exist — checked above)
    }
    let last = seq.len() - 1;
    let key = *seq.key(last);
    if seq.suffix_count(last, &key) >= 3 {
        return true; // threefold now
    }
    if clock >= 99 {
        for m in legal {
            if !leaf.is_zeroing(m) {
                let next = leaf.make(*m);
                if next.halfmove_clock() >= 100 && next.has_legal_move() {
                    return true; // fifty reachable by one quiet move
                }
            }
        }
    }
    for m in legal {
        let next = leaf.make(*m);
        let next_key = next.transposition_key();
        if seq.suffix_count(last, &next_key) >= 2 {
            return true; // threefold reachable by one move
        }
    }
    false
}

/// `board.is_game_over(claim_draw=True)` for `root` given the game's prior
/// positions (`history`, see [`GameHistory`]) — the H3 game loop's per-ply
/// termination check, the same probe the search itself uses at root/leaves.
pub fn is_game_over_claim_draw(root: &Position, history: &GameHistory) -> bool {
    let mut seq = Seq::new(history);
    seq.push_path(root, history.root_move);
    let legal = root.legal_moves();
    is_game_over_claim(&seq, root, &legal)
}

/// Run one search. `root` is the position to search; `history` carries the
/// game's prior positions for repetition claims (empty for a fresh game).
pub fn search(
    root: &Position,
    history: &GameHistory,
    cfg: &SearchConfig,
    evaluator: &mut dyn Evaluator,
) -> SearchResult {
    let zeros = || vec![0f32; POLICY_MOVES];
    let legal_pairs = root.legal_moves_python_order();
    let plain: Vec<chess::ChessMove> = legal_pairs.iter().map(|(_, m)| *m).collect();

    if plain.is_empty() {
        return SearchResult {
            visit_counts: zeros(),
            root_value: terminal_value(root),
            chosen_move: None,
            visited_root: false,
        };
    }

    let mut tree = Tree::new();
    let mut seq = Seq::new(history);
    seq.push_path(root, history.root_move);

    // Root expansion (python `_evaluate_and_expand(root, add_root_noise=True)`).
    if is_game_over_claim(&seq, root, &plain) {
        let tv = terminal_value(root);
        return SearchResult {
            visit_counts: zeros(),
            root_value: tv,
            chosen_move: None,
            visited_root: false,
        };
    }
    let root_path = seq.full_path();
    let (logits, values) =
        evaluator.evaluate_history(std::slice::from_ref(root), std::slice::from_ref(&root_path));
    debug_assert!(values.len() == 1 && logits.len() >= POLICY_MOVES);
    let noise: Option<Vec<f64>> = if cfg.root_noise && plain.len() > 1 {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        Some(dirichlet(&mut rng, cfg.dirichlet_alpha, plain.len()))
    } else {
        None
    };
    let root_idx = tree.root();
    expand(&mut tree, root_idx, &legal_pairs, &logits[..POLICY_MOVES], noise.as_deref(), cfg);

    // Forced move: python returns early with visits[move] = sims and the
    // (unvisited) root's value_mean — which is 0.0. Documented quirk.
    if plain.len() == 1 {
        let mut counts = zeros();
        counts[legal_pairs[0].0] = cfg.num_simulations as f32;
        return SearchResult {
            visit_counts: counts,
            root_value: tree.node(tree.root()).value_mean(),
            chosen_move: Some(plain[0]),
            visited_root: true,
        };
    }

    // Simulations in waves.
    let mut remaining = cfg.num_simulations;
    let batch = cfg.batch_size.max(1);
    while remaining > 0 {
        let wave = (remaining as usize).min(batch);
        // (path, leaf position, node) per in-wave leaf; terminals are backed
        // up immediately.
        let mut wave_leaves: Vec<(Vec<u32>, Position, u32, Vec<Position>)> =
            Vec::with_capacity(wave);
        for _ in 0..wave {
            let mut pos = *root;
            let mut node = tree.root();
            let mut path = vec![node];
            seq.clear_path();
            seq.push_path(&pos, None);
            loop {
                let descend = {
                    let n = tree.node(node);
                    n.expanded && !n.children.is_empty() && !n.terminal
                };
                if !descend {
                    break;
                }
                let child = select_child(&tree, node, cfg);
                tree.add_virtual_loss(child);
                let mv = tree.node(child).mv.expect("non-root node has a move");
                pos = pos.make(mv);
                seq.push_path(&pos, Some(mv));
                path.push(child);
                node = child;
            }

            let n = tree.node(node);
            if n.terminal {
                let v = n.terminal_value;
                tree.backup(&path, v);
            } else if wave_leaves.iter().any(|(_, _, id, _)| *id == node) {
                // Selected twice within one wave (virtual loss did not fully
                // deter it): python-sequential semantics expand immediately —
                // do the single eval inline.
                let value = expand_leaf(&mut tree, node, &pos, cfg, evaluator);
                tree.backup(&path, value);
            } else if is_game_over_claim(&seq, &pos, &pos.legal_moves()) {
                let tv = terminal_value(&pos);
                let n = tree.node_mut(node);
                n.terminal = true;
                n.terminal_value = tv;
                n.expanded = true;
                tree.backup(&path, tv);
            } else {
                let leaf_path = seq.full_path();
                wave_leaves.push((path, pos, node, leaf_path));
            }
        }

        if !wave_leaves.is_empty() {
            let batch_pos: Vec<Position> = wave_leaves.iter().map(|(_, p, _, _)| *p).collect();
            let batch_paths: Vec<Vec<Position>> =
                wave_leaves.iter().map(|(_, _, _, path)| path.clone()).collect();
            let (logits, values) = evaluator.evaluate_history(&batch_pos, &batch_paths);
            debug_assert_eq!(values.len(), batch_pos.len());
            debug_assert!(logits.len() >= batch_pos.len() * POLICY_MOVES);
            for (i, (path, pos, node, _leaf_path)) in wave_leaves.into_iter().enumerate() {
                // values[i] is THIS leaf eval - pass the matching row through
                // so the backup credits the right node.
                let value = expand_leaf_with_logits(
                    &mut tree,
                    node,
                    &pos,
                    &logits[i * POLICY_MOVES..(i + 1) * POLICY_MOVES],
                    values[i] as f64,
                    cfg,
                );
                tree.backup(&path, value);
            }
        }
        remaining -= wave as u32;
    }

    // Export: visit counts over the 4672 encoding + argmax-visit move (first
    // child wins ties, matching python's `max`).
    let mut counts = zeros();
    let root_children = tree.node(tree.root()).children.clone();
    let mut best: Option<(u32, chess::ChessMove)> = None;
    for &c in &root_children {
        let child = tree.node(c);
        let mv = child.mv.expect("non-root node has a move");
        counts[policy_index(root, &mv)] = child.visit_count as f32;
        let better = match best {
            None => true,
            Some((bc, _)) => child.visit_count > bc,
        };
        if better {
            best = Some((child.visit_count, mv));
        }
    }
    SearchResult {
        visit_counts: counts,
        root_value: tree.node(tree.root()).value_mean(),
        chosen_move: best.map(|(_, mv)| mv),
        visited_root: true,
    }
}

/// PUCT child selection (python `_select_child`), over effective (virtual
/// loss inclusive) statistics.
fn select_child(tree: &Tree, node: u32, cfg: &SearchConfig) -> u32 {
    let n = tree.node(node);
    let sqrt_parent = (n.effective_visits().max(1) as f64).sqrt();
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for &c in &n.children {
        let child = tree.node(c);
        let exploit = -child.effective_value_mean(cfg.virtual_loss);
        let explore =
            cfg.cpuct * child.prior * sqrt_parent / (1.0 + child.effective_visits() as f64);
        let score = exploit + explore;
        if score > best_score {
            best_score = score;
            best = Some(c);
        }
    }
    best.expect("select_child called on a childless node")
}

/// Expand an unexpanded leaf with a fresh evaluation (no noise below the
/// root) and return the eval value — the python `_evaluate_and_expand` leaf
/// path.
fn expand_leaf(
    tree: &mut Tree,
    node: u32,
    pos: &Position,
    cfg: &SearchConfig,
    evaluator: &mut dyn Evaluator,
) -> f64 {
    let (logits, values) = evaluator.evaluate(std::slice::from_ref(pos));
    let legal_pairs = pos.legal_moves_python_order();
    expand(tree, node, &legal_pairs, &logits[..POLICY_MOVES], None, cfg);
    debug_assert_eq!(values.len(), 1);
    values[0] as f64
}

/// Wave-leaf expansion from a batch row (no noise below the root). The value
/// is taken from the evaluator VALUES row when provided shape-wise; the
/// logits row drives the priors.
fn expand_leaf_with_logits(
    tree: &mut Tree,
    node: u32,
    pos: &Position,
    logits: &[f32],
    value: f64,
    cfg: &SearchConfig,
) -> f64 {
    let legal_pairs = pos.legal_moves_python_order();
    expand(tree, node, &legal_pairs, &logits[..POLICY_MOVES], None, cfg);
    value
}

/// Create children for `node` from its legal moves (python-chess order) with
/// softmax priors over the legal logits, optionally mixed with root noise.
fn expand(
    tree: &mut Tree,
    node: u32,
    legal_pairs: &[(usize, chess::ChessMove)],
    logits: &[f32],
    noise: Option<&[f64]>,
    cfg: &SearchConfig,
) {
    let raw: Vec<f64> = legal_pairs.iter().map(|(idx, _)| logits[*idx] as f64).collect();
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = raw.iter().map(|r| (r - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    for (i, (_, m)) in legal_pairs.iter().enumerate() {
        let mut prior = exps[i] / sum;
        if let Some(nz) = noise {
            prior = (1.0 - cfg.dirichlet_epsilon) * prior + cfg.dirichlet_epsilon * nz[i];
        }
        tree.push_child(node, Some(*m), prior);
    }
    tree.node_mut(node).expanded = true;
}

/// Dirichlet(alpha, n) sample via normalized Gammas (python draws from
/// numpy's Dirichlet — same distribution, independent stream).
fn dirichlet(rng: &mut StdRng, alpha: f64, n: usize) -> Vec<f64> {
    let gamma = Gamma::new(alpha, 1.0).expect("dirichlet alpha > 0");
    let xs: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f64 = xs.iter().sum();
    xs.into_iter().map(|x| x / sum).collect()
}

/// Mirror of python `select_move_from_visits`: sample a move index from the
/// visit distribution at temperature `tau` (tau <= 1e-6 degenerates to the
/// argmax — first index on ties, matching python's `max`).
pub fn select_by_temperature(
    legal_indices: &[usize],
    visit_counts: &[f32],
    temperature: f64,
    rng: &mut StdRng,
) -> usize {
    assert!(!legal_indices.is_empty(), "no legal moves to select from");
    if temperature <= 1e-6 {
        let mut best = legal_indices[0];
        for &idx in &legal_indices[1..] {
            if visit_counts[idx] > visit_counts[best] {
                best = idx;
            }
        }
        return best;
    }
    let counts: Vec<f64> = legal_indices
        .iter()
        .map(|&idx| visit_counts[idx].max(0.0) as f64)
        .collect();
    let powered: Vec<f64> = counts.iter().map(|c| c.powf(1.0 / temperature)).collect();
    let total: f64 = powered.iter().sum();
    if total <= 0.0 {
        return legal_indices[0];
    }
    let draw = rand::Rng::gen::<f64>(rng) * total;
    let mut acc = 0.0;
    for (i, p) in powered.iter().enumerate() {
        acc += p;
        if draw < acc {
            return legal_indices[i];
        }
    }
    legal_indices[legal_indices.len() - 1]
}
