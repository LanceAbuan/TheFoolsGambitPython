//! Arena-based MCTS tree (WP-F C1).
//!
//! The tree is a flat `Vec<Node>`; nodes reference each other by `u32` arena
//! index, so the whole tree lives in one contiguous growable buffer with no
//! per-node heap allocation (children are `Vec<u32>` index lists) and no
//! pointer chasing across allocations. No GIL: nothing here touches Python.
//!
//! Bookkeeping mirrors the python `_Node` (`training/selfplay.py`, WP-C)
//! field-for-field so the WP-F C6 parity gate can compare trees directly:
//!
//! | python `_Node`    | here                          |
//! |-------------------|-------------------------------|
//! | `move`            | `mv` (None at the root)       |
//! | `prior`           | `prior` (f64, post-noise)     |
//! | `children`        | `children: Vec<u32>` (order = python-chess move order) |
//! | `visit_count`     | `visit_count: u32`            |
//! | `value_sum`       | `value_sum: f64`              |
//! | `terminal` / `terminal_value` | same              |
//! | `expanded`        | `expanded`                    |
//!
//! Value convention: `value_sum` accumulates from THIS node's side-to-move
//! perspective (negamax backup flips the sign at every ply), and parent PUCT
//! exploitation reads `-child.value_mean()`.
//!
//! C2 adds one field python does not have: `pending`, the number of in-flight
//! (virtual-loss) selections not yet backed up. `visit_count`/`value_sum`
//! stay COMMITTED statistics; selectors use the effective accessors below.
//! With no in-flight selections (sequential mode) the effective values reduce
//! exactly to the python quantities.

use chess::ChessMove;

/// One search-tree node. See the module docs for the python `_Node` mapping.
#[derive(Clone, Debug)]
pub struct Node {
    /// Arena index of the parent (`None` for the root).
    pub parent: Option<u32>,
    /// Arena indices of the children, in python-chess `board.legal_moves`
    /// order (insertion order) — the order `_select_child`'s strict `>`
    /// tie-break resolves toward, and part of the parity contract.
    pub children: Vec<u32>,
    /// Move that led INTO this node (`None` only at the root).
    pub mv: Option<ChessMove>,
    /// PUCT prior (post-softmax, post-root-noise). f64 to match python floats.
    pub prior: f64,
    /// Committed visit count (virtual-loss pendings NOT included).
    pub visit_count: u32,
    /// Sum of backed-up values, this node's side-to-move perspective.
    pub value_sum: f64,
    /// Children were created, or the position was found terminal.
    pub expanded: bool,
    /// Position is game-over (mate/stalemate/insufficient/fifty/threefold).
    pub terminal: bool,
    /// Terminal value from this node's side-to-move perspective (mate -1.0,
    /// every other terminal 0.0). Meaningful only when `terminal`.
    pub terminal_value: f64,
    /// In-flight selections (virtual loss) not yet backed up. C2 parallelism
    /// bookkeeping; always 0 in sequential mode.
    pub pending: u32,
}

impl Node {
    /// A fresh, unvisited leaf node.
    pub fn new(parent: Option<u32>, mv: Option<ChessMove>, prior: f64) -> Node {
        Node {
            parent,
            children: Vec::new(),
            mv,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            expanded: false,
            terminal: false,
            terminal_value: 0.0,
            pending: 0,
        }
    }

    /// Mean value from this node's side-to-move perspective
    /// (python `_Node.value_mean`: 0.0 while unvisited).
    pub fn value_mean(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }

    /// Committed + in-flight selections.
    pub fn effective_visits(&self) -> u32 {
        self.visit_count + self.pending
    }

    /// Mean value with virtual loss folded in: every pending selection counts
    /// as a visit with value +`virtual_loss` (a LOSS for the PARENT, because
    /// parents exploit `-value_mean`), so concurrent descents diverge instead
    /// of piling onto the same leaf. Reduces to [`Node::value_mean`] when
    /// `pending == 0`.
    pub fn effective_value_mean(&self, virtual_loss: f64) -> f64 {
        let n = self.effective_visits();
        if n == 0 {
            0.0
        } else {
            (self.value_sum + virtual_loss * self.pending as f64) / n as f64
        }
    }
}

/// Arena owner: `nodes[0]` is always the root.
#[derive(Clone, Debug, Default)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// A tree holding only the root (unexpanded, unvisited) — the state
    /// `MCTSSearch.search` starts from.
    pub fn new() -> Tree {
        Tree {
            nodes: vec![Node::new(None, None, 0.0)],
        }
    }

    /// Arena index of the root.
    pub fn root(&self) -> u32 {
        0
    }

    /// Number of nodes allocated (root included).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the arena holds only the root.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    /// Read access to one node.
    #[inline]
    pub fn node(&self, idx: u32) -> &Node {
        &self.nodes[idx as usize]
    }

    /// Mutable access to one node.
    #[inline]
    pub fn node_mut(&mut self, idx: u32) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    /// Append a child to `parent` (at the END of the child list — callers
    /// preserve python-chess move order) and return its arena index.
    pub fn push_child(&mut self, parent: u32, mv: Option<ChessMove>, prior: f64) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node::new(Some(parent), mv, prior));
        self.nodes[parent as usize].children.push(idx);
        idx
    }

    /// Apply one in-flight selection's virtual loss to `child`: count it as a
    /// pending visit so selectors see the extra visit and the pessimistic
    /// effective value (see [`Node::effective_value_mean`]). Balanced by
    /// exactly one [`Tree::backup`] through the same node.
    pub fn add_virtual_loss(&mut self, child: u32) {
        self.node_mut(child).pending += 1;
    }

    /// Back one simulation up along `path` (root ..= leaf, in descent order):
    /// +1 visit and +`value` at the leaf, sign-flipped at every ply
    /// (negamax), removing one virtual-loss pending per node below the root.
    pub fn backup(&mut self, path: &[u32], mut value: f64) {
        for (depth, &idx) in path.iter().enumerate().rev() {
            let n = self.node_mut(idx);
            n.visit_count += 1;
            n.value_sum += value;
            if depth > 0 {
                n.pending = n.pending.saturating_sub(1);
            }
            value = -value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_linking_and_backup_signs() {
        let mut t = Tree::new();
        let a = t.push_child(0, None, 0.5);
        let b = t.push_child(0, None, 0.5);
        assert_eq!(t.len(), 3);
        assert_eq!(t.node(0).children, vec![a, b]);
        assert_eq!(t.node(a).parent, Some(0));

        // Leaf value +0.25 (leaf's perspective): leaf gets +0.25, root -0.25.
        t.backup(&[0, a], 0.25);
        assert_eq!(t.node(a).visit_count, 1);
        assert_eq!(t.node(a).value_sum, 0.25);
        assert_eq!(t.node(0).visit_count, 1);
        assert_eq!(t.node(0).value_sum, -0.25);

        // Virtual loss: one pending selection makes the child look like a
        // loss to the parent and counts as a visit.
        t.add_virtual_loss(b);
        assert_eq!(t.node(b).effective_visits(), 1);
        assert_eq!(t.node(b).effective_value_mean(1.0), 1.0);
        assert_eq!(t.node(b).value_mean(), 0.0); // committed stats untouched
        t.backup(&[0, b], -0.5);
        assert_eq!(t.node(b).pending, 0);
        assert_eq!(t.node(b).value_sum, -0.5);
        assert_eq!(t.node(b).visit_count, 1);
    }

    #[test]
    fn value_mean_zero_when_unvisited() {
        let mut t = Tree::new();
        let a = t.push_child(0, None, 1.0);
        assert_eq!(t.node(a).value_mean(), 0.0);
        assert_eq!(t.node(0).value_mean(), 0.0);
    }
}
