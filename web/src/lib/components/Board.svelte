<script lang="ts">
  import { untrack } from 'svelte';
  import type { Move } from 'chess.js';
  import { settings, GLYPHS } from '../settings.svelte';

  interface Props {
    fen: string;
    lastMove?: { from: string; to: string } | null;
    checkSquare?: string | null;
    orientation?: 'white' | 'black';
    mini?: boolean;
    /** When false (game over, spectator view) the board renders but ignores input. */
    interactive?: boolean;
    legalTargets?: (square: string) => Move[];
    onmove?: (from: string, to: string, promotion?: string) => void;
  }

  let {
    fen,
    lastMove = null,
    checkSquare = null,
    orientation = 'white',
    mini = false,
    interactive = true,
    legalTargets = () => [],
    onmove = () => {},
  }: Props = $props();

  const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const uid = `cg${Math.random().toString(36).slice(2, 8)}`;

  type Pc = { id: number; square: string; type: string; color: 'w' | 'b' };
  let nextId = 1;
  let pieces: Pc[] = $state([]);
  let selected: string | null = $state(null);
  let targets: Move[] = $state([]);
  let dragId: number | null = $state(null);
  let dragX = $state(0);
  let dragY = $state(0);
  let hoverSquare: string | null = $state(null);
  let awaiting: { from: string; to: string } | null = $state(null);
  let boardEl: HTMLDivElement | undefined = $state();
  /** Keyboard focus cursor (FS-8 a11y): arrows move it, Enter selects/moves, Escape cancels. */
  let kcursor: string | null = $state(null);
  let promoEl: HTMLDivElement | undefined = $state();

  // ---- move glide (2026-11-21: "the games are jumpy") ------------------------
  // The server coalesces live frames (newest fen wins), so a board can jump
  // several plies at once. Only a CLEAN single-ply delta that matches lastMove
  // may animate; everything else snaps. A wrong animation is worse than a snap.
  //
  // Mechanism (why two pieces of state): a CSS transition added in the SAME
  // style recalc as the value change does not run (verified in Chrome — the
  // element just jumps). So the gliding piece is first PINNED to its old
  // square with an inline style override (inline style beats the transform
  // attribute; `transition: none` keeps the pin un-animated), then the pin is
  // released two animation frames later — by then the .glide transition is
  // active in the before-state, and the fallback to the attribute position
  // animates as a real glide. No layout reads: SVG transforms never trigger
  // layout, and the release is a style-only change on ≤2 pieces per board.
  /** Piece ids allowed to glide for the CURRENT position update (drives .glide). */
  let glideIds = $state<Set<number>>(new Set());
  /** Glide start override per piece id (display xy) — cleared after pin release. */
  let glideFrom = $state<Map<number, { fx: number; fy: number }>>(new Map());
  /** The human's just-made move (drag drop, click-click, keyboard): snap it. */
  let localAttempt: { from: string; to: string; at: number } | null = null;
  /** Last orientation seen by the reconcile effect (null until the first run). */
  let lastOrientation: 'white' | 'black' | null = null;
  let glideRelease = 0;

  function scheduleGlideRelease(): void {
    cancelAnimationFrame(glideRelease);
    glideRelease = requestAnimationFrame(() => {
      glideRelease = requestAnimationFrame(() => {
        glideFrom = new Map(); // unpins: the .glide transition now animates to the new square
      });
    });
  }

  const active = $derived(!mini && interactive);

  /** Promotion generates up to four moves per destination square; dots are identical, render one. */
  const targetDots = $derived.by(() => {
    const seen = new Set<string>();
    const out: Move[] = [];
    for (const t of targets) {
      if (!seen.has(t.to)) {
        seen.add(t.to);
        out.push(t);
      }
    }
    return out;
  });

  function parse(f: string): Map<string, { type: string; color: 'w' | 'b' }> {
    const map = new Map<string, { type: string; color: 'w' | 'b' }>();
    const rows = f.split(' ')[0].split('/');
    for (let r = 0; r < 8; r++) {
      let file = 0;
      for (const ch of rows[r]) {
        if (ch >= '1' && ch <= '8') {
          file += Number(ch);
          continue;
        }
        map.set(FILES[file] + (8 - r), {
          type: ch.toLowerCase(),
          color: ch === ch.toUpperCase() ? 'w' : 'b',
        });
        file += 1;
      }
    }
    return map;
  }

  /** Diff new square map against existing pieces, preserving ids so moves glide. */
  function reconcile(oldP: Pc[], next: Map<string, { type: string; color: 'w' | 'b' }>): Pc[] {
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
          (v.type === info.type || (v.type === 'p' && ['q', 'r', 'b', 'n'].includes(info.type)))
      );
      if (vi) {
        used.add(vi.id);
        out.push({ ...vi, square: sq, type: info.type });
      } else {
        out.push({ id: nextId++, square: sq, type: info.type, color: info.color });
      }
    }
    return out;
  }

  /**
   * Decide which pieces may glide for this position update, and where each
   * glide starts. The bar is HIGH on purpose: only a clean single-ply
   * relocation that exactly matches the lastMove highlight animates —
   *  - the piece now on lastMove.to must be the SAME piece that was painted
   *    on lastMove.from (id continuity through the diff);
   *  - its type must be unchanged (a promotion swaps type on arrival: snap);
   *  - a coalesced multi-ply jump fails the from-square test for every piece
   *    whose rendered delta doesn't match the last ply → snap;
   *  - the human's own just-made move snaps (the piece already traveled
   *    under their pointer — animating it would feel laggy);
   *  - a board flip mid-update snaps everything;
   *  - castling is the one multi-piece exception, and only when BOTH the
   *    king's and the rook's deltas prove the exact castle pattern.
   */
  function computeGlide(prev: Pc[], curr: Pc[], flipped: boolean): { ids: Set<number>; from: Map<number, { fx: number; fy: number }> } {
    const ids = new Set<number>();
    const from = new Map<number, { fx: number; fy: number }>();
    const lm = lastMove;
    if (flipped || !lm || lm.from === lm.to) return { ids, from };
    if (dragId !== null) return { ids, from }; // a drag is in flight — keep the board still
    const la = localAttempt;
    if (la && Date.now() - la.at < 1500 && la.from === lm.from && la.to === lm.to) {
      localAttempt = null; // consume: this fen change IS the human's move
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
    // Castling: king moved exactly two files on its own rank. Glide the rook
    // too ONLY if its own delta is the matching corner→landing relocation.
    const fFrom = FILES.indexOf(lm.from[0]);
    const fTo = FILES.indexOf(lm.to[0]);
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
  }

  $effect.pre(() => {
    const next = parse(fen);
    // orientation is a tracked dependency ON PURPOSE: it can change without
    // the fen changing (e.g. the landing board re-oriented into a fresh game
    // whose start position is byte-identical). Without this, lastOrientation
    // goes stale and the flip guard wrongly snaps the next real move.
    const orient = orientation;
    untrack(() => {
      const prev = pieces;
      // A board flip mid-update (orientation changed between renders) snaps
      // everything — gliding the flip would sweep every piece across the board.
      const flipped = lastOrientation !== null && orient !== lastOrientation;
      lastOrientation = orient;
      pieces = reconcile(prev, next);
      const g = computeGlide(prev, pieces, flipped);
      glideIds = g.ids;
      glideFrom = g.from;
      if (g.from.size > 0) scheduleGlideRelease();
    });
  });

  /** Inline pin for a gliding piece: paint it on the from-square until release. */
  function glideOverride(id: number): string | undefined {
    const g = glideFrom.get(id);
    return g ? `transform: translate(${g.fx}px, ${g.fy}px); transition: none;` : undefined;
  }

  // Any position change invalidates in-progress selection / promotion state.
  $effect(() => {
    void fen;
    untrack(() => {
      awaiting = null;
      selected = null;
      targets = [];
      dragId = null;
      hoverSquare = null;
    });
  });

  const displayOrder = $derived.by(() => {
    const ranks = orientation === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
    const files = orientation === 'white' ? FILES : [...FILES].reverse();
    const arr: string[] = [];
    for (const r of ranks) for (const f of files) arr.push(f + r);
    return arr;
  });

  function isDark(sq: string): boolean {
    return (FILES.indexOf(sq[0]) + Number(sq[1])) % 2 === 1;
  }
  function xy(sq: string): { x: number; y: number } {
    let fi = FILES.indexOf(sq[0]);
    let ri = Number(sq[1]) - 1;
    if (orientation === 'black') {
      fi = 7 - fi;
      ri = 7 - ri;
    }
    return { x: fi * 100, y: (7 - ri) * 100 };
  }
  function pieceName(t: string): string {
    return (
      ({ k: 'king', q: 'queen', r: 'rook', b: 'bishop', n: 'knight', p: 'pawn' } as Record<string, string>)[t] ?? t
    );
  }

  const dragPiece = $derived(pieces.find((p) => p.id === dragId) ?? null);

  const promoColor = $derived.by(() => {
    const a = awaiting;
    if (!a) return 'w' as const;
    const p = pieces.find((pp) => pp.square === a.from);
    return p?.color ?? ('w' as const);
  });
  const promoStyle = $derived.by(() => {
    if (!awaiting) return '';
    let fi = FILES.indexOf(awaiting.to[0]);
    let top = awaiting.to[1] === '8';
    if (orientation === 'black') {
      fi = 7 - fi;
      top = !top;
    }
    return `left:${fi * 12.5}%;${top ? 'top:2.5%' : 'bottom:2.5%'}`;
  });

  function toSvgPt(e: PointerEvent): { x: number; y: number } {
    const r = boardEl!.getBoundingClientRect();
    const s = 800 / r.width;
    return { x: (e.clientX - r.left) * s, y: (e.clientY - r.top) * s };
  }
  function sqAt(pt: { x: number; y: number }): string | null {
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
  }

  function clearSel(): void {
    selected = null;
    targets = [];
  }
  function select(sq: string): void {
    selected = sq;
    targets = legalTargets(sq);
  }
  function attempt(from: string, to: string): void {
    const cands = legalTargets(from).filter((m) => m.to === to);
    if (cands.length === 0) {
      clearSel();
      return;
    }
    if (cands.some((m) => m.promotion)) {
      awaiting = { from, to };
      return;
    }
    // The human's own move must SNAP when the position updates: drag drops
    // already traveled under the pointer, and click-click/kbd moves land
    // instantly by intent. Engine/remote/stream moves glide instead.
    localAttempt = { from, to, at: Date.now() };
    onmove(from, to);
    clearSel();
  }

  function onSquareDown(sq: string, e: PointerEvent): void {
    if (!active || awaiting || dragId !== null) return;
    if (selected && selected !== sq && targets.some((t) => t.to === sq)) {
      attempt(selected, sq);
      return;
    }
    if (legalTargets(sq).length > 0) select(sq);
    else clearSel();
    e.preventDefault();
  }

  function onPieceDown(p: Pc, e: PointerEvent): void {
    if (!active || awaiting) return;
    e.preventDefault();
    if (selected && selected !== p.square && targets.some((t) => t.to === p.square)) {
      attempt(selected, p.square);
      return;
    }
    // F-10: a piece only responds — select or drag-lift — when the human may
    // actually move it. The engine's side, pinned pieces, and the enemy's turn
    // stay inert (legalTargets already encodes all of that).
    if (legalTargets(p.square).length === 0) {
      clearSel();
      return;
    }
    select(p.square);
    dragId = p.id;
    const { x, y } = xy(p.square);
    dragX = x;
    dragY = y;
    hoverSquare = p.square;
    try {
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    } catch {
      /* synthetic pointers cannot be captured; drag still works via move/up on the element */
    }
  }
  function onPieceMove(e: PointerEvent): void {
    if (dragId === null) return;
    const pt = toSvgPt(e);
    dragX = pt.x - 50;
    dragY = pt.y - 50;
    hoverSquare = sqAt(pt);
  }
  function onPieceUp(e: PointerEvent): void {
    if (dragId === null) return;
    const pt = toSvgPt(e);
    const target = sqAt(pt);
    const dragged = pieces.find((p) => p.id === dragId);
    dragId = null;
    hoverSquare = null;
    if (!dragged || !target || target === dragged.square) return;
    attempt(dragged.square, target);
  }

  // Safety net: if pointerup/pointercancel happens off-element (alt-tab, lost capture),
  // release the drag so the board never bricks.
  $effect(() => {
    const release = () => {
      dragId = null;
      hoverSquare = null;
    };
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
    window.addEventListener('blur', release);
    return () => {
      window.removeEventListener('pointerup', release);
      window.removeEventListener('pointercancel', release);
      window.removeEventListener('blur', release);
    };
  });

  function pick(ptype: string): void {
    if (!awaiting) return;
    if (awaiting.from && awaiting.to) localAttempt = { from: awaiting.from, to: awaiting.to, at: Date.now() };
    onmove(awaiting.from, awaiting.to, ptype);
    awaiting = null;
    clearSel();
  }
  function cancelPromo(): void {
    awaiting = null;
    clearSel();
  }

  // ---- keyboard accessibility (FS-8): arrows move a focus cursor, Enter
  // selects/moves, Escape cancels. The cursor is a visible dashed square in
  // the theme gold; direction is adjusted for board orientation. ----
  function stepCursor(sq: string, df: number, dr: number): string | null {
    let fi = FILES.indexOf(sq[0]) + df;
    let ri = Number(sq[1]) - 1 + dr;
    if (fi < 0 || fi > 7 || ri < 0 || ri > 7) return null;
    return FILES[fi] + (ri + 1);
  }

  function onBoardKeydown(e: KeyboardEvent): void {
    if (!active) return;
    if (e.key === 'Escape') {
      if (awaiting) cancelPromo();
      else clearSel();
      return;
    }
    if (awaiting) return; // promotion picker owns Enter/Tab while open
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
        kcursor = lastMove ? lastMove.to : orientation === 'white' ? 'e2' : 'e7';
        return;
      }
      const next = stepCursor(kcursor, df, dr);
      if (next) kcursor = next;
      return;
    }
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (kcursor === null) return;
      if (selected && selected !== kcursor && targets.some((t) => t.to === kcursor)) {
        const dest = kcursor;
        attempt(selected, dest);
        kcursor = dest; // stay on the destination for follow-up moves
        return;
      }
      if (legalTargets(kcursor).length > 0) {
        select(kcursor);
      } else {
        clearSel();
      }
    }
  }

  // When a promotion picker opens, pull focus into it so Enter/Tab can choose
  // without a pointer; the wrapper's Escape handler still cancels.
  $effect(() => {
    if (awaiting && promoEl) promoEl.focus();
  });
</script>

<!-- The board is a composite keyboard widget (roving cursor): role=application +
     tabindex is the standard accessible-chessboard pattern (cf. lichess). -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
  class="board-wrap"
  class:mini
  bind:this={boardEl}
  tabindex={active ? 0 : -1}
  role={active ? 'application' : undefined}
  aria-roledescription={active ? 'chess board' : undefined}
  aria-label={
    active ? 'Chess board. Arrow keys move the cursor, Enter selects or moves, Escape cancels.' : 'chess board'
  }
  onkeydown={onBoardKeydown}
>
  <svg viewBox="0 0 800 800" role="presentation" class:inert={!active}>
    {#if !mini}
      <defs>
        <radialGradient id={uid} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#c05a4a" stop-opacity="0.55" />
          <stop offset="100%" stop-color="#c05a4a" stop-opacity="0" />
        </radialGradient>
      </defs>
    {/if}

    {#each displayOrder as sq (sq)}
      {@const { x, y } = xy(sq)}
      <rect
        {x}
        {y}
        width="100"
        height="100"
        class={isDark(sq) ? 'dark' : 'light'}
        role={active ? 'button' : undefined}
        aria-label={`square ${sq}`}
        onpointerdown={(e) => onSquareDown(sq, e)}
      />
    {/each}

    {#if lastMove}
      {#each [lastMove.from, lastMove.to] as sq (sq)}
        {@const { x, y } = xy(sq)}
        <rect {x} {y} width="100" height="100" class="last" pointer-events="none" />
      {/each}
    {/if}

    {#if hoverSquare && dragId !== null}
      {@const { x, y } = xy(hoverSquare)}
      <rect {x} {y} width="100" height="100" class="hover" pointer-events="none" />
    {/if}

    {#if selected}
      {@const { x, y } = xy(selected)}
      <rect x={x + 2} y={y + 2} width="96" height="96" class="sel" pointer-events="none" />
    {/if}

    {#if active && kcursor}
      {@const { x, y } = xy(kcursor)}
      <rect x={x + 3} y={y + 3} width="94" height="94" class="kcursor" pointer-events="none" />
    {/if}

    {#if active}
      {#each targetDots as t (t.to)}
        {@const { x, y } = xy(t.to)}
        {#if t.captured}
          <circle cx={x + 50} cy={y + 50} r="40" class="cap" pointer-events="none" />
        {:else}
          <circle cx={x + 50} cy={y + 50} r="13" class="dot" pointer-events="none" />
        {/if}
      {/each}
    {/if}

    {#if checkSquare}
      {@const { x, y } = xy(checkSquare)}
      <circle cx={x + 50} cy={y + 50} r="55" fill="url(#{uid})" pointer-events="none" />
    {/if}

    {#each pieces as p (p.id)}
      {@const pos = xy(p.square)}
      <g
        class="piece"
        class:lifted={p.id === dragId}
        class:glide={glideIds.has(p.id)}
        style={glideOverride(p.id)}
        transform={`translate(${pos.x} ${pos.y})`}
        role={active ? 'button' : undefined}
        aria-label={`${p.color === 'w' ? 'white' : 'black'} ${pieceName(p.type)} on ${p.square}`}
        onpointerdown={(e) => onPieceDown(p, e)}
        onpointermove={onPieceMove}
        onpointerup={onPieceUp}
      >
        {#if settings.unicodePieces}
          <text class="uglyph {p.color}" x="50" y="54" text-anchor="middle">{GLYPHS[p.type] ?? p.type}</text>
        {:else}
          <image href={`/pieces/${p.color}${p.type.toUpperCase()}.svg`} x="3" y="3" width="94" height="94" />
        {/if}
      </g>
    {/each}

    {#if dragPiece}
      <g class="piece dragging" transform={`translate(${dragX} ${dragY})`}>
        {#if settings.unicodePieces}
          <text class="uglyph {dragPiece.color}" x="50" y="54" text-anchor="middle">
            {GLYPHS[dragPiece.type] ?? dragPiece.type}
          </text>
        {:else}
          <image href={`/pieces/${dragPiece.color}${dragPiece.type.toUpperCase()}.svg`} x="3" y="3" width="94" height="94" />
        {/if}
      </g>
    {/if}
  </svg>

  {#if awaiting}
    <div class="promo" style={promoStyle} bind:this={promoEl} tabindex="-1">
      {#each ['q', 'r', 'b', 'n'] as pt (pt)}
        <button class="promo-piece" onclick={() => pick(pt)} aria-label={`promote to ${pieceName(pt)}`}>
          {#if settings.unicodePieces}
            <span class="promo-glyph {promoColor}">{GLYPHS[pt]}</span>
          {:else}
            <img src={`/pieces/${promoColor}${pt.toUpperCase()}.svg`} alt="" />
          {/if}
        </button>
      {/each}
      <button class="promo-x" onclick={cancelPromo} aria-label="cancel promotion">✕</button>
    </div>
  {/if}
</div>

<style>
  .board-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    user-select: none;
    -webkit-user-select: none;
    border: 5px solid #4c3b31;
    background: #17141b;
    box-shadow: 0 7px 24px rgba(9, 7, 11, 0.5);
  }
  .board-wrap svg {
    display: block;
    width: 100%;
    height: 100%;
    touch-action: none;
  }
  rect.light {
    fill: var(--sq-light);
  }
  rect.dark {
    fill: var(--sq-dark);
  }
  rect.last {
    fill: rgba(213, 173, 100, 0.34);
    stroke: rgba(213, 173, 100, 0.85);
    stroke-width: 2.5;
  }
  rect.hover {
    fill: rgba(213, 173, 100, 0.2);
    stroke: #d5ad64;
    stroke-width: 3;
  }
  rect.sel {
    fill: none;
    stroke: #d5ad64;
    stroke-width: 5;
  }
  .dot {
    fill: rgba(23, 16, 33, 0.5);
  }
  .cap {
    fill: none;
    stroke: rgba(23, 16, 33, 0.45);
    stroke-width: 9;
  }
  .piece {
    /* Deliberately NO ambient transition: position changes snap by default.
       Only pieces that pass the honest-move guard get .glide (below) — the
       old blanket transition animated coalesced multi-ply teleports and even
       orientation flips (every piece gliding across the board at once). */
    cursor: grab;
  }
  .piece.glide {
    transition: transform 210ms cubic-bezier(0.22, 0.61, 0.36, 1);
  }
  /* Accessibility: the glide is decorative motion — reduce → snap. */
  @media (prefers-reduced-motion: reduce) {
    .piece.glide {
      transition: none;
    }
  }
  svg.inert .piece {
    cursor: default;
  }
  .piece.lifted {
    opacity: 0;
  }
  .piece.dragging {
    transition: none;
    cursor: grabbing;
  }
  .piece image {
    filter: drop-shadow(0 3px 3px rgba(10, 5, 25, 0.45));
  }
  .promo {
    position: absolute;
    width: 12.5%;
    display: flex;
    flex-direction: column;
    background: #19151c;
    border: 1px solid #6a543e;
    border-radius: 0;
    overflow: hidden;
    z-index: 5;
    box-shadow: 0 14px 40px rgba(8, 7, 10, 0.65);
  }
  .promo-piece {
    border: 0;
    background: transparent;
    padding: 6%;
    cursor: pointer;
    aspect-ratio: 1;
  }
  .promo-piece:hover {
    background: rgba(185, 149, 88, 0.22);
  }
  .promo-piece img {
    width: 100%;
    height: 100%;
    display: block;
  }
  .promo-x {
    border: 0;
    border-top: 1px solid var(--line-soft);
    background: rgba(0, 0, 0, 0.3);
    color: var(--muted);
    font-size: 12px;
    padding: 4px 0;
    cursor: pointer;
  }
  .promo-x:hover {
    color: var(--gold);
  }
  .board-wrap.mini {
    border-width: 3px;
    box-shadow: none;
  }
  /* keyboard accessibility (FS-8): visible focus + dashed gold cursor */
  .board-wrap:focus-visible {
    outline: 2px solid var(--gold);
    outline-offset: 3px;
  }
  rect.kcursor {
    fill: rgba(213, 173, 100, 0.14);
    stroke: var(--gold-bright);
    stroke-width: 3.5;
    stroke-dasharray: 10 7;
  }
  /* unicode-glyph piece mode (settings toggle, default off) */
  text.uglyph {
    font-size: 84px;
    text-anchor: middle;
    paint-order: stroke fill;
    stroke-width: 4;
    filter: drop-shadow(0 3px 3px rgba(10, 5, 25, 0.45));
    user-select: none;
  }
  text.uglyph.w {
    fill: #f3e6c8;
    stroke: #4c3b31;
  }
  text.uglyph.b {
    fill: #201a20;
    stroke: #c9a969;
  }
  .promo {
    container-type: inline-size;
  }
  .promo-glyph {
    display: block;
    text-align: center;
    line-height: 1;
    font-size: 34px;
    font-size: 58cqi;
  }
  .promo-glyph.w {
    color: #f3e6c8;
    text-shadow: 1px 1px 0 #4c3b31;
  }
  .promo-glyph.b {
    color: #241d24;
    text-shadow: 0 0 2px #c9a969;
  }
</style>
