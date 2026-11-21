<script lang="ts">
  import { untrack } from 'svelte';
  import type { Move } from 'chess.js';

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

  const active = $derived(!mini && interactive);

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

  $effect.pre(() => {
    const next = parse(fen);
    untrack(() => {
      pieces = reconcile(pieces, next);
    });
  });

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
    onmove(awaiting.from, awaiting.to, ptype);
    awaiting = null;
    clearSel();
  }
  function cancelPromo(): void {
    awaiting = null;
    clearSel();
  }
</script>

<div class="board-wrap" class:mini bind:this={boardEl}>
  <svg viewBox="0 0 800 800" role="group" aria-label="chess board" class:inert={!active}>
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

    {#if active}
      {#each targets as t (t.to)}
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
        transform={`translate(${pos.x} ${pos.y})`}
        role={active ? 'button' : undefined}
        aria-label={`${p.color === 'w' ? 'white' : 'black'} ${pieceName(p.type)} on ${p.square}`}
        onpointerdown={(e) => onPieceDown(p, e)}
        onpointermove={onPieceMove}
        onpointerup={onPieceUp}
      >
        <image href={`/pieces/${p.color}${p.type.toUpperCase()}.svg`} x="3" y="3" width="94" height="94" />
      </g>
    {/each}

    {#if dragPiece}
      <g class="piece dragging" transform={`translate(${dragX} ${dragY})`}>
        <image href={`/pieces/${dragPiece.color}${dragPiece.type.toUpperCase()}.svg`} x="3" y="3" width="94" height="94" />
      </g>
    {/if}
  </svg>

  {#if awaiting}
    <div class="promo" style={promoStyle}>
      {#each ['q', 'r', 'b', 'n'] as pt (pt)}
        <button class="promo-piece" onclick={() => pick(pt)} aria-label={`promote to ${pieceName(pt)}`}>
          <img src={`/pieces/${promoColor}${pt.toUpperCase()}.svg`} alt={pieceName(pt)} />
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
    transition: transform 170ms cubic-bezier(0.2, 0.7, 0.3, 1);
    cursor: grab;
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
</style>
