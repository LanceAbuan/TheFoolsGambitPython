<script module lang="ts">
  /** Instance counter for unique aria ids. */
  let tipSeq = 0;
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';

  /**
   * FS-8b — newcomer tooltip. A small "?" chip (or dotted-underline term) that
   * reveals a 1-3 sentence plain-English explanation. Bodies are pure clarity;
   * the mystical voice stays in headings, never here.
   *
   * A11y contract: the trigger is a real <button>, so Tab reaches it; focus
   * reveals the bubble, Enter/Space pins it, Escape dismisses it (and unpins);
   * aria-describedby + role="tooltip" wire the relationship for screen readers
   * and aria-expanded announces the pinned state. Touch: tap toggles, tap
   * elsewhere dismisses. Positioning is measured on open and clamped to the
   * viewport (flips above/below, never overflows horizontally); while open it
   * re-clamps on scroll (capture catches nested scrollers) and resize.
   */
  let {
    text,
    children,
    variant = 'q',
    label = 'What does this mean?',
  }: {
    text: string;
    /** Optional custom trigger content (dotted-underline term). */
    children?: Snippet;
    /** 'q' renders the round ? chip; 'underline' renders the slot as the trigger. */
    variant?: 'q' | 'underline';
    label?: string;
  } = $props();

  const id = `tip-body-${++tipSeq}`;

  let open = $state(false);
  let pinned = $state(false);
  let x = $state(0);
  let y = $state(0);
  let wrapEl = $state<HTMLSpanElement>();
  let bubbleEl = $state<HTMLSpanElement>();

  /** Center on the trigger, clamp inside the viewport, flip below when no headroom.
   *  The bubble is position:fixed (viewport coords), which also keeps *closed*
   *  bubbles out of the document's scrollable-overflow area — with absolute
   *  positioning a hidden 280px bubble sitting at an unclamped offset widened
   *  the page at narrow widths. */
  function place(): void {
    const w = wrapEl;
    const b = bubbleEl;
    if (!w || !b) return;
    const r = w.getBoundingClientRect();
    const bw = b.offsetWidth; // visibility:hidden keeps layout, so this works while closed
    const bh = b.offsetHeight;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const m = 8;
    let left = r.left + r.width / 2 - bw / 2;
    left = Math.max(m, Math.min(left, vw - bw - m));
    x = Math.round(left);
    const fitsAbove = r.top - bh - 8 >= m;
    const fitsBelow = r.bottom + bh + 8 <= vh - m;
    const above = fitsAbove || (!fitsBelow && r.top >= vh - r.bottom);
    y = Math.round(above ? r.top - bh - 8 : r.bottom + 8);
  }

  function reveal(): void {
    open = true;
    place();
  }

  function conceal(): void {
    if (!pinned) open = false;
  }

  function togglePin(): void {
    if (pinned) {
      pinned = false;
      open = false;
      return;
    }
    pinned = true;
    open = true;
    place();
  }

  function dismiss(): void {
    pinned = false;
    open = false;
  }

  function onTriggerKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.stopPropagation();
      dismiss();
    }
  }

  function onDocPointerdown(e: PointerEvent): void {
    if (pinned && wrapEl && !wrapEl.contains(e.target as Node)) dismiss();
  }

  $effect(() => {
    document.addEventListener('pointerdown', onDocPointerdown);
    return () => document.removeEventListener('pointerdown', onDocPointerdown);
  });

  // Keep the clamped position honest while open: capture-phase scroll also
  // catches nested scrollers (the journal feed), resize catches layout shifts.
  $effect(() => {
    if (!open) return;
    const re = (): void => place();
    window.addEventListener('scroll', re, true);
    window.addEventListener('resize', re);
    return () => {
      window.removeEventListener('scroll', re, true);
      window.removeEventListener('resize', re);
    };
  });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions a11y_no_mouse_event_without_blur -->
<span
  class="tip"
  bind:this={wrapEl}
  onmouseenter={() => reveal()}
  onmouseleave={() => conceal()}
>
  {#if variant === 'underline'}
    <button
      type="button"
      class="tip-term"
      aria-describedby={id}
      aria-expanded={open}
      onclick={togglePin}
      onfocus={() => reveal()}
      onblur={() => conceal()}
      onkeydown={onTriggerKeydown}
    >
      {@render children?.()}
    </button>
  {:else}
    <button
      type="button"
      class="tip-q"
      aria-label={label}
      aria-describedby={id}
      aria-expanded={open}
      onclick={togglePin}
      onfocus={() => reveal()}
      onblur={() => conceal()}
      onkeydown={onTriggerKeydown}
    >?</button>
  {/if}
  <span
    class="tip-bubble"
    role="tooltip"
    id={id}
    class:on={open}
    bind:this={bubbleEl}
    style:left="{x}px"
    style:top="{y}px"
  >
    {text}
  </span>
</span>

<style>
  .tip {
    position: relative;
    display: inline-flex;
    vertical-align: baseline;
  }
  .tip-q {
    width: 15px;
    height: 15px;
    padding: 0;
    border-radius: 50%;
    border: 1px solid var(--line);
    background: transparent;
    color: var(--muted);
    font: 10px/1 var(--mono);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: help;
    flex: none;
    vertical-align: middle;
  }
  .tip-q:hover {
    color: var(--gold-bright);
    border-color: var(--gold);
    background: var(--panel2);
  }
  .tip-q:focus-visible,
  .tip-term:focus-visible {
    outline: 1px solid var(--gold-bright);
    outline-offset: 2px;
  }
  .tip-term {
    background: none;
    border: 0;
    border-bottom: 1px dotted var(--muted);
    padding: 0;
    color: inherit;
    font: inherit;
    letter-spacing: inherit;
    cursor: help;
  }
  .tip-term:hover {
    border-bottom-color: var(--gold-bright);
  }
  .tip-bubble {
    position: fixed;
    z-index: 80;
    width: max-content;
    max-width: min(280px, calc(100vw - 16px));
    max-height: 45vh;
    overflow: auto;
    /* Tooltips must stay readable in every theme. `--panel2`/`--paper` are the
       one pair whose contrast holds everywhere; the old `--paper` background +
       `--ink-text` was white-on-cream in classical-dark. */
    background: var(--panel2);
    color: var(--paper);
    border: 1px solid var(--gold);
    padding: 9px 11px;
    font: 12px/1.55 var(--serif);
    box-shadow: var(--float-shadow);
    visibility: hidden;
    opacity: 0;
    pointer-events: none;
    transition: opacity 120ms ease;
  }
  .tip-bubble.on {
    visibility: visible;
    opacity: 1;
  }
</style>
