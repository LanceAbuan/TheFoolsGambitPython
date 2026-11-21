<script lang="ts">
  interface Props {
    sans: string[];
  }

  let { sans }: Props = $props();

  const rows = $derived.by(() => {
    const out: { no: number; w?: string; b?: string; cur: 'w' | 'b' | null }[] = [];
    for (let i = 0; i < sans.length; i += 2) {
      out.push({
        no: i / 2 + 1,
        w: sans[i],
        b: sans[i + 1],
        cur: sans.length - 1 === i ? 'w' : sans.length - 1 === i + 1 ? 'b' : null,
      });
    }
    return out;
  });
</script>

<div class="scroll">
  {#if rows.length === 0}
    <div class="move-row"><span>01.</span><b>—</b><b>—</b></div>
    <div class="empty-state">Your story has yet to unfold.</div>
  {:else}
    {#each rows as r (r.no)}
      <div class="move-row">
        <span>{String(r.no).padStart(2, '0')}.</span>
        <b class:cur={r.cur === 'w'}>{r.w}</b>
        <b class:cur={r.cur === 'b'}>{r.b ?? ''}</b>
      </div>
    {/each}
  {/if}
</div>

<style>
  .scroll {
    max-height: 220px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
</style>
