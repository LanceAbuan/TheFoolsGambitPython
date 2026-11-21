<script lang="ts">
  import HomeView from './views/HomeView.svelte';
  import PlayView from './views/PlayView.svelte';
  import WatchView from './views/WatchView.svelte';
  import AboutView from './views/AboutView.svelte';

  type Tab = 'home' | 'play' | 'watch' | 'about';
  const allTabs: readonly Tab[] = ['home', 'play', 'watch', 'about'];
  const tabFromPath = (): Tab => {
    const p = location.pathname.replace(/\/+$/, '');
    return allTabs.find((t) => p === `/${t}`) ?? 'home';
  };
  let tab: Tab = $state(tabFromPath());
  const goto = (t: Tab): void => {
    tab = t;
    if (tabFromPath() !== t) history.pushState(null, '', t === 'home' ? '/' : `/${t}`);
  };
  $effect(() => {
    const onpop = (): void => {
      tab = tabFromPath();
    };
    addEventListener('popstate', onpop);
    return () => removeEventListener('popstate', onpop);
  });
  const tabs: { id: Tab; label: string }[] = [
    { id: 'home', label: 'Home' },
    { id: 'play', label: 'Play' },
    { id: 'watch', label: 'Watch' },
    { id: 'about', label: 'About' },
  ];
  const footers: Record<Tab, [string, string]> = {
    home: ['THE FOOL · A CHESS EXPERIMENT', 'BEGIN ANYWHERE. LEARN EVERYTHING.'],
    play: ['THE FOOL · PLAY', 'EVERY MOVE IS A NEW BEGINNING.'],
    watch: ['THE FOOL · WATCH', 'ONE MOVE AT A TIME.'],
    about: ['THE FOOL · ABOUT', 'BEGIN ANYWHERE. LEARN EVERYTHING.'],
  };
</script>

<header class="topbar">
  <button class="brand" onclick={() => goto('home')}>
    <span class="brand-card">0</span><span>The Fool</span><i>✦</i>
  </button>
  <nav aria-label="sections">
    {#each tabs as t (t.id)}
      <button class:active={tab === t.id} onclick={() => goto(t.id)}>{t.label}</button>
    {/each}
  </nav>
  <span class="nav-star">✳</span>
</header>

<main>
  {#if tab === 'home'}
    <HomeView onnav={(t) => goto(t)} />
  {:else if tab === 'play'}
    <PlayView onnav={(t) => goto(t)} />
  {:else if tab === 'watch'}
    <WatchView onnav={(t) => goto(t)} />
  {:else}
    <AboutView onnav={(t) => goto(t)} />
  {/if}
</main>

<footer class="site-footer">
  <span>{footers[tab][0]}</span><span>{footers[tab][1]}</span><span>✦</span>
</footer>
