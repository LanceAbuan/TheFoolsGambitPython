<script lang="ts">
  interface Props {
    onnav?: (tab: 'play' | 'watch' | 'about') => void;
  }
  let { onnav }: Props = $props();
</script>

<main class="home-layout">
  <section class="hero-art" aria-label="The Fool on a cliff beneath a brass sun">
    <div class="sun"></div>
    <div class="mountain mountain-a"></div>
    <div class="mountain mountain-b"></div>
    <div class="cliff"></div>
    <div class="wanderer">
      <span class="wanderer-head"></span><span class="wanderer-body"></span><span class="wanderer-staff"></span
      ><span class="wanderer-pack">✧</span>
    </div>
    <div class="little-dog">♞</div>
    <span class="art-star star-one">✦</span><span class="art-star star-two">✧</span
    ><span class="art-star star-three">✶</span>
    <div class="art-caption">THE JOURNEY BEGINS</div>
  </section>

  <section class="home-copy">
    <div class="eyebrow"><span></span> AN EXPERIMENT IN ARTIFICIAL INTELLIGENCE <span></span></div>
    <h1>The Fool</h1>
    <p class="lead">A chess bot on a journey<br />of endless possibility.</p>
    <div class="ornament">─────── ✦ ───────</div>
    <p class="intro">
      Curious by nature. Uncertain by design. The Fool learns one move at a time, finding its way through the infinite
      game.
    </p>
    <div class="home-actions">
      <button class="action-card primary" onclick={() => onnav?.('play')}>
        <span class="action-icon">♜</span><strong>Play Chess</strong><small>Challenge The Fool</small><b>↗</b>
      </button>
      <button class="action-card" onclick={() => onnav?.('watch')}>
        <span class="action-icon">◉</span><strong>Watch Training</strong><small>Observe the journey</small><b>↗</b>
      </button>
      <button class="action-card" onclick={() => onnav?.('about')}>
        <span class="action-icon">✧</span><strong>About</strong><small>Meet the experiment</small><b>↗</b>
      </button>
    </div>
    <div class="bottom-motif">☾ &nbsp; ☽ &nbsp; ✳ &nbsp; ☾ &nbsp; ☽</div>
  </section>
</main>
