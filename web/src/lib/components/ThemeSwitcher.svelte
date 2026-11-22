<script lang="ts">
  import { applyTheme, setSetting, settings, THEMES } from '../settings.svelte';

  // Theme picker, mounted in the site footer (App.svelte) so it never floats
  // over page content. Radiogroup semantics: one tab stop, arrow keys move the
  // selection, Enter/Space are native button activation.
  const SHORT: Record<string, string> = {
    'refined-tarot': 'Tarot',
    'classical-light': 'Light',
    'classical-dark': 'Dark',
  };

  // Keep <html data-theme> in sync if the setting ever changes elsewhere.
  $effect(() => {
    applyTheme(settings.theme);
  });

  function onKeydown(e: KeyboardEvent): void {
    const group = e.currentTarget as HTMLElement;
    const radios = [...group.querySelectorAll<HTMLButtonElement>('[role="radio"]')];
    const index = radios.findIndex((r) => r.getAttribute('aria-checked') === 'true');
    let next = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (index + 1) % radios.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (index - 1 + radios.length) % radios.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = radios.length - 1;
    if (next < 0) return;
    e.preventDefault();
    const option = THEMES[next];
    if (!option) return;
    setSetting('theme', option.id);
    radios[next]?.focus();
  }
</script>

<div class="theme-switch" role="radiogroup" aria-label="Colour theme" tabindex="-1" onkeydown={onKeydown}>
  <span class="theme-switch-label" aria-hidden="true">Theme</span>
  <div class="theme-segments">
    {#each THEMES as t (t.id)}
      <button
        type="button"
        role="radio"
        aria-checked={settings.theme === t.id}
        aria-label={t.label}
        title={t.label}
        tabindex={settings.theme === t.id ? 0 : -1}
        class="theme-choice"
        class:active={settings.theme === t.id}
        data-swatch={t.id}
        onclick={() => setSetting('theme', t.id)}
      >
        <span class="swatch" aria-hidden="true"></span>
        <span class="theme-name" aria-hidden="true">{t.label}</span>
        <span class="theme-short" aria-hidden="true">{SHORT[t.id] ?? t.label}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .theme-switch {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
    flex: none;
  }
  .theme-switch-label {
    font: 8px var(--mono);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--faint);
  }
  .theme-segments {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 2px;
    border: 1px solid var(--line-soft);
    border-radius: var(--radius-pill);
    background: var(--accent-ghost);
  }
  .theme-choice {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid transparent;
    border-radius: var(--radius-pill);
    background: transparent;
    color: var(--muted);
    font: 9px var(--mono);
    letter-spacing: 0.04em;
    padding: 4px 8px;
    cursor: pointer;
    line-height: 1.2;
    white-space: nowrap;
  }
  .theme-choice:hover {
    color: var(--paper);
    background: var(--accent-ghost);
  }
  .theme-choice.active {
    color: var(--paper);
    border-color: var(--line-soft);
    background: var(--panel2);
  }
  .swatch {
    width: 13px;
    height: 13px;
    border-radius: 50%;
    border: 1px solid var(--line-soft);
    flex: none;
  }
  .theme-choice.active .swatch {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
  .theme-name {
    display: inline;
  }
  .theme-short {
    display: none;
  }
  /* Swatches sample each approved theme's ground + accent (no new palette). */
  [data-swatch='refined-tarot'] .swatch {
    background: linear-gradient(135deg, #17141b 0 50%, #c8a35f 50% 100%);
  }
  [data-swatch='classical-light'] .swatch {
    background: linear-gradient(135deg, #f7f5f2 0 50%, #8a3324 50% 100%);
  }
  [data-swatch='classical-dark'] .swatch {
    background: linear-gradient(135deg, #161412 0 50%, #cf8a68 50% 100%);
  }
  @media (max-width: 560px) {
    .theme-switch {
      gap: 6px;
    }
    .theme-switch-label {
      display: none;
    }
    .theme-choice {
      padding: 4px 7px;
      gap: 5px;
      letter-spacing: 0;
    }
    .theme-name {
      display: none;
    }
    .theme-short {
      display: inline;
    }
  }
</style>
