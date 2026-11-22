/**
 * User preferences persisted to localStorage (FS-8 polish).
 *
 * `unicodePieces`: render pieces as unicode glyphs (the mock look) instead of
 * the Cburnett SVGs — default OFF.
 * `theme`: one of the three full color themes (THEME SLICE 2026-11-21). The
 * value is written to <html data-theme="…">; app.css carries a token block per
 * theme. Stored under one key so more toggles can join later. localStorage
 * access is guarded: SSR/private-mode failures fall back to the in-memory
 * default.
 */
const KEY = 'fsg:settings';

export type ThemeName = 'classical-light' | 'classical-dark' | 'refined-tarot';

/** Cycle order for the temporary theme pill (and the validation set on load).
 *  classical-light is the hero candidate (classical chess palette, modern
 *  minimal system); refined-tarot is kept as the identity comparison so the
 *  choice is real. Default stays refined-tarot. */
export const THEMES: { id: ThemeName; label: string }[] = [
  { id: 'classical-light', label: 'Classical Light' },
  { id: 'classical-dark', label: 'Classical Dark' },
  { id: 'refined-tarot', label: 'Refined Tarot' },
];

const isTheme = (v: unknown): v is ThemeName => THEMES.some((t) => t.id === v);

export interface FsgSettings {
  unicodePieces: boolean;
  theme: ThemeName;
}

const DEFAULTS: FsgSettings = { unicodePieces: false, theme: 'refined-tarot' };

function load(): FsgSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<FsgSettings>;
    return {
      ...DEFAULTS,
      ...parsed,
      unicodePieces: parsed.unicodePieces === true,
      theme: isTheme(parsed.theme) ? parsed.theme : DEFAULTS.theme,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export const settings = $state<FsgSettings>(load());

export function setSetting<K extends keyof FsgSettings>(key: K, value: FsgSettings[K]): void {
  settings[key] = value;
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...settings }));
  } catch {
    /* storage unavailable — keep the session-local value */
  }
}

/**
 * Write the theme to <html data-theme>. Called from main.ts BEFORE mount (no
 * flash of the default theme for returning modern-theme visitors) and from the
 * theme pill's effect on every change. Pure attribute write: no layout work.
 */
export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme;
}

/** Glyphs for the unicode-piece look (filled set, tinted per color). */
export const GLYPHS: Record<string, string> = {
  k: '\u265A',
  q: '\u265B',
  r: '\u265C',
  b: '\u265D',
  n: '\u265E',
  p: '\u265F',
};
