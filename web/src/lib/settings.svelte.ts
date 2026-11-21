/**
 * User preferences persisted to localStorage (FS-8 polish).
 *
 * `unicodePieces`: render pieces as unicode glyphs (the mock look) instead of
 * the Cburnett SVGs — default OFF. Stored under one key so more toggles can
 * join later. localStorage access is guarded: SSR/private-mode failures fall
 * back to the in-memory default.
 */
const KEY = 'fsg:settings';

export interface FsgSettings {
  unicodePieces: boolean;
}

const DEFAULTS: FsgSettings = { unicodePieces: false };

function load(): FsgSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<FsgSettings>;
    return { ...DEFAULTS, ...parsed, unicodePieces: parsed.unicodePieces === true };
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

/** Glyphs for the unicode-piece look (filled set, tinted per color). */
export const GLYPHS: Record<string, string> = {
  k: '\u265A',
  q: '\u265B',
  r: '\u265C',
  b: '\u265D',
  n: '\u265E',
  p: '\u265F',
};
