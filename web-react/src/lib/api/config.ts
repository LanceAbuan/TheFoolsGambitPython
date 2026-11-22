/**
 * Shared client config — §14.0 pinned contract.
 *
 * Base resolution: `VITE_API_URL` always wins (tunnel / remote override).
 * Otherwise the base is RELATIVE, which is correct in every mode we run:
 * - Vite dev/preview proxy `/api` and `/ws` to 127.0.0.1:8000 (see vite.config.ts)
 * - the built site is served by the FastAPI backend itself (single origin)
 */
const fromEnv = import.meta.env.VITE_API_URL as string | undefined;

export const API_BASE: string = fromEnv ?? '';

/** http(s) base → ws(s) WebSocket endpoint URL. Handles the empty (same-origin) base. */
export function wsUrlFromBase(base: string): string {
  if (base === '') {
    if (typeof location === 'undefined') return '/ws';
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${location.host}/ws`;
  }
  return `${base.replace(/^http/, 'ws').replace(/\/+$/, '')}/ws`;
}
