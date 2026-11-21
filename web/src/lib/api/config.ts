/** Shared client config — §14.0 pinned contract.
 *
 * API base resolution:
 * - `VITE_API_URL` always wins (dev/tunnel override).
 * - Under the Vite dev server (port 5173) the dev backend at 127.0.0.1:8000 is used.
 * - Otherwise (built site served by the backend itself, single-origin) the base is
 *   RELATIVE: fetch('/api/…') and ws://location.host/ws hit the serving origin.
 */
function resolveApiBase(): string {
  const fromEnv = import.meta.env.VITE_API_URL;
  if (fromEnv) return fromEnv;
  if (typeof location !== 'undefined' && location.port === '5173') return 'http://127.0.0.1:8000';
  return '';
}

export const API_BASE: string = resolveApiBase();

/** http(s) base → ws(s) WebSocket endpoint URL. Handles the empty (same-origin) base. */
export function wsUrlFromBase(base: string): string {
  if (base === '') {
    if (typeof location === 'undefined') return '/ws';
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${location.host}/ws`;
  }
  return `${base.replace(/^http/, 'ws').replace(/\/+$/, '')}/ws`;
}
