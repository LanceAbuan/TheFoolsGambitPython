/** Shared client config — §14.0 pinned contract, base URL overridable for dev/tunnel. */
export const API_BASE: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

/** http(s) base → ws(s) WebSocket endpoint URL. */
export function wsUrlFromBase(base: string): string {
  return `${base.replace(/^http/, 'ws').replace(/\/+$/, '')}/ws`;
}
