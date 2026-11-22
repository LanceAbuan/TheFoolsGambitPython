import { useEffect, useState } from 'react';
import { metrics as fetchMetrics, type MetricsSnapshot } from '../api/http';

export interface MetricsResult {
  metrics: MetricsSnapshot | null;
  error: string | null;
  /** Wall clock of the last successful snapshot (ms) — lets a view date itself. */
  updatedAt: number | null;
}

/**
 * Live training snapshot polled every 10 s, matching the server's `train`
 * heartbeat cadence. The WS `metrics` frame supersedes this once the socket is
 * ported; until then polling keeps every page honest with zero extra deps.
 */
export function useMetrics(): MetricsResult {
  const [data, setData] = useState<MetricsSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const next = await fetchMetrics();
        if (!alive) return;
        setData(next);
        setUpdatedAt(Date.now());
        setError(null);
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (alive) timer = setTimeout(tick, 10_000);
      }
    };

    void tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return { metrics: data, error, updatedAt };
}
