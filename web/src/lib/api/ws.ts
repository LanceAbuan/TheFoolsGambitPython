/**
 * WebSocket transport for the §14.0 pinned contract: one connection per client
 * to `<API_BASE>/ws`, typed JSON, sequence-numbered messages.
 *
 * - C→S subscribe: `{seq, type: "subscribe", topic, last_seq}` — `last_seq` is the
 *   last seq seen on that topic so the server can resync on reconnect.
 * - S→C: `{seq, topic, type: "position"|"state"|"metrics"|"log"|"live"|"game_end"|"lobby"|"error", payload, ts}`.
 * - Auto-reconnect with exponential backoff (1s → 15s cap); all topics are
 *   resubscribed automatically when the socket reopens.
 *
 * Deliberately rune-free: reactive mirroring lives in the views/stores that
 * consume the `onMessage` / `onStatus` callbacks.
 */
import { wsUrlFromBase, API_BASE } from './config';

export type ServerEventType =
  | 'position'
  | 'state'
  | 'metrics'
  | 'log'
  | 'live'
  | 'game_end'
  | 'lobby'
  | 'error';
export type SocketStatus = 'idle' | 'connecting' | 'open' | 'closed';

export interface ServerMessage {
  seq: number;
  topic: string;
  type: ServerEventType;
  payload: Record<string, unknown>;
  ts: string;
}

export type MessageHandler = (msg: ServerMessage) => void;
export type StatusHandler = (status: SocketStatus) => void;

const MAX_BACKOFF_MS = 15_000;

/** The only topic with a server-guaranteed heartbeat (metrics every 10s). */
export const WATCHDOG_TOPIC = 'train';
/** Watchdog tick. Contract: must never exceed 20s (detection granularity). */
export const WATCHDOG_TICK_MS = 10_000;
/** Visible-tab silence that proves the `train` subscription is deaf (>25s). */
export const TRAIN_STALE_MS = 25_000;
/** Refocus silence threshold: recover immediately instead of waiting a tick. */
export const TRAIN_STALE_ON_REFOCUS_MS = 15_000;
/** Fallback if the browser never fires `onclose` after a forced close (half-open). */
export const FORCE_CLOSE_GRACE_MS = 2_000;

export class GambitSocket {
  private ws: WebSocket | null = null;
  private topics = new Map<string, number>(); // topic → last seq seen (sent back on resubscribe)
  /** topic → Date.now() of the last frame received on it (watchdog input). */
  private lastMessageAt = new Map<string, number>();
  /** Sockets whose close has already been processed (idempotent recovery). */
  private closedSockets = new WeakSet<WebSocket>();
  private messageHandlers = new Set<MessageHandler>();
  private statusHandlers = new Set<StatusHandler>();
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  private forceCloseTimer: ReturnType<typeof setTimeout> | null = null;
  private visibilityListening = false;
  private attempt = 0;
  private seq = 0;
  private stopped = true;
  private state: SocketStatus = 'idle';

  constructor(private readonly url: string) {}

  get status(): SocketStatus {
    return this.state;
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    return () => this.statusHandlers.delete(handler);
  }

  /** Register a topic and subscribe immediately if the socket is open. */
  subscribe(topic: string): void {
    if (!this.topics.has(topic)) {
      this.topics.set(topic, 0);
      this.lastMessageAt.set(topic, Date.now());
    }
    if (this.ws && this.state === 'open') this.sendSubscribe(topic);
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    // Re-baseline: a topic carried over from a previous mount must not look
    // stale before the fresh connection has had a chance to deliver.
    const now = Date.now();
    for (const topic of this.topics.keys()) this.lastMessageAt.set(topic, now);
    this.installWatchdog();
    this.open();
  }

  stop(): void {
    this.stopped = true;
    this.teardownWatchdog();
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      this.closedSockets.add(ws);
      try {
        ws.close();
      } catch {
        /* already closing */
      }
    }
    this.setStatus('idle');
  }

  /** Full teardown (page navigation/hot reload): no timers, listeners or handlers survive. */
  destroy(): void {
    this.stop();
    this.messageHandlers.clear();
    this.statusHandlers.clear();
    this.topics.clear();
    this.lastMessageAt.clear();
  }

  // ---- staleness watchdog --------------------------------------------------
  // Reconnects are still owned exclusively by `handleClose` (the body of
  // `ws.onclose`). The watchdog only asks the live socket to close; the normal
  // close → backoff → resubscribe path does the rest.

  private installWatchdog(): void {
    if (this.watchdogTimer === null) {
      this.watchdogTimer = setInterval(() => this.watchdogTick(), WATCHDOG_TICK_MS);
    }
    if (!this.visibilityListening && typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.handleVisibilityChange);
      this.visibilityListening = true;
    }
  }

  private teardownWatchdog(): void {
    if (this.watchdogTimer !== null) {
      clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
    if (this.visibilityListening && typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
    this.visibilityListening = false;
    if (this.forceCloseTimer !== null) {
      clearTimeout(this.forceCloseTimer);
      this.forceCloseTimer = null;
    }
  }

  /** Milliseconds since the last `train` frame, or null when not subscribed. */
  private trainSilentFor(): number | null {
    if (!this.topics.has(WATCHDOG_TOPIC)) return null;
    const last = this.lastMessageAt.get(WATCHDOG_TOPIC);
    return last === undefined ? null : Date.now() - last;
  }

  private watchdogTick(): void {
    // Only an established socket can be "deaf"; during backoff/connecting the
    // reconnect path already owns recovery, and force-closing a connecting
    // socket would just add churn.
    if (this.stopped || this.state !== 'open') return;
    // Hidden tabs are throttled/frozen; recovery happens on refocus below.
    if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
    const silent = this.trainSilentFor();
    if (silent !== null && silent > TRAIN_STALE_MS) this.forceReconnect();
  }

  private readonly handleVisibilityChange = (): void => {
    if (this.stopped || this.state !== 'open') return;
    if (typeof document === 'undefined' || document.visibilityState !== 'visible') return;
    const silent = this.trainSilentFor();
    if (silent !== null && silent > TRAIN_STALE_ON_REFOCUS_MS) this.forceReconnect();
  };

  /** Ask the live socket to close; `handleClose` remains the only reconnect owner. */
  private forceReconnect(): void {
    const ws = this.ws;
    if (!ws) return;
    if (ws.readyState === WebSocket.CLOSING || ws.readyState === WebSocket.CLOSED) return;
    if (this.forceCloseTimer !== null) return;
    try {
      ws.close();
    } catch {
      /* already closing */
    }
    // If `onclose` ran synchronously (non-spec, defensive), recovery is already
    // owned by `handleClose`; do not arm a grace timer for a retired socket.
    if (this.ws !== ws) return;
    // A half-open peer may never answer the close handshake, so finalize through
    // the same close path if the browser hasn't fired `onclose` promptly.
    this.forceCloseTimer = setTimeout(() => {
      this.forceCloseTimer = null;
      this.handleClose(ws);
    }, FORCE_CLOSE_GRACE_MS);
  }

  /** Single owner of connection teardown + reconnect scheduling. */
  private handleClose(ws: WebSocket): void {
    if (this.closedSockets.has(ws)) return;
    this.closedSockets.add(ws);
    if (this.forceCloseTimer !== null) {
      clearTimeout(this.forceCloseTimer);
      this.forceCloseTimer = null;
    }
    // A zombie socket must not keep delivering frames after recovery started.
    ws.onopen = null;
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    try {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
    } catch {
      /* already closing */
    }
    if (this.ws === ws) {
      this.ws = null;
    } else if (this.ws !== null) {
      return; // a newer socket already owns recovery
    }
    if (this.stopped) {
      this.setStatus('idle');
      return;
    }
    this.setStatus('closed');
    this.scheduleReconnect();
  }

  private setStatus(status: SocketStatus): void {
    if (this.state === status) return;
    this.state = status;
    for (const handler of this.statusHandlers) handler(status);
  }

  private open(): void {
    this.setStatus('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      // Fresh connection = fresh heartbeat baseline: give the server the full
      // staleness budget before the watchdog may judge it deaf.
      const now = Date.now();
      for (const topic of this.topics.keys()) this.lastMessageAt.set(topic, now);
      this.setStatus('open');
      // Resubscribe every topic; server answers each with a fresh snapshot.
      for (const topic of this.topics.keys()) this.sendSubscribe(topic);
    };
    ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      let msg: ServerMessage;
      try {
        msg = JSON.parse(ev.data) as ServerMessage;
      } catch {
        return;
      }
      if (!msg || typeof msg.topic !== 'string') return;
      if (msg.topic !== '') {
        this.lastMessageAt.set(msg.topic, Date.now());
        if (typeof msg.seq === 'number') this.topics.set(msg.topic, msg.seq);
      }
      for (const handler of this.messageHandlers) handler(msg);
    };
    ws.onerror = () => {
      /* onclose always follows; reconnect handled there */
    };
    ws.onclose = () => {
      this.handleClose(ws);
    };
  }

  private scheduleReconnect(): void {
    if (this.retryTimer !== null) return;
    const delay = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** this.attempt);
    this.attempt += 1;
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      if (!this.stopped) this.open();
    }, delay);
  }

  private sendSubscribe(topic: string): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    this.seq += 1;
    ws.send(JSON.stringify({ seq: this.seq, type: 'subscribe', topic, last_seq: this.topics.get(topic) ?? 0 }));
  }
}

/** App-wide singleton: one WS connection per client, per the contract. */
export const socket = new GambitSocket(wsUrlFromBase(API_BASE));
