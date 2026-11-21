/**
 * WebSocket transport for the §14.0 pinned contract: one connection per client
 * to `<API_BASE>/ws`, typed JSON, sequence-numbered messages.
 *
 * - C→S subscribe: `{seq, type: "subscribe", topic, last_seq}` — `last_seq` is the
 *   last seq seen on that topic so the server can resync on reconnect.
 * - S→C: `{seq, topic, type: "position"|"state"|"metrics"|"log"|"lobby"|"error", payload, ts}`.
 * - Auto-reconnect with exponential backoff (1s → 15s cap); all topics are
 *   resubscribed automatically when the socket reopens.
 *
 * Deliberately rune-free: reactive mirroring lives in the views/stores that
 * consume the `onMessage` / `onStatus` callbacks.
 */
import { wsUrlFromBase, API_BASE } from './config';

export type ServerEventType = 'position' | 'state' | 'metrics' | 'log' | 'lobby' | 'error';
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

export class GambitSocket {
  private ws: WebSocket | null = null;
  private topics = new Map<string, number>(); // topic → last seq seen (sent back on resubscribe)
  private messageHandlers = new Set<MessageHandler>();
  private statusHandlers = new Set<StatusHandler>();
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
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
    if (!this.topics.has(topic)) this.topics.set(topic, 0);
    if (this.ws && this.state === 'open') this.sendSubscribe(topic);
  }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
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
      try {
        ws.close();
      } catch {
        /* already closing */
      }
    }
    this.setStatus('idle');
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
      if (msg.topic !== '' && typeof msg.seq === 'number') this.topics.set(msg.topic, msg.seq);
      for (const handler of this.messageHandlers) handler(msg);
    };
    ws.onerror = () => {
      /* onclose always follows; reconnect handled there */
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      if (this.stopped) {
        this.setStatus('idle');
        return;
      }
      this.setStatus('closed');
      this.scheduleReconnect();
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
