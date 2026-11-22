import { useEffect, useRef, useState } from 'react';
import { socket, type ServerMessage, type SocketStatus } from '../api/ws';

/**
 * Subscribe to one or more WS topics and fan frames out to a handler.
 *
 * - Subscribes on mount, unsubscribes on unmount (or when `topics` changes).
 * - The handler is held in a ref, so passing an inline closure does **not**
 *   resubscribe on every render (a classic React WS footgun).
 * - `status` is reactive state; everything else stays out of the render path.
 */
export function useSocket(
  topics: string[],
  onMessage: (msg: ServerMessage) => void,
): { status: SocketStatus; lastFrameAt: number | null } {
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  const [status, setStatus] = useState<SocketStatus>(socket.status);
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null);
  const key = topics.join(',');

  useEffect(() => {
    socket.start();
    const list = key ? key.split(',') : [];

    const offStatus = socket.onStatus(setStatus);
    const offMsg = socket.onMessage((msg) => {
      if (!list.includes(msg.topic)) return;
      setLastFrameAt(Date.now());
      handlerRef.current(msg);
    });

    for (const topic of list) socket.subscribe(topic);

    return () => {
      offMsg();
      offStatus();
      for (const topic of list) socket.unsubscribe(topic);
    };
  }, [key]);

  return { status, lastFrameAt };
}
