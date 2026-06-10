"""Asyncio WebSocket broadcast server for the viz bridge.

Runs its own asyncio event loop on a background thread so it can live alongside
the synchronous rclpy executor in the same process. The ROS side calls the
thread-safe ``broadcast()`` from its subscription callbacks; the server fans each
frame out to every connected client.

Design notes
------------
* **Latest-frame-wins per channel.** Each client keeps a tiny per-channel mailbox
  holding only the most recent frame. A slow/stalled laptop can therefore never
  back-pressure the ROS callbacks or grow memory without bound — it just sees the
  newest data when it catches up. Realtime visualization wants freshness, not a
  complete history.
* **Resilient.** A client that errors or disconnects is dropped silently; the
  server and all other clients keep running.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Dict, Optional

try:
    import websockets
except ImportError as exc:  # pragma: no cover - surfaced at runtime with a clear msg
    raise ImportError(
        "The 'websockets' package is required by vam_viz_bridge. "
        "Install it with: pip install websockets"
    ) from exc


class _Client:
    """Per-connection state: a latest-frame-wins mailbox keyed by channel."""

    def __init__(self) -> None:
        self.mailbox: Dict[str, str] = {}
        self.event = asyncio.Event()

    def put(self, channel: str, payload: str) -> None:
        self.mailbox[channel] = payload
        self.event.set()

    def drain(self) -> list:
        items = list(self.mailbox.values())
        self.mailbox.clear()
        self.event.clear()
        return items


class WebSocketBroadcastServer:
    """Threaded WebSocket server with a thread-safe broadcast() entry point."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, logger=None) -> None:
        self._host = host
        self._port = port
        self._logger = logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: "set[_Client]" = set()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ws-broadcast", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())
        self._loop.run_forever()

    async def _serve(self) -> None:
        # max_queue keeps a misbehaving client from buffering unboundedly.
        await websockets.serve(
            self._handler, self._host, self._port, ping_interval=20, max_queue=8
        )
        self._ready.set()
        self._log_info(f"WebSocket server listening on ws://{self._host}:{self._port}")

    async def _handler(self, connection, path=None) -> None:
        # `path` is passed positionally by older websockets (<11) and omitted by
        # newer versions; the default keeps the handler compatible with both.
        client = _Client()
        self._clients.add(client)
        peer = getattr(connection, "remote_address", ("?", 0))
        self._log_info(f"client connected: {peer} (total={len(self._clients)})")
        try:
            while True:
                await client.event.wait()
                for payload in client.drain():
                    await connection.send(payload)
        except Exception:
            pass
        finally:
            self._clients.discard(client)
            self._log_info(f"client disconnected: {peer} (total={len(self._clients)})")

    # -- broadcast (thread-safe, called from ROS callbacks) ----------------- #
    def broadcast(self, channel: str, frame: dict) -> None:
        """Serialize a frame to JSON and queue it for every connected client.

        Safe to call from any thread (e.g. an rclpy subscription callback).
        """
        if self._loop is None or not self._clients:
            return
        payload = json.dumps(frame, separators=(",", ":"))
        self._loop.call_soon_threadsafe(self._enqueue, channel, payload)

    def _enqueue(self, channel: str, payload: str) -> None:
        for client in self._clients:
            client.put(channel, payload)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- logging ------------------------------------------------------------ #
    def _log_info(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(f"[ws] {msg}")
