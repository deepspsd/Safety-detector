"""
services/ws_broadcaster.py — Real-time platform WebSocket broadcaster
======================================================================
Provides asynchronous, non-blocking state and alert broadcasting to all
active dashboard / live-monitor WebSockets.

Thread-safe: background worker threads can call `broadcast_state_sync()`
or `broadcast_alert_sync()` safely without needing an active asyncio loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Set
from fastapi import WebSocket

log = logging.getLogger("ws_broadcaster")


class _WebSocketBroadcaster:
    """Thread-safe multi-client WebSocket broadcaster."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        with self._lock:
            self._clients.add(ws)
        log.info("[WSBroadcaster] Client connected (total: %d)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.discard(ws)
        log.info("[WSBroadcaster] Client disconnected (remaining: %d)", len(self._clients))

    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        """Async broadcast to all connected WebSocket clients."""
        with self._lock:
            active_clients = list(self._clients)

        if not active_clients:
            return

        dead_clients: List[WebSocket] = []
        payload_text = json.dumps(data)

        for ws in active_clients:
            try:
                await ws.send_text(payload_text)
            except Exception:
                dead_clients.append(ws)

        if dead_clients:
            with self._lock:
                for ws in dead_clients:
                    self._clients.discard(ws)

    def broadcast_sync(self, data: Dict[str, Any]) -> None:
        """Thread-safe synchronous broadcast method callable from background threads."""
        if not self._clients:
            return

        try:
            loop = self._loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast_json(data), loop)
            else:
                # Try getting existing event loop in thread
                try:
                    cur_loop = asyncio.get_event_loop()
                    if cur_loop.is_running():
                        cur_loop.create_task(self.broadcast_json(data))
                except Exception:
                    pass
        except Exception as exc:
            log.debug("[WSBroadcaster] broadcast_sync error: %s", exc)


broadcaster = _WebSocketBroadcaster()
