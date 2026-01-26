#!/usr/bin/env python3
"""
Utility module to manage Server-Sent Events (SSE) subscribers.

This manager keeps track of connected clients, exposes a simple publish API,
and ensures that slow or disconnected clients are cleaned up automatically.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Dict, Tuple


class SSEManager:
    """Thread-safe registry of SSE subscribers."""

    def __init__(self) -> None:
        self._clients: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def add_client(self, max_queue_size: int = 128) -> Tuple[str, queue.Queue]:
        """
        Register a new client and return its identifier and message queue.

        Args:
            max_queue_size: Maximum number of pending messages for this client.

        Returns:
            Tuple containing the client identifier and its queue.
        """
        client_id = uuid.uuid4().hex
        channel: queue.Queue = queue.Queue(maxsize=max_queue_size)

        with self._lock:
            self._clients[client_id] = channel

        return client_id, channel

    def remove_client(self, client_id: str) -> bool:
        """
        Remove a client from the registry and drain any pending messages.

        Args:
            client_id: Identifier of the client to remove.

        Returns:
            True if the client existed, False otherwise.
        """
        with self._lock:
            channel = self._clients.pop(client_id, None)

        if channel is None:
            return False

        # Drain queue to free resources
        try:
            while True:
                channel.get_nowait()
        except queue.Empty:
            pass

        return True

    def broadcast(self, event: str, data: dict) -> None:
        """
        Push an event to all connected clients.

        Args:
            event: Name of the SSE event.
            data: Payload to send to the client.
        """
        message = {
            "event": event,
            "data": data,
            "timestamp": time.time(),
        }

        with self._lock:
            clients_snapshot = list(self._clients.items())

        stale_clients = []
        for client_id, channel in clients_snapshot:
            try:
                channel.put_nowait(message)
            except queue.Full:
                # Drop unresponsive client
                stale_clients.append(client_id)

        for client_id in stale_clients:
            self.remove_client(client_id)

    def client_count(self) -> int:
        """Return the number of currently connected clients."""
        with self._lock:
            return len(self._clients)


# Shared singleton used throughout the backend
sse_manager = SSEManager()

