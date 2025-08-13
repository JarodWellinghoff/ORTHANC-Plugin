#!/usr/bin/env python3

"""
WebSocket server for real-time CHO calculation updates
This module provides WebSocket functionality for ORTHANC CHO analysis real-time updates
"""

import asyncio
import websockets
import json
import threading
import time
import logging
from typing import Set, Dict, Any
import socket
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketProgressBroadcaster:
    """WebSocket server for broadcasting calculation progress"""
    
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.series_clients: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.server = None
        self.loop = None
        self.thread = None
        self._running = False
        
    def start_server(self):
        """Start the WebSocket server in a separate thread"""
        if self._running:
            logger.warning("WebSocket server is already running")
            return
            
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
    def _run_server(self):
        """Run the WebSocket server event loop"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            start_server = websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.server = self.loop.run_until_complete(start_server)
            self._running = True
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            self.loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            self._running = False
            
    def stop_server(self):
        """Stop the WebSocket server"""
        if not self._running:
            return
            
        self._running = False
        if self.loop and self.server:
            # Schedule the server close in the event loop
            asyncio.run_coroutine_threadsafe(self._close_server(), self.loop)
            
    async def _close_server(self):
        """Close the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")
            
    async def handle_client(self, websocket, path):
        """Handle new WebSocket client connections"""
        try:
            logger.info(f"New WebSocket client connected from {websocket.remote_address}")
            self.clients.add(websocket)
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'welcome',
                'message': 'Connected to CHO Analysis WebSocket server',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
        finally:
            # Clean up client references
            self.clients.discard(websocket)
            self._remove_client_from_series(websocket)
            
    def _remove_client_from_series(self, websocket):
        """Remove client from all series subscriptions"""
        for series_id in list(self.series_clients.keys()):
            self.series_clients[series_id].discard(websocket)
            if not self.series_clients[series_id]:
                del self.series_clients[series_id]
                
    async def handle_message(self, websocket, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                series_id = data.get('series_id')
                if series_id:
                    await self.subscribe_to_series(websocket, series_id)
                    
            elif msg_type == 'unsubscribe':
                series_id = data.get('series_id')
                if series_id:
                    await self.unsubscribe_from_series(websocket, series_id)
                    
            elif msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received from {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
    async def subscribe_to_series(self, websocket, series_id):
        """Subscribe a client to updates for a specific series"""
        if series_id not in self.series_clients:
            self.series_clients[series_id] = set()
            
        self.series_clients[series_id].add(websocket)
        
        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'series_id': series_id,
            'message': f'Subscribed to updates for series {series_id}'
        }))
        
        logger.info(f"Client {websocket.remote_address} subscribed to series {series_id}")
        
    async def unsubscribe_from_series(self, websocket, series_id):
        """Unsubscribe a client from series updates"""
        if series_id in self.series_clients:
            self.series_clients[series_id].discard(websocket)
            if not self.series_clients[series_id]:
                del self.series_clients[series_id]
                
        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'unsubscribed',
            'series_id': series_id,
            'message': f'Unsubscribed from series {series_id}'
        }))
        
    def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self._running or not self.clients:
            return
            
        # Schedule the broadcast in the WebSocket event loop
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_to_all(message), 
                self.loop
            )
            
    async def _broadcast_to_all(self, message: Dict[str, Any]):
        """Internal method to broadcast to all clients"""
        if not self.clients:
            return
            
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.clients.copy():
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
            self._remove_client_from_series(client)
            
    def broadcast_to_series(self, series_id: str, message: Dict[str, Any]):
        """Broadcast a message to clients subscribed to a specific series"""
        if not self._running or series_id not in self.series_clients:
            return
            
        # Schedule the broadcast in the WebSocket event loop
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_to_series(series_id, message), 
                self.loop
            )
            
    async def _broadcast_to_series(self, series_id: str, message: Dict[str, Any]):
        """Internal method to broadcast to series subscribers"""
        if series_id not in self.series_clients:
            return
            
        clients = self.series_clients[series_id].copy()
        if not clients:
            return
            
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to series client: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        for client in disconnected:
            self.series_clients[series_id].discard(client)
            if not self.series_clients[series_id]:
                del self.series_clients[series_id]
                
    def get_stats(self):
        """Get server statistics"""
        return {
            'running': self._running,
            'total_clients': len(self.clients),
            'series_subscriptions': {
                series_id: len(clients) 
                for series_id, clients in self.series_clients.items()
            },
            'host': self.host,
            'port': self.port
        }

# Global WebSocket broadcaster instance
ws_broadcaster = WebSocketProgressBroadcaster()

def start_websocket_server():
    """Start the WebSocket server"""
    ws_broadcaster.start_server()

def stop_websocket_server():
    """Stop the WebSocket server"""
    ws_broadcaster.stop_server()

def broadcast_progress_update(series_id: str, progress_data: Dict[str, Any]):
    """Broadcast progress update for a specific series"""
    message = {
        'type': 'progress_update',
        'series_id': series_id,
        'data': progress_data,
        'timestamp': datetime.now().isoformat()
    }
    
    # Broadcast to both series subscribers and all clients
    ws_broadcaster.broadcast_to_series(series_id, message)
    ws_broadcaster.broadcast_to_all(message)

def broadcast_calculation_status(series_id: str, status: str, data: Dict[str, Any] = None):
    """Broadcast calculation status change"""
    message = {
        'type': 'status_update',
        'series_id': series_id,
        'status': status,
        'data': data or {},
        'timestamp': datetime.now().isoformat()
    }
    
    ws_broadcaster.broadcast_to_series(series_id, message)
    ws_broadcaster.broadcast_to_all(message)

def get_websocket_stats():
    """Get WebSocket server statistics"""
    return ws_broadcaster.get_stats()

# Auto-start the WebSocket server when module is imported
start_websocket_server()