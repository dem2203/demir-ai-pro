#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - WebSocket Manager

Manages WebSocket connections for real-time dashboard updates:
- Client connection pooling
- Broadcast AI predictions
- Auto-reconnect handling
- Production error handling
"""

import logging
import asyncio
import json
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None
        logger.info("✅ WebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket):
        """
        Accept new WebSocket connection
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"✅ WebSocket connected (total: {len(self.active_connections)})")
        
        # Start broadcast task if not running
        if not self.broadcast_task:
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection
        """
        self.active_connections.discard(websocket)
        logger.info(f"❌ WebSocket disconnected (total: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific client
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"❌ Send message error: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients
        """
        if not self.active_connections:
            return
        
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"❌ Broadcast error: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def _broadcast_loop(self):
        """
        Background task to broadcast AI predictions every 30 seconds
        """
        logger.info("🔄 Starting WebSocket broadcast loop (30s intervals)")
        
        while True:
            try:
                if self.active_connections:
                    # Get latest predictions
                    from core.ai_engine.prediction_engine import get_prediction_engine
                    engine = get_prediction_engine()
                    
                    # Broadcast predictions for each symbol
                    for symbol in ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']:
                        if symbol in engine.last_predictions:
                            prediction = engine.last_predictions[symbol]
                            await self.broadcast({
                                'type': 'ai_update',
                                'symbol': symbol,
                                **prediction
                            })
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Broadcast loop error: {e}")
                await asyncio.sleep(10)


# Singleton instance
_ws_manager = None

def get_ws_manager() -> WebSocketManager:
    """Get singleton WebSocketManager instance"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
