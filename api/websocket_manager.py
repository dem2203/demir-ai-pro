#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - WebSocket Manager

Real-time WebSocket connection management:
- Client connection/disconnection
- Broadcast AI predictions
- Latency tracking
- Reconnection handling
"""

import logging
import json
import asyncio
from typing import List, Dict
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_task = None
        
    async def connect(self, websocket: WebSocket):
        """
        Accept new WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"❌ WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send message to specific client
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"❌ Send message error: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Dictionary to broadcast (will be JSON serialized)
        """
        if not self.active_connections:
            return
        
        try:
            text = json.dumps(message, default=str)
            
            # Send to all clients
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(text)
                except Exception as e:
                    logger.error(f"❌ Broadcast error to client: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.disconnect(conn)
                
            logger.debug(f"📡 Broadcast to {len(self.active_connections)} clients")
            
        except Exception as e:
            logger.error(f"❌ Broadcast serialization error: {e}")
    
    async def broadcast_ai_snapshot(self, signals: Dict):
        """
        Broadcast AI prediction snapshot to all clients
        
        Args:
            signals: Dictionary of AI predictions by symbol
        """
        message = {
            "type": "ai_snapshot",
            "data": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message)
    
    async def start_broadcast_loop(self, interval: int = 30):
        """
        Start background task for periodic broadcasts
        
        Args:
            interval: Broadcast interval in seconds (default: 30)
        """
        logger.info(f"🔁 Starting broadcast loop (interval: {interval}s)")
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                if not self.active_connections:
                    continue
                
                # Get AI snapshot from prediction engine
                try:
                    from core.ai_engine.prediction_engine import get_prediction_engine
                    from api.coin_manager import get_monitored_coins
                    
                    pred_engine = get_prediction_engine()
                    coins = get_monitored_coins()
                    
                    signals = {}
                    for symbol in coins:
                        if symbol in pred_engine.last_predictions:
                            signals[symbol] = pred_engine.last_predictions[symbol]
                    
                    if signals:
                        await self.broadcast_ai_snapshot(signals)
                        logger.info(f"✅ Broadcast {len(signals)} AI signals")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Broadcast loop error: {e}")
                    
            except asyncio.CancelledError:
                logger.info("🛑 Broadcast loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Broadcast loop fatal error: {e}")
                await asyncio.sleep(5)

# Singleton instance
_ws_manager = None

def get_ws_manager() -> WebSocketManager:
    """
    Get WebSocket manager singleton instance
    """
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
