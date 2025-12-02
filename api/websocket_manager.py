#!/usr/bin/env python3
"""DEMIR AI PRO v11.0 - WebSocket Manager (FIXED)

Production-grade WebSocket management:
✅ Proper Python logging (NO extra kwargs)
✅ Real-time AI predictions broadcast
✅ Dashboard live updates
✅ Zero crashes
"""

import logging
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Production WebSocket Manager v11.0"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_task: Optional[asyncio.Task] = None
        self.connection_timestamps: Dict[WebSocket, float] = {}
        self.message_count: int = 0
        self.error_count: int = 0
        self.start_time: float = time.time()
        
        logger.info("✅ WebSocket Manager v11.0 initialized")
        
    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_timestamps[websocket] = time.time()
        
        # ✅ FIXED: Proper logging format
        logger.info(
            "WebSocket connected | Total: %d | Time: %s",
            len(self.active_connections),
            datetime.now(pytz.UTC).isoformat()
        )
        
        await self.send_welcome_message(websocket)
        
    async def send_welcome_message(self, websocket: WebSocket) -> None:
        """Send initial connection message"""
        try:
            welcome = {
                "type": "connection_established",
                "message": "DEMIR AI PRO v11.0 - WebSocket Connected",
                "features": [
                    "Real-time AI predictions",
                    "AI Brain activity streaming",
                    "127 Layer status updates",
                    "Market data broadcasting"
                ],
                "update_interval_seconds": 30,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            await websocket.send_text(json.dumps(welcome))
            logger.debug("Welcome message sent")
        except Exception as e:
            logger.error(f"Welcome message error: {e}")
        
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Calculate connection duration
            if websocket in self.connection_timestamps:
                duration = time.time() - self.connection_timestamps[websocket]
                del self.connection_timestamps[websocket]
            else:
                duration = 0
            
            # ✅ FIXED: Proper logging format
            logger.info(
                "WebSocket disconnected | Remaining: %d | Duration: %.2fs",
                len(self.active_connections),
                duration
            )
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> bool:
        """Send message to specific client"""
        try:
            text = json.dumps(message, default=str)
            await websocket.send_text(text)
            self.message_count += 1
            return True
        except Exception as e:
            logger.error(f"Send personal message error: {e}")
            self.error_count += 1
            self.disconnect(websocket)
            return False
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return 0
        
        try:
            text = json.dumps(message, default=str)
            
            successful = 0
            disconnected = []
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(text)
                    successful += 1
                    self.message_count += 1
                except Exception as e:
                    logger.error(f"Broadcast error: {e}")
                    self.error_count += 1
                    disconnected.append(connection)
            
            for conn in disconnected:
                self.disconnect(conn)
            
            if successful > 0:
                msg_type = message.get('type', 'unknown')
                logger.debug(
                    "Broadcast: %s → %d clients",
                    msg_type,
                    successful
                )
            
            return successful
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            self.error_count += 1
            return 0
    
    async def broadcast_ai_update(self, symbol: str, prediction_data: Dict[str, Any]) -> int:
        """Broadcast AI prediction update"""
        message = {
            "type": "ai_update",
            "symbol": symbol,
            "data": prediction_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_ai_brain_activity(self, brain_metrics: Dict[str, Any]) -> int:
        """Broadcast AI Brain activity metrics"""
        message = {
            "type": "ai_brain_activity",
            "metrics": brain_metrics,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_layer_status(self, layer_data: Dict[str, Any]) -> int:
        """Broadcast 127 Technical Layers status"""
        message = {
            "type": "layer_status",
            "layers": layer_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]) -> int:
        """Broadcast market data updates"""
        message = {
            "type": "market_update",
            "data": market_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_heartbeat(self) -> int:
        """Send heartbeat/keep-alive"""
        message = {
            "type": "heartbeat",
            "active_clients": len(self.active_connections),
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def start_broadcast_loop(self, interval: int = 30) -> None:
        """Start background broadcast task"""
        # ✅ FIXED: Proper logging format
        logger.info(
            "🔄 WebSocket broadcast loop starting | Interval: %ds",
            interval
        )
        
        loop_iteration = 0
        
        while True:
            try:
                await asyncio.sleep(interval)
                loop_iteration += 1
                
                if not self.active_connections:
                    logger.debug("No active connections, skipping broadcast")
                    continue
                
                # Get AI predictions
                try:
                    from core.ai_engine.prediction_engine import get_prediction_engine
                    from api.coin_manager import get_monitored_coins
                    
                    pred_engine = get_prediction_engine()
                    coins = get_monitored_coins()
                    
                    broadcasts_sent = 0
                    for symbol in coins:
                        if symbol in pred_engine.last_predictions:
                            pred = pred_engine.last_predictions[symbol]
                            result = await self.broadcast_ai_update(symbol, {
                                "direction": pred.direction,
                                "confidence": pred.confidence,
                                "model_predictions": pred.model_predictions,
                                "layer_summary": pred.layer_summary if hasattr(pred, 'layer_summary') else {}
                            })
                            broadcasts_sent += result
                    
                    # Heartbeat every 3rd iteration
                    if loop_iteration % 3 == 0:
                        await self.broadcast_heartbeat()
                    
                    if broadcasts_sent > 0:
                        logger.info(
                            "📊 Broadcast iteration #%d | Symbols: %d | Messages: %d",
                            loop_iteration,
                            len(coins),
                            broadcasts_sent
                        )
                    
                except ImportError:
                    logger.warning("Prediction engine not available")
                except Exception as e:
                    logger.error(f"Broadcast data error: {e}")
                    self.error_count += 1
                    
            except asyncio.CancelledError:
                logger.info("Broadcast loop cancelled")
                break
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
                self.error_count += 1
                await asyncio.sleep(5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        uptime = time.time() - self.start_time
        
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": self.message_count,
            "total_errors": self.error_count,
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "error_rate": round(self.error_count / max(self.message_count, 1), 4),
            "messages_per_minute": round((self.message_count / uptime) * 60, 2) if uptime > 0 else 0
        }

# Singleton
_ws_manager: Optional[WebSocketManager] = None

def get_ws_manager() -> WebSocketManager:
    """Get WebSocket manager singleton"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
        logger.info("WebSocket Manager singleton created v11.0")
    return _ws_manager

def reset_ws_manager() -> None:
    """Reset manager (testing)"""
    global _ws_manager
    _ws_manager = None
    logger.info("WebSocket Manager reset")
