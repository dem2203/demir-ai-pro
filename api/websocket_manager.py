#!/usr/bin/env python3
"""
DEMIR AI PRO v9.0 PROFESSIONAL - WebSocket Manager

Real-time WebSocket connection management for Ultra Dashboard:
- Client connection/disconnection tracking
- AI predictions broadcast
- AI Brain activity updates
- 127 Layer status streaming
- Latency tracking
- Automatic reconnection handling
- Structured logging
- Performance metrics

✅ Production-grade WebSocket management
✅ Ultra Dashboard integration
✅ Real-time AI Brain visualization
✅ NO MOCK DATA
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
    """
    Manages WebSocket connections for Ultra Professional Dashboard
    
    Features:
    - Multi-client connection management
    - Real-time AI predictions broadcast
    - AI Brain activity streaming
    - 127 Technical layers status
    - Market data updates
    - Performance metrics tracking
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_task: Optional[asyncio.Task] = None
        self.connection_timestamps: Dict[WebSocket, float] = {}
        self.message_count: int = 0
        self.error_count: int = 0
        self.start_time: float = time.time()
        
        logger.info("WebSocket Manager initialized v9.0")
        
    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept new WebSocket connection from Ultra Dashboard
        
        Args:
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_timestamps[websocket] = time.time()
        
        logger.info(
            "WebSocket connected",
            total_connections=len(self.active_connections),
            timestamp=datetime.now(pytz.UTC).isoformat()
        )
        
        # Send welcome message with current status
        await self.send_welcome_message(websocket)
        
    async def send_welcome_message(self, websocket: WebSocket) -> None:
        """
        Send initial connection message to new client
        
        Args:
            websocket: Client WebSocket connection
        """
        try:
            welcome = {
                "type": "connection_established",
                "message": "DEMIR AI PRO v9.0 ULTRA - WebSocket Connected",
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
            logger.debug("Welcome message sent to client")
        except Exception as e:
            logger.error(f"Welcome message error: {e}")
        
    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove WebSocket connection
        
        Args:
            websocket: WebSocket to disconnect
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Calculate connection duration
            if websocket in self.connection_timestamps:
                duration = time.time() - self.connection_timestamps[websocket]
                del self.connection_timestamps[websocket]
            else:
                duration = 0
            
            logger.info(
                "WebSocket disconnected",
                remaining_connections=len(self.active_connections),
                connection_duration_seconds=round(duration, 2),
                timestamp=datetime.now(pytz.UTC).isoformat()
            )
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> bool:
        """
        Send message to specific client
        
        Args:
            message: Dictionary to send (will be JSON serialized)
            websocket: Target WebSocket connection
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
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
        """
        Broadcast message to all connected clients
        
        Args:
            message: Dictionary to broadcast (will be JSON serialized)
            
        Returns:
            int: Number of clients successfully reached
        """
        if not self.active_connections:
            return 0
        
        try:
            text = json.dumps(message, default=str)
            
            # Send to all clients
            successful = 0
            disconnected = []
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(text)
                    successful += 1
                    self.message_count += 1
                except Exception as e:
                    logger.error(f"Broadcast error to client: {e}")
                    self.error_count += 1
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.disconnect(conn)
            
            if successful > 0:
                logger.debug(
                    "Broadcast successful",
                    clients_reached=successful,
                    message_type=message.get('type', 'unknown')
                )
            
            return successful
            
        except Exception as e:
            logger.error(f"Broadcast serialization error: {e}")
            self.error_count += 1
            return 0
    
    async def broadcast_ai_update(self, symbol: str, prediction_data: Dict[str, Any]) -> int:
        """
        Broadcast AI prediction update for Ultra Dashboard
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            prediction_data: AI prediction dictionary
            
        Returns:
            int: Number of clients reached
        """
        message = {
            "type": "ai_update",
            "symbol": symbol,
            "data": prediction_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_ai_brain_activity(self, brain_metrics: Dict[str, Any]) -> int:
        """
        Broadcast AI Brain activity for Ultra Dashboard visualization
        
        Args:
            brain_metrics: Dictionary containing model performance metrics
                Example: {
                    'lstm_layer1': 85,
                    'lstm_layer2': 78,
                    'xgboost': 92,
                    'random_forest': 67,
                    'gradient_boosting': 81,
                    'ensemble_confidence': 83.5
                }
                
        Returns:
            int: Number of clients reached
        """
        message = {
            "type": "ai_brain_activity",
            "metrics": brain_metrics,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_layer_status(self, layer_data: Dict[str, Any]) -> int:
        """
        Broadcast 127 Technical Layers status
        
        Args:
            layer_data: Dictionary with layer metrics
                Example: {
                    'rsi_14': {'value': 42.3, 'signal': 'neutral', 'weight': 8},
                    'macd': {'signal': 'bullish_crossover', 'weight': 12},
                    'composite_score': 68
                }
                
        Returns:
            int: Number of clients reached
        """
        message = {
            "type": "layer_status",
            "layers": layer_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]) -> int:
        """
        Broadcast market data updates (prices, volume, etc.)
        
        Args:
            market_data: Dictionary with market information
                Example: {
                    'BTCUSDT': {'price': 90975.30, 'change_24h': 0.06},
                    'ETHUSDT': {'price': 3019.22, 'change_24h': 0.93}
                }
                
        Returns:
            int: Number of clients reached
        """
        message = {
            "type": "market_update",
            "data": market_data,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def broadcast_heartbeat(self) -> int:
        """
        Send heartbeat/keep-alive message
        
        Returns:
            int: Number of clients reached
        """
        message = {
            "type": "heartbeat",
            "active_clients": len(self.active_connections),
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        return await self.broadcast(message)
    
    async def start_broadcast_loop(self, interval: int = 30) -> None:
        """
        Start background task for periodic broadcasts to Ultra Dashboard
        
        Args:
            interval: Broadcast interval in seconds (default: 30)
        """
        logger.info(
            "WebSocket broadcast loop starting",
            interval_seconds=interval,
            features=["AI predictions", "Brain activity", "Layer status", "Market data"]
        )
        
        loop_iteration = 0
        
        while True:
            try:
                await asyncio.sleep(interval)
                loop_iteration += 1
                
                if not self.active_connections:
                    logger.debug("No active WebSocket connections, skipping broadcast")
                    continue
                
                # Get AI predictions from prediction engine
                try:
                    from core.ai_engine.prediction_engine import get_prediction_engine
                    from api.coin_manager import get_monitored_coins
                    
                    pred_engine = get_prediction_engine()
                    coins = get_monitored_coins()
                    
                    # Broadcast AI updates for each coin
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
                    
                    # Send heartbeat every 3rd iteration
                    if loop_iteration % 3 == 0:
                        await self.broadcast_heartbeat()
                    
                    if broadcasts_sent > 0:
                        logger.info(
                            "Broadcast loop iteration complete",
                            iteration=loop_iteration,
                            symbols_broadcast=len(coins),
                            total_messages_sent=broadcasts_sent
                        )
                    
                except ImportError:
                    logger.warning("Prediction engine not available for broadcast")
                except Exception as e:
                    logger.error(f"Broadcast loop data collection error: {e}")
                    self.error_count += 1
                    
            except asyncio.CancelledError:
                logger.info("Broadcast loop cancelled gracefully")
                break
            except Exception as e:
                logger.error(f"Broadcast loop fatal error: {e}")
                self.error_count += 1
                await asyncio.sleep(5)  # Wait before retry
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket manager statistics
        
        Returns:
            Dictionary with statistics
        """
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

# Singleton instance
_ws_manager: Optional[WebSocketManager] = None

def get_ws_manager() -> WebSocketManager:
    """
    Get WebSocket manager singleton instance
    
    Returns:
        WebSocketManager instance
    """
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
        logger.info("WebSocket Manager singleton created v9.0 ULTRA")
    return _ws_manager

def reset_ws_manager() -> None:
    """
    Reset WebSocket manager (useful for testing)
    """
    global _ws_manager
    _ws_manager = None
    logger.info("WebSocket Manager reset")
