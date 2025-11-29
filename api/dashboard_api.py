"""
Dashboard API - WebSocket Live Data
===================================
Production-grade WebSocket API for real-time dashboard updates.
- Live P&L streaming
- Position updates
- Trade notifications
- Performance metrics
- Zero mock/fallback

Author: DEMIR AI PRO
Version: 8.0
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
import os
from pathlib import Path

logger = logging.getLogger("api.dashboard")

router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.append(connection)

        # Cleanup disconnected
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


@router.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live dashboard updates
    
    Sends:
    - Live P&L updates
    - Position updates
    - Trade notifications
    - Performance metrics
    """
    await manager.connect(websocket)

    try:
        # Send initial data
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "Dashboard connected"
        })

        # Keep connection alive and listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle client requests if needed
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """
    Get current dashboard statistics
    
    Returns:
        JSON with current stats
    """
    # Import here to avoid circular dependency
    try:
        from database.trade_logger import TradeLogger

        logger_instance = TradeLogger()
        summary = await logger_instance.get_performance_summary()
        await logger_instance.close()

        return {
            "success": True,
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/api/dashboard/trades/recent")
async def get_recent_trades(limit: int = 20):
    """
    Get recent trades
    
    Args:
        limit: Number of trades to return
        
    Returns:
        JSON with recent trades
    """
    try:
        from database.trade_logger import TradeLogger

        logger_instance = TradeLogger()
        trades = await logger_instance.get_trade_history(limit=limit)
        await logger_instance.close()

        return {
            "success": True,
            "data": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get recent trades: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/dashboard")
async def serve_dashboard():
    """
    Serve live dashboard HTML
    Railway-compatible path resolution using pathlib
    """
    try:
        dashboard_path = os.path.join(os.path.dirname(__file__), "..", "ui", "live_dashboard.html")
        # Use pathlib for cross-platform and Railway-compatible path resolution
        dashboard_path = Path(__file__).parent.parent / "ui" / "live_dashboard.html"
        
        # Log the resolved path for debugging
        logger.info(f"Attempting to load dashboard from: {dashboard_path}")
        
        if not dashboard_path.exists():
            logger.error(f"Dashboard file not found at: {dashboard_path}")
            return HTMLResponse(
                content=f"<html><body><h1>Dashboard Not Found</h1><p>Path: {dashboard_path}</p></body></html>",
                status_code=404
            )
        
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        logger.info("✅ Dashboard HTML loaded successfully")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Failed to serve dashboard: {e}")
        return HTMLResponse(
            content=f"<html><body><h1>Error loading dashboard</h1><p>{str(e)}</p></body></html>",
            status_code=500
        )


# Broadcasting helper functions (to be called from trading engine)
async def broadcast_trade_update(trade_data: Dict[str, Any]):
    """Broadcast trade update to all connected clients"""
    await manager.broadcast({
        "type": "trade_update",
        "data": trade_data,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_position_update(position_data: Dict[str, Any]):
    """Broadcast position update to all connected clients"""
    await manager.broadcast({
        "type": "position_update",
        "data": position_data,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_pnl_update(pnl_data: Dict[str, Any]):
    """Broadcast P&L update to all connected clients"""
    await manager.broadcast({
        "type": "pnl_update",
        "data": pnl_data,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_performance_update(perf_data: Dict[str, Any]):
    """Broadcast performance metrics update"""
    await manager.broadcast({
        "type": "performance_update",
        "data": perf_data,
        "timestamp": datetime.now().isoformat()
    })
