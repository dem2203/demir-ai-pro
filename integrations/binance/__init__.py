"""
Binance Integration Module

Production-grade Binance Futures API integration.
Features:
- REST API with rate limiting
- WebSocket real-time streams (optional)
- Order execution
- Position management

Graceful degradation: Missing modules won't break the system
"""

# Try to import client (required)
try:
    from .client import BinanceClient
    BINANCE_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Binance client import failed: {e}")
    BinanceClient = None
    BINANCE_CLIENT_AVAILABLE = False

# Try to import websocket (optional)
try:
    from .websocket import BinanceWebSocket
    BINANCE_WEBSOCKET_AVAILABLE = True
except ImportError:
    BinanceWebSocket = None
    BINANCE_WEBSOCKET_AVAILABLE = False

__all__ = [
    'BinanceClient',
    'BinanceWebSocket',
    'BINANCE_CLIENT_AVAILABLE',
    'BINANCE_WEBSOCKET_AVAILABLE'
]
