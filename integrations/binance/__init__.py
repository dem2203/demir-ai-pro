"""
Binance Integration Module

Production-grade Binance Futures API integration.
Features:
- REST API with rate limiting
- WebSocket real-time streams
- Order execution
- Position management
"""

from .client import BinanceClient
from .websocket import BinanceWebSocket

__all__ = ['BinanceClient', 'BinanceWebSocket']
