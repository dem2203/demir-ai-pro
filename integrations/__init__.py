"""
DEMIR AI PRO - External Integrations

Production-grade integrations:
- Binance Futures API
- Telegram notifications
- Future: More exchanges, webhooks

Graceful degradation: Missing integrations won't break the system
"""

# Binance Integration (optional)
try:
    from .binance_integration import BinanceIntegration
    BINANCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Binance integration not available: {e}")
    BinanceIntegration = None
    BINANCE_AVAILABLE = False

# Telegram Integration (optional)
try:
    from .telegram import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TelegramNotifier = None
    TELEGRAM_AVAILABLE = False

__all__ = [
    'BinanceIntegration',
    'TelegramNotifier',
    'BINANCE_AVAILABLE',
    'TELEGRAM_AVAILABLE'
]
