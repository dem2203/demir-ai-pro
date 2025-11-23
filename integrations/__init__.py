"""
DEMIR AI PRO - External Integrations

Production-grade integrations:
- Binance Futures API
- Telegram notifications
- Future: More exchanges, webhooks
"""

from .binance import BinanceIntegration
from .telegram import TelegramNotifier

__all__ = ['BinanceIntegration', 'TelegramNotifier']
