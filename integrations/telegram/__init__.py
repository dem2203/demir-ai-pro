"""
Telegram Integration Module

Real-time notifications via Telegram bot.
Production-grade with message queuing and retry.
"""

from .notifier import TelegramNotifier

__all__ = ['TelegramNotifier']
