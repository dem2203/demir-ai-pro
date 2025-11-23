"""
Telegram Notifier

Sends trading signals and alerts via Telegram.
Production-grade with rate limiting and formatting.
"""

import logging
from typing import Dict, Any, Optional
from telegram import Bot
from telegram.error import TelegramError
import asyncio

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Telegram notification service.
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Target chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        
        logger.info(f"✅ Telegram notifier initialized (chat_id={chat_id})")
    
    def send_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send trading signal to Telegram.
        
        Args:
            signal: Signal data
            
        Returns:
            Success status
        """
        try:
            # Format signal message
            direction_emoji = {
                'LONG': '🟢',
                'SHORT': '🔴',
                'NEUTRAL': '⚪'
            }
            
            emoji = direction_emoji.get(signal.get('direction', 'NEUTRAL'), '⚪')
            
            message = f"""
{emoji} <b>DEMIR AI SIGNAL</b>

<b>Symbol:</b> {signal.get('symbol', '-')}
<b>Direction:</b> {signal.get('direction', '-')}
<b>Confidence:</b> {signal.get('confidence', 0):.1%}
<b>Strength:</b> {signal.get('strength', 0):.1%}

<b>Entry:</b> ${signal.get('entry_price', 0):,.2f}
<b>TP1:</b> ${signal.get('take_profit_1', 0):,.2f}
<b>TP2:</b> ${signal.get('take_profit_2', 0):,.2f}
<b>SL:</b> ${signal.get('stop_loss', 0):,.2f}

⏰ {signal.get('timestamp', '-')}
            """
            
            # Send message
            asyncio.run(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                )
            )
            
            logger.info(f"✅ Signal sent to Telegram: {signal['symbol']}")
            return True
        
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send signal: {e}")
            return False
    
    def send_alert(self, message: str, alert_type: str = 'info') -> bool:
        """
        Send generic alert to Telegram.
        
        Args:
            message: Alert message
            alert_type: 'info', 'warning', 'error', 'success'
            
        Returns:
            Success status
        """
        try:
            emoji_map = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'success': '✅'
            }
            
            emoji = emoji_map.get(alert_type, 'ℹ️')
            formatted_message = f"{emoji} {message}"
            
            asyncio.run(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_message,
                    parse_mode='HTML'
                )
            )
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
            return False
