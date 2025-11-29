#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - Telegram Notifier

Sends trading signals and alerts via Telegram:
- Strong BUY/SELL signals
- System status updates
- Error notifications
- Production-ready async implementation
"""

import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Async Telegram bot for trading notifications
    """
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        logger.info("✅ TelegramNotifier initialized")
    
    async def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """
        Send message to Telegram chat
        
        Args:
            text: Message text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.info("✅ Telegram message sent")
                        return True
                    else:
                        logger.error(f"❌ Telegram API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    async def send_strong_buy_alert(self, symbol: str, confidence: float, 
                                    agreement: float, price: float):
        """
        Send STRONG BUY alert
        """
        message = (
            f"🚀 <b>STRONG BUY SIGNAL</b>\n\n"
            f"📊 <b>Symbol:</b> {symbol}\n"
            f"💰 <b>Price:</b> ${price:.2f}\n"
            f"💪 <b>Confidence:</b> {confidence*100:.1f}%\n"
            f"🤝 <b>Model Agreement:</b> {agreement*100:.0f}%\n\n"
            f"🤖 <b>DEMIR AI PRO v8.0</b>\n"
            f"⏰ {self._get_timestamp()}"
        )
        await self.send_message(message)
    
    async def send_strong_sell_alert(self, symbol: str, confidence: float, 
                                     agreement: float, price: float):
        """
        Send STRONG SELL alert
        """
        message = (
            f"⚠️ <b>STRONG SELL SIGNAL</b>\n\n"
            f"📊 <b>Symbol:</b> {symbol}\n"
            f"💰 <b>Price:</b> ${price:.2f}\n"
            f"💪 <b>Confidence:</b> {confidence*100:.1f}%\n"
            f"🤝 <b>Model Agreement:</b> {agreement*100:.0f}%\n\n"
            f"🤖 <b>DEMIR AI PRO v8.0</b>\n"
            f"⏰ {self._get_timestamp()}"
        )
        await self.send_message(message)
    
    async def send_error_alert(self, error_message: str):
        """
        Send error notification
        """
        message = (
            f"❌ <b>SYSTEM ERROR</b>\n\n"
            f"{error_message}\n\n"
            f"⏰ {self._get_timestamp()}"
        )
        await self.send_message(message)
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get formatted timestamp"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
