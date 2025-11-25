"""
Telegram Notification Integration
================================
Production-grade emergency & trade event notifications.
- Sends alerts to Telegram for critical events (emergency, execution, pnl, drawdown, error)
- Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID variables (Railway/Production safe)
- Async, high-resilience, idempotent

Author: DEMIR AI PRO
Version: 8.0
"""

import aiohttp
import os
import logging
from typing import Optional

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

logger = logging.getLogger("notifications.telegram")


class TelegramAlert:
    """Production-grade Telegram alert system"""
    
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.bot_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None
        
        if not self.bot_token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - alerts disabled")
    
    async def send_alert(self, message: str, silent: bool = False, parse_mode: str = "Markdown") -> bool:
        """
        Send alert to Telegram
        
        Args:
            message: Alert text (max 4096 chars)
            silent: If True, no notification sound
            parse_mode: Markdown or HTML
            
        Returns:
            bool: Success status
        """
        if not self.bot_url or not self.chat_id:
            logger.warning("Telegram not configured - alert skipped")
            return False
        
        payload = {
            "chat_id": self.chat_id,
            "text": message[:4096],  # Telegram max message length
            "disable_notification": silent,
            "parse_mode": parse_mode
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.bot_url, json=payload, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Telegram alert failed: {resp.status} - {error_text}")
                        return False
                    
                    logger.info(f"✅ Telegram alert sent: {message[:60]}...")
                    return True
        except asyncio.TimeoutError:
            logger.error("Telegram send timeout")
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def send_trade_alert(self, action: str, symbol: str, size: float, price: float, **kwargs):
        """Send formatted trade execution alert"""
        message = f"🔔 *Trade {action.upper()}*\n"
        message += f"Symbol: `{symbol}`\n"
        message += f"Size: `{size:.6f}`\n"
        message += f"Price: `${price:,.2f}`\n"
        
        if 'stop_loss' in kwargs:
            message += f"Stop Loss: `${kwargs['stop_loss']:,.2f}`\n"
        if 'take_profit' in kwargs:
            message += f"Take Profit: `${kwargs['take_profit']:,.2f}`\n"
        if 'pnl' in kwargs:
            pnl = kwargs['pnl']
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            message += f"PNL: `${pnl:,.2f}` {pnl_emoji}\n"
        
        return await self.send_alert(message)
    
    async def send_emergency_alert(self, event_type: str, title: str, description: str):
        """Send critical emergency alert"""
        message = f"🚨 *EMERGENCY: {event_type}*\n\n"
        message += f"*{title}*\n"
        message += f"{description}\n\n"
        message += "⚠️ Trading may be halted - check system immediately"
        
        return await self.send_alert(message, silent=False)
    
    async def send_performance_alert(self, daily_pnl: float, win_rate: float, total_trades: int):
        """Send daily performance summary"""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        message = f"{pnl_emoji} *Daily Performance*\n\n"
        message += f"PNL: `${daily_pnl:,.2f}`\n"
        message += f"Win Rate: `{win_rate:.1f}%`\n"
        message += f"Total Trades: `{total_trades}`\n"
        
        return await self.send_alert(message)


# Global instance for easy import
_telegram_alert = TelegramAlert()


async def send_telegram_alert(message: str, silent: bool = False) -> bool:
    """Convenience function for quick alerts"""
    return await _telegram_alert.send_alert(message, silent)


# Usage examples:
# await send_telegram_alert("🚀 Trading started!")
# await _telegram_alert.send_trade_alert("OPEN", "BTCUSDT", 0.1, 45000, stop_loss=44000)
# await _telegram_alert.send_emergency_alert("EXCHANGE_DOWN", "Binance API Error", "Connection lost")
