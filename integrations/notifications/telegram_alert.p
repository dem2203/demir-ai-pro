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

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

logger = logging.getLogger("notifications.telegram")

async def send_telegram_alert(message: str, silent: bool = False) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram bot config missing, alert skipped.")
        return False
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],  # Telegram max msg len
        "disable_notification": silent,
        "parse_mode": "Markdown"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(BOT_URL, data=payload, timeout=12) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram alert failed: {resp.status}")
                    return False
                logger.info(f"Sent telegram alert: {message[:60]}")
                return True
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False

# Usage examples:
# await send_telegram_alert("Emergency event detected! Trading halted.")
# await send_telegram_alert(f"Trade opened: {symbol} size={size:.4f} PNL={pnl}")
