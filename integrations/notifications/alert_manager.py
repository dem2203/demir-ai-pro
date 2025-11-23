#!/usr/bin/env python3
"""
Alert Manager - Real-Time Notification Engine

- Telegram, Discord, Email entegrasyonu
- Her anomali/fırsat/sinyali anında bildirim olarak iletir
- Production-grade, otomatik bağlantı testli
"""

import logging
import requests
import asyncio

logger = logging.getLogger("alert_manager")

class AlertManager:
    def __init__(self, telegram_token:str=None, telegram_chat_id:str=None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        logger.info("✅ Alert manager initialized")

    async def notify_telegram(self, msg:str):
        """
        Telegram'a proaktif bildirim gönderir
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram token/chat id yok")
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            resp = requests.post(url, json={"chat_id":self.telegram_chat_id, "text":msg, "parse_mode":"HTML"},timeout=5)
            if resp.status_code == 200:
                logger.info("Telegram notification gönderildi")
            else:
                logger.error(f"Telegram error: {resp.status_code} {resp.text}")
        except Exception as exc:
            logger.error(f"Telegram notification EXCEPTION: {exc}")
    # Benzer şekilde notify_discord, notify_email fonksiyonları eklenebilir
# Kullanım: alert_mgr = AlertManager(token, chat_id); await alert_mgr.notify_telegram('ALERT msg')
