#!/usr/bin/env python3
"""DEMIR AI PRO v10.0 - Ultra Telegram Notifier

Professional Telegram integration with:
- Real-time signal alerts (<30 sec)
- Rich formatting with emojis
- Position recommendations
- Risk/reward analysis
- Priority-based routing
- Rate limiting

❌ NO SPAM
✅ Professional Alert System
"""

import logging
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta
import aiohttp
import pytz

logger = logging.getLogger(__name__)

class TelegramUltra:
    """Ultra-enhanced Telegram notification system"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_message_time = {}
        self.rate_limit_seconds = 30  # Min 30 sec between similar messages
        logger.info("Telegram Ultra initialized")
    
    async def send_signal_alert(self, signal) -> bool:
        """Send professional trading signal alert"""
        try:
            from core.signal_engine import SignalType, SignalPriority
            
            # Format signal message
            emoji_map = {
                SignalType.STRONG_BUY: "🚀",
                SignalType.BUY: "🟢",
                SignalType.NEUTRAL: "⚪",
                SignalType.SELL: "🔴",
                SignalType.STRONG_SELL: "⚠️"
            }
            
            priority_emoji = {
                SignalPriority.CRITICAL: "🚨",
                SignalPriority.HIGH: "🔔",
                SignalPriority.MEDIUM: "🔵",
                SignalPriority.LOW: "🟡"
            }
            
            signal_emoji = emoji_map.get(signal.signal_type, "❓")
            priority_icon = priority_emoji.get(signal.priority, "")
            
            # Calculate risk/reward ratio
            if signal.signal_type.value in ["BUY", "STRONG_BUY"]:
                risk = signal.entry_price - signal.stop_loss
                reward = signal.target_price - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.target_price
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Build message
            message = f"{priority_icon} {signal_emoji} **{signal.signal_type.value}** SIGNAL\n\n"
            message += f"📊 **Symbol:** {signal.symbol}\n"
            message += f"🎯 **Priority:** {signal.priority.value}\n"
            message += f"💪 **Confidence:** {signal.confidence:.1f}%\n"
            message += f"⚡ **Strength:** {signal.strength:.1f}/100\n\n"
            
            message += f"**💰 TRADING LEVELS:**\n"
            message += f"Entry: ${signal.entry_price:,.2f}\n"
            message += f"Target: ${signal.target_price:,.2f} (+{((signal.target_price/signal.entry_price-1)*100):.2f}%)\n"
            message += f"Stop Loss: ${signal.stop_loss:,.2f} ({((signal.stop_loss/signal.entry_price-1)*100):.2f}%)\n"
            message += f"Risk/Reward: 1:{rr_ratio:.2f}\n\n"
            
            message += f"**📈 POSITION SIZING:**\n"
            message += f"Recommended: {signal.position_size_percent:.1f}% of portfolio\n\n"
            
            message += f"**💡 ANALYSIS:**\n"
            message += f"Technical: {signal.technical_score:.0f}/100\n"
            message += f"AI Models: {signal.ai_score:.0f}/100\n"
            message += f"Market Intel: {signal.market_intelligence_score:.0f}/100\n"
            message += f"Risk: {signal.risk_score:.0f}/100\n\n"
            
            if signal.reasons:
                message += f"**✅ REASONS:**\n"
                for reason in signal.reasons:
                    message += f"\u2022 {reason}\n"
                message += "\n"
            
            if signal.warnings:
                message += f"**⚠️ WARNINGS:**\n"
                for warning in signal.warnings:
                    message += f"\u2022 {warning}\n"
                message += "\n"
            
            message += f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            message += f"⏱️ Expires: {signal.expires_at[:19]} UTC"
            
            # Check rate limiting
            key = f"{signal.symbol}_{signal.signal_type.value}"
            if self._is_rate_limited(key):
                logger.info(f"Signal alert rate limited: {key}")
                return False
            
            # Send message
            success = await self._send_message(message)
            
            if success:
                self.last_message_time[key] = datetime.now(pytz.UTC)
                logger.info(f"Signal alert sent: {signal.symbol} {signal.signal_type.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Signal alert error: {e}")
            return False
    
    async def send_market_update(self, symbol: str, analysis: dict) -> bool:
        """Send market intelligence update"""
        try:
            message = f"📊 **MARKET UPDATE: {symbol}**\n\n"
            
            # Market depth
            if analysis.get('market_depth'):
                depth = analysis['market_depth']
                buy_pressure_pct = depth.buy_pressure * 100
                pressure_emoji = "🟢" if buy_pressure_pct > 55 else ("🔴" if buy_pressure_pct < 45 else "⚪")
                
                message += f"**📊 ORDER BOOK:**\n"
                message += f"{pressure_emoji} Buy Pressure: {buy_pressure_pct:.1f}%\n"
                message += f"📈 Liquidity: {depth.liquidity_score:.0f}/100\n"
                message += f"💰 Spread: {depth.spread_percent:.3f}%\n\n"
            
            # Sentiment
            if analysis.get('sentiment'):
                sentiment = analysis['sentiment']
                sentiment_emoji = "🟢" if sentiment.sentiment_score > 0.2 else ("🔴" if sentiment.sentiment_score < -0.2 else "⚪")
                
                message += f"**🧠 MARKET SENTIMENT:**\n"
                message += f"{sentiment_emoji} Score: {sentiment.sentiment_score:.2f}\n"
                message += f"😨🤑 Fear/Greed: {sentiment.fear_greed_index:.0f}/100\n"
                message += f"💸 Funding: {sentiment.funding_rate:.4f}%\n\n"
            
            # Whale activity
            if analysis.get('whale_activity'):
                whale = analysis['whale_activity']
                if whale.detected_whales > 0:
                    whale_emoji = "🐳"
                    direction_emoji = "🟢" if whale.whale_direction == "BUY" else ("🔴" if whale.whale_direction == "SELL" else "⚪")
                    
                    message += f"**{whale_emoji} WHALE ACTIVITY:**\n"
                    message += f"Detected: {whale.detected_whales} large orders\n"
                    message += f"{direction_emoji} Direction: {whale.whale_direction}\n"
                    message += f"💪 Confidence: {whale.confidence*100:.0f}%\n"
                    message += f"💰 Volume: ${whale.total_whale_volume:,.0f}\n\n"
            
            message += f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Market update error: {e}")
            return False
    
    async def send_hourly_summary(self, coins: List[str], prices: dict) -> bool:
        """Send hourly market summary"""
        try:
            message = f"🔔 **HOURLY MARKET SUMMARY**\n\n"
            
            for coin in coins:
                if coin in prices:
                    price_data = prices[coin]
                    price = price_data.get('price', 0)
                    change_24h = price_data.get('change_24h', 0)
                    
                    emoji = "🟢" if change_24h > 0 else ("🔴" if change_24h < 0 else "⚪")
                    message += f"{emoji} **{coin}:** ${price:,.2f} ({change_24h:+.2f}%)\n"
            
            message += f"\n⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Hourly summary error: {e}")
            return False
    
    async def send_risk_alert(self, symbol: str, risk_type: str, severity: str, details: str) -> bool:
        """Send risk warning alert"""
        try:
            severity_emoji = {
                "CRITICAL": "🚨",
                "HIGH": "⚠️",
                "MEDIUM": "🔶",
                "LOW": "🟡"
            }
            
            emoji = severity_emoji.get(severity, "⚠️")
            
            message = f"{emoji} **RISK ALERT**\n\n"
            message += f"📊 Symbol: {symbol}\n"
            message += f"🚨 Type: {risk_type}\n"
            message += f"⚠️ Severity: {severity}\n\n"
            message += f"📝 {details}\n\n"
            message += f"⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Risk alert error: {e}")
            return False
    
    async def _send_message(self, text: str) -> bool:
        """Internal method to send Telegram message"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"Telegram API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def _is_rate_limited(self, key: str) -> bool:
        """Check if message is rate limited"""
        if key not in self.last_message_time:
            return False
        
        elapsed = (datetime.now(pytz.UTC) - self.last_message_time[key]).total_seconds()
        return elapsed < self.rate_limit_seconds

# Singleton instance
_telegram_ultra: Optional[TelegramUltra] = None

def get_telegram_ultra(token: str = None, chat_id: str = None) -> Optional[TelegramUltra]:
    """Get singleton TelegramUltra instance"""
    global _telegram_ultra
    
    if _telegram_ultra is None and token and chat_id:
        _telegram_ultra = TelegramUltra(token, chat_id)
    
    return _telegram_ultra
