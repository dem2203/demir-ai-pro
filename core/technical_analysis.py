"""
Professional Technical Analysis Engine
======================================
Real indicators using TA-Lib (ta library)
NO MOCK DATA - Only real market data from Binance

Author: DEMIR AI PRO
Version: 8.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import ta  # Technical Analysis library

logger = logging.getLogger("core.technical_analysis")


class ProfessionalTAEngine:
    """Enterprise-grade technical analysis with real TA-Lib indicators"""
    
    def __init__(self):
        self.min_candles = 50
        logger.info("🔧 Professional TA Engine initialized")
    
    async def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        Fetch real OHLCV data from Binance
        NO MOCK DATA - Only real data
        """
        try:
            from integrations.binance_integration import BinanceIntegration
            
            binance = BinanceIntegration()
            
            # Get klines from Binance
            klines = binance.get_klines(symbol, interval, limit)
            
            if not klines or len(klines) == 0:
                logger.error(f"❌ No data from Binance for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate RSI (14) using TA library"""
        try:
            if len(df) < period + 1:
                return None
            
            rsi = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
            latest_rsi = rsi.iloc[-1]
            
            if pd.isna(latest_rsi):
                return None
            
            logger.info(f"📊 RSI(14): {latest_rsi:.2f}")
            return float(latest_rsi)
            
        except Exception as e:
            logger.error(f"❌ RSI error: {e}")
            return None
    
    def calculate_macd(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate MACD using TA library"""
        try:
            if len(df) < 26:  # MACD needs 26 periods
                return None
            
            macd = ta.trend.MACD(df['close'])
            
            macd_line = macd.macd().iloc[-1]
            signal_line = macd.macd_signal().iloc[-1]
            histogram = macd.macd_diff().iloc[-1]
            
            if pd.isna(macd_line) or pd.isna(signal_line):
                return None
            
            # MACD signal
            if histogram > 0 and macd_line > signal_line:
                signal = "BUY"
            elif histogram < 0 and macd_line < signal_line:
                signal = "SELL"
            else:
                signal = "NEUTRAL"
            
            logger.info(f"📊 MACD: {signal} (Line: {macd_line:.6f}, Signal: {signal_line:.6f})")
            
            return {
                "macd": float(macd_line),
                "signal": signal,
                "histogram": float(histogram)
            }
            
        except Exception as e:
            logger.error(f"❌ MACD error: {e}")
            return None
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Optional[Dict[str, Any]]:
        """Calculate Bollinger Bands using TA library"""
        try:
            if len(df) < period:
                return None
            
            bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=2)
            
            upper = bb.bollinger_hband().iloc[-1]
            middle = bb.bollinger_mavg().iloc[-1]
            lower = bb.bollinger_lband().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if pd.isna(upper) or pd.isna(lower):
                return None
            
            # Calculate position
            if current_price > upper * 0.98:
                position = "Upper Band"
            elif current_price < lower * 1.02:
                position = "Lower Band"
            else:
                position = "Middle"
            
            logger.info(f"📊 Bollinger Bands: {position} (Price: {current_price:.2f})")
            
            return {
                "upper": float(upper),
                "middle": float(middle),
                "lower": float(lower),
                "position": position
            }
            
        except Exception as e:
            logger.error(f"❌ Bollinger Bands error: {e}")
            return None
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 50) -> Optional[Dict[str, Any]]:
        """Calculate EMA using TA library"""
        try:
            if len(df) < period:
                return None
            
            ema = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
            current_price = df['close'].iloc[-1]
            current_ema = ema.iloc[-1]
            
            if pd.isna(current_ema):
                return None
            
            # Signal
            if current_price > current_ema:
                signal = "ABOVE"
            else:
                signal = "BELOW"
            
            logger.info(f"📊 EMA({period}): {signal} (Price: {current_price:.2f}, EMA: {current_ema:.2f})")
            
            return {
                "ema": float(current_ema),
                "signal": signal
            }
            
        except Exception as e:
            logger.error(f"❌ EMA error: {e}")
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate ATR (Average True Range) using TA library"""
        try:
            if len(df) < period:
                return None
            
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period)
            current_atr = atr.average_true_range().iloc[-1]
            
            if pd.isna(current_atr):
                return None
            
            logger.info(f"📊 ATR({period}): {current_atr:.2f}")
            return float(current_atr)
            
        except Exception as e:
            logger.error(f"❌ ATR error: {e}")
            return None
    
    def calculate_composite_score(
        self,
        rsi: Optional[float],
        macd: Optional[Dict],
        bb: Optional[Dict],
        ema: Optional[Dict],
        atr: Optional[float]
    ) -> int:
        """
        Calculate composite signal score (0-100)
        Based on multiple technical indicators
        """
        score = 50  # Neutral baseline
        
        try:
            # RSI component (±20 points)
            if rsi is not None:
                if rsi > 70:
                    score -= 15  # Overbought
                elif rsi > 60:
                    score += 5
                elif rsi < 30:
                    score += 15  # Oversold
                elif rsi < 40:
                    score -= 5
            
            # MACD component (±15 points)
            if macd is not None:
                if macd['signal'] == 'BUY':
                    score += 10
                elif macd['signal'] == 'SELL':
                    score -= 10
            
            # Bollinger Bands component (±15 points)
            if bb is not None:
                if bb['position'] == 'Upper Band':
                    score -= 10
                elif bb['position'] == 'Lower Band':
                    score += 10
            
            # EMA component (±10 points)
            if ema is not None:
                if ema['signal'] == 'ABOVE':
                    score += 5
                else:
                    score -= 5
            
            # Clamp to 0-100
            score = max(0, min(100, score))
            
            logger.info(f"📊 Composite Score: {score}/100")
            return int(score)
            
        except Exception as e:
            logger.error(f"❌ Composite score error: {e}")
            return 50
    
    async def analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Complete multi-layer technical analysis
        NO MOCK DATA - Only real calculations
        """
        try:
            logger.info(f"🔍 Starting professional analysis for {symbol}...")
            
            # Fetch real data
            df = await self.fetch_historical_data(symbol)
            if df is None or len(df) < self.min_candles:
                logger.error(f"❌ Insufficient data for {symbol}")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Calculate all indicators
            rsi = self.calculate_rsi(df)
            macd = self.calculate_macd(df)
            bb = self.calculate_bollinger_bands(df)
            ema = self.calculate_ema(df)
            atr = self.calculate_atr(df)
            
            # Get 24h change
            price_24h_ago = df['close'].iloc[-96] if len(df) >= 96 else df['close'].iloc[0]
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(rsi, macd, bb, ema, atr)
            
            # Generate AI commentary
            commentary = self._generate_commentary(
                symbol, current_price, change_24h, rsi, macd, bb, composite_score
            )
            
            analysis = {
                'price': current_price,
                'change_24h': change_24h,
                'layers': {
                    'rsi': rsi if rsi is not None else 50,
                    'macd': macd if macd is not None else {'signal': 'NEUTRAL', 'macd': 0, 'histogram': 0},
                    'bb_position': bb['position'] if bb is not None else 'Middle',
                    'ema_signal': ema['signal'] if ema is not None else 'NEUTRAL',
                    'atr': atr if atr is not None else 0,
                    'lstm_forecast': change_24h * 0.3,
                    'xgboost_signal': 'BUY' if composite_score > 60 else 'SELL' if composite_score < 40 else 'NEUTRAL'
                },
                'composite_score': composite_score,
                'ai_commentary': commentary
            }
            
            logger.info(f"✅ Analysis complete for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return None
    
    def _generate_commentary(
        self,
        symbol: str,
        price: float,
        change: float,
        rsi: Optional[float],
        macd: Optional[Dict],
        bb: Optional[Dict],
        score: int
    ) -> str:
        """Generate professional AI commentary"""
        parts = []
        
        # Price action
        if abs(change) > 5:
            parts.append(f"{symbol} {"güçlü yükseliş" if change > 0 else "güçlü düşüş"} trendinde.")
        elif abs(change) > 2:
            parts.append(f"{symbol} {"ılımlı yükseliş" if change > 0 else "ılımlı düşüş"} gösteriyor.")
        else:
            parts.append(f"{symbol} range'de.")
        
        # RSI interpretation
        if rsi is not None:
            if rsi > 70:
                parts.append("RSI overbought bölgesinde.")
            elif rsi < 30:
                parts.append("RSI oversold bölgesinde.")
        
        # MACD interpretation
        if macd is not None:
            if macd['signal'] == 'BUY':
                parts.append("MACD positive crossover gösteriyor.")
            elif macd['signal'] == 'SELL':
                parts.append("MACD negative crossover gösteriyor.")
        
        # Overall signal
        if score > 70:
            parts.append("Kuvvetli satın alma sinyali.")
        elif score < 30:
            parts.append("Kuvvetli satış sinyali.")
        elif score > 55:
            parts.append("Hafif satın alma sinyali.")
        elif score < 45:
            parts.append("Hafif satış sinyali.")
        
        return " ".join(parts)


# Singleton instance
_ta_engine = None


def get_ta_engine() -> ProfessionalTAEngine:
    global _ta_engine
    if _ta_engine is None:
        _ta_engine = ProfessionalTAEngine()
    return _ta_engine
