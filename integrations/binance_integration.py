"""
Binance Integration Wrapper
===========================
Simple wrapper around Binance Client for trading engine use.
Provides ticker data and OHLCV klines for technical analysis.

Author: DEMIR AI PRO
Version: 8.0
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("integrations.binance_integration")


class BinanceIntegration:
    """
    Simple Binance integration wrapper
    Uses existing BinanceClient if available
    """
    
    def __init__(self):
        """Initialize Binance integration"""
        self.client = None
        self.base_url = "https://api.binance.com"
        
        try:
            from .binance.client import BinanceClient
            self.client = BinanceClient()
            logger.info("✅ Binance client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Binance client not available: {e}")
            # Try direct API approach
            try:
                from config import BINANCE_API_KEY, BINANCE_API_SECRET
                if BINANCE_API_KEY and BINANCE_API_SECRET:
                    # Use requests library for simple API calls
                    self.api_key = BINANCE_API_KEY
                    self.api_secret = BINANCE_API_SECRET
                    logger.info("✅ Binance API keys loaded (direct mode)")
            except Exception as e2:
                logger.error(f"❌ Binance initialization failed: {e2}")
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker/price data for symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Dict with ticker data or None
        """
        try:
            if self.client:
                return self.client.get_ticker(symbol)
            else:
                # Fallback to direct API call
                return self._fetch_ticker_direct(symbol)
        except Exception as e:
            logger.error(f"❌ Ticker fetch error for {symbol}: {e}")
            return None
    
    def get_klines(
        self, 
        symbol: str, 
        interval: str = "15m", 
        limit: int = 500
    ) -> Optional[List[List]]:
        """
        Get historical klines/candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Number of klines to fetch (max 1000, default 500)
            
        Returns:
            List of klines data or None
            Each kline: [
                Open time,
                Open,
                High,
                Low,
                Close,
                Volume,
                Close time,
                Quote asset volume,
                Number of trades,
                Taker buy base asset volume,
                Taker buy quote asset volume,
                Ignore
            ]
        """
        try:
            if self.client and hasattr(self.client, 'get_klines'):
                return self.client.get_klines(symbol, interval, limit)
            else:
                # Direct API call
                return self._fetch_klines_direct(symbol, interval, limit)
        except Exception as e:
            logger.error(f"❌ Klines fetch error for {symbol}: {e}")
            return None
    
    def _fetch_ticker_direct(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Direct API call using requests
        """
        try:
            import requests
            url = f"{self.base_url}/api/v3/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"✅ Fetched ticker for {symbol}")
                return data
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Direct ticker fetch failed: {e}")
            return None
    
    def _fetch_klines_direct(
        self, 
        symbol: str, 
        interval: str, 
        limit: int
    ) -> Optional[List[List]]:
        """
        Direct klines API call using requests
        
        Args:
            symbol: Trading pair
            interval: Timeframe interval
            limit: Number of candles
            
        Returns:
            List of klines or None
        """
        try:
            import requests
            
            # Binance klines endpoint
            url = f"{self.base_url}/api/v3/klines"
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Max 1000 per request
            }
            
            logger.info(f"🔍 Fetching {limit} klines for {symbol} ({interval})...")
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                klines = response.json()
                logger.info(f"✅ Fetched {len(klines)} klines for {symbol}")
                return klines
            else:
                logger.error(f"❌ Binance API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Direct klines fetch failed for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
