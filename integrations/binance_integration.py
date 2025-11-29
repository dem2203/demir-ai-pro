"""
Binance Integration Wrapper
===========================
Simple wrapper around Binance Client for trading engine use.

Author: DEMIR AI PRO
Version: 8.0
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("integrations.binance_integration")


class BinanceIntegration:
    """
    Simple Binance integration wrapper
    Uses existing BinanceClient if available
    """
    
    def __init__(self):
        """Initialize Binance integration"""
        self.client = None
        
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
    
    def _fetch_ticker_direct(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Direct API call using requests
        """
        try:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=5)
            
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
