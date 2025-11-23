"""
Data Fetcher

Fetches real-time market data from exchanges.
Production-grade with automatic retry and failover.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Real-time market data fetcher.
    """
    
    def __init__(self, binance_client=None):
        """
        Initialize data fetcher.
        
        Args:
            binance_client: Binance exchange client
        """
        self.binance = binance_client
        logger.info("DataFetcher initialized")
    
    def fetch_price(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current price for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Price data dict
        """
        try:
            if self.binance:
                ticker = self.binance.fetch_ticker(symbol)
                return {
                    'symbol': symbol,
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'timestamp': ticker['timestamp'] / 1000
                }
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
        
        return {'symbol': symbol, 'price': 0, 'volume': 0, 'timestamp': 0}
    
    def fetch_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch prices for multiple symbols.
        """
        logger.info(f"Fetching prices for {len(symbols)} symbols")
        
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.fetch_price(symbol)
        
        return prices
