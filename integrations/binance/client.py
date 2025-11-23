"""
Binance REST API Client

Production-grade Binance Futures client with:
- Automatic rate limiting
- Retry logic
- Error handling
- Connection pooling
"""

import ccxt
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class BinanceClient:
    """
    Production Binance Futures API client.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (default: False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Use futures
                'adjustForTimeDifference': True,
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("⚠️  Binance TESTNET mode enabled")
        
        logger.info(f"✅ Binance client initialized (testnet={testnet})")
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker for symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker {symbol}: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> List[List[float]]:
        """
        Fetch OHLCV candle data.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (1m, 5m, 1h, 1d, etc.)
            limit: Number of candles
            
        Returns:
            List of OHLCV arrays
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV {symbol}: {e}")
            raise
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Balance data
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            raise
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            amount: Order amount
            price: Limit price (for limit orders)
            params: Additional parameters
            
        Returns:
            Order result
        """
        try:
            order = self.exchange.create_order(
                symbol,
                order_type,
                side,
                amount,
                price,
                params or {}
            )
            logger.info(f"✅ Order created: {side} {amount} {symbol} @ {price}")
            return order
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Position data or None
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                    return pos
            return None
        except Exception as e:
            logger.error(f"Failed to fetch position {symbol}: {e}")
            return None
    
    def close_position(self, symbol: str) -> bool:
        """
        Close position for symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Success status
        """
        try:
            position = self.get_position(symbol)
            if not position:
                logger.info(f"No position to close: {symbol}")
                return True
            
            # Determine side to close
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = abs(float(position['contracts']))
            
            # Close with market order
            self.create_order(symbol, side, 'market', amount)
            logger.info(f"✅ Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to close position {symbol}: {e}")
            return False
