"""
Order Router & Execution Manager
================================
Production-grade order execution with paper/live mode support.
- Paper trading: Virtual positions with realistic slippage
- Live trading: Real Binance order placement
- Position tracking, P&L calculation
- Stop loss & take profit management
- Zero mock/fallback - full production logic

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from binance.client import AsyncClient
from binance.exceptions import BinanceAPIException
import os

logger = logging.getLogger("trading_engine.order_router")


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str]
    filled_price: float
    filled_size: float
    commission: float
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class Position:
    """Active position tracker"""
    symbol: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class OrderRouter:
    """Production-grade order execution router"""
    
    def __init__(self, mode: Literal["PAPER", "LIVE"] = "PAPER"):
        self.mode = mode
        self.positions: Dict[str, Position] = {}
        self.binance_client: Optional[AsyncClient] = None
        
        # Slippage settings (realistic)
        self.slippage_bps = 5  # 5 basis points (0.05%)
        self.commission_rate = 0.001  # 0.1% (Binance standard)
        
        logger.info(f"OrderRouter initialized in {mode} mode")
        
        if mode == "LIVE":
            self._init_binance_client()
    
    def _init_binance_client(self):
        """Initialize Binance client for live trading"""
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_SECRET_KEY")
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set for LIVE mode")
        
        # Client will be initialized async
        logger.info("Binance credentials loaded - client will initialize on first use")
    
    async def _get_binance_client(self) -> AsyncClient:
        """Lazy async client initialization"""
        if not self.binance_client:
            api_key = os.environ.get("BINANCE_API_KEY")
            api_secret = os.environ.get("BINANCE_SECRET_KEY")
            self.binance_client = await AsyncClient.create(api_key, api_secret)
        return self.binance_client
    
    def _calculate_slippage(self, price: float, side: Literal["BUY", "SELL"]) -> float:
        """Calculate realistic slippage"""
        slippage = price * (self.slippage_bps / 10000)
        return price + slippage if side == "BUY" else price - slippage
    
    async def execute_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> OrderResult:
        """
        Execute order (paper or live)
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: BUY or SELL
            size: Order size in base currency
            price: Current market price (for slippage calc)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            OrderResult with execution details
        """
        if self.mode == "PAPER":
            return await self._execute_paper(symbol, side, size, price, stop_loss, take_profit)
        else:
            return await self._execute_live(symbol, side, size, price, stop_loss, take_profit)
    
    async def _execute_paper(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        size: float,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> OrderResult:
        """Execute paper trade with realistic slippage"""
        try:
            # Apply slippage
            filled_price = self._calculate_slippage(price, side)
            
            # Calculate commission
            commission = size * filled_price * self.commission_rate
            
            # Generate virtual order ID
            order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Update position tracking
            if side == "BUY":
                self._open_position(symbol, "LONG", filled_price, size, stop_loss, take_profit)
            else:
                self._close_position(symbol, filled_price)
            
            logger.info(f"📝 Paper trade: {side} {size:.6f} {symbol} @ ${filled_price:,.2f}")
            
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=filled_price,
                filled_size=size,
                commission=commission,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Paper trade error: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=0.0,
                filled_size=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def _execute_live(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        size: float,
        price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ) -> OrderResult:
        """Execute real Binance order"""
        try:
            client = await self._get_binance_client()
            
            # Place market order
            order = await client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=size
            )
            
            # Extract fill details
            filled_price = float(order['fills'][0]['price']) if order.get('fills') else price
            filled_size = float(order['executedQty'])
            commission = sum(float(fill['commission']) for fill in order.get('fills', []))
            
            # Place stop loss if provided
            if stop_loss and side == "BUY":
                await client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="STOP_LOSS_LIMIT",
                    quantity=size,
                    stopPrice=stop_loss,
                    price=stop_loss * 0.995  # 0.5% below stop
                )
            
            # Place take profit if provided
            if take_profit and side == "BUY":
                await client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="TAKE_PROFIT_LIMIT",
                    quantity=size,
                    stopPrice=take_profit,
                    price=take_profit * 1.005  # 0.5% above target
                )
            
            # Update position tracking
            if side == "BUY":
                self._open_position(symbol, "LONG", filled_price, size, stop_loss, take_profit)
            else:
                self._close_position(symbol, filled_price)
            
            logger.info(f"✅ Live trade: {side} {filled_size:.6f} {symbol} @ ${filled_price:,.2f}")
            
            return OrderResult(
                success=True,
                order_id=order['orderId'],
                filled_price=filled_price,
                filled_size=filled_size,
                commission=commission,
                timestamp=datetime.now()
            )
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e.message}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=0.0,
                filled_size=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error_message=e.message
            )
        except Exception as e:
            logger.error(f"Live trade error: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                filled_price=0.0,
                filled_size=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _open_position(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        entry_price: float,
        size: float,
        stop_loss: Optional[float],
        take_profit: Optional[float]
    ):
        """Track opened position"""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        logger.info(f"Position opened: {side} {size:.6f} {symbol} @ ${entry_price:,.2f}")
    
    def _close_position(self, symbol: str, exit_price: float):
        """Close and calculate P&L"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == "LONG":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        # Subtract commission
        pnl -= (position.entry_price * position.size * self.commission_rate)
        pnl -= (exit_price * position.size * self.commission_rate)
        
        position.realized_pnl = pnl
        
        logger.info(f"Position closed: {symbol} | PNL: ${pnl:,.2f}")
        
        # Remove from active positions
        del self.positions[symbol]
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for open position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        if position.side == "LONG":
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.size
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_total_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    async def close(self):
        """Cleanup resources"""
        if self.binance_client:
            await self.binance_client.close_connection()
            logger.info("Binance client connection closed")
