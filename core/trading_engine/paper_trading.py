"""
Paper Trading Engine
===================
Full paper trading simulation with realistic execution.
- Virtual balance tracking
- Trade history logging
- Performance metrics calculation
- Integration with OrderRouter
- Zero mock data - production-ready structure

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass, field
import pandas as pd

from core.trading_engine.order_router import OrderRouter, OrderResult, Position
from integrations.notifications.telegram_alert import send_telegram_alert

logger = logging.getLogger("trading_engine.paper_trading")


@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    size: float
    price: float
    commission: float
    pnl: float = 0.0
    is_winning_trade: bool = False


@dataclass
class PaperTradingStats:
    """Performance statistics"""
    initial_balance: float
    current_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commission: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)


class PaperTradingEngine:
    """Production-grade paper trading engine"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        self.order_router = OrderRouter(mode="PAPER")
        self.trade_history: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"Paper trading initialized with ${initial_balance:,.2f}")
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> OrderResult:
        """
        Execute paper trade and update balance
        
        Args:
            symbol: Trading pair
            side: BUY or SELL
            size: Position size
            price: Current market price
            stop_loss: Optional stop loss
            take_profit: Optional take profit
            
        Returns:
            OrderResult
        """
        # Execute order through router
        result = await self.order_router.execute_order(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if not result.success:
            logger.error(f"Paper trade failed: {result.error_message}")
            return result
        
        # Update balance
        trade_value = result.filled_size * result.filled_price
        
        if side == "BUY":
            self.current_balance -= (trade_value + result.commission)
        else:
            # Calculate P&L for SELL
            position = self.order_router.get_position(symbol)
            if position:
                pnl = position.realized_pnl
                self.current_balance += (trade_value - result.commission + pnl)
                
                # Record trade
                self._record_trade(result, symbol, side, pnl)
        
        # Update peak for drawdown calculation
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Send Telegram alert
        await self._send_trade_alert(result, symbol, side)
        
        logger.info(f"Balance: ${self.current_balance:,.2f} (PNL: ${self.current_balance - self.initial_balance:,.2f})")
        
        return result
    
    def _record_trade(self, result: OrderResult, symbol: str, side: str, pnl: float):
        """Record completed trade"""
        trade = TradeRecord(
            timestamp=result.timestamp,
            symbol=symbol,
            side=side,
            size=result.filled_size,
            price=result.filled_price,
            commission=result.commission,
            pnl=pnl,
            is_winning_trade=(pnl > 0)
        )
        
        self.trade_history.append(trade)
        logger.info(f"Trade recorded: {side} {symbol} | PNL: ${pnl:,.2f}")
    
    async def _send_trade_alert(self, result: OrderResult, symbol: str, side: str):
        """Send Telegram notification for trade"""
        message = f"📊 *Paper Trade Executed*\n"
        message += f"Symbol: `{symbol}`\n"
        message += f"Side: `{side}`\n"
        message += f"Size: `{result.filled_size:.6f}`\n"
        message += f"Price: `${result.filled_price:,.2f}`\n"
        message += f"Commission: `${result.commission:.2f}`\n"
        message += f"Balance: `${self.current_balance:,.2f}`"
        
        await send_telegram_alert(message)
    
    def calculate_statistics(self) -> PaperTradingStats:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return PaperTradingStats(
                initial_balance=self.initial_balance,
                current_balance=self.current_balance
            )
        
        # Basic metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.is_winning_trade)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t.pnl for t in self.trade_history)
        total_commission = sum(t.commission for t in self.trade_history)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # Win/Loss averages
        winning_pnls = [t.pnl for t in self.trade_history if t.is_winning_trade]
        losing_pnls = [t.pnl for t in self.trade_history if not t.is_winning_trade]
        
        average_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        average_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0
        
        # Profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0.0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 0.0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0
        
        # Max drawdown
        max_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
        
        # Sharpe ratio (simplified)
        if len(self.daily_returns) > 1:
            returns_std = pd.Series(self.daily_returns).std()
            avg_return = pd.Series(self.daily_returns).mean()
            sharpe_ratio = (avg_return / returns_std * (252 ** 0.5)) if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        return PaperTradingStats(
            initial_balance=self.initial_balance,
            current_balance=self.current_balance,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            total_commission=total_commission,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trade_history
        )
    
    async def send_daily_report(self):
        """Send daily performance report via Telegram"""
        stats = self.calculate_statistics()
        
        message = "📊 *Daily Paper Trading Report*\n\n"
        message += f"Balance: `${stats.current_balance:,.2f}`\n"
        message += f"P&L: `${stats.total_pnl:,.2f}` ({(stats.total_pnl/stats.initial_balance*100):+.2f}%)\n\n"
        message += f"Total Trades: `{stats.total_trades}`\n"
        message += f"Win Rate: `{stats.win_rate:.1f}%`\n"
        message += f"Profit Factor: `{stats.profit_factor:.2f}`\n"
        message += f"Max Drawdown: `{stats.max_drawdown:.2f}%`\n\n"
        message += f"Avg Win: `${stats.average_win:,.2f}`\n"
        message += f"Avg Loss: `${stats.average_loss:,.2f}`\n"
        message += f"Sharpe Ratio: `{stats.sharpe_ratio:.2f}`"
        
        await send_telegram_alert(message)
        
        logger.info("Daily report sent")
    
    def export_trades_to_csv(self, filepath: str = "paper_trades.csv"):
        """Export trade history to CSV"""
        if not self.trade_history:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'side': t.side,
                'size': t.size,
                'price': t.price,
                'commission': t.commission,
                'pnl': t.pnl,
                'is_winning': t.is_winning_trade
            }
            for t in self.trade_history
        ])
        
        df.to_csv(filepath, index=False)
        logger.info(f"Trades exported to {filepath}")
    
    async def close(self):
        """Cleanup and final report"""
        await self.send_daily_report()
        self.export_trades_to_csv()
        await self.order_router.close()
        logger.info("Paper trading engine closed")
