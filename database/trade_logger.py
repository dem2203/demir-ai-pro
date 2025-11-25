"""
Trade Logger - Database Persistence
===================================
Production-grade trade history logging to PostgreSQL.
- Async database writes
- Trade execution logging
- Performance metrics tracking
- Position history
- Zero mock/fallback

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json

from database.connection import DatabaseConnection

logger = logging.getLogger("database.trade_logger")


class TradeLogger:
    """Production-grade trade logging system"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self._ensure_tables()
        logger.info("TradeLogger initialized")
    
    def _ensure_tables(self):
        """Ensure all required tables exist"""
        # Trade history table
        create_trades_sql = """
        CREATE TABLE IF NOT EXISTS trade_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            order_type VARCHAR(20) NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            exit_price DECIMAL(20, 8),
            quantity DECIMAL(20, 8) NOT NULL,
            commission DECIMAL(20, 8) NOT NULL,
            pnl DECIMAL(20, 8),
            pnl_percent DECIMAL(10, 4),
            status VARCHAR(20) DEFAULT 'open',
            stop_loss DECIMAL(20, 8),
            take_profit DECIMAL(20, 8),
            signal_confidence DECIMAL(5, 4),
            regime VARCHAR(20),
            order_id VARCHAR(100),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_trade_history_status ON trade_history(status);
        CREATE INDEX IF NOT EXISTS idx_trade_history_pnl ON trade_history(pnl DESC NULLS LAST);
        """
        
        # Performance metrics table
        create_metrics_sql = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            symbol VARCHAR(20),
            value DECIMAL(20, 8) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_perf_metrics_type ON performance_metrics(metric_type);
        CREATE INDEX IF NOT EXISTS idx_perf_metrics_timestamp ON performance_metrics(timestamp DESC);
        """
        
        # Position snapshots table
        create_positions_sql = """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            entry_price DECIMAL(20, 8) NOT NULL,
            current_price DECIMAL(20, 8) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            unrealized_pnl DECIMAL(20, 8) NOT NULL,
            stop_loss DECIMAL(20, 8),
            take_profit DECIMAL(20, 8),
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_position_snapshots_timestamp ON position_snapshots(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_position_snapshots_symbol ON position_snapshots(symbol);
        """
        
        try:
            self.db.execute(create_trades_sql, commit=True)
            self.db.execute(create_metrics_sql, commit=True)
            self.db.execute(create_positions_sql, commit=True)
            logger.info("✅ Trade logging tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def log_trade_open(
        self,
        symbol: str,
        side: str,
        order_type: str,
        entry_price: float,
        quantity: float,
        commission: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_confidence: Optional[float] = None,
        regime: Optional[str] = None,
        order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log trade opening
        
        Returns:
            int: Trade ID
        """
        sql = """
        INSERT INTO trade_history (
            timestamp, symbol, side, order_type, entry_price, quantity,
            commission, status, stop_loss, take_profit, signal_confidence,
            regime, order_id, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
        """
        
        params = (
            datetime.now(),
            symbol,
            side,
            order_type,
            entry_price,
            quantity,
            commission,
            'open',
            stop_loss,
            take_profit,
            signal_confidence,
            regime,
            order_id,
            json.dumps(metadata) if metadata else None
        )
        
        try:
            result = await asyncio.to_thread(self.db.execute, sql, params, commit=True, fetch=True)
            trade_id = result[0]['id']
            logger.info(f"📝 Trade opened logged: ID={trade_id} | {symbol} {side} @ ${entry_price:,.2f}")
            return trade_id
        except Exception as e:
            logger.error(f"Failed to log trade open: {e}")
            raise
    
    async def log_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ):
        """
        Log trade closing
        
        Args:
            trade_id: Trade ID from log_trade_open
            exit_price: Exit price
            pnl: Profit/Loss amount
            pnl_percent: P&L percentage
        """
        sql = """
        UPDATE trade_history
        SET exit_price = %s,
            pnl = %s,
            pnl_percent = %s,
            status = 'closed',
            updated_at = NOW()
        WHERE id = %s;
        """
        
        params = (exit_price, pnl, pnl_percent, trade_id)
        
        try:
            await asyncio.to_thread(self.db.execute, sql, params, commit=True)
            logger.info(f"📝 Trade closed logged: ID={trade_id} | PNL=${pnl:,.2f} ({pnl_percent:+.2f}%)")
        except Exception as e:
            logger.error(f"Failed to log trade close: {e}")
            raise
    
    async def log_performance_metric(
        self,
        metric_type: str,
        value: float,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log performance metric
        
        Args:
            metric_type: Type of metric (win_rate, sharpe_ratio, total_pnl, etc.)
            value: Metric value
            symbol: Optional symbol
            metadata: Optional metadata
        """
        sql = """
        INSERT INTO performance_metrics (timestamp, metric_type, symbol, value, metadata)
        VALUES (%s, %s, %s, %s, %s);
        """
        
        params = (
            datetime.now(),
            metric_type,
            symbol,
            value,
            json.dumps(metadata) if metadata else None
        )
        
        try:
            await asyncio.to_thread(self.db.execute, sql, params, commit=True)
            logger.debug(f"📊 Metric logged: {metric_type}={value:.4f}")
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
    
    async def log_position_snapshot(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        unrealized_pnl: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Log position snapshot (for monitoring)
        
        Args:
            symbol: Trading pair
            side: LONG or SHORT
            entry_price: Entry price
            current_price: Current market price
            quantity: Position size
            unrealized_pnl: Unrealized P&L
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        sql = """
        INSERT INTO position_snapshots (
            timestamp, symbol, side, entry_price, current_price,
            quantity, unrealized_pnl, stop_loss, take_profit
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        params = (
            datetime.now(),
            symbol,
            side,
            entry_price,
            current_price,
            quantity,
            unrealized_pnl,
            stop_loss,
            take_profit
        )
        
        try:
            await asyncio.to_thread(self.db.execute, sql, params, commit=True)
        except Exception as e:
            logger.error(f"Failed to log position snapshot: {e}")
    
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Get trade history
        
        Args:
            symbol: Filter by symbol
            status: Filter by status (open/closed)
            limit: Max results
            
        Returns:
            List of trade records
        """
        sql = "SELECT * FROM trade_history WHERE 1=1"
        params = []
        
        if symbol:
            sql += " AND symbol = %s"
            params.append(symbol)
        
        if status:
            sql += " AND status = %s"
            params.append(status)
        
        sql += " ORDER BY timestamp DESC LIMIT %s;"
        params.append(limit)
        
        try:
            result = await asyncio.to_thread(self.db.execute, sql, tuple(params), fetch=True)
            return result
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics
        
        Returns:
            Dict with performance metrics
        """
        sql = """
        SELECT
            COUNT(*) as total_trades,
            COUNT(*) FILTER (WHERE status = 'closed' AND pnl > 0) as winning_trades,
            COUNT(*) FILTER (WHERE status = 'closed' AND pnl <= 0) as losing_trades,
            COALESCE(SUM(pnl) FILTER (WHERE status = 'closed'), 0) as total_pnl,
            COALESCE(AVG(pnl) FILTER (WHERE status = 'closed' AND pnl > 0), 0) as avg_win,
            COALESCE(AVG(pnl) FILTER (WHERE status = 'closed' AND pnl <= 0), 0) as avg_loss,
            COALESCE(SUM(commission), 0) as total_commission
        FROM trade_history;
        """
        
        try:
            result = await asyncio.to_thread(self.db.execute, sql, fetch=True)
            if result:
                data = result[0]
                winning = data['winning_trades'] or 0
                total_closed = winning + (data['losing_trades'] or 0)
                
                return {
                    'total_trades': data['total_trades'],
                    'winning_trades': winning,
                    'losing_trades': data['losing_trades'],
                    'win_rate': (winning / total_closed * 100) if total_closed > 0 else 0,
                    'total_pnl': float(data['total_pnl']),
                    'avg_win': float(data['avg_win']),
                    'avg_loss': float(data['avg_loss']),
                    'profit_factor': abs(winning * data['avg_win'] / ((data['losing_trades'] or 1) * data['avg_loss'])) if data['avg_loss'] != 0 else 0,
                    'total_commission': float(data['total_commission'])
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    async def close(self):
        """Cleanup database connection"""
        self.db.close()
        logger.info("TradeLogger closed")
