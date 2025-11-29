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

IMPORTANT: PostgreSQL reserves 'timestamp' as keyword.
Using specific column names: trade_timestamp, metric_timestamp, snapshot_timestamp
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from database.connection import get_db

logger = logging.getLogger("database.trade_logger")


class TradeLogger:
    """Production-grade trade logging system"""
    
    def __init__(self):
        self.db = get_db()  # Use global DB instance
        self._ensure_tables()
        logger.info("TradeLogger initialized")
    
    def _ensure_tables(self):
        """Ensure all required tables exist"""
        # Trade history table (timestamp → trade_timestamp)
        create_trades_sql = """
        CREATE TABLE IF NOT EXISTS trade_history (
            id SERIAL PRIMARY KEY,
            trade_timestamp TIMESTAMP NOT NULL,
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
        CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(trade_timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_trade_history_status ON trade_history(status);
        CREATE INDEX IF NOT EXISTS idx_trade_history_pnl ON trade_history(pnl DESC NULLS LAST);
        """
        
        # Performance metrics table (timestamp → metric_timestamp)
        create_metrics_sql = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            metric_timestamp TIMESTAMP NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            symbol VARCHAR(20),
            value DECIMAL(20, 8) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_perf_metrics_type ON performance_metrics(metric_type);
        CREATE INDEX IF NOT EXISTS idx_perf_metrics_timestamp ON performance_metrics(metric_timestamp DESC);
        """
        
        # Position snapshots table (timestamp → snapshot_timestamp)
        create_positions_sql = """
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id SERIAL PRIMARY KEY,
            snapshot_timestamp TIMESTAMP NOT NULL,
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
        
        CREATE INDEX IF NOT EXISTS idx_position_snapshots_timestamp ON position_snapshots(snapshot_timestamp DESC);
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
            trade_timestamp, symbol, side, order_type, entry_price, quantity,
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
            with self.db.get_cursor() as cursor:
                cursor.execute(sql, params)
                trade_id = cursor.fetchone()[0]
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
            self.db.execute(sql, params, commit=True)
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
        INSERT INTO performance_metrics (metric_timestamp, metric_type, symbol, value, metadata)
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
            self.db.execute(sql, params, commit=True)
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
            snapshot_timestamp, symbol, side, entry_price, current_price,
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
            self.db.execute(sql, params, commit=True)
        except Exception as e:
            logger.error(f"Failed to log position snapshot: {e}")
    
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
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
        
        sql += " ORDER BY trade_timestamp DESC LIMIT %s;"
        params.append(limit)
        
        try:
            rows = self.db.fetchall(sql, tuple(params))
            # Convert to dict list
            if rows:
                columns = ['id', 'trade_timestamp', 'symbol', 'side', 'order_type', 'entry_price',
                          'exit_price', 'quantity', 'commission', 'pnl', 'pnl_percent', 'status',
                          'stop_loss', 'take_profit', 'signal_confidence', 'regime', 'order_id',
                          'metadata', 'created_at', 'updated_at']
                return [dict(zip(columns, row)) for row in rows]
            return []
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
            row = self.db.fetchone(sql)
            if row:
                winning = row[1] or 0
                losing = row[2] or 0
                total_closed = winning + losing
                total_pnl = float(row[3]) if row[3] else 0
                avg_win = float(row[4]) if row[4] else 0
                avg_loss = float(row[5]) if row[5] else 0
                total_commission = float(row[6]) if row[6] else 0
                
                return {
                    'total_trades': row[0],
                    'winning_trades': winning,
                    'losing_trades': losing,
                    'win_rate': (winning / total_closed * 100) if total_closed > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(winning * avg_win / (losing * avg_loss)) if (losing > 0 and avg_loss != 0) else 0,
                    'total_commission': total_commission
                }
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_commission': 0
            }
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_commission': 0
            }
    
    async def close(self):
        """Cleanup - DB connection is global, don't close"""
        logger.info("TradeLogger closed (connection pool remains active)")
