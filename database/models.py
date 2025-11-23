"""
Database Models

SQLAlchemy models for database tables.
Production-grade schema with constraints and indices.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import json

class Signal:
    """
    Signal model - AI trading signals.
    """
    
    @staticmethod
    def create_table_sql() -> str:
        return """
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            signal_group VARCHAR(50) NOT NULL,
            direction VARCHAR(10) NOT NULL,
            strength FLOAT NOT NULL,
            confidence FLOAT NOT NULL,
            entry_price FLOAT,
            take_profit_1 FLOAT,
            take_profit_2 FLOAT,
            stop_loss FLOAT,
            metadata JSONB,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_signals_active ON signals(active);
        """

class Trade:
    """
    Trade model - executed trades.
    """
    
    @staticmethod
    def create_table_sql() -> str:
        return """
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            entry_price FLOAT NOT NULL,
            exit_price FLOAT,
            quantity FLOAT NOT NULL,
            pnl FLOAT,
            pnl_percent FLOAT,
            status VARCHAR(20) DEFAULT 'open',
            signal_id INTEGER REFERENCES signals(id),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
        """

class UserSetting:
    """
    User settings model - persistent configuration.
    """
    
    @staticmethod
    def create_table_sql() -> str:
        return """
        CREATE TABLE IF NOT EXISTS user_settings (
            id SERIAL PRIMARY KEY,
            setting_key VARCHAR(100) UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_settings_key ON user_settings(setting_key);
        """

class PerformanceMetric:
    """
    Performance metrics model - system monitoring.
    """
    
    @staticmethod
    def create_table_sql() -> str:
        return """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp DESC);
        """

def create_all_tables(db_connection):
    """
    Create all database tables.
    
    Args:
        db_connection: DatabaseConnection instance
    """
    models = [Signal, Trade, UserSetting, PerformanceMetric]
    
    for model in models:
        try:
            db_connection.execute(model.create_table_sql(), commit=True)
            print(f"✅ Created table: {model.__name__}")
        except Exception as e:
            print(f"❌ Failed to create table {model.__name__}: {e}")
            raise
