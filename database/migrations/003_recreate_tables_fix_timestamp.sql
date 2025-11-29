-- Drop and recreate all tables with correct column names
-- This fixes the "timestamp does not exist" error by removing old schema

-- Drop existing tables (if they have old schema)
DROP TABLE IF EXISTS trade_history CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS position_snapshots CASCADE;

-- Recreate trade_history with trade_timestamp
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

-- Recreate performance_metrics with metric_timestamp
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

-- Recreate position_snapshots with snapshot_timestamp
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
