-- Fix trades table - rename 'timestamp' to 'trade_timestamp' (timestamp is reserved keyword)
-- This migration fixes the "column timestamp does not exist" error

-- First, check if the trade_history table exists and has old schema
DO $$
BEGIN
    -- If column 'timestamp' exists (old schema), rename it
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'trade_history' 
        AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE trade_history RENAME COLUMN timestamp TO trade_timestamp;
    END IF;
    
    -- If column doesn't exist yet, table will be created correctly by TradeLogger
END $$;

-- Same fix for position_snapshots
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'position_snapshots' 
        AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE position_snapshots RENAME COLUMN timestamp TO trade_timestamp;
    END IF;
END $$;

-- Same fix for performance_metrics  
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'performance_metrics' 
        AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE performance_metrics RENAME COLUMN timestamp TO metric_timestamp;
    END IF;
END $$;
