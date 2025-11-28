-- Fix Signal table schema
ALTER TABLE signals ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT true;
