#!/bin/bash
# Railway Startup Script
# DEMIR AI PRO v8.0

set -e  # Exit on error

echo "🚀 DEMIR AI PRO - Railway Startup"
echo "================================"

# Step 1: Run migrations
echo "📝 Running database migrations..."
python scripts/run_migrations.py

if [ $? -eq 0 ]; then
    echo "✅ Migrations completed successfully"
else
    echo "❌ Migration failed!"
    exit 1
fi

# Step 2: Start FastAPI server
echo ""
echo "🚀 Starting FastAPI server..."
echo "================================"
python main.py
