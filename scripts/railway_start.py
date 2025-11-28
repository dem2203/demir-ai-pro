#!/usr/bin/env python3
"""
Railway Startup Script (Python)
DEMIR AI PRO v8.0

Runs migrations then starts main.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"\n✅ {description} - SUCCESS\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED with code {e.returncode}\n")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("🚀 DEMIR AI PRO - Railway Startup (Python)")
    print("="*60 + "\n")
    
    # Step 1: Run migrations
    run_command(
        ["python", "scripts/run_migrations.py"],
        "📝 Step 1: Running Database Migrations"
    )
    
    # Step 2: Start main.py
    print("\n" + "="*60)
    print("🚀 Step 2: Starting FastAPI Server")
    print("="*60 + "\n")
    
    # Use exec to replace process (important for Railway)
    os.execvp("python", ["python", "main.py"])

if __name__ == "__main__":
    main()
