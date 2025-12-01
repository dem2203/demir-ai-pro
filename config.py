#!/usr/bin/env python3
"""DEMIR AI PRO v10.1 - Configuration"""

import os
from typing import List
import logging

logger = logging.getLogger(__name__)

# Application Info
VERSION = "10.1"
APP_NAME = "DEMIR AI PRO ULTRA"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Database (optional)
DATABASE_URL = os.getenv("DATABASE_URL", "")
DATABASE_PUBLIC_URL = os.getenv("DATABASE_PUBLIC_URL", "")

# Exchange APIs
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Trading Configuration
ADVISORY_MODE = os.getenv("ADVISORY_MODE", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

def validate_or_exit() -> None:
    """Validate critical configuration"""
    warnings = []
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        warnings.append("Binance API credentials missing")
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        warnings.append("Telegram credentials missing - notifications disabled")
    
    if warnings:
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
    
    logger.info(f"Configuration validated", version=VERSION, environment=ENVIRONMENT, advisory_mode=ADVISORY_MODE)
