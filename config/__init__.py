"""
DEMIR AI PRO - Configuration Management

Production-grade configuration with environment validation.
Zero tolerance for missing critical parameters.
"""

from .settings import *
from .validation import validate_production_config

__all__ = [
    'VERSION',
    'APP_NAME',
    'ENVIRONMENT',
    'DATABASE_URL',
    'BINANCE_API_KEY',
    'BINANCE_API_SECRET',
    'TRACKED_SYMBOLS',
    'OPPORTUNITY_THRESHOLDS',
    'validate_production_config',
]
