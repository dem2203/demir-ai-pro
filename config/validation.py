"""
Production Configuration Validation

Strictly validates all configuration parameters before application startup.
Zero tolerance for invalid or missing critical parameters.
"""

import sys
from typing import List, Tuple

def validate_production_config() -> Tuple[bool, List[str]]:
    """
    Validate production configuration.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    from . import settings
    
    errors = []
    warnings = []
    
    # ====================================================================
    # CRITICAL VALIDATIONS (Will halt startup)
    # ====================================================================
    
    # Database
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL must be set")
    elif not settings.DATABASE_URL.startswith(('postgresql://', 'postgres://')):
        errors.append("DATABASE_URL must be a valid PostgreSQL connection string")
    
    # Binance API
    if not settings.BINANCE_API_KEY:
        errors.append("BINANCE_API_KEY must be set")
    if not settings.BINANCE_API_SECRET:
        errors.append("BINANCE_API_SECRET must be set")
    
    # Environment
    if settings.ENVIRONMENT not in ['production', 'development']:
        errors.append(f"Invalid ENVIRONMENT: {settings.ENVIRONMENT}")
    
    # Tracked symbols
    if not settings.TRACKED_SYMBOLS:
        errors.append("No symbols tracked - at least one symbol required")
    
    # Risk parameters validation
    if settings.MAX_POSITION_SIZE <= 0 or settings.MAX_POSITION_SIZE > 1:
        errors.append(f"Invalid MAX_POSITION_SIZE: {settings.MAX_POSITION_SIZE} (must be 0-1)")
    
    if settings.MAX_LEVERAGE < 1 or settings.MAX_LEVERAGE > 125:
        errors.append(f"Invalid MAX_LEVERAGE: {settings.MAX_LEVERAGE} (must be 1-125)")
    
    # Opportunity thresholds
    thresholds = settings.OPPORTUNITY_THRESHOLDS
    if thresholds['min_confidence'] < 0 or thresholds['min_confidence'] > 1:
        errors.append("min_confidence must be between 0 and 1")
    
    if thresholds['min_risk_reward'] < 1:
        errors.append("min_risk_reward must be >= 1.0")
    
    # ====================================================================
    # WARNING VALIDATIONS (Non-blocking, but logged)
    # ====================================================================
    
    # Redis
    if not settings.REDIS_ENABLED:
        warnings.append("Redis not configured - in-memory cache will be used (not recommended for production)")
    
    # Telegram
    if not settings.TELEGRAM_ENABLED:
        warnings.append("Telegram notifications disabled (optional feature)")
    
    # Advanced data sources
    if not settings.COINGLASS_API_KEY:
        warnings.append("COINGLASS_API_KEY not set - funding rate data will be limited")
    
    if not settings.COINMARKETCAP_API_KEY:
        warnings.append("COINMARKETCAP_API_KEY not set - market cap rankings will be limited")
    
    if not settings.GLASSNODE_API_KEY:
        warnings.append("GLASSNODE_API_KEY not set - on-chain analytics disabled")
    
    # Advisory mode check
    if not settings.ADVISORY_MODE and not settings.DRY_RUN:
        warnings.append("⚠️  LIVE TRADING ENABLED - Real money at risk!")
    
    # ====================================================================
    # DISPLAY RESULTS
    # ====================================================================
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   • {error}")
        print("\n❌ Startup aborted due to configuration errors.\n")
        return False, errors
    
    print("\n✅ Configuration validation passed\n")
    return True, []

def validate_or_exit():
    """
    Validate configuration and exit if invalid.
    """
    is_valid, errors = validate_production_config()
    if not is_valid:
        sys.exit(1)

if __name__ == "__main__":
    # Run validation when executed directly
    validate_or_exit()
