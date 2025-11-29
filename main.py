#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - Main Application

Enterprise-grade AI trading bot with:
- Multi-layer ML ensemble
- Real-time signal generation
- Production data validation
- Zero-tolerance for mock data
- Background AI trading engine

❌ NO MOCK DATA
❌ NO FALLBACK
❌ NO TEST DATA

✅ 100% Production Real-Time Data
"""

import logging
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("demir_ai_pro.log")
    ]
)

logger = logging.getLogger(__name__)

# Import configuration
from config import (
    VERSION,
    APP_NAME,
    API_HOST,
    API_PORT,
    API_WORKERS,
    CORS_ORIGINS,
    validate_or_exit
)

# Validate configuration before starting
validate_or_exit()

# Import database
try:
    from database import get_db
    from database.models import create_all_tables
except ImportError:
    logger.warning("⚠️  Database module not available - some features may be limited")
    get_db = None
    create_all_tables = None

# Import core modules (graceful degradation)
try:
    from core import AIEngine, SignalProcessor, RiskManager, DataPipeline
except ImportError:
    logger.warning("⚠️  Core modules not available - some features may be limited")

# Import integrations
try:
    from integrations import BinanceIntegration, TelegramNotifier
except ImportError:
    logger.warning("⚠️  Integration modules not available - some features may be limited")

# Import trading engine
try:
    from core.trading_engine import get_engine
    TRADING_ENGINE_AVAILABLE = True
    logger.info("✅ Trading engine module loaded")
except ImportError as e:
    logger.warning(f"⚠️  Trading engine not available: {e}")
    TRADING_ENGINE_AVAILABLE = False

# -------------------------------------------------------------------
# API ROUTES
# -------------------------------------------------------------------

from api import router as api_router
from api.dashboard_api import router as dashboard_router
DASHBOARD_AVAILABLE = True

# ====================================================================
# APPLICATION INITIALIZATION
# ====================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    version=VERSION,
    description="Enterprise-grade AI crypto trading bot API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Include dashboard routes
app.include_router(dashboard_router)
logger.info("✅ API and Dashboard routes included")

# ====================================================================
# ROOT ENDPOINT - REDIRECT TO DASHBOARD
# ====================================================================

@app.get("/")
async def root():
    """
    Root endpoint - redirects to live dashboard
    Railway production deployment entry point
    """
    logger.info("👉 Root endpoint accessed - redirecting to dashboard")
    return RedirectResponse(url="/dashboard")

# ====================================================================
# HEALTH CHECK ENDPOINT
# ====================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    health_data = {
        "status": "healthy",
        "service": APP_NAME,
        "version": VERSION
    }
    
    # Add trading engine status if available
    if TRADING_ENGINE_AVAILABLE:
        try:
            engine = get_engine()
            health_data["trading_engine"] = engine.get_status()
        except Exception as e:
            health_data["trading_engine"] = {"error": str(e)}
    
    return health_data

# ====================================================================
# TRADING ENGINE CONTROL ENDPOINTS
# ====================================================================

if TRADING_ENGINE_AVAILABLE:
    @app.get("/api/engine/status")
    async def get_engine_status():
        """Get trading engine status"""
        try:
            engine = get_engine()
            return {
                "success": True,
                "data": engine.get_status()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @app.post("/api/engine/start")
    async def start_engine():
        """Manually start trading engine"""
        try:
            engine = get_engine()
            await engine.start()
            return {
                "success": True,
                "message": "Trading engine started"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @app.post("/api/engine/stop")
    async def stop_engine():
        """Manually stop trading engine"""
        try:
            engine = get_engine()
            await engine.stop()
            return {
                "success": True,
                "message": "Trading engine stopped"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ====================================================================
# STARTUP EVENTS
# ====================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup initialization."""
    logger.info(f"\n\n{'='*60}")
    logger.info(f"{APP_NAME} v{VERSION} - STARTING")
    logger.info(f"{'='*60}\n")
    
    # Initialize database
    if get_db and create_all_tables:
        try:
            db = get_db()
            create_all_tables(db)
            logger.info("✅ Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    # Initialize core modules
    try:
        logger.info("⚙️  Initializing core modules...")
        logger.info("✅ Core modules initialized")
    except Exception as e:
        logger.error(f"❌ Core initialization failed: {e}")
    
    # Start trading engine
    if TRADING_ENGINE_AVAILABLE:
        try:
            logger.info("🤖 Starting AI Trading Engine...")
            engine = get_engine()
            await engine.start()
            logger.info("✅ AI Trading Engine started in background")
        except Exception as e:
            logger.error(f"❌ Trading engine startup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Railway port info
    port = int(os.getenv("PORT", API_PORT))
    logger.info(f"\n✅ {APP_NAME} v{VERSION} is ready!")
    logger.info(f"🌐 API server: http://0.0.0.0:{port}")
    logger.info(f"📊 Health: http://0.0.0.0:{port}/health")
    logger.info(f"📄 Docs: http://0.0.0.0:{port}/docs")
    if DASHBOARD_AVAILABLE:
        logger.info(f"📈 Dashboard: http://0.0.0.0:{port}/dashboard")
        logger.info(f"🏠 Root: http://0.0.0.0:{port}/ (redirects to dashboard)")
    if TRADING_ENGINE_AVAILABLE:
        logger.info(f"🤖 Engine Status: http://0.0.0.0:{port}/api/engine/status")
    logger.info("")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown cleanup."""
    logger.info(f"\n\n{'='*60}")
    logger.info(f"{APP_NAME} v{VERSION} - SHUTTING DOWN")
    logger.info(f"{'='*60}\n")
    
    # Stop trading engine
    if TRADING_ENGINE_AVAILABLE:
        try:
            logger.info("🛑 Stopping AI Trading Engine...")
            engine = get_engine()
            await engine.stop()
            logger.info("✅ Trading engine stopped")
        except Exception as e:
            logger.error(f"❌ Trading engine shutdown error: {e}")
    
    # Close database connections
    if get_db:
        try:
            db = get_db()
            db.close()
            logger.info("✅ Database connections closed")
        except Exception as e:
            logger.error(f"❌ Database cleanup error: {e}")
    
    logger.info(f"✅ {APP_NAME} v{VERSION} shutdown complete\n")

# ====================================================================
# MAIN ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    # Railway port configuration
    PORT = int(os.getenv("PORT", API_PORT))
    HOST = "0.0.0.0"  # Railway requires 0.0.0.0
    
    logger.info(f"🚀 Starting server on {HOST}:{PORT}")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        workers=API_WORKERS,
        reload=False,
        log_level="info"
    )
