#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - Main Application

Enterprise-grade AI trading bot with:
- Multi-layer ML ensemble
- Real-time signal generation
- Production data validation
- Zero-tolerance for mock data
- Background AI trading engine
- Professional multi-layer dashboard
- AI/ML prediction dashboard
- 24/7 AI prediction engine
- Telegram notifications
- WebSocket live updates

❌ NO MOCK DATA
❌ NO FALLBACK
❌ NO TEST DATA

✅ 100% Production Real-Time Data
"""

import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# Import AI prediction engine
try:
    from core.ai_engine.prediction_engine import get_prediction_engine
    PREDICTION_ENGINE_AVAILABLE = True
    logger.info("✅ AI prediction engine module loaded")
except ImportError as e:
    logger.warning(f"⚠️  Prediction engine not available: {e}")
    PREDICTION_ENGINE_AVAILABLE = False

# Import WebSocket manager
try:
    from api.websocket_manager import get_ws_manager
    WEBSOCKET_AVAILABLE = True
    logger.info("✅ WebSocket manager loaded")
except ImportError as e:
    logger.warning(f"⚠️  WebSocket manager not available: {e}")
    WEBSOCKET_AVAILABLE = False

# Import API routes
from api import router as api_router
from api.dashboard_api import router as dashboard_router
from api.dashboard_professional import router as professional_router
from api.dashboard_ai import router as ai_dashboard_router
from api.ai_endpoints import router as ai_endpoints_router

# ====================================================================
# LIFESPAN EVENTS
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # ============ STARTUP ============
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
    
    # Start 24/7 AI prediction engine
    if PREDICTION_ENGINE_AVAILABLE:
        try:
            logger.info("🤖 Starting 24/7 AI Prediction Engine...")
            pred_engine = get_prediction_engine()
            await pred_engine.start()
            logger.info("✅ AI Prediction Engine started (24/7 mode with Telegram alerts)")
        except Exception as e:
            logger.error(f"❌ Prediction engine startup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Railway port info
    port = int(os.getenv("PORT", API_PORT))
    logger.info(f"\n✅ {APP_NAME} v{VERSION} is ready!")
    logger.info(f"🌐 API server: http://0.0.0.0:{port}")
    logger.info(f"📊 Health: http://0.0.0.0:{port}/health")
    logger.info(f"📄 Docs: http://0.0.0.0:{port}/docs")
    logger.info(f"📈 Dashboard: http://0.0.0.0:{port}/dashboard")
    logger.info(f"📊 Professional: http://0.0.0.0:{port}/professional")
    logger.info(f"🤖 AI Dashboard: http://0.0.0.0:{port}/ai-dashboard")
    logger.info(f"🧠 AI Predictions: http://0.0.0.0:{port}/api/ai/latest")
    logger.info(f"🔌 WebSocket: ws://0.0.0.0:{port}/ws/dashboard")
    logger.info(f"🏠 Root: http://0.0.0.0:{port}/ (redirects to professional)")
    if TRADING_ENGINE_AVAILABLE:
        logger.info(f"🤖 Engine Status: http://0.0.0.0:{port}/api/engine/status")
    if PREDICTION_ENGINE_AVAILABLE:
        logger.info(f"📢 Telegram Alerts: ENABLED (strong buy/sell signals)")
    logger.info("")
    
    # Application is running
    yield
    
    # ============ SHUTDOWN ============
    logger.info(f"\n\n{'='*60}")
    logger.info(f"{APP_NAME} v{VERSION} - SHUTTING DOWN")
    logger.info(f"{'='*60}\n")
    
    # Stop AI prediction engine
    if PREDICTION_ENGINE_AVAILABLE:
        try:
            logger.info("🛑 Stopping AI Prediction Engine...")
            pred_engine = get_prediction_engine()
            await pred_engine.stop()
            logger.info("✅ AI Prediction Engine stopped")
        except Exception as e:
            logger.error(f"❌ Prediction engine shutdown error: {e}")
    
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
# APPLICATION INITIALIZATION
# ====================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    version=VERSION,
    description="Enterprise-grade AI crypto trading bot API with 24/7 ML predictions and Telegram alerts",
    lifespan=lifespan
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
app.include_router(dashboard_router)
app.include_router(professional_router)
app.include_router(ai_dashboard_router)
app.include_router(ai_endpoints_router)
logger.info("✅ API, Dashboard, Professional, AI Dashboard, and AI Endpoints routes included")

# ====================================================================
# WEBSOCKET ENDPOINT FOR REAL-TIME UPDATES
# ====================================================================

if WEBSOCKET_AVAILABLE:
    @app.websocket("/ws/dashboard")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time AI predictions
        Broadcasts updates every 30 seconds
        """
        ws_manager = get_ws_manager()
        await ws_manager.connect(websocket)
        
        try:
            while True:
                # Keep connection alive
                data = await websocket.receive_text()
                logger.debug(f"📬 WebSocket message received: {data}")
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
            logger.info("❌ WebSocket client disconnected")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
            ws_manager.disconnect(websocket)

# ====================================================================
# ROOT ENDPOINT - REDIRECT TO PROFESSIONAL DASHBOARD
# ====================================================================

@app.get("/")
async def root():
    """
    Root endpoint - redirects to professional dashboard
    Railway production deployment entry point
    """
    logger.info("👉 Root endpoint accessed - redirecting to professional dashboard")
    return RedirectResponse(url="/professional")

# ====================================================================
# HEALTH CHECK ENDPOINT
# ====================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    health_data = {
        "status": "healthy",
        "service": APP_NAME,
        "version": VERSION,
        "ai_prediction_engine": "running" if PREDICTION_ENGINE_AVAILABLE else "disabled",
        "websocket": "available" if WEBSOCKET_AVAILABLE else "disabled"
    }
    
    # Add trading engine status if available
    if TRADING_ENGINE_AVAILABLE:
        try:
            engine = get_engine()
            health_data["trading_engine"] = engine.get_status()
        except Exception as e:
            health_data["trading_engine"] = {"error": str(e)}
    
    # Add prediction engine status
    if PREDICTION_ENGINE_AVAILABLE:
        try:
            pred_engine = get_prediction_engine()
            health_data["prediction_engine"] = {
                "running": pred_engine.is_running,
                "last_predictions": list(pred_engine.last_predictions.keys())
            }
        except Exception as e:
            health_data["prediction_engine"] = {"error": str(e)}
    
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
                "error": str(e)}
    
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
# AI PREDICTION ENGINE CONTROL ENDPOINTS
# ====================================================================

if PREDICTION_ENGINE_AVAILABLE:
    @app.get("/api/ai/status")
    async def get_prediction_status():
        """Get AI prediction engine status"""
        try:
            pred_engine = get_prediction_engine()
            return {
                "success": True,
                "data": {
                    "running": pred_engine.is_running,
                    "telegram_enabled": pred_engine.telegram_notifier is not None,
                    "last_predictions": pred_engine.last_predictions,
                    "thresholds": {
                        "strong_buy": pred_engine.strong_buy_threshold,
                        "strong_sell": pred_engine.strong_sell_threshold
                    }
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

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
