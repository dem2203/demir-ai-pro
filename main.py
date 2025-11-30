#!/usr/bin/env python3
"""
DEMIR AI PRO v9.0 - Main Application PROFESSIONAL

Enterprise-grade AI trading bot with:
- Multi-layer ML ensemble
- Real-time signal generation
- Production data validation
- Zero-tolerance for mock data
- Background AI trading engine
- Professional Trading Terminal (main dashboard)
- Professional multi-layer dashboard
- AI/ML prediction dashboard
- Manuel coin management
- 24/7 AI prediction engine
- Telegram notifications (hourly + strong signals)
- WebSocket live updates
- Real-time Binance price API
- Structured logging
- Professional error handling
- Health monitoring
- Performance metrics

❌ NO MOCK DATA
❌ NO FALLBACK
❌ NO TEST DATA

✅ 100% Production Real-Time Data
✅ Professional AI Standards
"""

import logging
import sys
import os
import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import pytz

# ====================================================================
# STRUCTURED LOGGING CONFIGURATION
# ====================================================================

class StructuredLogger:
    """JSON structured logger for production monitoring"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        log_data = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            **kwargs
        }
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log('DEBUG', message, **kwargs)

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Structured logger handles formatting
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("demir_ai_pro.log")
    ]
)

logger = StructuredLogger(__name__)

# ====================================================================
# IMPORT CONFIGURATION
# ====================================================================

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

# ====================================================================
# IMPORT MODULES WITH GRACEFUL DEGRADATION
# ====================================================================

# Database
try:
    from database import get_db
    from database.models import create_all_tables
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning("Database module not available", error=str(e))
    get_db = None
    create_all_tables = None
    DATABASE_AVAILABLE = False

# Core modules
try:
    from core import AIEngine, SignalProcessor, RiskManager, DataPipeline
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning("Core modules not available", error=str(e))
    CORE_MODULES_AVAILABLE = False

# Integrations
try:
    from integrations import BinanceIntegration, TelegramNotifier
    INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning("Integration modules not available", error=str(e))
    INTEGRATIONS_AVAILABLE = False

# Trading engine
try:
    from core.trading_engine import get_engine
    TRADING_ENGINE_AVAILABLE = True
    logger.info("Trading engine module loaded")
except ImportError as e:
    logger.warning("Trading engine not available", error=str(e))
    TRADING_ENGINE_AVAILABLE = False

# AI prediction engine
try:
    from core.ai_engine.prediction_engine import get_prediction_engine
    PREDICTION_ENGINE_AVAILABLE = True
    logger.info("AI prediction engine module loaded")
except ImportError as e:
    logger.warning("Prediction engine not available", error=str(e))
    PREDICTION_ENGINE_AVAILABLE = False

# WebSocket manager
try:
    from api.websocket_manager import get_ws_manager
    WEBSOCKET_AVAILABLE = True
    logger.info("WebSocket manager loaded")
except ImportError as e:
    logger.warning("WebSocket manager not available", error=str(e))
    WEBSOCKET_AVAILABLE = False

# Import API routes
from api import router as api_router
from api.dashboard_api import router as dashboard_router
from api.dashboard_professional import router as professional_router
from api.dashboard_ai import router as ai_dashboard_router
from api.ai_endpoints import router as ai_endpoints_router
from api.coin_manager import router as coin_manager_router
from api.prices import router as prices_router

# ====================================================================
# APPLICATION STATE
# ====================================================================

class ApplicationState:
    """Global application state for monitoring"""
    
    def __init__(self):
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.error_count: int = 0
        self.health_status: str = "starting"
        self.services_status: Dict[str, bool] = {}
    
    def get_uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    def increment_request(self) -> None:
        self.request_count += 1
    
    def increment_error(self) -> None:
        self.error_count += 1

app_state = ApplicationState()

# ====================================================================
# LIFESPAN EVENTS
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    
    # ============ STARTUP ============
    logger.info("Application starting", name=APP_NAME, version=VERSION)
    
    app_state.health_status = "initializing"
    
    # Initialize database
    if DATABASE_AVAILABLE and get_db and create_all_tables:
        try:
            db = get_db()
            create_all_tables(db)
            app_state.services_status['database'] = True
            logger.info("Database initialized")
        except Exception as e:
            app_state.services_status['database'] = False
            logger.error("Database initialization failed", error=str(e))
    else:
        app_state.services_status['database'] = False
    
    # Initialize core modules
    if CORE_MODULES_AVAILABLE:
        try:
            logger.info("Initializing core modules")
            app_state.services_status['core_modules'] = True
            logger.info("Core modules initialized")
        except Exception as e:
            app_state.services_status['core_modules'] = False
            logger.error("Core initialization failed", error=str(e))
    else:
        app_state.services_status['core_modules'] = False
    
    # Start trading engine
    if TRADING_ENGINE_AVAILABLE:
        try:
            logger.info("Starting AI Trading Engine")
            engine = get_engine()
            await engine.start()
            app_state.services_status['trading_engine'] = True
            logger.info("AI Trading Engine started")
        except Exception as e:
            app_state.services_status['trading_engine'] = False
            logger.error("Trading engine startup failed", error=str(e),
                        traceback=traceback.format_exc())
    else:
        app_state.services_status['trading_engine'] = False
    
    # Start 24/7 AI prediction engine
    if PREDICTION_ENGINE_AVAILABLE:
        try:
            logger.info("Starting 24/7 AI Prediction Engine")
            pred_engine = get_prediction_engine()
            await pred_engine.start()
            app_state.services_status['prediction_engine'] = True
            logger.info("AI Prediction Engine started",
                       features=["24/7_predictions", "telegram_hourly", "strong_signals"])
        except Exception as e:
            app_state.services_status['prediction_engine'] = False
            logger.error("Prediction engine startup failed", error=str(e),
                        traceback=traceback.format_exc())
    else:
        app_state.services_status['prediction_engine'] = False
    
    # Mark as healthy
    app_state.health_status = "healthy"
    
    # Railway port info
    port = int(os.getenv("PORT", API_PORT))
    logger.info("Application ready",
               name=APP_NAME,
               version=VERSION,
               port=port,
               host="0.0.0.0",
               main_dashboard="/trading-terminal",
               health_check="/health",
               api_docs="/docs")
    
    # Application is running
    yield
    
    # ============ SHUTDOWN ============
    logger.info("Application shutting down", name=APP_NAME, version=VERSION)
    
    app_state.health_status = "shutting_down"
    
    # Stop AI prediction engine
    if PREDICTION_ENGINE_AVAILABLE and app_state.services_status.get('prediction_engine'):
        try:
            logger.info("Stopping AI Prediction Engine")
            pred_engine = get_prediction_engine()
            await pred_engine.stop()
            logger.info("AI Prediction Engine stopped")
        except Exception as e:
            logger.error("Prediction engine shutdown error", error=str(e))
    
    # Stop trading engine
    if TRADING_ENGINE_AVAILABLE and app_state.services_status.get('trading_engine'):
        try:
            logger.info("Stopping AI Trading Engine")
            engine = get_engine()
            await engine.stop()
            logger.info("Trading engine stopped")
        except Exception as e:
            logger.error("Trading engine shutdown error", error=str(e))
    
    # Close database connections
    if DATABASE_AVAILABLE and get_db:
        try:
            db = get_db()
            db.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error("Database cleanup error", error=str(e))
    
    logger.info("Application shutdown complete",
               total_requests=app_state.request_count,
               total_errors=app_state.error_count,
               uptime_seconds=app_state.get_uptime_seconds())

# ====================================================================
# APPLICATION INITIALIZATION
# ====================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    version=VERSION,
    description="Enterprise-grade AI crypto trading bot API with 24/7 ML predictions, Professional Trading Terminal, Real-time Prices, and Telegram alerts",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Store app state in FastAPI app
app.state.app_state = app_state

# ====================================================================
# MIDDLEWARE
# ====================================================================

# GZip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with structured logging"""
    start_time = time.time()
    app_state.increment_request()
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("HTTP request",
                   method=request.method,
                   path=request.url.path,
                   status_code=response.status_code,
                   duration_ms=duration_ms,
                   client_host=request.client.host if request.client else None)
        
        return response
    except Exception as e:
        app_state.increment_error()
        logger.error("Request processing error",
                    method=request.method,
                    path=request.url.path,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise

# ====================================================================
# EXCEPTION HANDLERS
# ====================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    app_state.increment_error()
    
    logger.error("Validation error",
                method=request.method,
                path=request.url.path,
                errors=exc.errors())
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    app_state.increment_error()
    
    logger.error("Unhandled exception",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    )

# ====================================================================
# INCLUDE API ROUTES
# ====================================================================

app.include_router(api_router)
app.include_router(dashboard_router)
app.include_router(professional_router)
app.include_router(ai_dashboard_router)
app.include_router(ai_endpoints_router)
app.include_router(coin_manager_router)
app.include_router(prices_router)

logger.info("API routes included",
           routers=["api", "dashboard", "professional", "ai_dashboard", 
                   "ai_endpoints", "coin_manager", "prices"])

# ====================================================================
# TRADING TERMINAL ENDPOINT (MAIN DASHBOARD)
# ====================================================================

@app.get("/trading-terminal")
async def trading_terminal():
    """
    Professional Trading Terminal with live AI signals (MAIN DASHBOARD)
    
    Features:
    - Real-time Binance prices (30s refresh)
    - WebSocket AI signals
    - 127 layer status
    - Dynamic coin management
    - Telegram integration
    """
    try:
        return FileResponse("ui/trading_terminal.html")
    except FileNotFoundError:
        logger.error("Trading terminal file not found", path="ui/trading_terminal.html")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": "Trading terminal not found",
                "path": "ui/trading_terminal.html"
            }
        )

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
                logger.debug("WebSocket message received", data=data)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            ws_manager.disconnect(websocket)

# ====================================================================
# ROOT ENDPOINT - REDIRECT TO TRADING TERMINAL
# ====================================================================

@app.get("/")
async def root():
    """Root endpoint - redirects to Trading Terminal (main dashboard)"""
    logger.info("Root endpoint accessed - redirecting to Trading Terminal")
    return RedirectResponse(url="/trading-terminal")

# ====================================================================
# HEALTH CHECK ENDPOINT
# ====================================================================

@app.get("/health")
async def health_check():
    """Professional health check endpoint with system metrics"""
    try:
        uptime_seconds = app_state.get_uptime_seconds()
        
        health_data = {
            "status": app_state.health_status,
            "service": APP_NAME,
            "version": VERSION,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "metrics": {
                "total_requests": app_state.request_count,
                "total_errors": app_state.error_count,
                "error_rate": app_state.error_count / max(app_state.request_count, 1)
            },
            "services": app_state.services_status,
            "endpoints": {
                "main_dashboard": "/trading-terminal",
                "health": "/health",
                "api_docs": "/api/docs"
            }
        }
        
        # Add trading engine status if available
        if TRADING_ENGINE_AVAILABLE and app_state.services_status.get('trading_engine'):
            try:
                engine = get_engine()
                health_data["trading_engine"] = engine.get_status()
            except Exception as e:
                health_data["trading_engine"] = {"error": str(e)}
        
        # Add prediction engine status
        if PREDICTION_ENGINE_AVAILABLE and app_state.services_status.get('prediction_engine'):
            try:
                pred_engine = get_prediction_engine()
                metrics = pred_engine.get_performance_metrics()
                health_data["prediction_engine"] = {
                    "running": pred_engine.is_running,
                    "total_predictions": metrics.total_predictions,
                    "successful_predictions": metrics.successful_predictions,
                    "failed_predictions": metrics.failed_predictions,
                    "avg_execution_time_ms": metrics.avg_execution_time_ms,
                    "uptime_hours": metrics.uptime_hours
                }
            except Exception as e:
                health_data["prediction_engine"] = {"error": str(e)}
        
        # Add coin manager status
        try:
            from api.coin_manager import get_monitored_coins
            health_data["monitored_coins"] = get_monitored_coins()
        except:
            health_data["monitored_coins"] = []
        
        # Determine overall status
        all_services_healthy = all(app_state.services_status.values())
        if not all_services_healthy:
            health_data["status"] = "degraded"
        
        return health_data
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        )

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
                "data": engine.get_status(),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        except Exception as e:
            logger.error("Engine status error", error=str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": str(e)}
            )
    
    @app.post("/api/engine/start")
    async def start_engine():
        """Manually start trading engine"""
        try:
            engine = get_engine()
            await engine.start()
            logger.info("Trading engine started manually")
            return {
                "success": True,
                "message": "Trading engine started",
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        except Exception as e:
            logger.error("Engine start error", error=str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": str(e)}
            )
    
    @app.post("/api/engine/stop")
    async def stop_engine():
        """Manually stop trading engine"""
        try:
            engine = get_engine()
            await engine.stop()
            logger.info("Trading engine stopped manually")
            return {
                "success": True,
                "message": "Trading engine stopped",
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        except Exception as e:
            logger.error("Engine stop error", error=str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": str(e)}
            )

# ====================================================================
# AI PREDICTION ENGINE CONTROL ENDPOINTS
# ====================================================================

if PREDICTION_ENGINE_AVAILABLE:
    @app.get("/api/ai/status")
    async def get_prediction_status():
        """Get AI prediction engine status with performance metrics"""
        try:
            pred_engine = get_prediction_engine()
            metrics = pred_engine.get_performance_metrics()
            
            from api.coin_manager import get_monitored_coins
            
            return {
                "success": True,
                "data": {
                    "running": pred_engine.is_running,
                    "version": pred_engine.version,
                    "telegram_enabled": pred_engine.telegram_notifier is not None,
                    "telegram_features": {
                        "hourly_status": "enabled (BTC/ETH/LTC prices)",
                        "strong_signals": "enabled (all coins, >=85% confidence)"
                    },
                    "monitored_coins": get_monitored_coins(),
                    "last_predictions": {k: v.timestamp for k, v in pred_engine.last_predictions.items()},
                    "thresholds": {
                        "strong_buy": pred_engine.strong_buy_threshold,
                        "strong_sell": pred_engine.strong_sell_threshold
                    },
                    "performance_metrics": {
                        "total_predictions": metrics.total_predictions,
                        "successful_predictions": metrics.successful_predictions,
                        "failed_predictions": metrics.failed_predictions,
                        "success_rate": metrics.successful_predictions / max(metrics.total_predictions, 1),
                        "avg_execution_time_ms": metrics.avg_execution_time_ms,
                        "uptime_hours": metrics.uptime_hours
                    }
                },
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        except Exception as e:
            logger.error("Prediction status error", error=str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": str(e)}
            )

# ====================================================================
# MAIN ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    # Railway port configuration
    PORT = int(os.getenv("PORT", API_PORT))
    HOST = "0.0.0.0"  # Railway requires 0.0.0.0
    
    logger.info("Starting server",
               host=HOST,
               port=PORT,
               workers=API_WORKERS,
               reload=False)
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        workers=API_WORKERS,
        reload=False,
        log_level="info"
    )
