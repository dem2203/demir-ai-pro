#!/usr/bin/env python3
"""DEMIR AI PRO v10.0 PROFESSIONAL ULTRA - Main Application

Enterprise-grade AI trading bot with:
- Real-time signal generation with multi-component fusion
- Market intelligence (order book, whale detection, sentiment)
- Advanced ML predictions (LSTM, XGBoost, RF, GB)
- Professional Telegram alerts (<30 sec latency)
- Ultra Trading Terminal (TradingView-style)
- 24/7 monitoring and alerts
- WebSocket real-time updates
- Comprehensive risk management

❌ NO MOCK DATA
✅ 100% Real-Time Professional Trading System
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
# STRUCTURED LOGGING
# ====================================================================

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        log_data = {'timestamp': datetime.now(pytz.UTC).isoformat(), 'level': level, 'logger': self.name, 'message': message, **kwargs}
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log('DEBUG', message, **kwargs)

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("demir_ai_pro.log")])

logger = StructuredLogger(__name__)

# ====================================================================
# CONFIGURATION
# ====================================================================

VERSION = "10.0"
APP_NAME = "DEMIR AI PRO ULTRA"

try:
    from config import API_HOST, API_PORT, API_WORKERS, CORS_ORIGINS, validate_or_exit
    validate_or_exit()
except:
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 1
    CORS_ORIGINS = ["*"]

# ====================================================================
# MODULE IMPORTS
# ====================================================================

try:
    from database import get_db
    from database.models import create_all_tables
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning("Database not available", error=str(e))
    get_db = None
    create_all_tables = None
    DATABASE_AVAILABLE = False

try:
    from core.trading_engine import get_engine
    TRADING_ENGINE_AVAILABLE = True
    logger.info("Trading engine loaded")
except ImportError as e:
    logger.warning("Trading engine not available", error=str(e))
    TRADING_ENGINE_AVAILABLE = False

try:
    from core.ai_engine.prediction_engine import get_prediction_engine
    PREDICTION_ENGINE_AVAILABLE = True
    logger.info("Prediction engine loaded")
except ImportError as e:
    logger.warning("Prediction engine not available", error=str(e))
    PREDICTION_ENGINE_AVAILABLE = False

try:
    from core.signal_engine import get_signal_engine
    SIGNAL_ENGINE_AVAILABLE = True
    logger.info("Signal engine loaded")
except ImportError as e:
    logger.warning("Signal engine not available", error=str(e))
    SIGNAL_ENGINE_AVAILABLE = False

try:
    from core.market_intelligence import get_market_intelligence
    MARKET_INTELLIGENCE_AVAILABLE = True
    logger.info("Market intelligence loaded")
except ImportError as e:
    logger.warning("Market intelligence not available", error=str(e))
    MARKET_INTELLIGENCE_AVAILABLE = False

try:
    from api.websocket_manager import get_ws_manager
    WEBSOCKET_AVAILABLE = True
    logger.info("WebSocket manager loaded")
except ImportError as e:
    logger.warning("WebSocket not available", error=str(e))
    WEBSOCKET_AVAILABLE = False

# Import API routers
from api import router as api_router
from api.dashboard_api import router as dashboard_router
from api.dashboard_professional import router as professional_router
from api.dashboard_ai import router as ai_dashboard_router
from api.ai_endpoints import router as ai_endpoints_router
from api.coin_manager import router as coin_manager_router
from api.prices import router as prices_router
from api.signal_endpoints import router as signal_router

# ====================================================================
# APPLICATION STATE
# ====================================================================

class ApplicationState:
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
            logger.warning("Database failed - memory-only mode", error=str(e))
    else:
        app_state.services_status['database'] = False
    
    # Start Market Intelligence
    if MARKET_INTELLIGENCE_AVAILABLE:
        try:
            logger.info("Initializing Market Intelligence")
            mi = get_market_intelligence()
            await mi.initialize()
            app_state.services_status['market_intelligence'] = True
            logger.info("Market Intelligence ready")
        except Exception as e:
            app_state.services_status['market_intelligence'] = False
            logger.error("Market Intelligence failed", error=str(e))
    
    # Start Signal Engine
    if SIGNAL_ENGINE_AVAILABLE:
        try:
            logger.info("Initializing Signal Engine")
            signal_engine = get_signal_engine()
            await signal_engine.initialize()
            app_state.services_status['signal_engine'] = True
            logger.info("Signal Engine ready")
        except Exception as e:
            app_state.services_status['signal_engine'] = False
            logger.error("Signal Engine failed", error=str(e))
    
    # Start Trading Engine
    if TRADING_ENGINE_AVAILABLE:
        try:
            logger.info("Starting Trading Engine")
            engine = get_engine()
            await engine.start()
            app_state.services_status['trading_engine'] = True
            logger.info("Trading Engine started")
        except Exception as e:
            app_state.services_status['trading_engine'] = False
            logger.error("Trading Engine failed", error=str(e), traceback=traceback.format_exc())
    
    # Start AI Prediction Engine
    if PREDICTION_ENGINE_AVAILABLE:
        try:
            logger.info("Starting AI Prediction Engine")
            pred_engine = get_prediction_engine()
            await pred_engine.start()
            app_state.services_status['prediction_engine'] = True
            logger.info("AI Prediction Engine started")
        except Exception as e:
            app_state.services_status['prediction_engine'] = False
            logger.error("Prediction Engine failed", error=str(e), traceback=traceback.format_exc())
    
    # Initialize Telegram Ultra
    try:
        from integrations.telegram_ultra import get_telegram_ultra
        telegram_token = os.getenv('TELEGRAM_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if telegram_token and telegram_chat_id:
            telegram = get_telegram_ultra(telegram_token, telegram_chat_id)
            app_state.services_status['telegram_ultra'] = True
            logger.info("Telegram Ultra ready")
        else:
            app_state.services_status['telegram_ultra'] = False
            logger.warning("Telegram credentials missing")
    except Exception as e:
        app_state.services_status['telegram_ultra'] = False
        logger.error("Telegram Ultra failed", error=str(e))
    
    app_state.health_status = "healthy"
    port = int(os.getenv("PORT", API_PORT))
    logger.info("Application ready", name=APP_NAME, version=VERSION, port=port, host="0.0.0.0")
    
    yield
    
    # ============ SHUTDOWN ============
    logger.info("Application shutting down", name=APP_NAME, version=VERSION)
    app_state.health_status = "shutting_down"
    
    if PREDICTION_ENGINE_AVAILABLE and app_state.services_status.get('prediction_engine'):
        try:
            pred_engine = get_prediction_engine()
            await pred_engine.stop()
            logger.info("AI Prediction Engine stopped")
        except Exception as e:
            logger.error("Prediction Engine shutdown error", error=str(e))
    
    if TRADING_ENGINE_AVAILABLE and app_state.services_status.get('trading_engine'):
        try:
            engine = get_engine()
            await engine.stop()
            logger.info("Trading Engine stopped")
        except Exception as e:
            logger.error("Trading Engine shutdown error", error=str(e))
    
    if DATABASE_AVAILABLE and get_db:
        try:
            db = get_db()
            db.close()
            logger.info("Database closed")
        except Exception as e:
            logger.error("Database cleanup error", error=str(e))
    
    logger.info("Shutdown complete", requests=app_state.request_count, errors=app_state.error_count, uptime=app_state.get_uptime_seconds())

# ====================================================================
# APPLICATION INITIALIZATION
# ====================================================================

app = FastAPI(title=f"{APP_NAME} API", version=VERSION, description="Enterprise AI crypto trading bot with real-time signals, market intelligence, and professional alerts", lifespan=lifespan, docs_url="/api/docs", redoc_url="/api/redoc")
app.state.app_state = app_state

# ====================================================================
# MIDDLEWARE
# ====================================================================

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    app_state.increment_request()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info("HTTP request", method=request.method, path=request.url.path, status=response.status_code, duration_ms=duration_ms)
        return response
    except Exception as e:
        app_state.increment_error()
        logger.error("Request error", method=request.method, path=request.url.path, error=str(e))
        raise

# ====================================================================
# EXCEPTION HANDLERS
# ====================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    app_state.increment_error()
    logger.error("Validation error", method=request.method, path=request.url.path, errors=exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"success": False, "error": "Validation error", "details": exc.errors(), "timestamp": datetime.now(pytz.UTC).isoformat()})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    app_state.increment_error()
    logger.error("Unhandled exception", method=request.method, path=request.url.path, error=str(exc), type=type(exc).__name__)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": "Internal server error", "timestamp": datetime.now(pytz.UTC).isoformat()})

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
app.include_router(signal_router)

logger.info("API routes included", routers=["api", "dashboard", "professional", "ai_dashboard", "ai_endpoints", "coin_manager", "prices", "signals"])

# ====================================================================
# MAIN ENDPOINTS
# ====================================================================

@app.get("/")
async def root_ultra_dashboard():
    """Ultra Professional Trading Terminal v10.0"""
    try:
        return FileResponse("ui/trading_terminal_ultra.html")
    except FileNotFoundError:
        logger.error("Dashboard not found")
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"success": False, "error": "Dashboard not found"})

@app.get("/trading-terminal")
async def trading_terminal_legacy():
    """Legacy endpoint - redirects to ultra dashboard"""
    return RedirectResponse(url="/")

if WEBSOCKET_AVAILABLE:
    @app.websocket("/ws/dashboard")
    async def websocket_endpoint(websocket: WebSocket):
        ws_manager = get_ws_manager()
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                logger.debug("WebSocket message", data=data)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            ws_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check with comprehensive system metrics v10.0"""
    try:
        uptime_seconds = app_state.get_uptime_seconds()
        health_data = {
            "status": app_state.health_status,
            "service": APP_NAME,
            "version": VERSION,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "uptime_seconds": uptime_seconds,
            "uptime_hours": uptime_seconds / 3600,
            "metrics": {"total_requests": app_state.request_count, "total_errors": app_state.error_count, "error_rate": app_state.error_count / max(app_state.request_count, 1)},
            "services": app_state.services_status,
            "endpoints": {"main_dashboard": "/", "health": "/health", "api_docs": "/api/docs", "signals": "/api/signals", "market_intelligence": "/api/signals/market-intelligence/{symbol}"}
        }
        
        if TRADING_ENGINE_AVAILABLE and app_state.services_status.get('trading_engine'):
            try:
                engine = get_engine()
                health_data["trading_engine"] = engine.get_status()
            except Exception as e:
                health_data["trading_engine"] = {"error": str(e)}
        
        if PREDICTION_ENGINE_AVAILABLE and app_state.services_status.get('prediction_engine'):
            try:
                pred_engine = get_prediction_engine()
                metrics = pred_engine.get_performance_metrics()
                health_data["prediction_engine"] = {"running": pred_engine.is_running, "total_predictions": metrics.total_predictions, "successful_predictions": metrics.successful_predictions, "avg_execution_time_ms": metrics.avg_execution_time_ms, "uptime_hours": metrics.uptime_hours}
            except Exception as e:
                health_data["prediction_engine"] = {"error": str(e)}
        
        if SIGNAL_ENGINE_AVAILABLE and app_state.services_status.get('signal_engine'):
            try:
                signal_engine = get_signal_engine()
                health_data["signal_engine"] = {"initialized": signal_engine.technical_analyzer is not None}
            except Exception as e:
                health_data["signal_engine"] = {"error": str(e)}
        
        try:
            from api.coin_manager import get_monitored_coins
            health_data["monitored_coins"] = get_monitored_coins()
        except:
            health_data["monitored_coins"] = []
        
        if not all(app_state.services_status.values()):
            health_data["status"] = "degraded"
        
        return health_data
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "unhealthy", "error": str(e), "timestamp": datetime.now(pytz.UTC).isoformat()})

if TRADING_ENGINE_AVAILABLE:
    @app.get("/api/engine/status")
    async def get_engine_status():
        try:
            engine = get_engine()
            return {"success": True, "data": engine.get_status(), "timestamp": datetime.now(pytz.UTC).isoformat()}
        except Exception as e:
            logger.error("Engine status error", error=str(e))
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": str(e)})
    
    @app.post("/api/engine/start")
    async def start_engine():
        try:
            engine = get_engine()
            await engine.start()
            logger.info("Engine started manually")
            return {"success": True, "message": "Trading engine started", "timestamp": datetime.now(pytz.UTC).isoformat()}
        except Exception as e:
            logger.error("Engine start error", error=str(e))
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": str(e)})
    
    @app.post("/api/engine/stop")
    async def stop_engine():
        try:
            engine = get_engine()
            await engine.stop()
            logger.info("Engine stopped manually")
            return {"success": True, "message": "Trading engine stopped", "timestamp": datetime.now(pytz.UTC).isoformat()}
        except Exception as e:
            logger.error("Engine stop error", error=str(e))
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": str(e)})

if PREDICTION_ENGINE_AVAILABLE:
    @app.get("/api/ai/status")
    async def get_prediction_status():
        try:
            pred_engine = get_prediction_engine()
            metrics = pred_engine.get_performance_metrics()
            from api.coin_manager import get_monitored_coins
            return {"success": True, "data": {"running": pred_engine.is_running, "version": pred_engine.version, "telegram_enabled": pred_engine.telegram_notifier is not None, "monitored_coins": get_monitored_coins(), "performance_metrics": {"total_predictions": metrics.total_predictions, "successful_predictions": metrics.successful_predictions, "success_rate": metrics.successful_predictions / max(metrics.total_predictions, 1), "avg_execution_time_ms": metrics.avg_execution_time_ms, "uptime_hours": metrics.uptime_hours}}, "timestamp": datetime.now(pytz.UTC).isoformat()}
        except Exception as e:
            logger.error("Prediction status error", error=str(e))
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", API_PORT))
    HOST = "0.0.0.0"
    logger.info("Starting server", host=HOST, port=PORT, workers=API_WORKERS)
    uvicorn.run("main:app", host=HOST, port=PORT, workers=API_WORKERS, reload=False, log_level="info")
