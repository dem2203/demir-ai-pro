#!/usr/bin/env python3
"""DEMIR AI PRO v10.5 PROFESSIONAL - Full Production System

Enterprise-grade AI trading bot with:
- PURE AI predictions (NO FALLBACK)
- Real-time WebSocket updates
- Professional Dashboard with 127-layer analysis
- Market intelligence and sentiment analysis
- 24/7 monitoring and alerts
- Comprehensive risk management

❌ NO MOCK DATA
❌ NO FALLBACK PREDICTIONS
✅ 100% Pure AI Real-Time Professional Trading System
"""

import logging
import sys
import os
import json
import time
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = StructuredLogger(__name__)

VERSION = "10.5"
APP_NAME = "DEMIR AI PRO PROFESSIONAL"

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
    
    # Initialize WebSocket Manager
    try:
        from api.websocket_manager import get_ws_manager
        ws_manager = get_ws_manager()
        app_state.services_status['websocket'] = True
        
        # Start WebSocket broadcast loop
        asyncio.create_task(ws_manager.start_broadcast_loop(interval=30))
        logger.info("WebSocket manager started")
    except Exception as e:
        app_state.services_status['websocket'] = False
        logger.error("WebSocket failed", error=str(e))
    
    # Initialize Technical Analysis Engine
    try:
        from core.technical_analysis import get_ta_engine
        ta_engine = get_ta_engine()
        app_state.services_status['technical_analysis'] = True
        logger.info("Technical Analysis engine ready")
    except Exception as e:
        app_state.services_status['technical_analysis'] = False
        logger.error("Technical Analysis failed", error=str(e))
    
    # Initialize Market Intelligence (optional)
    try:
        from core.market_intelligence import get_market_intelligence
        mi = get_market_intelligence()
        await mi.initialize()
        app_state.services_status['market_intelligence'] = True
        logger.info("Market Intelligence ready")
    except Exception as e:
        app_state.services_status['market_intelligence'] = False
        logger.warning("Market Intelligence unavailable", error=str(e))
    
    app_state.health_status = "healthy"
    logger.info(
        "Application ready",
        name=APP_NAME,
        version=VERSION,
        services=app_state.services_status
    )
    
    yield
    
    # ============ SHUTDOWN ============
    logger.info("Application shutting down", name=APP_NAME, version=VERSION)
    app_state.health_status = "shutting_down"
    
    logger.info(
        "Shutdown complete",
        requests=app_state.request_count,
        errors=app_state.error_count,
        uptime=app_state.get_uptime_seconds()
    )

# ====================================================================
# APPLICATION INITIALIZATION
# ====================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    version=VERSION,
    description="Enterprise AI crypto trading bot with real-time analysis",
    lifespan=lifespan
)
app.state.app_state = app_state

# ====================================================================
# MIDDLEWARE
# ====================================================================

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    app_state.increment_request()
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms
        )
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
    app_state.increment_error()
    logger.error(
        "Unhandled exception",
        method=request.method,
        path=request.url.path,
        error=str(exc),
        type=type(exc).__name__
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    )

# ====================================================================
# MAIN ENDPOINTS
# ====================================================================

@app.get("/")
async def root_dashboard():
    """Serve Professional Dashboard - 127-Layer Analysis"""
    try:
        possible_paths = [
            Path("ui/professional_dashboard.html"),
            Path("/app/ui/professional_dashboard.html"),
            Path("app/ui/professional_dashboard.html")
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Loading dashboard from: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return HTMLResponse(content=content)
        
        logger.error("❌ Dashboard HTML not found")
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Dashboard not found",
                "searched_paths": [str(p) for p in possible_paths]
            }
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    from api.websocket_manager import get_ws_manager
    ws_manager = get_ws_manager()
    
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            logger.debug("WebSocket message received", data=data)
            
            # Parse client request
            try:
                request = json.loads(data)
                if request.get('type') == 'ping':
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Analyze crypto symbol with 127-layer professional TA"""
    try:
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        
        logger.info(f"🔍 Analyzing {symbol}...")
        
        from core.technical_analysis import get_ta_engine
        ta_engine = get_ta_engine()
        analysis = await ta_engine.analyze(symbol)
        
        if not analysis:
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Could not analyze {symbol}",
                    "symbol": symbol
                },
                status_code=404
            )
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": analysis.get('timestamp', datetime.now().isoformat()),
            "analysis": {
                "price": analysis['price'],
                "change_24h": analysis['change_24h'],
                "composite_score": analysis['composite_score'],
                "ai_commentary": analysis['ai_commentary'],
                "layer_count": analysis.get('layer_count', 127),
                "layers": analysis['layers']
            }
        }
    except Exception as e:
        logger.error(f"❌ Analysis error for {symbol}: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e), "symbol": symbol},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check with comprehensive metrics"""
    try:
        uptime_seconds = app_state.get_uptime_seconds()
        
        # Get monitored coins
        try:
            from api.coin_manager import get_monitored_coins
            monitored_coins = get_monitored_coins()
        except:
            monitored_coins = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
        
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
            "monitored_coins": monitored_coins
        }
        
        # Get WebSocket stats
        if app_state.services_status.get('websocket'):
            try:
                from api.websocket_manager import get_ws_manager
                ws_manager = get_ws_manager()
                health_data["websocket"] = ws_manager.get_stats()
            except:
                pass
        
        return health_data
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
        )

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "online",
        "version": VERSION,
        "uptime_seconds": app_state.get_uptime_seconds(),
        "timestamp": datetime.now().isoformat()
    }

# ====================================================================
# RUN SERVER
# ====================================================================

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    HOST = "0.0.0.0"
    
    logger.info(f"🚀 Starting {APP_NAME} v{VERSION}")
    logger.info(f"📍 Server: {HOST}:{PORT}")
    logger.info(f"🌐 Dashboard: http://{HOST}:{PORT}/")
    logger.info(f"📊 API: http://{HOST}:{PORT}/api/analyze/BTCUSDT")
    logger.info(f"❤️ Health: http://{HOST}:{PORT}/health")
    logger.info(f"🔌 WebSocket: ws://{HOST}:{PORT}/ws/dashboard")
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        workers=1,
        reload=False,
        log_level="info",
        access_log=True
    )
