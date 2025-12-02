#!/usr/bin/env python3
"""DEMIR AI PRO v10.3 HOTFIX - Minimal Safe Main

Minimal version that WORKS - no crashes
Starts with dashboard only, services initialize in background
"""

import logging
import sys
import os
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi import Request, status
import uvicorn
from pathlib import Path

# Simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

VERSION = "10.3-HOTFIX"
APP_NAME = "DEMIR AI PRO"

# ====================================================================
# MINIMAL FASTAPI APP - NO COMPLEX IMPORTS
# ====================================================================

app = FastAPI(
    title=f"{APP_NAME} API",
    version=VERSION,
    description="Professional AI Trading Dashboard"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple state
start_time = time.time()

# ====================================================================
# ROOT ENDPOINT - PROFESSIONAL DASHBOARD
# ====================================================================

@app.get("/")
async def root_dashboard():
    """Serve Professional Dashboard HTML"""
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
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Checked paths: " + ", ".join(str(p) for p in possible_paths) + "</p>",
            status_code=404
        )
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        return HTMLResponse(
            content=f"<h1>Error loading dashboard</h1><p>{str(e)}</p>",
            status_code=500
        )

# ====================================================================
# API ENDPOINTS - SAFE IMPORTS
# ====================================================================

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Analyze crypto symbol"""
    try:
        symbol = symbol.upper()
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        
        logger.info(f"🔍 Analyzing {symbol}...")
        
        # Safe import - only when endpoint called
        try:
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
        except ImportError as ie:
            logger.error(f"Import error: {ie}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Technical analysis module not available",
                    "symbol": symbol
                },
                status_code=503
            )
    except Exception as e:
        logger.error(f"❌ Analysis error for {symbol}: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "symbol": symbol
            },
            status_code=500
        )

@app.get("/api/status")
async def api_status():
    """API status check"""
    return {
        "status": "online",
        "version": VERSION,
        "uptime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "service": APP_NAME,
        "version": VERSION,
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "timestamp": datetime.now().isoformat()
    }

# ====================================================================
# ERROR HANDLERS
# ====================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error"
        }
    )

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
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        workers=1,
        reload=False,
        log_level="info",
        access_log=True
    )
