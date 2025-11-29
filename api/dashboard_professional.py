"""
Professional Dashboard Routes
=============================
127-Layer Multi-Dimensional Analysis API
Real Technical Analysis (TA-Lib) with NO MOCK DATA

Author: DEMIR AI PRO
Version: 8.0
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("api.dashboard_professional")

router = APIRouter(prefix="", tags=["professional_dashboard"])

# Fixed symbols to monitor
FIXED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]


@router.get("/professional")
async def get_professional_dashboard():
    """
    Serve professional dashboard HTML with 127-layer support
    """
    try:
        # Try multiple possible paths
        possible_paths = [
            Path("ui/professional_dashboard.html"),
            Path("/app/ui/professional_dashboard.html"),
            Path("app/ui/professional_dashboard.html")
        ]
        
        dashboard_path = None
        for path in possible_paths:
            if path.exists():
                dashboard_path = path
                break
        
        if not dashboard_path:
            logger.error(f"❌ Professional dashboard HTML not found")
            return HTMLResponse(
                content="<h1>Dashboard file not found</h1>",
                status_code=404
            )
        
        logger.info(f"📄 Loading professional dashboard from: {dashboard_path}")
        
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        logger.info("✅ Professional dashboard HTML loaded")
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"❌ Error loading dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return HTMLResponse(
            content=f"<h1>Error: {str(e)}</h1>",
            status_code=500
        )


@router.get("/api/analyze/{symbol}")
async def analyze_coin(symbol: str):
    """
    Analyze any coin with 127-layer professional TA engine
    
    Returns comprehensive analysis including:
    - Technical Analysis (40 layers)
    - Volume Analysis (10 layers)
    - Volatility (9 layers)
    - Pattern Recognition (27 layers)
    - Statistical Features (17 layers)
    - AI/ML Models (4 layers)
    - Sentiment (4 layers)
    - Market Regime (4 layers)
    - Multi-Timeframe (8 layers)
    - Ensemble Meta (4 layers)
    
    NO MOCK DATA - Only real market data and calculations
    """
    try:
        symbol = symbol.upper()
        
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        
        logger.info(f"🔍 Analyzing {symbol} with 127-layer professional TA engine...")
        
        # Use professional TA engine (127 layers)
        from core.technical_analysis import get_ta_engine
        
        ta_engine = get_ta_engine()
        analysis = await ta_engine.analyze(symbol)
        
        if not analysis:
            logger.error(f"❌ Professional analysis failed for {symbol}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": f"Could not analyze {symbol}. Check if symbol exists on Binance Futures.",
                    "symbol": symbol
                },
                status_code=404
            )
        
        # Construct professional response with all 127 layers
        response = {
            "success": True,
            "symbol": symbol,
            "timestamp": analysis.get('timestamp', datetime.now().isoformat()),
            "analysis": {
                "price": analysis['price'],
                "change_24h": analysis['change_24h'],
                "composite_score": analysis['composite_score'],
                "ai_commentary": analysis['ai_commentary'],
                "layer_count": analysis.get('layer_count', 127),
                "layers": analysis['layers']  # All 127 layers included
            }
        }
        
        logger.info(f"✅ 127-layer analysis complete for {symbol} | Score: {analysis['composite_score']}/100")
        return response
        
    except Exception as e:
        logger.error(f"❌ Analysis error for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/api/status")
async def get_status():
    """
    API health check
    """
    return {
        "status": "online",
        "version": "8.0",
        "features": {
            "layer_count": 127,
            "categories": [
                "Technical Analysis (40)",
                "Volume Analysis (10)",
                "Volatility (9)",
                "Pattern Recognition (27)",
                "Statistical Features (17)",
                "AI/ML Models (4)",
                "Sentiment (4)",
                "Market Regime (4)",
                "Multi-Timeframe (8)",
                "Ensemble Meta (4)"
            ],
            "fixed_symbols": FIXED_SYMBOLS
        },
        "timestamp": datetime.now().isoformat()
    }
