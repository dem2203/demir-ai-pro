#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - AI Dashboard Route Handler

Serves the enterprise AI/ML prediction dashboard
with real-time model updates and WebSocket support.
"""

import logging
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Dashboard"])

# Resolve dashboard path
BASE_DIR = Path(__file__).resolve().parent.parent
AI_DASHBOARD_PATH = BASE_DIR / "ui" / "ai_dashboard.html"

@router.get("/ai-dashboard")
async def serve_ai_dashboard():
    """
    Serve AI/ML prediction dashboard
    
    Features:
    - Real-time AI model predictions (LSTM, XGBoost, RF, GB)
    - Ensemble weighted voting
    - Model performance metrics
    - Feature importance visualization
    - WebSocket live updates
    - REST API fallback
    - Zero mock data architecture
    
    Returns:
        HTML dashboard with embedded JavaScript for real-time updates
    """
    try:
        if not AI_DASHBOARD_PATH.exists():
            logger.error(f"❌ AI Dashboard not found at {AI_DASHBOARD_PATH}")
            return {
                "error": "AI Dashboard file not found",
                "path": str(AI_DASHBOARD_PATH)
            }
        
        logger.info(f"✅ Serving AI Dashboard from {AI_DASHBOARD_PATH}")
        return FileResponse(
            path=str(AI_DASHBOARD_PATH),
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"❌ Error serving AI dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "error": "Failed to serve AI dashboard",
            "detail": str(e)
        }
