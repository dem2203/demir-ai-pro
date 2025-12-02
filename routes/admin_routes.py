#!/usr/bin/env python3
"""🏋️ Admin Routes - Training Control
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/train")
async def trigger_training():
    """Start AI model training"""
    try:
        from api.train_endpoint import start_training
        result = await start_training()
        return result
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@router.get("/train/status")
async def training_status():
    """Get training status"""
    try:
        from api.train_endpoint import training_in_progress, training_start_time
        
        if not training_in_progress:
            return {
                "training_in_progress": False,
                "last_training": training_start_time.isoformat() if training_start_time else None
            }
        
        elapsed = (datetime.now(pytz.UTC) - training_start_time).total_seconds()
        return {
            "training_in_progress": True,
            "started_at": training_start_time.isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
            "estimated_remaining": max(0, 10 - elapsed / 60)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
