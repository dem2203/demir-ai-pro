"""
DEMIR AI PRO - API Routes

FastAPI routes for:
- Health monitoring
- Price data
- Signal data
- Trading operations
- System status
"""

from fastapi import APIRouter

from .health import router as health_router
from .prices import router as prices_router
from .signals import router as signals_router
from .status import router as status_router

# Main router
router = APIRouter()

# Include sub-routers
router.include_router(health_router, prefix="/health", tags=["Health"])
router.include_router(prices_router, prefix="/api/prices", tags=["Prices"])
router.include_router(signals_router, prefix="/api/signals", tags=["Signals"])
router.include_router(status_router, prefix="/api/status", tags=["Status"])

__all__ = ['router']
