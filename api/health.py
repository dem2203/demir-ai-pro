"""
Health Check Endpoints

System health monitoring and status checks.
"""

from fastapi import APIRouter
from datetime import datetime
import psutil

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Basic health check.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "DEMIR AI PRO"
    }

@router.get("/detailed")
async def detailed_health():
    """
    Detailed health check with system metrics.
    
    Returns:
        Detailed health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "service": {
            "name": "DEMIR AI PRO",
            "version": "8.0.0",
            "uptime_seconds": 0  # TODO: Track actual uptime
        }
    }
