"""
System Status Endpoints

System metrics and performance data.
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
async def get_status():
    """
    Get system status.
    
    Returns:
        System status and metrics
    """
    # TODO: Integrate with actual system metrics
    return {
        "status": "active",
        "global_state": {
            "metrics": {
                "sentiment_score": 0.0,
                "market_regime": 0.0,
                "risk_var": 0.0
            },
            "signal_stats": {},
            "signals_count": {}
        },
        "timestamp": datetime.now().isoformat()
    }
