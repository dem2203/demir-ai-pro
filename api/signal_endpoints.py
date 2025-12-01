#!/usr/bin/env python3
"""DEMIR AI PRO v10.0 - Signal API Endpoints

RESTful API for trading signals and market intelligence
"""

import logging
from typing import Optional
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pytz

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signals", tags=["signals"])

# Response models
class SignalResponse(BaseModel):
    success: bool
    data: Optional[dict]
    error: Optional[str]
    timestamp: str

class MarketIntelligenceResponse(BaseModel):
    success: bool
    data: Optional[dict]
    error: Optional[str]
    timestamp: str

# Background monitoring
_monitoring_active = False
_monitored_symbols = set()

@router.get("/", summary="Get all active signals")
async def get_all_signals():
    """Get all active trading signals"""
    try:
        from api.coin_manager import get_monitored_coins
        from core.signal_engine import get_signal_engine
        
        coins = get_monitored_coins()
        signal_engine = get_signal_engine()
        
        if not signal_engine.technical_analyzer:
            await signal_engine.initialize()
        
        # Generate signals for all monitored coins
        signals = {}
        for symbol in coins:
            try:
                signal = await signal_engine.generate_signal(symbol)
                if signal:
                    from dataclasses import asdict
                    signal_dict = asdict(signal)
                    # Convert enums to strings
                    signal_dict['signal_type'] = signal.signal_type.value
                    signal_dict['priority'] = signal.priority.value
                    signals[symbol] = signal_dict
            except Exception as e:
                logger.error(f"Signal generation error for {symbol}: {e}")
        
        return SignalResponse(
            success=True,
            data={"signals": signals, "count": len(signals)},
            error=None,
            timestamp=datetime.now(pytz.UTC).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get all signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}", summary="Get signal for specific symbol")
async def get_signal(symbol: str):
    """Get trading signal for specific symbol"""
    try:
        from core.signal_engine import get_signal_engine
        
        signal_engine = get_signal_engine()
        
        if not signal_engine.technical_analyzer:
            await signal_engine.initialize()
        
        signal = await signal_engine.generate_signal(symbol.upper())
        
        if not signal:
            return SignalResponse(
                success=False,
                data=None,
                error="Failed to generate signal",
                timestamp=datetime.now(pytz.UTC).isoformat()
            )
        
        from dataclasses import asdict
        signal_dict = asdict(signal)
        signal_dict['signal_type'] = signal.signal_type.value
        signal_dict['priority'] = signal.priority.value
        
        return SignalResponse(
            success=True,
            data=signal_dict,
            error=None,
            timestamp=datetime.now(pytz.UTC).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get signal error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-intelligence/{symbol}", summary="Get market intelligence for symbol")
async def get_market_intelligence(symbol: str):
    """Get comprehensive market intelligence for symbol"""
    try:
        from core.market_intelligence import get_market_intelligence
        
        mi = get_market_intelligence()
        
        if not mi.binance_client:
            await mi.initialize()
        
        analysis = await mi.get_comprehensive_analysis(symbol.upper())
        
        # Convert dataclasses to dicts
        from dataclasses import asdict
        
        result = {
            "symbol": analysis["symbol"],
            "timestamp": analysis["timestamp"]
        }
        
        if analysis["market_depth"]:
            result["market_depth"] = asdict(analysis["market_depth"])
        
        if analysis["sentiment"]:
            result["sentiment"] = asdict(analysis["sentiment"])
        
        if analysis["whale_activity"]:
            result["whale_activity"] = asdict(analysis["whale_activity"])
        
        return MarketIntelligenceResponse(
            success=True,
            data=result,
            error=None,
            timestamp=datetime.now(pytz.UTC).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Market intelligence error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/start", summary="Start signal monitoring")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start background signal monitoring with Telegram alerts"""
    global _monitoring_active
    
    if _monitoring_active:
        return {
            "success": False,
            "message": "Monitoring already active",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    
    try:
        background_tasks.add_task(_signal_monitoring_loop)
        _monitoring_active = True
        
        return {
            "success": True,
            "message": "Signal monitoring started",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor/stop", summary="Stop signal monitoring")
async def stop_monitoring():
    """Stop background signal monitoring"""
    global _monitoring_active
    
    _monitoring_active = False
    
    return {
        "success": True,
        "message": "Signal monitoring stopped",
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

@router.get("/monitor/status", summary="Get monitoring status")
async def get_monitoring_status():
    """Get current monitoring status"""
    return {
        "active": _monitoring_active,
        "monitored_symbols": list(_monitored_symbols),
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

# Background monitoring task
async def _signal_monitoring_loop():
    """Background task for continuous signal monitoring"""
    global _monitoring_active, _monitored_symbols
    
    logger.info("Signal monitoring loop started")
    
    from core.signal_engine import get_signal_engine
    from integrations.telegram_ultra import get_telegram_ultra
    from api.coin_manager import get_monitored_coins
    import os
    
    signal_engine = get_signal_engine()
    await signal_engine.initialize()
    
    # Initialize Telegram
    telegram_token = os.getenv('TELEGRAM_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    telegram = get_telegram_ultra(telegram_token, telegram_chat_id) if telegram_token and telegram_chat_id else None
    
    while _monitoring_active:
        try:
            coins = get_monitored_coins()
            _monitored_symbols = set(coins)
            
            for symbol in coins:
                try:
                    # Generate signal
                    signal = await signal_engine.generate_signal(symbol)
                    
                    if signal and telegram:
                        # Send alert for high-priority signals
                        from core.signal_engine import SignalPriority
                        if signal.priority in [SignalPriority.CRITICAL, SignalPriority.HIGH]:
                            if signal.confidence >= 75:
                                await telegram.send_signal_alert(signal)
                    
                except Exception as e:
                    logger.error(f"Monitoring error for {symbol}: {e}")
            
            # Wait 5 minutes before next check
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            await asyncio.sleep(60)
    
    logger.info("Signal monitoring loop stopped")
