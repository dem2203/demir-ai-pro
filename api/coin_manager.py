#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - Coin Manager API

Manage monitored trading pairs dynamically:
- Add/remove coins
- Get active coin list
- Persist to cache/database

Default coins: BTCUSDT, ETHUSDT, LTCUSDT
User can add more for 24/7 monitoring + Telegram alerts
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/coins", tags=["Coin Management"])

# In-memory coin list (production: use Redis/Database)
# Default 3 coins always monitored
DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
active_coins: List[str] = DEFAULT_COINS.copy()

class AddCoinRequest(BaseModel):
    """Request to add a coin"""
    symbol: str

class RemoveCoinRequest(BaseModel):
    """Request to remove a coin"""
    symbol: str

class CoinsResponse(BaseModel):
    """Response with coin list"""
    success: bool
    coins: List[str]
    message: str = ""

@router.get("", response_model=CoinsResponse)
async def get_active_coins():
    """
    Get list of currently monitored coins
    
    Returns:
        List of trading pairs being monitored 24/7
    """
    global active_coins
    return CoinsResponse(
        success=True,
        coins=active_coins,
        message=f"Monitoring {len(active_coins)} trading pairs"
    )

@router.post("/add", response_model=CoinsResponse)
async def add_coin(request: AddCoinRequest):
    """
    Add a new coin to monitoring list
    
    Args:
        request: AddCoinRequest with symbol (e.g., XRPUSDT)
        
    Returns:
        Updated coin list
        
    Notes:
        - Symbol must be valid Binance Futures pair
        - Duplicate symbols are ignored
        - AI will start monitoring immediately
        - Telegram alerts will include this coin
    """
    global active_coins
    
    symbol = request.symbol.upper().strip()
    
    # Validation
    if len(symbol) < 6:
        raise HTTPException(status_code=400, detail="Invalid symbol (too short)")
    
    if not symbol.endswith("USDT"):
        raise HTTPException(status_code=400, detail="Symbol must end with USDT")
    
    if symbol in active_coins:
        logger.info(f"⚠️ Coin {symbol} already monitored")
        return CoinsResponse(
            success=True,
            coins=active_coins,
            message=f"{symbol} already in watchlist"
        )
    
    # Add to list
    active_coins.append(symbol)
    logger.info(f"✅ Added coin: {symbol} (Total: {len(active_coins)})")
    
    # TODO: Persist to database
    # TODO: Notify prediction engine to start monitoring
    
    return CoinsResponse(
        success=True,
        coins=active_coins,
        message=f"Added {symbol} to monitoring"
    )

@router.post("/remove", response_model=CoinsResponse)
async def remove_coin(request: RemoveCoinRequest):
    """
    Remove a coin from monitoring list
    
    Args:
        request: RemoveCoinRequest with symbol
        
    Returns:
        Updated coin list
        
    Notes:
        - Cannot remove default coins (BTC, ETH, LTC)
        - Only user-added coins can be removed
    """
    global active_coins
    
    symbol = request.symbol.upper().strip()
    
    # Cannot remove default coins
    if symbol in DEFAULT_COINS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot remove default coin: {symbol}"
        )
    
    if symbol not in active_coins:
        raise HTTPException(
            status_code=404,
            detail=f"Coin {symbol} not in watchlist"
        )
    
    # Remove from list
    active_coins.remove(symbol)
    logger.info(f"✅ Removed coin: {symbol} (Total: {len(active_coins)})")
    
    return CoinsResponse(
        success=True,
        coins=active_coins,
        message=f"Removed {symbol} from monitoring"
    )

@router.get("/count")
async def get_coin_count():
    """
    Get count of monitored coins
    """
    return {
        "total": len(active_coins),
        "default": len(DEFAULT_COINS),
        "user_added": len(active_coins) - len(DEFAULT_COINS)
    }

def get_monitored_coins() -> List[str]:
    """
    Helper function for other modules to get active coin list
    
    Returns:
        List of currently monitored trading pairs
    """
    return active_coins.copy()
