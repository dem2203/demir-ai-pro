#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - Real-Time Price Data API

Enterprise-grade price data endpoints:
- Real Binance Futures API integration
- Multi-symbol batch pricing
- 24h change tracking
- Volume analysis
- NO MOCK DATA - 100% Production

❌ NO FALLBACK
❌ NO TEST DATA

✅ 100% Real-Time Binance Data
"""

import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException, Query
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prices", tags=["Price Data"])

# Binance Futures API Base URL
BINANCE_FUTURES_API = "https://fapi.binance.com"

# ====================================================================
# PRICE DATA FUNCTIONS
# ====================================================================

async def fetch_binance_price(symbol: str) -> Dict:
    """
    Fetch real-time price from Binance Futures API
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        
    Returns:
        Price data dictionary
    """
    try:
        url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
        params = {"symbol": symbol}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "price": float(data.get("lastPrice", 0)),
                        "change_24h": float(data.get("priceChangePercent", 0)),
                        "volume_24h": float(data.get("volume", 0)),
                        "high_24h": float(data.get("highPrice", 0)),
                        "low_24h": float(data.get("lowPrice", 0)),
                        "quote_volume": float(data.get("quoteVolume", 0))
                    }
                else:
                    logger.error(f"❌ Binance API error for {symbol}: HTTP {response.status}")
                    return None
                    
    except asyncio.TimeoutError:
        logger.error(f"❌ Binance API timeout for {symbol}")
        return None
    except Exception as e:
        logger.error(f"❌ Binance API fetch error for {symbol}: {e}")
        return None

async def fetch_multiple_prices(symbols: List[str]) -> Dict[str, Dict]:
    """
    Fetch prices for multiple symbols concurrently
    
    Args:
        symbols: List of trading pairs
        
    Returns:
        Dictionary mapping symbol to price data
    """
    tasks = [fetch_binance_price(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    prices = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"❌ Error fetching {symbol}: {result}")
            prices[symbol] = None
        else:
            prices[symbol] = result
    
    return prices

# ====================================================================
# API ENDPOINTS
# ====================================================================

@router.get("")
async def get_prices(
    symbols: str = Query(
        "BTCUSDT,ETHUSDT,LTCUSDT",
        description="Comma-separated list of trading pairs"
    )
):
    """
    Get real-time prices for multiple symbols from Binance Futures
    
    Query params:
        symbols: Comma-separated trading pairs (e.g., BTCUSDT,ETHUSDT)
        
    Returns:
        Dictionary with price data for each symbol
        
    Example:
        GET /api/prices?symbols=BTCUSDT,ETHUSDT,LTCUSDT
        
        {
            "success": true,
            "prices": {
                "BTCUSDT": {
                    "price": 95432.50,
                    "change_24h": 2.45,
                    "volume_24h": 123456.78,
                    "high_24h": 96200.00,
                    "low_24h": 93800.00,
                    "quote_volume": 11765432100.00
                },
                ...
            },
            "count": 3
        }
    """
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        
        if not symbol_list:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols per request")
        
        logger.info(f"💰 Fetching prices for {len(symbol_list)} symbols: {symbol_list}")
        
        # Fetch prices concurrently
        prices = await fetch_multiple_prices(symbol_list)
        
        # Filter out failed fetches
        valid_prices = {k: v for k, v in prices.items() if v is not None}
        
        if not valid_prices:
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch price data from Binance"
            )
        
        logger.info(f"✅ Successfully fetched {len(valid_prices)}/{len(symbol_list)} prices")
        
        return {
            "success": True,
            "prices": valid_prices,
            "count": len(valid_prices),
            "failed": [s for s in symbol_list if s not in valid_prices]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Price fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Price fetch failed: {str(e)}")

@router.get("/{symbol}")
async def get_single_price(symbol: str):
    """
    Get real-time price for a single symbol
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        
    Returns:
        Price data for the symbol
        
    Example:
        GET /api/prices/BTCUSDT
        
        {
            "success": true,
            "symbol": "BTCUSDT",
            "data": {
                "price": 95432.50,
                "change_24h": 2.45,
                "volume_24h": 123456.78,
                "high_24h": 96200.00,
                "low_24h": 93800.00,
                "quote_volume": 11765432100.00
            }
        }
    """
    try:
        symbol = symbol.upper().strip()
        
        logger.info(f"💰 Fetching price for {symbol}")
        
        price_data = await fetch_binance_price(symbol)
        
        if price_data is None:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to fetch price for {symbol} from Binance"
            )
        
        logger.info(f"✅ Price fetched for {symbol}: ${price_data['price']:.2f}")
        
        return {
            "success": True,
            "symbol": symbol,
            "data": price_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Price fetch error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Price fetch failed: {str(e)}")
