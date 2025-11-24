"""
Binance Market Data Integration
===============================
Full async, railway-production compliant, real-time OHLCV & orderbook feed
for use in main trading loop (no fallback/mock, only production pathways).

Author: DEMIR AI PRO
Version: 8.0
"""

import aiohttp
import asyncio
import json
import logging
from typing import List, Dict, Any

BINANCE_API_URL = "https://api.binance.com"
WS_STREAM_URL = "wss://stream.binance.com:9443/ws"
logger = logging.getLogger("binance.market_data")

class BinanceMarketData:
    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.ohlcv_url = f"{BINANCE_API_URL}/api/v3/klines"
        self.loop = asyncio.get_event_loop()

    async def fetch_ohlcv(self, interval: str = "1m", limit: int = 150) -> List[Dict[str, Any]]:
        params = {
            "symbol": self.symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.ohlcv_url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Failed to fetch OHLCV: {resp.status} {body}")
                    raise Exception(f"Binance OHLCV fetch error: {resp.status}")
                klines = await resp.json()
                # Returns list: [open time, open, high, low, close, volume, ...]
                parsed = [{
                    "open_time": x[0],
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[5]),
                    "close_time": x[6],
                } for x in klines]
                return parsed

    async def stream_orderbook(self, depth_levels: int = 20):
        pair = self.symbol
        ws_url = f"{WS_STREAM_URL}/{pair}@depth@100ms"
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=30) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        # Format: {'bids': [[price, qty], ...], 'asks': ...}
                        orderbook = {
                            'bids': [(float(p), float(q)) for p, q in data.get('bids', [])[:depth_levels]],
                            'asks': [(float(p), float(q)) for p, q in data.get('asks', [])[:depth_levels]]
                        }
                        yield orderbook
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error("WS error: %s", msg.data)
                        break

    @staticmethod
    async def fetch_orderbook_snapshot(symbol: str, limit: int = 20) -> Dict[str, List]:
        url = f"{BINANCE_API_URL}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": limit}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"Orderbook snapshot fetch failed: {resp.status}")
                    raise Exception(f"Orderbook fetch error: {resp.status}")
                data = await resp.json()
                return {
                    'bids': [(float(p), float(q)) for p, q in data.get('bids', [])],
                    'asks': [(float(p), float(q)) for p, q in data.get('asks', [])]
                }

# Usage example (real code should run async):
#
# binance_data = BinanceMarketData(symbol="BTCUSDT")
# ohlcv = await binance_data.fetch_ohlcv()
# async for ob in binance_data.stream_orderbook():
#     ... # process_orderbook(ob)
