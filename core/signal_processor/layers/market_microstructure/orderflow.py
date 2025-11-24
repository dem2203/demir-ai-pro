"""
Orderflow Analysis Module
=========================
Real-time orderbook and trade flow analysis for market microstructure insights.

Components:
- OrderbookAnalyzer: Bid-ask spread, depth imbalance, liquidity analysis
- TapeReader: Large order detection, aggressive buy/sell identification
- LiquidityHeatmap: Volume clustering at price levels
- OrderFlowImbalance: Net buying/selling pressure tracking

Author: DEMIR AI PRO
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """Orderbook snapshot at a point in time"""
    timestamp: str
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    spread_pct: float


@dataclass
class LargeOrder:
    """Large order detected in tape"""
    timestamp: str
    price: float
    size: float
    side: str  # 'BUY' or 'SELL'
    is_aggressive: bool
    value_usd: float


@dataclass
class LiquidityLevel:
    """Liquidity concentration at price level"""
    price: float
    bid_volume: float
    ask_volume: float
    net_volume: float
    is_significant: bool


class OrderbookAnalyzer:
    """
    Real-time Orderbook Analysis
    
    Analyzes bid-ask spread, depth imbalance, and liquidity distribution.
    Identifies support/resistance levels based on order clustering.
    """
    
    def __init__(self, depth_levels: int = 20, imbalance_threshold: float = 0.3):
        """
        Args:
            depth_levels: Number of orderbook levels to analyze
            imbalance_threshold: Threshold for significant imbalance (0-1)
        """
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
        self.history = deque(maxlen=100)  # Last 100 snapshots
        
    def analyze(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Dict[str, any]:
        """
        Analyze current orderbook state
        
        Args:
            bids: List of (price, size) tuples, sorted descending
            asks: List of (price, size) tuples, sorted ascending
            
        Returns:
            Dictionary with orderbook metrics
        """
        try:
            if not bids or not asks:
                logger.warning("Orderbook: Empty bids or asks")
                return self._empty_result()
            
            # Best bid/ask
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # Spread analysis
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100
            
            # Take only specified depth levels
            bids_depth = bids[:self.depth_levels]
            asks_depth = asks[:self.depth_levels]
            
            # Calculate total volume
            bid_volume = sum(size for _, size in bids_depth)
            ask_volume = sum(size for _, size in asks_depth)
            total_volume = bid_volume + ask_volume
            
            # Depth imbalance
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            else:
                imbalance = 0
            
            # Weighted mid price (volume-weighted)
            if total_volume > 0:
                weighted_bid = sum(price * size for price, size in bids_depth) / bid_volume if bid_volume > 0 else best_bid
                weighted_ask = sum(price * size for price, size in asks_depth) / ask_volume if ask_volume > 0 else best_ask
                weighted_mid = (weighted_bid * bid_volume + weighted_ask * ask_volume) / total_volume
            else:
                weighted_mid = mid_price
            
            # Identify large walls (significant single orders)
            bid_walls = self._find_walls(bids_depth, is_bid=True)
            ask_walls = self._find_walls(asks_depth, is_bid=False)
            
            # Calculate depth ratios at different levels
            depth_ratios = self._calculate_depth_ratios(bids_depth, asks_depth)
            
            # Store snapshot
            snapshot = OrderbookSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                bids=bids_depth,
                asks=asks_depth,
                mid_price=mid_price,
                spread=spread,
                spread_pct=spread_pct
            )
            self.history.append(snapshot)
            
            # Determine market pressure
            if imbalance > self.imbalance_threshold:
                pressure = "BULLISH"
            elif imbalance < -self.imbalance_threshold:
                pressure = "BEARISH"
            else:
                pressure = "NEUTRAL"
            
            return {
                'mid_price': mid_price,
                'weighted_mid': weighted_mid,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'imbalance': imbalance,
                'market_pressure': pressure,
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'depth_ratios': depth_ratios,
                'is_tight_spread': spread_pct < 0.1,
                'is_wide_spread': spread_pct > 0.5,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Orderbook analysis error: {e}")
            return self._empty_result()
    
    def _find_walls(
        self,
        orders: List[Tuple[float, float]],
        is_bid: bool
    ) -> List[Dict[str, any]]:
        """Identify significant order walls"""
        if not orders:
            return []
        
        # Calculate average size
        sizes = [size for _, size in orders]
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Wall threshold: 2 standard deviations above mean
        wall_threshold = avg_size + 2 * std_size
        
        walls = []
        for price, size in orders:
            if size >= wall_threshold:
                walls.append({
                    'price': price,
                    'size': size,
                    'side': 'BID' if is_bid else 'ASK',
                    'size_ratio': size / avg_size if avg_size > 0 else 0
                })
        
        return walls
    
    def _calculate_depth_ratios(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """Calculate bid/ask volume ratios at different depth levels"""
        ratios = {}
        
        for level in [5, 10, 20]:
            if len(bids) >= level and len(asks) >= level:
                bid_vol = sum(size for _, size in bids[:level])
                ask_vol = sum(size for _, size in asks[:level])
                
                if ask_vol > 0:
                    ratios[f'ratio_{level}'] = bid_vol / ask_vol
                else:
                    ratios[f'ratio_{level}'] = 0
            else:
                ratios[f'ratio_{level}'] = 1.0
        
        return ratios
    
    def _empty_result(self) -> Dict:
        return {
            'mid_price': None,
            'weighted_mid': None,
            'best_bid': None,
            'best_ask': None,
            'spread': 0,
            'spread_pct': 0,
            'bid_volume': 0,
            'ask_volume': 0,
            'imbalance': 0,
            'market_pressure': 'NEUTRAL',
            'bid_walls': [],
            'ask_walls': [],
            'depth_ratios': {},
            'is_tight_spread': False,
            'is_wide_spread': False,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_spread_history(self, periods: int = 20) -> List[float]:
        """Get historical spread percentages"""
        return [s.spread_pct for s in list(self.history)[-periods:]]
    
    def get_imbalance_trend(self, periods: int = 10) -> str:
        """Analyze imbalance trend over recent periods"""
        if len(self.history) < periods:
            return "INSUFFICIENT_DATA"
        
        recent = list(self.history)[-periods:]
        imbalances = []
        
        for snapshot in recent:
            bid_vol = sum(size for _, size in snapshot.bids)
            ask_vol = sum(size for _, size in snapshot.asks)
            total = bid_vol + ask_vol
            if total > 0:
                imbalances.append((bid_vol - ask_vol) / total)
        
        if not imbalances:
            return "NEUTRAL"
        
        avg_imbalance = np.mean(imbalances)
        
        if avg_imbalance > 0.2:
            return "INCREASING_BUY_PRESSURE"
        elif avg_imbalance < -0.2:
            return "INCREASING_SELL_PRESSURE"
        else:
            return "BALANCED"


class TapeReader:
    """
    Trade Tape Analysis
    
    Reads real-time trade flow to detect:
    - Large orders (whale activity)
    - Aggressive buying/selling
    - Trade imbalance
    - Volume surges
    """
    
    def __init__(
        self,
        large_order_threshold: float = 10000,  # USD value
        window_seconds: int = 60
    ):
        """
        Args:
            large_order_threshold: Minimum order size (USD) to classify as large
            window_seconds: Time window for aggregation
        """
        self.large_order_threshold = large_order_threshold
        self.window_seconds = window_seconds
        self.trade_history = deque(maxlen=1000)
        self.large_orders = deque(maxlen=100)
        
    def process_trade(
        self,
        price: float,
        size: float,
        is_buyer_maker: bool,
        timestamp: Optional[str] = None
    ) -> Optional[LargeOrder]:
        """
        Process individual trade
        
        Args:
            price: Trade price
            size: Trade size (quantity)
            is_buyer_maker: True if buyer was maker (sell aggression)
            timestamp: Trade timestamp
            
        Returns:
            LargeOrder if trade exceeds threshold, None otherwise
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow().isoformat()
            
            # Determine side (from taker perspective)
            # is_buyer_maker=True means taker sold (hit bid)
            # is_buyer_maker=False means taker bought (hit ask)
            side = 'SELL' if is_buyer_maker else 'BUY'
            is_aggressive = True  # Taker orders are aggressive by definition
            
            value_usd = price * size
            
            trade = {
                'timestamp': timestamp,
                'price': price,
                'size': size,
                'side': side,
                'is_aggressive': is_aggressive,
                'value_usd': value_usd
            }
            
            self.trade_history.append(trade)
            
            # Check if large order
            if value_usd >= self.large_order_threshold:
                large_order = LargeOrder(
                    timestamp=timestamp,
                    price=price,
                    size=size,
                    side=side,
                    is_aggressive=is_aggressive,
                    value_usd=value_usd
                )
                self.large_orders.append(large_order)
                logger.info(f"Large {side} order detected: ${value_usd:,.0f} at {price}")
                return large_order
            
            return None
            
        except Exception as e:
            logger.error(f"Trade processing error: {e}")
            return None
    
    def get_flow_imbalance(self, seconds: Optional[int] = None) -> Dict[str, any]:
        """
        Calculate trade flow imbalance over time window
        
        Returns:
            Dictionary with buy/sell volumes and imbalance metrics
        """
        try:
            if seconds is None:
                seconds = self.window_seconds
            
            cutoff = datetime.utcnow() - timedelta(seconds=seconds)
            
            # Filter trades within window
            recent_trades = [
                t for t in self.trade_history
                if datetime.fromisoformat(t['timestamp']) >= cutoff
            ]
            
            if not recent_trades:
                return self._empty_flow_result()
            
            # Aggregate by side
            buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'BUY')
            sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'SELL')
            
            buy_value = sum(t['value_usd'] for t in recent_trades if t['side'] == 'BUY')
            sell_value = sum(t['value_usd'] for t in recent_trades if t['side'] == 'SELL')
            
            total_volume = buy_volume + sell_volume
            total_value = buy_value + sell_value
            
            # Calculate imbalance
            if total_volume > 0:
                volume_imbalance = (buy_volume - sell_volume) / total_volume
            else:
                volume_imbalance = 0
            
            if total_value > 0:
                value_imbalance = (buy_value - sell_value) / total_value
            else:
                value_imbalance = 0
            
            # Classify pressure
            if volume_imbalance > 0.3:
                pressure = "STRONG_BUY"
            elif volume_imbalance > 0.1:
                pressure = "MODERATE_BUY"
            elif volume_imbalance < -0.3:
                pressure = "STRONG_SELL"
            elif volume_imbalance < -0.1:
                pressure = "MODERATE_SELL"
            else:
                pressure = "BALANCED"
            
            # Average trade size
            avg_trade_size = total_volume / len(recent_trades) if recent_trades else 0
            
            return {
                'window_seconds': seconds,
                'trade_count': len(recent_trades),
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_value_usd': buy_value,
                'sell_value_usd': sell_value,
                'volume_imbalance': volume_imbalance,
                'value_imbalance': value_imbalance,
                'pressure': pressure,
                'avg_trade_size': avg_trade_size,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Flow imbalance calculation error: {e}")
            return self._empty_flow_result()
    
    def get_large_orders(self, minutes: int = 15) -> List[Dict[str, any]]:
        """Get large orders from recent time period"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_large = [
            asdict(order) for order in self.large_orders
            if datetime.fromisoformat(order.timestamp) >= cutoff
        ]
        
        return recent_large
    
    def _empty_flow_result(self) -> Dict:
        return {
            'window_seconds': self.window_seconds,
            'trade_count': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_value_usd': 0,
            'sell_value_usd': 0,
            'volume_imbalance': 0,
            'value_imbalance': 0,
            'pressure': 'BALANCED',
            'avg_trade_size': 0,
            'timestamp': datetime.utcnow().isoformat()
        }


class LiquidityHeatmap:
    """
    Liquidity Heatmap Generator
    
    Visualizes liquidity concentration across price levels.
    Identifies key support/resistance zones based on order clustering.
    """
    
    def __init__(self, price_precision: int = 2, bin_size: float = 0.001):
        """
        Args:
            price_precision: Decimal precision for price levels
            bin_size: Price bin size as percentage (0.001 = 0.1%)
        """
        self.price_precision = price_precision
        self.bin_size = bin_size
        
    def generate(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        current_price: float
    ) -> Dict[str, any]:
        """
        Generate liquidity heatmap from orderbook
        
        Args:
            bids: Bid orders [(price, size), ...]
            asks: Ask orders [(price, size), ...]
            current_price: Current market price
            
        Returns:
            Dictionary with liquidity levels and zones
        """
        try:
            # Combine all orders
            all_orders = []
            
            for price, size in bids:
                all_orders.append({
                    'price': round(price, self.price_precision),
                    'bid_volume': size,
                    'ask_volume': 0
                })
            
            for price, size in asks:
                all_orders.append({
                    'price': round(price, self.price_precision),
                    'bid_volume': 0,
                    'ask_volume': size
                })
            
            # Aggregate by price level
            df = pd.DataFrame(all_orders)
            aggregated = df.groupby('price').sum().reset_index()
            
            # Calculate net volume
            aggregated['net_volume'] = aggregated['bid_volume'] - aggregated['ask_volume']
            aggregated['total_volume'] = aggregated['bid_volume'] + aggregated['ask_volume']
            
            # Sort by total volume
            aggregated = aggregated.sort_values('total_volume', ascending=False)
            
            # Identify significant levels (top 20%)
            volume_threshold = aggregated['total_volume'].quantile(0.80)
            significant_levels = aggregated[aggregated['total_volume'] >= volume_threshold]
            
            # Create liquidity levels
            liquidity_levels = []
            for _, row in significant_levels.iterrows():
                level = LiquidityLevel(
                    price=row['price'],
                    bid_volume=row['bid_volume'],
                    ask_volume=row['ask_volume'],
                    net_volume=row['net_volume'],
                    is_significant=True
                )
                liquidity_levels.append(asdict(level))
            
            # Find nearest support/resistance
            support_levels = significant_levels[
                (significant_levels['price'] < current_price) &
                (significant_levels['bid_volume'] > significant_levels['ask_volume'])
            ].sort_values('price', ascending=False)
            
            resistance_levels = significant_levels[
                (significant_levels['price'] > current_price) &
                (significant_levels['ask_volume'] > significant_levels['bid_volume'])
            ].sort_values('price', ascending=True)
            
            nearest_support = support_levels.iloc[0]['price'] if len(support_levels) > 0 else None
            nearest_resistance = resistance_levels.iloc[0]['price'] if len(resistance_levels) > 0 else None
            
            return {
                'liquidity_levels': liquidity_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_levels': support_levels['price'].tolist()[:5],
                'resistance_levels': resistance_levels['price'].tolist()[:5],
                'current_price': current_price,
                'total_bid_liquidity': aggregated['bid_volume'].sum(),
                'total_ask_liquidity': aggregated['ask_volume'].sum(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Liquidity heatmap generation error: {e}")
            return {
                'liquidity_levels': [],
                'nearest_support': None,
                'nearest_resistance': None,
                'support_levels': [],
                'resistance_levels': [],
                'current_price': current_price,
                'total_bid_liquidity': 0,
                'total_ask_liquidity': 0,
                'timestamp': datetime.utcnow().isoformat()
            }


class OrderFlowImbalance:
    """
    Order Flow Imbalance Calculator
    
    Tracks net buying/selling pressure by comparing:
    - Bid-side liquidity changes
    - Ask-side liquidity changes
    - Trade aggression direction
    """
    
    def __init__(self, history_length: int = 50):
        self.history_length = history_length
        self.snapshots = deque(maxlen=history_length)
        
    def update(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Dict[str, any]:
        """
        Update with new orderbook snapshot and calculate imbalance
        
        Returns:
            Dictionary with current and historical imbalance metrics
        """
        try:
            timestamp = datetime.utcnow().isoformat()
            
            # Calculate current state
            bid_liquidity = sum(size for _, size in bids)
            ask_liquidity = sum(size for _, size in asks)
            total_liquidity = bid_liquidity + ask_liquidity
            
            if total_liquidity > 0:
                imbalance = (bid_liquidity - ask_liquidity) / total_liquidity
            else:
                imbalance = 0
            
            snapshot = {
                'timestamp': timestamp,
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'imbalance': imbalance
            }
            
            self.snapshots.append(snapshot)
            
            # Calculate trends
            if len(self.snapshots) >= 2:
                prev = self.snapshots[-2]
                bid_change = bid_liquidity - prev['bid_liquidity']
                ask_change = ask_liquidity - prev['ask_liquidity']
                imbalance_change = imbalance - prev['imbalance']
            else:
                bid_change = 0
                ask_change = 0
                imbalance_change = 0
            
            # Moving average imbalance
            if len(self.snapshots) >= 10:
                recent_imbalances = [s['imbalance'] for s in list(self.snapshots)[-10:]]
                avg_imbalance = np.mean(recent_imbalances)
            else:
                avg_imbalance = imbalance
            
            return {
                'current_imbalance': imbalance,
                'avg_imbalance': avg_imbalance,
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'bid_change': bid_change,
                'ask_change': ask_change,
                'imbalance_change': imbalance_change,
                'trend': self._classify_trend(imbalance_change),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Order flow imbalance calculation error: {e}")
            return {
                'current_imbalance': 0,
                'avg_imbalance': 0,
                'bid_liquidity': 0,
                'ask_liquidity': 0,
                'bid_change': 0,
                'ask_change': 0,
                'imbalance_change': 0,
                'trend': 'NEUTRAL',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _classify_trend(self, change: float) -> str:
        """Classify imbalance trend"""
        if change > 0.1:
            return "INCREASING_BUY_PRESSURE"
        elif change < -0.1:
            return "INCREASING_SELL_PRESSURE"
        else:
            return "STABLE"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Market Microstructure modules
    """
    
    # Sample orderbook data
    bids = [
        (50000, 1.5),
        (49995, 2.0),
        (49990, 0.8),
        (49985, 3.2),
        (49980, 1.1)
    ]
    
    asks = [
        (50005, 1.2),
        (50010, 1.8),
        (50015, 0.9),
        (50020, 2.5),
        (50025, 1.4)
    ]
    
    # Orderbook Analysis
    print("\n=== ORDERBOOK ANALYSIS ===")
    ob_analyzer = OrderbookAnalyzer(depth_levels=5)
    ob_result = ob_analyzer.analyze(bids, asks)
    print(f"Mid Price: ${ob_result['mid_price']:,.2f}")
    print(f"Spread: {ob_result['spread_pct']:.3f}%")
    print(f"Imbalance: {ob_result['imbalance']:.2%}")
    print(f"Market Pressure: {ob_result['market_pressure']}")
    
    # Tape Reader
    print("\n=== TAPE READER ===")
    tape = TapeReader(large_order_threshold=50000)
    
    # Simulate trades
    tape.process_trade(50010, 2.5, False)  # Aggressive buy
    large_order = tape.process_trade(50000, 1.2, True)  # Large sell
    if large_order:
        print(f"Large Order: {large_order.side} ${large_order.value_usd:,.0f}")
    
    flow = tape.get_flow_imbalance()
    print(f"Flow Pressure: {flow['pressure']}")
    print(f"Volume Imbalance: {flow['volume_imbalance']:.2%}")
    
    # Liquidity Heatmap
    print("\n=== LIQUIDITY HEATMAP ===")
    heatmap = LiquidityHeatmap()
    liq_result = heatmap.generate(bids, asks, 50002.5)
    print(f"Nearest Support: ${liq_result['nearest_support']:,.2f}")
    print(f"Nearest Resistance: ${liq_result['nearest_resistance']:,.2f}")
    print(f"Significant Levels: {len(liq_result['liquidity_levels'])}")
