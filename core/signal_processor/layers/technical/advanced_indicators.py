"""
Advanced Technical Indicators Module
=====================================
Enterprise-grade advanced technical analysis indicators for institutional trading.

Indicators:
- Volume Profile: Price level volume distribution analysis
- Choppiness Index: Market regime detection (trending vs ranging)
- Fibonacci Analyzer: Automated Fibonacci retracement/extension detection
- Advanced ADX: Full Directional Movement System with DI+/DI-
- Aroon Oscillator: Trend strength and direction
- VWMA: Volume Weighted Moving Average
- Cumulative Delta Volume: Buyer/seller pressure analysis

Author: DEMIR AI PRO
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileLevel:
    """Volume Profile price level data"""
    price: float
    volume: float
    percentage: float
    is_poc: bool  # Point of Control


@dataclass
class FibonacciLevel:
    """Fibonacci retracement/extension level"""
    level: float
    price: float
    ratio: float
    label: str


class VolumeProfile:
    """
    Volume Profile Analyzer
    
    Analyzes volume distribution across price levels to identify:
    - Point of Control (POC): Price with highest volume
    - Value Area: Price range containing 70% of volume
    - High Volume Nodes (HVN): Support/resistance levels
    - Low Volume Nodes (LVN): Breakout zones
    """
    
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
        
    def calculate(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Dict[str, any]:
        """
        Calculate Volume Profile for given price data
        
        Args:
            df: DataFrame with 'close', 'high', 'low', 'volume' columns
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary with POC, value area, HVN/LVN levels
        """
        try:
            data = df.tail(lookback).copy()
            
            # Price range
            price_min = data['low'].min()
            price_max = data['high'].max()
            
            # Create price bins
            bins = np.linspace(price_min, price_max, self.num_bins + 1)
            
            # Calculate volume per price level
            volume_profile = []
            for i in range(len(bins) - 1):
                price_level = (bins[i] + bins[i + 1]) / 2
                
                # Filter candles overlapping this price range
                mask = (data['low'] <= bins[i + 1]) & (data['high'] >= bins[i])
                level_volume = data[mask]['volume'].sum()
                
                volume_profile.append({
                    'price': price_level,
                    'volume': level_volume
                })
            
            # Convert to DataFrame
            vp_df = pd.DataFrame(volume_profile)
            total_volume = vp_df['volume'].sum()
            
            if total_volume == 0:
                logger.warning("Volume Profile: Zero total volume detected")
                return self._empty_result()
            
            vp_df['percentage'] = (vp_df['volume'] / total_volume) * 100
            
            # Find Point of Control (POC)
            poc_idx = vp_df['volume'].idxmax()
            poc_price = vp_df.loc[poc_idx, 'price']
            
            # Calculate Value Area (70% volume)
            vp_sorted = vp_df.sort_values('volume', ascending=False).reset_index(drop=True)
            cumulative_vol = 0
            value_area_prices = []
            
            for idx, row in vp_sorted.iterrows():
                cumulative_vol += row['volume']
                value_area_prices.append(row['price'])
                if cumulative_vol >= total_volume * 0.70:
                    break
            
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
            
            # Identify High Volume Nodes (top 20% volume)
            volume_threshold = vp_df['volume'].quantile(0.80)
            hvn = vp_df[vp_df['volume'] >= volume_threshold]['price'].tolist()
            
            # Identify Low Volume Nodes (bottom 20% volume)
            lvn_threshold = vp_df['volume'].quantile(0.20)
            lvn = vp_df[vp_df['volume'] <= lvn_threshold]['price'].tolist()
            
            current_price = df['close'].iloc[-1]
            
            return {
                'poc': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn,
                'lvn_levels': lvn,
                'current_price': current_price,
                'position_in_value_area': value_area_low <= current_price <= value_area_high,
                'distance_from_poc': ((current_price - poc_price) / poc_price) * 100,
                'profile_data': vp_df.to_dict('records'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Volume Profile calculation error: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'poc': None,
            'value_area_high': None,
            'value_area_low': None,
            'hvn_levels': [],
            'lvn_levels': [],
            'current_price': None,
            'position_in_value_area': False,
            'distance_from_poc': 0,
            'profile_data': [],
            'timestamp': datetime.utcnow().isoformat()
        }


class ChoppinessIndex:
    """
    Choppiness Index - Market Regime Detection
    
    Measures market choppiness to determine if market is:
    - Trending (CI < 38.2)
    - Transitioning (38.2 <= CI <= 61.8)
    - Ranging/Choppy (CI > 61.8)
    
    Values range from 0 to 100.
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Choppiness Index
        
        Formula:
        CI = 100 * LOG10(SUM(ATR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / LOG10(n)
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            Series with Choppiness Index values
        """
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Sum of ATR over period
            sum_tr = tr.rolling(window=self.period).sum()
            
            # Highest high - Lowest low over period
            high_low_range = (
                df['high'].rolling(window=self.period).max() -
                df['low'].rolling(window=self.period).min()
            )
            
            # Choppiness Index formula
            ci = 100 * np.log10(sum_tr / high_low_range) / np.log10(self.period)
            
            return ci.fillna(50)  # Neutral value for initial periods
            
        except Exception as e:
            logger.error(f"Choppiness Index calculation error: {e}")
            return pd.Series([50] * len(df))
    
    def get_regime(self, ci_value: float) -> str:
        """Classify market regime based on CI value"""
        if ci_value < 38.2:
            return "TRENDING"
        elif ci_value > 61.8:
            return "RANGING"
        else:
            return "TRANSITIONING"


class FibonacciAnalyzer:
    """
    Automated Fibonacci Retracement/Extension Analyzer
    
    Identifies swing highs/lows and calculates Fibonacci levels:
    - Retracement: 0.236, 0.382, 0.500, 0.618, 0.786
    - Extension: 1.272, 1.414, 1.618, 2.000, 2.618
    """
    
    RETRACEMENT_LEVELS = [0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000]
    EXTENSION_LEVELS = [1.272, 1.414, 1.618, 2.000, 2.618]
    
    def __init__(self, swing_period: int = 10):
        self.swing_period = swing_period
        
    def find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
        """
        Find most recent significant swing high and swing low
        
        Returns:
            Tuple of (swing_high, swing_low) as (index, price)
        """
        try:
            data = df.tail(lookback).copy()
            
            # Find swing highs (local maxima)
            data['swing_high'] = data['high'].rolling(
                window=self.swing_period * 2 + 1,
                center=True
            ).max() == data['high']
            
            # Find swing lows (local minima)
            data['swing_low'] = data['low'].rolling(
                window=self.swing_period * 2 + 1,
                center=True
            ).min() == data['low']
            
            # Get most recent swing high
            swing_highs = data[data['swing_high']].tail(2)
            swing_high = None
            if len(swing_highs) > 0:
                idx = swing_highs.index[-1]
                swing_high = (idx, swing_highs.loc[idx, 'high'])
            
            # Get most recent swing low
            swing_lows = data[data['swing_low']].tail(2)
            swing_low = None
            if len(swing_lows) > 0:
                idx = swing_lows.index[-1]
                swing_low = (idx, swing_lows.loc[idx, 'low'])
            
            return swing_high, swing_low
            
        except Exception as e:
            logger.error(f"Swing point detection error: {e}")
            return None, None
    
    def calculate_levels(
        self,
        df: pd.DataFrame,
        trend: str = "auto"
    ) -> Dict[str, List[FibonacciLevel]]:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            df: Price DataFrame
            trend: "up", "down", or "auto" (detect automatically)
            
        Returns:
            Dictionary with retracement and extension levels
        """
        try:
            swing_high, swing_low = self.find_swing_points(df)
            
            if swing_high is None or swing_low is None:
                logger.warning("Fibonacci: Could not identify swing points")
                return {'retracement': [], 'extension': []}
            
            # Determine trend
            if trend == "auto":
                trend = "up" if swing_low[0] < swing_high[0] else "down"
            
            # Calculate levels based on trend
            if trend == "up":
                # Uptrend: retracement from high to low
                high_price = swing_high[1]
                low_price = swing_low[1]
                price_range = high_price - low_price
                
                retracements = [
                    FibonacciLevel(
                        level=level,
                        price=high_price - (price_range * level),
                        ratio=level,
                        label=f"{level:.1%} Retracement"
                    )
                    for level in self.RETRACEMENT_LEVELS
                ]
                
                extensions = [
                    FibonacciLevel(
                        level=level,
                        price=high_price + (price_range * (level - 1)),
                        ratio=level,
                        label=f"{level:.3f} Extension"
                    )
                    for level in self.EXTENSION_LEVELS
                ]
                
            else:
                # Downtrend: retracement from low to high
                high_price = swing_high[1]
                low_price = swing_low[1]
                price_range = high_price - low_price
                
                retracements = [
                    FibonacciLevel(
                        level=level,
                        price=low_price + (price_range * level),
                        ratio=level,
                        label=f"{level:.1%} Retracement"
                    )
                    for level in self.RETRACEMENT_LEVELS
                ]
                
                extensions = [
                    FibonacciLevel(
                        level=level,
                        price=low_price - (price_range * (level - 1)),
                        ratio=level,
                        label=f"{level:.3f} Extension"
                    )
                    for level in self.EXTENSION_LEVELS
                ]
            
            current_price = df['close'].iloc[-1]
            
            # Find nearest level
            all_levels = retracements + extensions
            nearest = min(all_levels, key=lambda x: abs(x.price - current_price))
            
            return {
                'retracement': [
                    {
                        'level': r.level,
                        'price': r.price,
                        'ratio': r.ratio,
                        'label': r.label,
                        'distance_pct': ((current_price - r.price) / current_price) * 100
                    }
                    for r in retracements
                ],
                'extension': [
                    {
                        'level': e.level,
                        'price': e.price,
                        'ratio': e.ratio,
                        'label': e.label,
                        'distance_pct': ((current_price - e.price) / current_price) * 100
                    }
                    for e in extensions
                ],
                'nearest_level': {
                    'price': nearest.price,
                    'label': nearest.label,
                    'distance_pct': ((current_price - nearest.price) / current_price) * 100
                },
                'trend': trend,
                'swing_high': swing_high[1] if swing_high else None,
                'swing_low': swing_low[1] if swing_low else None,
                'current_price': current_price,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return {'retracement': [], 'extension': []}


class AdvancedADX:
    """
    Advanced ADX with Directional Movement System
    
    Complete implementation of:
    - ADX: Average Directional Index (trend strength)
    - +DI: Positive Directional Indicator
    - -DI: Negative Directional Indicator
    
    Interpretation:
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging
    - +DI > -DI: Bullish
    - -DI > +DI: Bearish
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, +DI, -DI
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            DataFrame with ADX, plus_di, minus_di columns
        """
        try:
            result = df.copy()
            
            # Calculate True Range
            high_low = result['high'] - result['low']
            high_close = np.abs(result['high'] - result['close'].shift())
            low_close = np.abs(result['low'] - result['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            plus_dm = result['high'].diff()
            minus_dm = -result['low'].diff()
            
            # Rules for directional movement
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            plus_dm[(plus_dm > 0) & (plus_dm <= minus_dm)] = 0
            minus_dm[(minus_dm > 0) & (minus_dm <= plus_dm)] = 0
            
            # Smooth with Wilder's moving average
            atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(alpha=1/self.period, adjust=False).mean() / atr
            minus_di = 100 * minus_dm.ewm(alpha=1/self.period, adjust=False).mean() / atr
            
            # Calculate DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX
            adx = dx.ewm(alpha=1/self.period, adjust=False).mean()
            
            result['adx'] = adx
            result['plus_di'] = plus_di
            result['minus_di'] = minus_di
            
            return result[['adx', 'plus_di', 'minus_di']]
            
        except Exception as e:
            logger.error(f"Advanced ADX calculation error: {e}")
            return pd.DataFrame({
                'adx': [0] * len(df),
                'plus_di': [0] * len(df),
                'minus_di': [0] * len(df)
            })
    
    def get_signal(self, adx: float, plus_di: float, minus_di: float) -> Dict[str, any]:
        """
        Generate trading signal from ADX system
        
        Returns:
            Dictionary with trend strength and direction
        """
        # Trend strength
        if adx > 50:
            strength = "VERY_STRONG"
        elif adx > 25:
            strength = "STRONG"
        elif adx > 20:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        # Trend direction
        if plus_di > minus_di:
            direction = "BULLISH"
            di_spread = plus_di - minus_di
        else:
            direction = "BEARISH"
            di_spread = minus_di - plus_di
        
        # Overall signal
        if adx > 25 and plus_di > minus_di:
            signal = "STRONG_BUY"
        elif adx > 25 and minus_di > plus_di:
            signal = "STRONG_SELL"
        elif adx < 20:
            signal = "NO_TREND"
        else:
            signal = "NEUTRAL"
        
        return {
            'signal': signal,
            'trend_strength': strength,
            'trend_direction': direction,
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'di_spread': di_spread,
            'timestamp': datetime.utcnow().isoformat()
        }


class AroonOscillator:
    """
    Aroon Oscillator - Trend Strength and Direction
    
    Measures time since highest high and lowest low.
    Values range from -100 to +100.
    
    - Aroon > 50: Strong uptrend
    - Aroon < -50: Strong downtrend
    - Aroon near 0: Consolidation
    """
    
    def __init__(self, period: int = 25):
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Aroon Oscillator"""
        try:
            # Aroon Up: ((period - periods since highest high) / period) * 100
            aroon_up = df['high'].rolling(window=self.period).apply(
                lambda x: ((self.period - (self.period - 1 - x.argmax())) / self.period) * 100
            )
            
            # Aroon Down: ((period - periods since lowest low) / period) * 100
            aroon_down = df['low'].rolling(window=self.period).apply(
                lambda x: ((self.period - (self.period - 1 - x.argmin())) / self.period) * 100
            )
            
            # Aroon Oscillator = Aroon Up - Aroon Down
            aroon_osc = aroon_up - aroon_down
            
            return aroon_osc.fillna(0)
            
        except Exception as e:
            logger.error(f"Aroon Oscillator calculation error: {e}")
            return pd.Series([0] * len(df))


class VWMA:
    """
    Volume Weighted Moving Average
    
    Places more weight on periods with higher volume.
    More responsive to volume surges than simple MA.
    """
    
    def __init__(self, period: int = 20):
        self.period = period
        
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWMA"""
        try:
            # VWMA = SUM(price * volume) / SUM(volume)
            pv = df['close'] * df['volume']
            vwma = pv.rolling(window=self.period).sum() / df['volume'].rolling(window=self.period).sum()
            
            return vwma.fillna(df['close'])
            
        except Exception as e:
            logger.error(f"VWMA calculation error: {e}")
            return df['close'].copy()


class CumulativeDeltaVolume:
    """
    Cumulative Delta Volume - Buyer/Seller Pressure
    
    Tracks net buying/selling pressure by comparing:
    - Uptick volume (close > open): Buying pressure
    - Downtick volume (close < open): Selling pressure
    
    Cumulative Delta = SUM(uptick volume - downtick volume)
    """
    
    def calculate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Cumulative Delta Volume"""
        try:
            # Determine tick direction
            df = df.copy()
            df['price_change'] = df['close'] - df['open']
            
            # Assign volume to buyers or sellers
            df['delta'] = np.where(
                df['price_change'] > 0,
                df['volume'],  # Buying volume
                np.where(
                    df['price_change'] < 0,
                    -df['volume'],  # Selling volume
                    0  # Neutral
                )
            )
            
            # Cumulative sum
            cumulative_delta = df['delta'].cumsum()
            
            return cumulative_delta.fillna(0)
            
        except Exception as e:
            logger.error(f"Cumulative Delta Volume calculation error: {e}")
            return pd.Series([0] * len(df))


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

class AdvancedIndicatorSuite:
    """
    Unified interface for all advanced technical indicators
    
    Usage:
        suite = AdvancedIndicatorSuite()
        results = suite.calculate_all(df)
    """
    
    def __init__(self):
        self.volume_profile = VolumeProfile(num_bins=50)
        self.choppiness = ChoppinessIndex(period=14)
        self.fibonacci = FibonacciAnalyzer(swing_period=10)
        self.adx = AdvancedADX(period=14)
        self.aroon = AroonOscillator(period=25)
        self.vwma = VWMA(period=20)
        self.delta_volume = CumulativeDeltaVolume()
        
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all advanced indicators
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with all indicator results
        """
        try:
            logger.info(f"Calculating advanced indicators for {len(df)} candles")
            
            results = {
                'volume_profile': self.volume_profile.calculate(df),
                'choppiness_index': self.choppiness.calculate(df).iloc[-1],
                'choppiness_regime': self.choppiness.get_regime(
                    self.choppiness.calculate(df).iloc[-1]
                ),
                'fibonacci': self.fibonacci.calculate_levels(df),
                'adx_system': None,
                'aroon_oscillator': self.aroon.calculate(df).iloc[-1],
                'vwma': self.vwma.calculate(df).iloc[-1],
                'cumulative_delta': self.delta_volume.calculate(df).iloc[-1],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # ADX with signals
            adx_data = self.adx.calculate(df)
            if len(adx_data) > 0:
                adx_last = adx_data.iloc[-1]
                results['adx_system'] = self.adx.get_signal(
                    adx_last['adx'],
                    adx_last['plus_di'],
                    adx_last['minus_di']
                )
            
            logger.info("Advanced indicators calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"Advanced indicator suite calculation error: {e}")
            return {}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Advanced Technical Indicators
    """
    import pandas as pd
    
    # Sample OHLCV data
    df = pd.DataFrame({
        'open': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
        'high': [103, 104, 103, 105, 107, 106, 108, 110, 109, 111],
        'low': [99, 101, 100, 102, 104, 103, 105, 107, 106, 108],
        'close': [102, 103, 102, 104, 106, 105, 107, 109, 108, 110],
        'volume': [1000, 1200, 900, 1500, 1800, 1100, 1600, 2000, 1300, 1700]
    })
    
    # Initialize suite
    suite = AdvancedIndicatorSuite()
    
    # Calculate all indicators
    results = suite.calculate_all(df)
    
    print("\n=== ADVANCED INDICATORS RESULTS ===\n")
    print(f"Volume Profile POC: {results['volume_profile']['poc']}")
    print(f"Choppiness Index: {results['choppiness_index']:.2f}")
    print(f"Market Regime: {results['choppiness_regime']}")
    print(f"ADX Signal: {results['adx_system']['signal']}")
    print(f"Aroon Oscillator: {results['aroon_oscillator']:.2f}")
    print(f"VWMA: {results['vwma']:.2f}")
    print(f"Cumulative Delta: {results['cumulative_delta']:.0f}")
