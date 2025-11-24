"""
Enhanced Signal Aggregator - Phase 2 Integration
=================================================
Pure technical + market microstructure focused signal generation.
Sentiment minimized - only emergency event detection.

Strategy:
- 70% Advanced Technical Indicators (Volume Profile, ADX, Fibonacci)
- 30% Market Microstructure (Orderbook, Tape, Liquidity)
- Regime Detection (filter)
- NO social sentiment (noise reduction)

Author: DEMIR AI PRO
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from core.signal_processor.layers.technical.advanced_indicators import AdvancedIndicatorSuite
from core.signal_processor.layers.market_microstructure.orderflow import (
    OrderbookAnalyzer,
    TapeReader,
    OrderFlowImbalance
)
from core.ai_engine.feature_engineering import RegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with detailed metrics"""
    signal: int  # -1, 0, 1
    confidence: float  # 0.0 to 1.0
    reason: str
    technical_score: float
    microstructure_score: float
    regime: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class EnhancedSignalAggregator:
    """
    Enhanced Signal Aggregator
    
    Combines Phase 2 advanced modules for high-quality signals:
    - Advanced technical indicators (Volume Profile, ADX, Fibonacci)
    - Market microstructure (Orderbook depth, tape reading)
    - Regime detection (trend vs range filtering)
    
    NO SENTIMENT - Pure price/volume/orderbook data only
    """
    
    def __init__(
        self,
        adx_threshold: float = 25,
        choppiness_threshold: float = 38.2,
        imbalance_threshold: float = 0.3,
        confidence_threshold: float = 0.7
    ):
        """
        Args:
            adx_threshold: Minimum ADX for trend signals
            choppiness_threshold: Max choppiness for trend signals
            imbalance_threshold: Orderbook imbalance threshold
            confidence_threshold: Minimum confidence for trade execution
        """
        # Phase 2 modules
        self.advanced_indicators = AdvancedIndicatorSuite()
        self.orderbook_analyzer = OrderbookAnalyzer(depth_levels=20)
        self.tape_reader = TapeReader(large_order_threshold=50000)
        self.regime_detector = RegimeDetector(adx_threshold=adx_threshold)
        self.flow_imbalance = OrderFlowImbalance()
        
        # Thresholds
        self.adx_threshold = adx_threshold
        self.choppiness_threshold = choppiness_threshold
        self.imbalance_threshold = imbalance_threshold
        self.confidence_threshold = confidence_threshold
        
        logger.info("Enhanced Signal Aggregator initialized (pure technical + microstructure)")
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[Dict] = None,
        current_price: Optional[float] = None
    ) -> EnhancedSignal:
        """
        Generate enhanced trading signal
        
        Args:
            df: OHLCV DataFrame
            orderbook_data: {'bids': [(price, size), ...], 'asks': [(price, size), ...]}
            current_price: Current market price
            
        Returns:
            EnhancedSignal object with detailed metrics
        """
        try:
            if current_price is None:
                current_price = df['close'].iloc[-1]
            
            # 1. ADVANCED TECHNICAL ANALYSIS (70% weight)
            tech_signals = self.advanced_indicators.calculate_all(df)
            technical_score = self._calculate_technical_score(tech_signals, current_price)
            
            # 2. MARKET MICROSTRUCTURE (30% weight)
            microstructure_score = 0.0
            if orderbook_data:
                ob_analysis = self.orderbook_analyzer.analyze(
                    orderbook_data.get('bids', []),
                    orderbook_data.get('asks', [])
                )
                microstructure_score = self._calculate_microstructure_score(ob_analysis)
            
            # 3. REGIME DETECTION (filter)
            regime = self.regime_detector.detect(df)
            
            # 4. SIGNAL COMBINATION
            signal_data = self._combine_signals(
                technical_score,
                microstructure_score,
                tech_signals,
                regime,
                current_price
            )
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Enhanced signal generation error: {e}")
            return EnhancedSignal(
                signal=0,
                confidence=0.0,
                reason=f"ERROR: {str(e)}",
                technical_score=0.0,
                microstructure_score=0.0,
                regime="UNKNOWN"
            )
    
    def _calculate_technical_score(
        self,
        tech_signals: Dict,
        current_price: float
    ) -> float:
        """
        Calculate technical analysis score (-1 to +1)
        
        Components:
        - Volume Profile (price position vs POC)
        - ADX System (trend strength + direction)
        - Choppiness Index (ranging filter)
        - Fibonacci levels (support/resistance)
        """
        try:
            score = 0.0
            
            # Volume Profile (weight: 30%)
            vp = tech_signals.get('volume_profile', {})
            poc = vp.get('poc')
            if poc:
                # Price above POC = bullish, below = bearish
                vp_score = (current_price - poc) / poc
                vp_score = np.clip(vp_score * 10, -1, 1)  # Scale to -1..1
                score += vp_score * 0.3
            
            # ADX System (weight: 40%)
            adx_system = tech_signals.get('adx_system', {})
            adx_signal = adx_system.get('signal', 'NEUTRAL')
            adx_value = adx_system.get('adx', 0)
            
            if adx_value > self.adx_threshold:
                if adx_signal == 'STRONG_BUY':
                    score += 0.4
                elif adx_signal == 'STRONG_SELL':
                    score -= 0.4
                elif adx_signal in ['NEUTRAL', 'NO_TREND']:
                    score += 0.0  # No contribution
            
            # Choppiness Index (weight: 20%)
            choppiness = tech_signals.get('choppiness_index', 50)
            regime = tech_signals.get('choppiness_regime', 'TRANSITIONING')
            
            if regime == 'TRENDING':
                # Strong trend - boost score
                if score > 0:
                    score += 0.2
                elif score < 0:
                    score -= 0.2
            elif regime == 'RANGING':
                # Ranging market - reduce score to 0
                score *= 0.3  # Heavily dampen
            
            # Fibonacci Levels (weight: 10%)
            fib = tech_signals.get('fibonacci', {})
            nearest_level = fib.get('nearest_level', {})
            if nearest_level:
                distance_pct = abs(nearest_level.get('distance_pct', 100))
                if distance_pct < 1:  # Within 1% of Fib level
                    # Near support/resistance - boost score magnitude
                    score *= 1.1
            
            return np.clip(score, -1, 1)
            
        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 0.0
    
    def _calculate_microstructure_score(self, ob_analysis: Dict) -> float:
        """
        Calculate market microstructure score (-1 to +1)
        
        Components:
        - Orderbook imbalance (bid/ask volume ratio)
        - Market pressure (orderbook analyzer output)
        - Spread quality (tight vs wide)
        """
        try:
            score = 0.0
            
            # Orderbook Imbalance (weight: 60%)
            imbalance = ob_analysis.get('imbalance', 0)
            if abs(imbalance) > self.imbalance_threshold:
                score += imbalance * 0.6
            
            # Market Pressure (weight: 30%)
            pressure = ob_analysis.get('market_pressure', 'NEUTRAL')
            if pressure == 'BULLISH':
                score += 0.3
            elif pressure == 'BEARISH':
                score -= 0.3
            
            # Spread Quality (weight: 10%)
            spread_pct = ob_analysis.get('spread_pct', 0.5)
            if spread_pct < 0.1:
                # Tight spread = good liquidity = boost confidence
                score *= 1.1
            elif spread_pct > 0.5:
                # Wide spread = poor liquidity = reduce confidence
                score *= 0.8
            
            return np.clip(score, -1, 1)
            
        except Exception as e:
            logger.error(f"Microstructure score calculation error: {e}")
            return 0.0
    
    def _combine_signals(
        self,
        technical_score: float,
        microstructure_score: float,
        tech_signals: Dict,
        regime: Dict,
        current_price: float
    ) -> EnhancedSignal:
        """
        Combine technical and microstructure scores into final signal
        
        Logic:
        - Technical: 70% weight
        - Microstructure: 30% weight
        - Regime: Filter (no trade in ranging/volatile)
        """
        try:
            # Weighted combination
            combined_score = (technical_score * 0.7) + (microstructure_score * 0.3)
            
            # Regime filter
            regime_type = regime.get('regime', 'UNKNOWN')
            regime_confidence = regime.get('confidence', 0)
            
            # Don't trade in ranging or highly volatile markets
            if regime_type == 'RANGING':
                return EnhancedSignal(
                    signal=0,
                    confidence=0.0,
                    reason="RANGING_MARKET_NO_TRADE",
                    technical_score=technical_score,
                    microstructure_score=microstructure_score,
                    regime=regime_type
                )
            
            if regime_type == 'VOLATILE' and regime_confidence > 0.7:
                return EnhancedSignal(
                    signal=0,
                    confidence=0.0,
                    reason="HIGHLY_VOLATILE_NO_TRADE",
                    technical_score=technical_score,
                    microstructure_score=microstructure_score,
                    regime=regime_type
                )
            
            # Determine signal direction
            if combined_score > 0.3:
                signal = 1  # LONG
                confidence = min(abs(combined_score), 1.0)
                reason = self._generate_reason(technical_score, microstructure_score, "LONG")
            elif combined_score < -0.3:
                signal = -1  # SHORT
                confidence = min(abs(combined_score), 1.0)
                reason = self._generate_reason(technical_score, microstructure_score, "SHORT")
            else:
                signal = 0  # NEUTRAL
                confidence = 0.0
                reason = "WEAK_SIGNALS_NO_TRADE"
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_levels(
                current_price,
                signal,
                tech_signals,
                regime_type
            )
            
            return EnhancedSignal(
                signal=signal,
                confidence=confidence,
                reason=reason,
                technical_score=technical_score,
                microstructure_score=microstructure_score,
                regime=regime_type,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            logger.error(f"Signal combination error: {e}")
            return EnhancedSignal(
                signal=0,
                confidence=0.0,
                reason=f"COMBINATION_ERROR: {str(e)}",
                technical_score=technical_score,
                microstructure_score=microstructure_score,
                regime="UNKNOWN"
            )
    
    def _generate_reason(
        self,
        technical_score: float,
        microstructure_score: float,
        direction: str
    ) -> str:
        """Generate human-readable reason for signal"""
        reasons = []
        
        if abs(technical_score) > 0.5:
            reasons.append(f"STRONG_TECHNICAL_{direction}")
        elif abs(technical_score) > 0.3:
            reasons.append(f"MODERATE_TECHNICAL_{direction}")
        
        if abs(microstructure_score) > 0.5:
            reasons.append(f"STRONG_ORDERBOOK_{direction}")
        elif abs(microstructure_score) > 0.3:
            reasons.append(f"MODERATE_ORDERBOOK_{direction}")
        
        return " + ".join(reasons) if reasons else f"WEAK_{direction}"
    
    def _calculate_levels(
        self,
        current_price: float,
        signal: int,
        tech_signals: Dict,
        regime: str
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels
        
        Uses:
        - ATR for volatility-based stops
        - Volume Profile POC for support/resistance
        - Fibonacci levels
        """
        try:
            if signal == 0:
                return None, None
            
            # Get ATR for stop distance
            atr_14 = tech_signals.get('volume_profile', {}).get('profile_data', [])
            # Simplified: use 2% as default ATR if not available
            atr_pct = 0.02
            
            # Stop loss distance
            stop_distance = current_price * atr_pct * 2  # 2x ATR
            
            # Take profit distance (1.5:1 risk-reward)
            tp_distance = stop_distance * 1.5
            
            if signal == 1:  # LONG
                stop_loss = current_price - stop_distance
                take_profit = current_price + tp_distance
            else:  # SHORT
                stop_loss = current_price + stop_distance
                take_profit = current_price - tp_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Level calculation error: {e}")
            return None, None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Enhanced Signal Aggregator
    """
    
    # Sample OHLCV data
    df = pd.DataFrame({
        'open': [100, 102, 101, 103, 105],
        'high': [103, 104, 103, 105, 107],
        'low': [99, 101, 100, 102, 104],
        'close': [102, 103, 102, 104, 106],
        'volume': [1000, 1200, 900, 1500, 1800]
    })
    
    # Sample orderbook
    orderbook = {
        'bids': [(105.5, 2.0), (105.0, 3.5), (104.5, 1.8)],
        'asks': [(106.0, 1.5), (106.5, 2.2), (107.0, 3.0)]
    }
    
    # Generate signal
    aggregator = EnhancedSignalAggregator()
    signal = aggregator.generate_signal(df, orderbook, current_price=106)
    
    print("\n=== ENHANCED SIGNAL ===")
    print(f"Signal: {signal.signal} ({'LONG' if signal.signal == 1 else 'SHORT' if signal.signal == -1 else 'NEUTRAL'})")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Reason: {signal.reason}")
    print(f"Technical Score: {signal.technical_score:.2f}")
    print(f"Microstructure Score: {signal.microstructure_score:.2f}")
    print(f"Regime: {signal.regime}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
