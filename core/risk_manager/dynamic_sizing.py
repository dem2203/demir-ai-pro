"""
Dynamic Position Sizing Module
===============================
ATR-based position sizing with regime adjustments.
Volatility-adaptive risk management for professional trading.

Strategy:
- Base risk: 1-2% per trade
- ATR-based stop distance
- Regime-adjusted sizing (reduce in volatile, increase in trending)
- Kelly Criterion optional

Author: DEMIR AI PRO
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing calculation result"""
    size: float  # Position size in base currency
    size_usd: float  # Position size in USD
    risk_amount: float  # Dollar risk amount
    stop_distance: float  # Stop loss distance
    leverage_used: float  # Actual leverage
    reason: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class DynamicPositionSizer:
    """
    Dynamic Position Sizing
    
    Features:
    - ATR-based stop loss distance
    - Regime-adjusted sizing
    - Volatility scaling
    - Max risk per trade enforcement
    - Kelly Criterion (optional)
    """
    
    def __init__(
        self,
        max_risk_pct: float = 0.02,  # 2% max risk per trade
        max_leverage: float = 3.0,
        use_kelly: bool = False,
        kelly_fraction: float = 0.25  # Fractional Kelly (conservative)
    ):
        """
        Args:
            max_risk_pct: Maximum risk per trade (as decimal)
            max_leverage: Maximum leverage allowed
            use_kelly: Whether to use Kelly Criterion
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.max_risk_pct = max_risk_pct
        self.max_leverage = max_leverage
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        
        logger.info(f"Dynamic Position Sizer initialized: max_risk={max_risk_pct:.1%}, "
                   f"max_leverage={max_leverage}x, kelly={use_kelly}")
    
    def calculate_position(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        atr: float,
        regime: str = "UNKNOWN",
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size
        
        Args:
            account_balance: Current account balance
            entry_price: Entry price
            stop_loss_price: Stop loss price
            atr: Average True Range (in price units)
            regime: Market regime (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win amount (for Kelly)
            avg_loss: Average loss amount (for Kelly)
            
        Returns:
            PositionSize object with calculated size and metrics
        """
        try:
            # 1. BASE RISK AMOUNT
            base_risk_amount = account_balance * self.max_risk_pct
            
            # 2. STOP DISTANCE
            stop_distance = abs(entry_price - stop_loss_price)
            
            if stop_distance == 0:
                logger.warning("Stop distance is zero, using ATR")
                stop_distance = atr * 2  # 2x ATR as default
            
            # 3. BASE POSITION SIZE
            # Size = Risk Amount / Stop Distance
            base_size = base_risk_amount / stop_distance
            
            # 4. REGIME ADJUSTMENT
            regime_multiplier = self._get_regime_multiplier(regime)
            adjusted_size = base_size * regime_multiplier
            
            # 5. VOLATILITY ADJUSTMENT
            atr_pct = atr / entry_price
            volatility_multiplier = self._get_volatility_multiplier(atr_pct)
            adjusted_size *= volatility_multiplier
            
            # 6. KELLY CRITERION (if enabled)
            if self.use_kelly and win_rate and avg_win and avg_loss:
                kelly_size = self._calculate_kelly_size(
                    account_balance,
                    entry_price,
                    win_rate,
                    avg_win,
                    avg_loss
                )
                # Use minimum of Kelly and ATR-based size
                adjusted_size = min(adjusted_size, kelly_size)
            
            # 7. ENFORCE MAX LEVERAGE
            max_position_value = account_balance * self.max_leverage
            max_size = max_position_value / entry_price
            final_size = min(adjusted_size, max_size)
            
            # 8. CALCULATE METRICS
            position_value = final_size * entry_price
            leverage_used = position_value / account_balance
            risk_amount = final_size * stop_distance
            
            # 9. GENERATE REASON
            reason = self._generate_reason(
                regime,
                regime_multiplier,
                volatility_multiplier,
                leverage_used
            )
            
            return PositionSize(
                size=final_size,
                size_usd=position_value,
                risk_amount=risk_amount,
                stop_distance=stop_distance,
                leverage_used=leverage_used,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            # Return minimal position on error
            return PositionSize(
                size=0.0,
                size_usd=0.0,
                risk_amount=0.0,
                stop_distance=0.0,
                leverage_used=0.0,
                reason=f"ERROR: {str(e)}"
            )
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Get position size multiplier based on market regime
        
        Logic:
        - Strong trend: Increase size (1.2x)
        - Weak trend: Normal size (1.0x)
        - Volatile: Reduce size (0.5x)
        - Ranging: Reduce size (0.7x)
        """
        multipliers = {
            'TRENDING_UP': 1.2,
            'TRENDING_DOWN': 1.2,
            'RANGING': 0.7,
            'VOLATILE': 0.5,
            'TRANSITIONING': 0.9,
            'UNKNOWN': 1.0
        }
        
        return multipliers.get(regime, 1.0)
    
    def _get_volatility_multiplier(self, atr_pct: float) -> float:
        """
        Get position size multiplier based on volatility
        
        Logic:
        - Low volatility (<1%): Increase size (1.2x)
        - Normal volatility (1-3%): Normal size (1.0x)
        - High volatility (3-5%): Reduce size (0.7x)
        - Very high volatility (>5%): Reduce size heavily (0.5x)
        """
        if atr_pct < 0.01:
            return 1.2
        elif atr_pct < 0.03:
            return 1.0
        elif atr_pct < 0.05:
            return 0.7
        else:
            return 0.5
    
    def _calculate_kelly_size(
        self,
        account_balance: float,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion
        
        Formula:
        Kelly % = W - [(1 - W) / R]
        Where:
        - W = Win rate
        - R = Win/Loss ratio
        
        Returns fractional Kelly for conservatism
        """
        try:
            # Win/Loss ratio
            if avg_loss == 0:
                return 0
            
            win_loss_ratio = abs(avg_win / avg_loss)
            
            # Kelly percentage
            kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
            
            # Fractional Kelly (conservative)
            kelly_pct *= self.kelly_fraction
            
            # Ensure positive and reasonable
            kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
            
            # Calculate size
            position_value = account_balance * kelly_pct
            size = position_value / entry_price
            
            return size
            
        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return 0
    
    def _generate_reason(self, regime: str, regime_mult: float, vol_mult: float, leverage: float) -> str:
        """Generate human-readable reason for position size"""
        reasons = []
        
        if regime_mult > 1.0:
            reasons.append(f"INCREASED_{regime}")
        elif regime_mult < 1.0:
            reasons.append(f"REDUCED_{regime}")
        
        if vol_mult < 1.0:
            reasons.append("HIGH_VOLATILITY_REDUCTION")
        elif vol_mult > 1.0:
            reasons.append("LOW_VOLATILITY_INCREASE")
        
        reasons.append(f"LEVERAGE_{leverage:.1f}x")
        
        return " | ".join(reasons) if reasons else "NORMAL_SIZING"


class StopLossCalculator:
    """
    Advanced Stop Loss Calculator
    
    Methods:
    - ATR-based stops
    - Support/Resistance based stops
    - Trailing stops
    - Time-based stops
    """
    
    def __init__(self, atr_multiplier: float = 2.0):
        """
        Args:
            atr_multiplier: ATR multiplier for stop distance (2.0 = 2x ATR)
        """
        self.atr_multiplier = atr_multiplier
    
    def calculate_atr_stop(
        self,
        entry_price: float,
        atr: float,
        position_side: str
    ) -> float:
        """
        Calculate ATR-based stop loss
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            position_side: 'LONG' or 'SHORT'
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * self.atr_multiplier
        
        if position_side == 'LONG':
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        atr: float,
        position_side: str
    ) -> float:
        """
        Calculate trailing stop loss
        
        Trails by ATR distance from highest/lowest price
        """
        trail_distance = atr * self.atr_multiplier
        
        if position_side == 'LONG':
            return highest_price - trail_distance
        else:  # SHORT
            return highest_price + trail_distance
    
    def calculate_support_resistance_stop(
        self,
        entry_price: float,
        support_level: Optional[float],
        resistance_level: Optional[float],
        position_side: str,
        buffer_pct: float = 0.005  # 0.5% buffer
    ) -> Optional[float]:
        """
        Calculate stop based on support/resistance levels
        
        Args:
            entry_price: Entry price
            support_level: Support price level
            resistance_level: Resistance price level
            position_side: 'LONG' or 'SHORT'
            buffer_pct: Buffer below/above support/resistance
            
        Returns:
            Stop loss price or None if levels not available
        """
        if position_side == 'LONG' and support_level:
            # Place stop below support
            buffer = support_level * buffer_pct
            return support_level - buffer
        elif position_side == 'SHORT' and resistance_level:
            # Place stop above resistance
            buffer = resistance_level * buffer_pct
            return resistance_level + buffer
        
        return None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Dynamic Position Sizing
    """
    
    # Account parameters
    account_balance = 10000  # $10,000
    entry_price = 50000  # $50,000 BTC
    atr = 1500  # $1,500 ATR
    
    # Initialize sizer
    sizer = DynamicPositionSizer(
        max_risk_pct=0.02,  # 2% risk
        max_leverage=3.0,
        use_kelly=False
    )
    
    # Calculate stop loss
    stop_calculator = StopLossCalculator(atr_multiplier=2.0)
    stop_loss = stop_calculator.calculate_atr_stop(entry_price, atr, 'LONG')
    
    print("\n=== POSITION SIZING EXAMPLES ===")
    
    # Example 1: Trending market
    print("\n1. TRENDING MARKET:")
    pos1 = sizer.calculate_position(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        atr=atr,
        regime='TRENDING_UP'
    )
    print(f"Size: {pos1.size:.4f} BTC (${pos1.size_usd:,.0f})")
    print(f"Risk: ${pos1.risk_amount:,.0f}")
    print(f"Leverage: {pos1.leverage_used:.2f}x")
    print(f"Reason: {pos1.reason}")
    
    # Example 2: Volatile market
    print("\n2. VOLATILE MARKET:")
    pos2 = sizer.calculate_position(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        atr=atr * 2,  # Higher volatility
        regime='VOLATILE'
    )
    print(f"Size: {pos2.size:.4f} BTC (${pos2.size_usd:,.0f})")
    print(f"Risk: ${pos2.risk_amount:,.0f}")
    print(f"Leverage: {pos2.leverage_used:.2f}x")
    print(f"Reason: {pos2.reason}")
    
    # Example 3: Trailing stop
    print("\n3. TRAILING STOP:")
    current_price = 52000
    highest_price = 52500
    trailing_stop = stop_calculator.calculate_trailing_stop(
        current_price, highest_price, atr, 'LONG'
    )
    print(f"Entry: ${entry_price:,}")
    print(f"Current: ${current_price:,}")
    print(f"Highest: ${highest_price:,}")
    print(f"Trailing Stop: ${trailing_stop:,.0f}")
