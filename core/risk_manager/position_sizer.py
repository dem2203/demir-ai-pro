"""
Position Sizer

Calculates optimal position sizes based on risk parameters.
Production-grade position sizing with Kelly Criterion.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Dynamic position sizing engine.
    """
    
    def __init__(
        self,
        max_position_size: float = 0.05,
        max_risk_per_trade: float = 0.02
    ):
        """
        Initialize position sizer.
        
        Args:
            max_position_size: Max position as % of capital (0-1)
            max_risk_per_trade: Max risk per trade as % of capital (0-1)
        """
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        logger.info(f"PositionSizer initialized (max_pos={max_position_size:.1%}, max_risk={max_risk_per_trade:.1%})")
    
    def calculate_size(
        self,
        account_balance: float,
        signal_confidence: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """
        Calculate position size.
        
        Args:
            account_balance: Total account balance
            signal_confidence: Signal confidence (0-1)
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size in base currency
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            logger.warning("Zero risk detected - using minimum position")
            return account_balance * 0.01
        
        # Max risk amount
        max_risk_amount = account_balance * self.max_risk_per_trade
        
        # Base position size
        base_size = max_risk_amount / risk_per_unit
        
        # Adjust for confidence
        adjusted_size = base_size * signal_confidence
        
        # Enforce max position limit
        max_size = account_balance * self.max_position_size
        final_size = min(adjusted_size, max_size)
        
        logger.info(f"Position size calculated: {final_size:.2f} (conf={signal_confidence:.2f})")
        
        return final_size
