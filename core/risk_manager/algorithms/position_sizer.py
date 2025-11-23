#!/usr/bin/env python3
"""
Dynamic Position Sizer

Advanced position sizing with Kelly Criterion.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DynamicPositionSizer:
    """
    Kelly Criterion-based position sizing.
    """
    
    def __init__(self, max_position: float = 0.05):
        self.max_position = max_position
        logger.info("✅ Dynamic Position Sizer initialized")
    
    def calculate(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade %
            avg_loss: Average losing trade %
            
        Returns:
            Position size as % of capital
        """
        try:
            if avg_loss == 0:
                return self.max_position
            
            # Kelly Criterion: f = (p*b - q) / b
            # p = win rate, q = loss rate, b = win/loss ratio
            b = avg_win / abs(avg_loss)
            kelly = (win_rate * b - (1 - win_rate)) / b
            
            # Use half Kelly for safety
            kelly = kelly / 2
            
            # Cap at max position
            position = min(kelly, self.max_position)
            
            return max(0, position)
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.01
