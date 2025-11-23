#!/usr/bin/env python3
"""
Dynamic Stop Loss Calculator

ATR-based dynamic stop loss levels.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

class DynamicStopLoss:
    """
    ATR-based stop loss calculator.
    """
    
    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier
        logger.info("✅ Dynamic Stop Loss initialized")
    
    def calculate(self, entry_price: float, atr: float, direction: str) -> float:
        """
        Calculate stop loss level.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Stop loss price
        """
        try:
            if direction == 'LONG':
                return entry_price - (atr * self.atr_multiplier)
            else:  # SHORT
                return entry_price + (atr * self.atr_multiplier)
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            return entry_price * 0.98  # 2% default
