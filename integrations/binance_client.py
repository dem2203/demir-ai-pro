#!/usr/bin/env python3
"""Binance Client Wrapper - Compatibility Layer

Provides get_binance_client() function for backward compatibility.
Delegates to binance_integration.py for actual implementation.
"""

from integrations.binance_integration import get_binance

# Compatibility wrapper
def get_binance_client():
    """
    Get Binance client instance (compatibility wrapper)
    Delegates to binance_integration.get_binance()
    """
    return get_binance()

__all__ = ['get_binance_client']
