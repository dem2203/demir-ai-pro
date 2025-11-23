"""
Signal Processor Module

Generates, validates, and manages trading signals.
Multi-group signal consensus engine.
"""

from .generator import SignalGenerator
from .validator import SignalValidator
from .consensus import ConsensusEngine

__all__ = ['SignalGenerator', 'SignalValidator', 'ConsensusEngine']
