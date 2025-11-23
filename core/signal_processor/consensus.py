"""
Consensus Engine

Aggregates signals from multiple groups into consensus.
Production-grade consensus algorithm.
"""

import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class ConsensusEngine:
    """
    Multi-group signal consensus engine.
    
    Combines signals from:
    - Technical group
    - Sentiment group
    - ML group
    - On-chain group
    - Risk group
    """
    
    def __init__(self, min_agreement: float = 0.6):
        """
        Initialize consensus engine.
        
        Args:
            min_agreement: Minimum agreement threshold (0-1)
        """
        self.min_agreement = min_agreement
        logger.info(f"ConsensusEngine initialized (min_agreement={min_agreement})")
    
    def calculate_consensus(
        self,
        group_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate consensus from group signals.
        
        Args:
            group_signals: Dict of signals from each group
            
        Returns:
            Consensus signal
        """
        if not group_signals:
            return self._neutral_consensus()
        
        # Extract directions and confidences
        directions = []
        confidences = []
        strengths = []
        
        for group_name, signal in group_signals.items():
            directions.append(signal.get('direction', 'NEUTRAL'))
            confidences.append(signal.get('confidence', 0))
            strengths.append(signal.get('strength', 0))
        
        # Calculate agreement
        long_count = directions.count('LONG')
        short_count = directions.count('SHORT')
        neutral_count = directions.count('NEUTRAL')
        
        total = len(directions)
        long_ratio = long_count / total
        short_ratio = short_count / total
        
        # Determine consensus direction
        if long_ratio >= self.min_agreement:
            consensus_direction = 'LONG'
        elif short_ratio >= self.min_agreement:
            consensus_direction = 'SHORT'
        else:
            consensus_direction = 'NEUTRAL'
        
        # Calculate consensus confidence
        avg_confidence = np.mean(confidences)
        avg_strength = np.mean(strengths)
        
        consensus = {
            'direction': consensus_direction,
            'confidence': avg_confidence,
            'strength': avg_strength,
            'active_groups': len(group_signals),
            'agreement_ratio': max(long_ratio, short_ratio),
            'group_breakdown': {
                'long': long_count,
                'short': short_count,
                'neutral': neutral_count
            }
        }
        
        logger.info(f"Consensus: {consensus_direction} (conf={avg_confidence:.2f})")
        
        return consensus
    
    def _neutral_consensus(self) -> Dict[str, Any]:
        """Return neutral consensus when no signals available."""
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.0,
            'strength': 0.0,
            'active_groups': 0,
            'agreement_ratio': 0.0,
            'group_breakdown': {'long': 0, 'short': 0, 'neutral': 0}
        }
