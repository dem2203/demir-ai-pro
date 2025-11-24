"""
Emergency Event Detector
=========================
Minimal event detection for critical market events only.
NO social sentiment - only emergency situations.

Purpose:
- Detect critical events (hacks, regulations, exchange halts)
- Automatically close positions on emergency
- Risk management focused

Author: DEMIR AI PRO
Version: 8.0
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EmergencyEvent:
    """Critical event that requires immediate action"""
    event_type: str
    severity: str  # 'CRITICAL', 'HIGH'
    title: str
    action_required: str  # 'CLOSE_ALL_POSITIONS', 'HALT_TRADING', 'MONITOR'
    timestamp: str
    source: str


class EmergencyEventDetector:
    """
    Emergency Event Detector
    
    Detects ONLY critical events that require immediate trading action:
    - Exchange hacks/exploits
    - Regulatory actions (SEC bans, delisting)
    - Network/protocol critical bugs
    - Exchange halts/outages
    
    NO social sentiment analysis - pure emergency detection
    """
    
    # Critical keywords that trigger emergency actions
    CRITICAL_KEYWORDS = {
        'HACK': [
            r'\bhack(ed|s)?\b',
            r'\bexploit(ed|s)?\b',
            r'\bstolen\b',
            r'\bbreach(ed)?\b',
            r'\b51%\s+attack\b',
            r'\brug\s+pull\b'
        ],
        'REGULATORY': [
            r'\bsec\s+(ban|halt|investigation)\b',
            r'\bdelisting\b',
            r'\bregulator\s+shutdown\b',
            r'\bemergency\s+regulation\b',
            r'\btrading\s+suspended\b'
        ],
        'NETWORK': [
            r'\bnetwork\s+halt(ed)?\b',
            r'\bcritical\s+bug\b',
            r'\bhard\s+fork\s+failed\b',
            r'\bconsensus\s+failure\b',
            r'\bchain\s+split\b'
        ],
        'EXCHANGE': [
            r'\bexchange\s+hack(ed)?\b',
            r'\bexchange\s+halt(ed)?\b',
            r'\bwithdraw(al)?\s+suspended\b',
            r'\binsolvency\b',
            r'\bbankrupt(cy)?\b'
        ]
    }
    
    def __init__(self, history_length: int = 100):
        """
        Args:
            history_length: Number of recent events to keep in memory
        """
        self.detected_events = deque(maxlen=history_length)
        self.compile_patterns()
        
        logger.info("Emergency Event Detector initialized (critical events only)")
    
    def compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.compiled_patterns = {}
        
        for event_type, patterns in self.CRITICAL_KEYWORDS.items():
            self.compiled_patterns[event_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def check_emergency(
        self,
        text: str,
        source: str = "unknown"
    ) -> Optional[EmergencyEvent]:
        """
        Check if text describes an emergency event
        
        Args:
            text: Text to analyze (news headline, alert, etc.)
            source: Source of the text
            
        Returns:
            EmergencyEvent if critical event detected, None otherwise
        """
        try:
            text_lower = text.lower()
            
            # Check each event type
            for event_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(text_lower):
                        # Critical event detected
                        event = self._create_emergency_event(
                            event_type,
                            text,
                            source
                        )
                        
                        self.detected_events.append(event)
                        
                        logger.critical(
                            f"EMERGENCY EVENT DETECTED: {event_type} - {text[:100]}"
                        )
                        
                        return event
            
            return None
            
        except Exception as e:
            logger.error(f"Emergency check error: {e}")
            return None
    
    def _create_emergency_event(
        self,
        event_type: str,
        text: str,
        source: str
    ) -> EmergencyEvent:
        """
        Create EmergencyEvent object with appropriate action
        """
        # Determine severity and action based on event type
        if event_type in ['HACK', 'EXCHANGE']:
            severity = 'CRITICAL'
            action = 'CLOSE_ALL_POSITIONS'
        elif event_type == 'REGULATORY':
            severity = 'CRITICAL'
            action = 'CLOSE_ALL_POSITIONS'
        elif event_type == 'NETWORK':
            severity = 'HIGH'
            action = 'HALT_TRADING'
        else:
            severity = 'HIGH'
            action = 'MONITOR'
        
        return EmergencyEvent(
            event_type=event_type,
            severity=severity,
            title=text[:150],  # First 150 chars
            action_required=action,
            timestamp=datetime.utcnow().isoformat(),
            source=source
        )
    
    def get_recent_emergencies(
        self,
        hours: int = 24
    ) -> List[EmergencyEvent]:
        """
        Get recent emergency events
        
        Args:
            hours: Time window in hours
            
        Returns:
            List of recent emergency events
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent = [
            event for event in self.detected_events
            if datetime.fromisoformat(event.timestamp) >= cutoff
        ]
        
        return recent
    
    def is_trading_halted(self) -> bool:
        """
        Check if trading should be halted due to recent emergencies
        
        Returns:
            True if trading should be halted
        """
        recent_emergencies = self.get_recent_emergencies(hours=1)
        
        for event in recent_emergencies:
            if event.action_required in ['CLOSE_ALL_POSITIONS', 'HALT_TRADING']:
                return True
        
        return False
    
    def get_active_emergency_actions(self) -> List[str]:
        """
        Get list of active emergency actions required
        
        Returns:
            List of action strings
        """
        recent = self.get_recent_emergencies(hours=2)
        actions = [e.action_required for e in recent]
        return list(set(actions))  # Unique actions


class EmergencyActionHandler:
    """
    Handles emergency actions based on detected events
    
    Actions:
    - CLOSE_ALL_POSITIONS: Immediately close all open positions
    - HALT_TRADING: Stop opening new positions
    - MONITOR: Log and notify, but continue trading
    """
    
    def __init__(self, detector: EmergencyEventDetector):
        self.detector = detector
        self.trading_halted = False
        
    def handle_event(self, event: EmergencyEvent) -> Dict[str, any]:
        """
        Handle emergency event
        
        Args:
            event: EmergencyEvent object
            
        Returns:
            Dictionary with action results
        """
        logger.critical(f"HANDLING EMERGENCY: {event.event_type} - {event.action_required}")
        
        if event.action_required == 'CLOSE_ALL_POSITIONS':
            return self._close_all_positions(event)
        elif event.action_required == 'HALT_TRADING':
            return self._halt_trading(event)
        else:  # MONITOR
            return self._monitor_event(event)
    
    def _close_all_positions(self, event: EmergencyEvent) -> Dict:
        """
        Close all open positions immediately
        """
        logger.critical("EMERGENCY: Closing all positions")
        
        # In production: Call position manager to close all
        # position_manager.close_all_positions(reason=event.title)
        
        self.trading_halted = True
        
        return {
            'action': 'POSITIONS_CLOSED',
            'reason': event.title,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True  # In production, check actual closure
        }
    
    def _halt_trading(self, event: EmergencyEvent) -> Dict:
        """
        Halt opening new positions
        """
        logger.warning("EMERGENCY: Trading halted - no new positions")
        
        self.trading_halted = True
        
        return {
            'action': 'TRADING_HALTED',
            'reason': event.title,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        }
    
    def _monitor_event(self, event: EmergencyEvent) -> Dict:
        """
        Monitor event but continue trading
        """
        logger.info(f"MONITORING: {event.title}")
        
        return {
            'action': 'MONITORING',
            'reason': event.title,
            'timestamp': datetime.utcnow().isoformat(),
            'success': True
        }
    
    def resume_trading(self) -> bool:
        """
        Resume trading after emergency
        
        Returns:
            True if trading resumed
        """
        # Check if any active emergencies
        if self.detector.is_trading_halted():
            logger.warning("Cannot resume: Active emergency still present")
            return False
        
        self.trading_halted = False
        logger.info("Trading resumed")
        return True
    
    def can_open_position(self) -> bool:
        """
        Check if new positions can be opened
        
        Returns:
            True if safe to open positions
        """
        return not self.trading_halted and not self.detector.is_trading_halted()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Emergency Event Detector
    """
    
    # Initialize detector
    detector = EmergencyEventDetector()
    handler = EmergencyActionHandler(detector)
    
    # Test emergency scenarios
    test_texts = [
        "BREAKING: Binance exchange hacked, withdrawals suspended",
        "SEC announces emergency ban on all crypto trading",
        "Bitcoin network halted due to critical consensus bug",
        "Normal market volatility, prices fluctuating",  # Not emergency
        "Large whale sell-off causes market dip"  # Not emergency
    ]
    
    print("\n=== EMERGENCY EVENT DETECTION TEST ===")
    
    for text in test_texts:
        print(f"\nText: {text}")
        event = detector.check_emergency(text, source="test")
        
        if event:
            print(f"  ⚠️ EMERGENCY DETECTED!")
            print(f"  Type: {event.event_type}")
            print(f"  Severity: {event.severity}")
            print(f"  Action: {event.action_required}")
            
            # Handle the event
            result = handler.handle_event(event)
            print(f"  Result: {result['action']}")
        else:
            print(f"  ✅ Normal - No emergency")
    
    # Check trading status
    print(f"\n\nCan open new positions: {handler.can_open_position()}")
    print(f"Recent emergencies (24h): {len(detector.get_recent_emergencies(24))}")
