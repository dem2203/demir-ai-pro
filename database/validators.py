# Production validators transferred from old repo
# Full content from real_data_validators.py with enhancements

"""
DEMIR AI PRO - Production Data Validators

Zero-tolerance mock/fake/test data detection.
All data must come from real exchange APIs.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MockDataDetector:
    """Detect mock, fake, test, fallback, prototype data patterns."""
    
    FAKE_KEYWORDS = [
        'mock', 'fake', 'test', 'fallback', 'prototype', 'dummy', 
        'sample', 'example', 'placeholder', 'stub', 'patch',
        'demo', 'trial', 'temp', 'temporary', 'staging'
    ]
    
    SUSPICIOUS_PATTERNS = [
        r'^0\.0+$',
        r'^1\.0+$',
        r'^\d{5,}$',
        r'^9+\.9+$',
        r'^\d\.0{5,}$',
    ]
    
    @staticmethod
    def detect_in_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        
        def check_value(key: str, value: Any, path: str = ""):
            current_path = f"{path}.{key}" if path else key
            
            for keyword in MockDataDetector.FAKE_KEYWORDS:
                if keyword.lower() in key.lower():
                    violations.append(f"Key '{current_path}' contains mock keyword '{keyword}'")
            
            if isinstance(value, str):
                for keyword in MockDataDetector.FAKE_KEYWORDS:
                    if keyword.lower() in value.lower():
                        violations.append(f"Value at '{current_path}' contains mock keyword '{keyword}'")
            
            if isinstance(value, (int, float)):
                str_val = str(value)
                for pattern in MockDataDetector.SUSPICIOUS_PATTERNS:
                    if re.match(pattern, str_val):
                        violations.append(f"Suspicious pattern at '{current_path}': {value}")
            
            if isinstance(value, dict):
                for k, v in value.items():
                    check_value(k, v, current_path)
        
        for key, value in data.items():
            check_value(key, value)
        
        return len(violations) == 0, violations

class RealDataValidator:
    """Verify data comes from real exchange APIs."""
    
    def __init__(self, binance_client=None):
        self.binance = binance_client
        logger.info("RealDataValidator initialized")
    
    def verify_price_data(self, symbol: str, price: float) -> Tuple[bool, str]:
        try:
            if self.binance:
                ticker = self.binance.fetch_ticker(symbol)
                real_price = ticker['last']
                deviation = abs(price - real_price) / real_price
                
                if deviation > 0.02:
                    return False, f"Price deviation too high: {deviation:.2%}"
                
                return True, f"Price verified: {real_price}"
            
            return False, "Exchange client not available"
        except Exception as e:
            logger.error(f"Price verification failed: {e}")
            return False, str(e)
    
    def verify_timestamp(self, timestamp: float, max_age_seconds: int = 300) -> Tuple[bool, str]:
        try:
            current_time = datetime.now().timestamp()
            age = current_time - timestamp
            
            if age < 0:
                return False, "Timestamp is in the future"
            
            if age > max_age_seconds:
                return False, f"Data is stale: {age} seconds old"
            
            return True, f"Timestamp valid: {age:.1f} seconds old"
        except Exception as e:
            return False, str(e)

class SignalValidator:
    """Master validator combining all checks."""
    
    REQUIRED_FIELDS = ['direction', 'strength', 'confidence', 'timestamp']
    VALID_DIRECTIONS = ['LONG', 'SHORT', 'NEUTRAL']
    
    def __init__(self, binance_client=None):
        self.mock_detector = MockDataDetector()
        self.real_validator = RealDataValidator(binance_client)
        logger.info("SignalValidator initialized")
    
    def validate_signal(self, signal: Dict[str, Any], group_type: str) -> Tuple[bool, List[str]]:
        issues = []
        
        # Check mock data
        is_real, mock_issues = self.mock_detector.detect_in_data(signal)
        issues.extend(mock_issues)
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in signal:
                issues.append(f"Missing required field: {field}")
        
        # Check direction
        if 'direction' in signal and signal['direction'] not in self.VALID_DIRECTIONS:
            issues.append(f"Invalid direction: {signal['direction']}")
        
        # Check ranges
        if 'strength' in signal and not (0 <= signal['strength'] <= 1):
            issues.append(f"Strength out of range: {signal['strength']}")
        
        if 'confidence' in signal and not (0 <= signal['confidence'] <= 1):
            issues.append(f"Confidence out of range: {signal['confidence']}")
        
        # Check timestamp
        if 'timestamp' in signal:
            is_valid, msg = self.real_validator.verify_timestamp(signal['timestamp'])
            if not is_valid:
                issues.append(msg)
        
        return len(issues) == 0, issues
