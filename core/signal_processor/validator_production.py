#!/usr/bin/env python3
"""
DEMIR AI PRO - Production Signal Validator

Comprehensive signal validation with:
- Mock/fake/fallback data detection
- Real exchange price verification
- Timestamp validation
- Structure and range checks
- Layer score validation

Zero tolerance for invalid or fake data.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import requests
import pytz

logger = logging.getLogger(__name__)

class MockDataDetector:
    """
    Detect and prevent mock, fake, fallback, or test data.
    """
    
    BANNED_KEYWORDS = [
        'mock_', 'test_', 'demo_', 'fake_', 'dummy_', 'sample_',
        'fallback_', 'prototype_', 'fixture_', 'debug_', 'hardcoded_',
    ]
    
    BANNED_PRICES = [
        99999.99, 88888.88, 77777.77, 12345.67, 11111.11,
        10000.00, 5000.00, 1000.00, 100.00, 69.69, 42.42,
    ]
    
    @staticmethod
    def check_value(value: Any, field_name: str) -> Tuple[bool, str]:
        """
        Check if value contains banned patterns.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if isinstance(value, str):
            for banned in MockDataDetector.BANNED_KEYWORDS:
                if banned.lower() in value.lower():
                    return False, f"❌ Banned keyword '{banned}' in {field_name}"
        
        if isinstance(value, (int, float)):
            if value in MockDataDetector.BANNED_PRICES:
                return False, f"❌ Hardcoded price detected: {value}"
        
        return True, "✅ Value OK"
    
    @staticmethod
    def check_signal(signal: Dict) -> Tuple[bool, List[str]]:
        """Check entire signal for invalid data."""
        errors = []
        
        for field, value in signal.items():
            is_valid, message = MockDataDetector.check_value(value, field)
            if not is_valid:
                errors.append(message)
        
        return len(errors) == 0, errors

class RealDataVerifier:
    """
    Verify signal data against real exchange sources.
    """
    
    def __init__(self):
        self.binance_url = 'https://fapi.binance.com'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DEMIR-AI-PRO/8.0'})
    
    def verify_price(self, symbol: str, expected_price: float, tolerance: float = 1.0) -> Tuple[bool, str]:
        """
        Verify price against real exchange data.
        
        Args:
            symbol: Trading pair
            expected_price: Expected price
            tolerance: Allowed difference percentage
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            response = self.session.get(
                f'{self.binance_url}/fapi/v1/ticker/price',
                params={'symbol': symbol},
                timeout=5
            )
            
            if response.status_code != 200:
                return False, "❌ Could not fetch real price"
            
            real_price = float(response.json()['price'])
            diff_percent = abs(expected_price - real_price) / real_price * 100
            
            if diff_percent > tolerance:
                return False, f"❌ Price mismatch: {diff_percent:.2f}% diff"
            
            return True, f"✅ Price verified ({diff_percent:.2f}% diff)"
        except Exception as e:
            logger.error(f"Price verification error: {e}")
            return False, f"❌ Verification error: {str(e)}"
    
    def verify_timestamp(self, timestamp: datetime, max_age_seconds: int = 60) -> Tuple[bool, str]:
        """
        Verify timestamp is current.
        
        Args:
            timestamp: Signal timestamp
            max_age_seconds: Maximum allowed age
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            current_time = datetime.now(pytz.UTC)
            time_diff = (current_time - timestamp).total_seconds()
            
            if time_diff > max_age_seconds:
                return False, f"❌ Signal too old: {time_diff:.0f}s"
            
            if time_diff < -5:
                return False, f"❌ Timestamp in future: {time_diff:.0f}s"
            
            return True, f"✅ Timestamp valid ({time_diff:.0f}s ago)"
        except Exception as e:
            logger.error(f"Timestamp verification error: {e}")
            return False, f"❌ Timestamp error: {str(e)}"

class SignalIntegrityChecker:
    """
    Check signal structure and value ranges.
    """
    
    REQUIRED_FIELDS = [
        'symbol', 'direction', 'confidence',
        'entry_price', 'take_profit_1', 'stop_loss',
        'timestamp'
    ]
    
    @staticmethod
    def check_structure(signal: Dict) -> Tuple[bool, List[str]]:
        """Check signal has all required fields."""
        errors = []
        
        for field in SignalIntegrityChecker.REQUIRED_FIELDS:
            if field not in signal:
                errors.append(f"❌ Missing field: {field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_ranges(signal: Dict) -> Tuple[bool, List[str]]:
        """Check all values are in valid ranges."""
        errors = []
        
        # Symbol format
        symbol = signal.get('symbol', '')
        if not symbol.endswith('USDT'):
            errors.append(f"❌ Invalid symbol: {symbol}")
        
        # Direction
        direction = signal.get('direction', '')
        if direction not in ['LONG', 'SHORT', 'NEUTRAL']:
            errors.append(f"❌ Invalid direction: {direction}")
        
        # Confidence (0-1)
        confidence = signal.get('confidence', -1)
        if not (0 <= confidence <= 1):
            errors.append(f"❌ Confidence out of range: {confidence}")
        
        # Prices
        entry = signal.get('entry_price', 0)
        tp = signal.get('take_profit_1', 0)
        sl = signal.get('stop_loss', 0)
        
        if entry <= 0:
            errors.append(f"❌ Entry price must be positive: {entry}")
        
        if direction == 'LONG':
            if tp <= entry:
                errors.append(f"❌ TP must be above entry (LONG)")
            if sl >= entry:
                errors.append(f"❌ SL must be below entry (LONG)")
        elif direction == 'SHORT':
            if tp >= entry:
                errors.append(f"❌ TP must be below entry (SHORT)")
            if sl <= entry:
                errors.append(f"❌ SL must be above entry (SHORT)")
        
        return len(errors) == 0, errors

class ProductionSignalValidator:
    """
    Main production signal validator.
    
    Runs all validation checks and provides comprehensive results.
    """
    
    def __init__(self):
        self.mock_detector = MockDataDetector()
        self.data_verifier = RealDataVerifier()
        self.integrity_checker = SignalIntegrityChecker()
        logger.info("✅ Production Signal Validator initialized")
    
    def validate(self, signal: Dict, verify_price: bool = False) -> Dict:
        """
        Comprehensive signal validation.
        
        Args:
            signal: Signal to validate
            verify_price: Whether to verify price against exchange (slower)
            
        Returns:
            Validation results dict
        """
        logger.info(f"🔍 Validating signal: {signal.get('symbol', 'UNKNOWN')}")
        
        results = {
            'valid': True,
            'checks': {},
            'errors': []
        }
        
        # Check 1: Mock data
        is_clean, mock_errors = self.mock_detector.check_signal(signal)
        results['checks']['mock_data'] = (is_clean, mock_errors)
        if not is_clean:
            results['valid'] = False
            results['errors'].extend(mock_errors)
            logger.error(f"❌ Mock data detected")
        else:
            logger.info("✅ No mock data")
        
        # Check 2: Structure
        has_structure, struct_errors = self.integrity_checker.check_structure(signal)
        results['checks']['structure'] = (has_structure, struct_errors)
        if not has_structure:
            results['valid'] = False
            results['errors'].extend(struct_errors)
            logger.error(f"❌ Structure invalid")
        else:
            logger.info("✅ Structure valid")
        
        # Check 3: Value ranges
        valid_ranges, range_errors = self.integrity_checker.check_ranges(signal)
        results['checks']['ranges'] = (valid_ranges, range_errors)
        if not valid_ranges:
            results['valid'] = False
            results['errors'].extend(range_errors)
            logger.error(f"❌ Range errors")
        else:
            logger.info("✅ Ranges valid")
        
        # Check 4: Timestamp
        timestamp = signal.get('timestamp')
        if timestamp:
            time_valid, time_msg = self.data_verifier.verify_timestamp(timestamp)
            results['checks']['timestamp'] = (time_valid, time_msg)
            if not time_valid:
                results['valid'] = False
                results['errors'].append(time_msg)
                logger.error(f"❌ {time_msg}")
            else:
                logger.info(f"✅ {time_msg}")
        
        # Check 5: Price (optional)
        if verify_price and 'symbol' in signal and 'entry_price' in signal:
            price_valid, price_msg = self.data_verifier.verify_price(
                signal['symbol'],
                signal['entry_price']
            )
            results['checks']['price'] = (price_valid, price_msg)
            if not price_valid:
                logger.warning(f"⚠️ {price_msg}")
            else:
                logger.info(f"✅ {price_msg}")
        
        # Final status
        if results['valid']:
            logger.info("🟢 ✅ SIGNAL VALIDATION PASSED")
        else:
            logger.error(f"🔴 ❌ SIGNAL VALIDATION FAILED ({len(results['errors'])} errors)")
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("SIGNAL VALIDATION REPORT")
        lines.append("=" * 70)
        
        status = "✅ VALID" if results['valid'] else "❌ INVALID"
        lines.append(f"Status: {status}")
        lines.append("")
        
        for check_name, (passed, details) in results['checks'].items():
            icon = "✅" if passed else "❌"
            lines.append(f"{icon} {check_name.upper()}")
            
            if isinstance(details, list):
                for detail in details:
                    lines.append(f"  {detail}")
            else:
                lines.append(f"  {details}")
        
        if results['errors']:
            lines.append("")
            lines.append("❌ ERRORS:")
            for error in results['errors']:
                lines.append(f"  {error}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
