"""
Resilience Manager - Advanced Error Recovery
============================================
Production-grade error recovery and resilience system.
- Auto-reconnection with exponential backoff
- Circuit breaker pattern
- Graceful degradation
- State recovery
- Health monitoring
- Zero mock/fallback

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger("monitoring.resilience")


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure detected, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes before closing from half-open
    timeout: int = 60  # Seconds before trying half-open
    half_open_timeout: int = 30  # Seconds for half-open state


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: datetime = datetime.now()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to call
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception if circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                logger.info(f"🔄 Circuit '{self.name}' moved to HALF_OPEN")
            else:
                raise Exception(f"Circuit '{self.name}' is OPEN - blocking call")
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Success
            self._on_success()
            return result
            
        except Exception as e:
            # Failure
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                self.last_state_change = datetime.now()
                logger.info(f"✅ Circuit '{self.name}' CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Immediate open on half-open failure
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.last_state_change = datetime.now()
            logger.warning(f"⚠️ Circuit '{self.name}' OPENED - service still failing")
        
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.error(f"🚨 Circuit '{self.name}' OPENED - too many failures ({self.failure_count})")
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset from open state"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).seconds
        return time_since_failure >= self.config.timeout
    
    def reset(self):
        """Manually reset circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = datetime.now()
        logger.info(f"🔄 Circuit '{self.name}' manually reset")


class ExponentialBackoff:
    """Exponential backoff for retries"""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempts = 0
    
    def get_delay(self) -> float:
        """Calculate next delay"""
        delay = min(self.base_delay * (self.multiplier ** self.attempts), self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        self.attempts += 1
        return delay
    
    def reset(self):
        """Reset backoff"""
        self.attempts = 0


class ResilienceManager:
    """Central resilience and error recovery management"""
    
    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.backoff_strategies: dict[str, ExponentialBackoff] = {}
        logger.info("ResilienceManager initialized")
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def get_backoff_strategy(self, name: str) -> ExponentialBackoff:
        """Get or create backoff strategy"""
        if name not in self.backoff_strategies:
            self.backoff_strategies[name] = ExponentialBackoff()
        return self.backoff_strategies[name]
    
    async def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        backoff_name: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """
        Retry function with exponential backoff
        
        Args:
            func: Async function to retry
            max_retries: Maximum retry attempts
            backoff_name: Name for backoff strategy
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        backoff = self.get_backoff_strategy(backoff_name)
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                backoff.reset()
                return result
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = backoff.get_delay()
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s - Error: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retries failed for {func.__name__}")
        
        raise last_exception
    
    async def safe_call(
        self,
        func: Callable,
        circuit_name: str,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        max_retries: int = 3,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Safe call with circuit breaker + retry
        
        Args:
            func: Async function to call
            circuit_name: Circuit breaker name
            circuit_config: Optional circuit config
            max_retries: Max retry attempts
            *args, **kwargs: Function arguments
            
        Returns:
            Function result or None on failure
        """
        circuit = self.get_circuit_breaker(circuit_name, circuit_config)
        
        try:
            result = await circuit.call(
                self.retry_with_backoff,
                func,
                max_retries,
                circuit_name,
                *args,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Safe call failed for {circuit_name}: {e}")
            return None
    
    def get_health_status(self) -> dict:
        """Get resilience system health status"""
        return {
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'last_state_change': cb.last_state_change.isoformat()
                }
                for name, cb in self.circuit_breakers.items()
            },
            'total_circuits': len(self.circuit_breakers),
            'open_circuits': sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        }
    
    def reset_all_circuits(self):
        """Reset all circuit breakers"""
        for circuit in self.circuit_breakers.values():
            circuit.reset()
        logger.info("All circuits reset")


# Global instance
_resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance"""
    return _resilience_manager
