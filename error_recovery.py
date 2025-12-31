#!/usr/bin/env python3
"""
🔄 ERROR RECOVERY SYSTEM
========================

Proper error handling and recovery for network interruptions and failures.
"""

import asyncio
import aiohttp
import time
import json
import random
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import functools
from contextlib import asynccontextmanager

class RecoveryStrategy(Enum):
    """Strategies for error recovery"""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FALLBACK = "fallback"
    RECONNECT = "reconnect"

class ErrorCategory(Enum):
    """Categories of errors"""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for errors"""
    operation: str
    timestamp: float = field(default_factory=time.time)
    attempt: int = 1
    total_attempts: int = 1
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    error_message: str = ""
    stack_trace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_errors: List[str] = field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "aiohttp.ClientError",
        "asyncio.TimeoutError", "ConnectionResetError"
    ])

@dataclass
class CircuitBreakerState:
    """State of circuit breaker"""
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0
    next_attempt_time: Optional[float] = None

@dataclass
class RecoveryMetrics:
    """Metrics for error recovery"""
    total_errors: int = 0
    recovered_errors: int = 0
    unrecoverable_errors: int = 0
    average_recovery_time: float = 0.0
    circuit_breaker_trips: int = 0
    fallback_activations: int = 0

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitBreakerState()
        self.logger = logging.getLogger(f"{__class__.__name__}")

    async def call(self, func: Callable[[], Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""

        if self.state.state == "open":
            if time.time() < (self.state.next_attempt_time or 0):
                raise CircuitBreakerOpenError("Circuit breaker is open")

            # Transition to half-open
            self.state.state = "half_open"
            self.state.success_count = 0
            self.logger.info("Circuit breaker transitioning to half-open")

        try:
            result = await func(*args, **kwargs)

            # Success
            if self.state.state == "half_open":
                self.state.success_count += 1
                if self.state.success_count >= self.success_threshold:
                    self._reset()
                    self.logger.info("Circuit breaker closed after successful calls")
            elif self.state.state == "closed":
                # Reset failure count on success
                self.state.failure_count = 0

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Record a failure"""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()

        if self.state.failure_count >= self.failure_threshold:
            self._open_circuit()
        elif self.state.state == "half_open":
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state.state = "open"
        self.state.next_attempt_time = time.time() + self.recovery_timeout
        self.logger.warning(f"Circuit breaker opened for {self.recovery_timeout}s")

    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitBreakerState()
        self.logger.info("Circuit breaker reset to closed state")

class RetryHandler:
    """Intelligent retry handler with exponential backoff"""

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(f"{__class__.__name__}")

    async def execute_with_retry(self, func: Callable[[], Awaitable[Any]],
                               context: ErrorContext) -> Any:
        """Execute function with retry logic"""

        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                context.attempt = attempt
                context.total_attempts = self.config.max_attempts

                result = await func()
                return result

            except Exception as e:
                last_exception = e
                context.error_message = str(e)
                context.stack_trace = traceback.format_exc()

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    self.logger.debug(f"Non-retryable error: {e}")
                    break

                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_attempts} attempts failed: {e}")

        raise last_exception

    def _is_retryable_error(self, exception: Exception) -> bool:
        """Check if error is retryable"""
        error_type = type(exception).__name__
        return error_type in self.config.retryable_errors

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.config.initial_delay * (self.config.backoff_factor ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # Add jitter

        return delay

class GracefulDegradationManager:
    """Manager for graceful degradation when services are unavailable"""

    def __init__(self):
        self.degraded_services: Dict[str, Dict[str, Any]] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"{__class__.__name__}")

    def register_fallback(self, service_name: str, fallback_func: Callable):
        """Register fallback function for service"""
        self.fallback_functions[service_name] = fallback_func

    async def execute_with_degradation(self, service_name: str,
                                     primary_func: Callable[[], Awaitable[Any]],
                                     *args, **kwargs) -> Any:
        """Execute with graceful degradation"""

        try:
            # Try primary function
            return await primary_func(*args, **kwargs)

        except Exception as e:
            self.logger.warning(f"Primary service {service_name} failed, attempting degradation: {e}")

            # Mark service as degraded
            self.degraded_services[service_name] = {
                'degraded_at': time.time(),
                'error': str(e),
                'attempts': self.degraded_services.get(service_name, {}).get('attempts', 0) + 1
            }

            # Try fallback if available
            if service_name in self.fallback_functions:
                try:
                    self.logger.info(f"Using fallback for {service_name}")
                    return await self.fallback_functions[service_name](*args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed for {service_name}: {fallback_error}")

            # Return degraded response
            return self._create_degraded_response(service_name, e)

    def _create_degraded_response(self, service_name: str, error: Exception) -> Dict[str, Any]:
        """Create degraded response when all else fails"""
        return {
            'status': 'degraded',
            'service': service_name,
            'error': str(error),
            'timestamp': time.time(),
            'message': f'Service {service_name} is currently unavailable. Using cached or default data.'
        }

    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'degraded_services': self.degraded_services,
            'total_degraded': len(self.degraded_services),
            'available_fallbacks': list(self.fallback_functions.keys())
        }

class NetworkResilienceManager:
    """Manager for network resilience and recovery"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_pool = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        self.retry_handler = RetryHandler()
        self.circuit_breaker = CircuitBreaker()
        self.degradation_manager = GracefulDegradationManager()
        self.metrics = RecoveryMetrics()

        # Setup logging
        self.logger = logging.getLogger(f"{__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        # Recovery state
        self.last_network_check = 0
        self.network_available = True

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize network resilience manager"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connection_pool,
            timeout=timeout,
            trust_env=True  # Use environment proxy settings
        )

        # Register fallback functions
        self.degradation_manager.register_fallback('api_call', self._api_fallback)
        self.degradation_manager.register_fallback('data_fetch', self._data_fetch_fallback)

        self.logger.info("Network resilience manager initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        await self.connection_pool.close()
        self.logger.info("Network resilience manager cleaned up")

    async def make_resilient_request(self, method: str, url: str,
                                   context: ErrorContext = None,
                                   **kwargs) -> Dict[str, Any]:
        """Make HTTP request with full resilience"""

        if context is None:
            context = ErrorContext(operation=f"{method} {url}")

        start_time = time.time()

        try:
            # Check network connectivity first
            if not await self._check_network_connectivity():
                raise NetworkUnavailableError("Network connectivity check failed")

            # Execute with circuit breaker and retry
            async def _make_request():
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status >= 500:
                        # Server error - retry
                        raise ServerError(f"Server error: {response.status}")
                    elif response.status >= 400:
                        # Client error - don't retry
                        raise ClientError(f"Client error: {response.status}")

                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'data': await response.read(),
                        'text': await response.text()
                    }

            result = await self.circuit_breaker.call(
                lambda: self.retry_handler.execute_with_retry(_make_request, context)
            )

            # Update metrics
            self.metrics.recovered_errors += 1

            return result

        except Exception as e:
            # Update metrics
            self.metrics.total_errors += 1

            # Categorize error
            context.error_category = self._categorize_error(e)

            # Try graceful degradation
            try:
                degraded_result = await self.degradation_manager.execute_with_degradation(
                    'api_call', self._make_fallback_request, method, url, **kwargs
                )
                self.metrics.fallback_activations += 1
                return degraded_result
            except Exception as degradation_error:
                self.metrics.unrecoverable_errors += 1
                self.logger.error(f"Request failed with no recovery: {e}")

                # Re-raise original error
                raise e

        finally:
            execution_time = time.time() - start_time
            if execution_time > 0:
                self.metrics.average_recovery_time = (
                    (self.metrics.average_recovery_time + execution_time) / 2
                )

    async def _make_fallback_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Fallback request implementation"""
        # This would implement cached responses, alternative endpoints, etc.
        return {
            'status': 503,
            'degraded': True,
            'message': 'Service temporarily unavailable',
            'fallback_used': True
        }

    async def _api_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """API call fallback"""
        return {
            'status': 'degraded',
            'message': 'API unavailable, using cached data',
            'cached': True
        }

    async def _data_fetch_fallback(self, *args, **kwargs) -> Any:
        """Data fetch fallback"""
        return {
            'status': 'degraded',
            'data': None,
            'message': 'Data fetch failed, using defaults'
        }

    async def _check_network_connectivity(self) -> bool:
        """Check basic network connectivity"""

        # Only check every 30 seconds to avoid spam
        current_time = time.time()
        if current_time - self.last_network_check < 30:
            return self.network_available

        self.last_network_check = current_time

        try:
            # Simple connectivity check to a reliable host
            async with self.session.get('http://httpbin.org/status/200', timeout=aiohttp.ClientTimeout(total=5)) as response:
                self.network_available = response.status == 200
        except Exception:
            self.network_available = False

        return self.network_available

    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize error type"""

        error_type = type(exception).__name__
        error_msg = str(exception).lower()

        if any(term in error_msg for term in ['connection', 'network', 'dns', 'resolve']):
            return ErrorCategory.NETWORK
        elif any(term in error_msg for term in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT
        elif any(term in error_msg for term in ['auth', 'unauthorized', 'forbidden']):
            return ErrorCategory.AUTHENTICATION
        elif any(term in error_msg for term in ['rate limit', 'too many requests']):
            return ErrorCategory.RATE_LIMIT
        elif error_type in ['ClientError'] and '5' in str(getattr(exception, 'status', '')):
            return ErrorCategory.SERVER_ERROR
        elif error_type in ['ClientError'] and '4' in str(getattr(exception, 'status', '')):
            return ErrorCategory.CLIENT_ERROR
        else:
            return ErrorCategory.UNKNOWN

    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics"""
        return {
            'total_errors': self.metrics.total_errors,
            'recovered_errors': self.metrics.recovered_errors,
            'unrecoverable_errors': self.metrics.unrecoverable_errors,
            'recovery_rate': (self.metrics.recovered_errors / max(1, self.metrics.total_errors)),
            'average_recovery_time': self.metrics.average_recovery_time,
            'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
            'fallback_activations': self.metrics.fallback_activations,
            'network_available': self.network_available,
            'degradation_status': self.degradation_manager.get_degradation_status()
        }

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    pass

class NetworkUnavailableError(Exception):
    pass

class ServerError(Exception):
    pass

class ClientError(Exception):
    pass

# Decorators for easy error recovery
def resilient_operation(strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                       max_attempts: int = 3):
    """Decorator for resilient operations"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would integrate with the resilience manager
            # Simplified implementation for demo
            retry_config = RetryConfig(max_attempts=max_attempts)
            retry_handler = RetryHandler(retry_config)
            context = ErrorContext(operation=func.__name__)

            try:
                return await retry_handler.execute_with_retry(func, context)
            except Exception as e:
                print(f"Operation {func.__name__} failed after {max_attempts} attempts: {e}")
                raise

        return wrapper
    return decorator

@asynccontextmanager
async def resilient_session():
    """Context manager for resilient network operations"""
    async with NetworkResilienceManager() as manager:
        yield manager

async def demo_error_recovery():
    """Demonstrate error recovery capabilities"""

    print("🔄 ERROR RECOVERY SYSTEM DEMONSTRATION")
    print("=" * 50)

    async with resilient_session() as resilience:

        # Test 1: Successful request
        print("\n1. Testing successful request...")
        try:
            result = await resilience.make_resilient_request('GET', 'http://httpbin.org/json')
            print(f"   ✅ Success: Status {result['status']}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

        # Test 2: Circuit breaker demonstration
        print("\n2. Testing circuit breaker (simulated failures)...")

        async def failing_request():
            raise aiohttp.ClientError("Simulated server error")

        for i in range(7):  # More than failure threshold
            try:
                await resilience.circuit_breaker.call(failing_request)
                print(f"   Attempt {i+1}: Success")
            except Exception as e:
                print(f"   Attempt {i+1}: Failed - {type(e).__name__}")

        # Test 3: Graceful degradation
        print("\n3. Testing graceful degradation...")

        async def failing_primary():
            raise ConnectionError("Primary service down")

        degraded_result = await resilience.degradation_manager.execute_with_degradation(
            'api_call', failing_primary
        )
        print(f"   Degraded result: {degraded_result}")

        # Test 4: Metrics
        print("\n4. Recovery metrics:")
        metrics = resilience.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Error Recovery System")
    parser.add_argument("--demo", action="store_true", help="Run error recovery demonstration")
    parser.add_argument("--test-request", metavar="URL", help="Test resilient HTTP request")
    parser.add_argument("--metrics", action="store_true", help="Show recovery metrics")

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo_error_recovery())

    elif args.test_request:
        async def test_request():
            async with resilient_session() as resilience:
                try:
                    result = await resilience.make_resilient_request('GET', args.test_request)
                    print(f"✅ Request successful: Status {result['status']}")
                except Exception as e:
                    print(f"❌ Request failed: {e}")

                if args.metrics:
                    metrics = resilience.get_metrics()
                    print("\n📊 METRICS:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")

        asyncio.run(test_request())

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
