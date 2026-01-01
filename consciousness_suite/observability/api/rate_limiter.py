"""Rate Limiter Monitor

Rate limiting metrics and enforcement tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import logging

from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)


class LimitScope(str, Enum):
    """Scope of rate limiting."""
    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    PER_ENDPOINT = "per_endpoint"


class LimitStrategy(str, Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    name: str
    limit: int  # Maximum requests
    window_seconds: int  # Time window
    scope: LimitScope = LimitScope.PER_IP
    strategy: LimitStrategy = LimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # For token bucket
    penalty_seconds: int = 0  # Extra wait on violation
    enabled: bool = True
    priority: int = 0  # Higher = checked first


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: datetime
    limit: int
    retry_after_seconds: float = 0
    config_name: str = ""


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""
    config_name: str
    scope: LimitScope
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    avg_usage: float = 0
    window_start: datetime = field(default_factory=datetime.now)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0
        return self.rejected_requests / self.total_requests


class TokenBucket:
    """Token bucket rate limiter implementation.

    Usage:
        bucket = TokenBucket(capacity=100, refill_rate=10)

        if bucket.consume():
            # Request allowed
            process_request()
        else:
            # Rate limited
            return 429
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # tokens per second
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if rate limited
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            self._refill()
            return self.tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get seconds to wait for tokens.

        Args:
            tokens: Required tokens

        Returns:
            Seconds to wait (0 if available)
        """
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                return 0
            needed = tokens - self.tokens
            return needed / self.refill_rate


class SlidingWindow:
    """Sliding window rate limiter implementation.

    Usage:
        window = SlidingWindow(limit=100, window_seconds=60)

        if window.allow("client_123"):
            process_request()
        else:
            return 429
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _clean_old_requests(self, key: str, now: float):
        """Remove requests outside the window."""
        if key in self._requests:
            cutoff = now - self.window_seconds
            self._requests[key] = [
                t for t in self._requests[key]
                if t > cutoff
            ]

    def allow(self, key: str) -> bool:
        """Check if request is allowed.

        Args:
            key: Client/user identifier

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()

        with self._lock:
            self._clean_old_requests(key, now)

            if key not in self._requests:
                self._requests[key] = []

            if len(self._requests[key]) < self.limit:
                self._requests[key].append(now)
                return True

            return False

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window.

        Args:
            key: Client/user identifier

        Returns:
            Remaining requests
        """
        now = time.time()

        with self._lock:
            self._clean_old_requests(key, now)
            current = len(self._requests.get(key, []))
            return max(0, self.limit - current)

    def get_reset_time(self, key: str) -> float:
        """Get when window resets.

        Args:
            key: Client/user identifier

        Returns:
            Timestamp when oldest request expires
        """
        now = time.time()

        with self._lock:
            self._clean_old_requests(key, now)
            requests = self._requests.get(key, [])

            if not requests or len(requests) < self.limit:
                return now

            oldest = min(requests)
            return oldest + self.window_seconds


class RateLimitMonitor:
    """Monitors and enforces rate limits.

    Usage:
        monitor = RateLimitMonitor()

        # Add rate limit configs
        monitor.add_config(RateLimitConfig(
            name="api_default",
            limit=100,
            window_seconds=60,
            scope=LimitScope.PER_API_KEY,
        ))

        # Check rate limit
        result = monitor.check_rate_limit(
            config_name="api_default",
            key="api_key_123",
        )

        if not result.allowed:
            return Response(status=429, headers={
                "Retry-After": str(result.retry_after_seconds),
                "X-RateLimit-Remaining": str(result.remaining),
            })
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._configs: Dict[str, RateLimitConfig] = {}
        self._limiters: Dict[str, Dict[str, Any]] = {}  # config -> key -> limiter
        self._stats: Dict[str, RateLimitStats] = {}
        self._lock = threading.Lock()

        self._violation_callbacks: List[Callable[[str, str, RateLimitResult], None]] = []

        # Prometheus metrics
        self.rate_limit_requests = Counter(
            f"{namespace}_api_rate_limit_requests_total",
            "Total rate limit checks",
            ["config", "scope", "result"],
        )

        self.rate_limit_remaining = Gauge(
            f"{namespace}_api_rate_limit_remaining",
            "Remaining requests in window",
            ["config", "key"],
        )

        self.rate_limit_rejections = Counter(
            f"{namespace}_api_rate_limit_rejections_total",
            "Total rate limit rejections",
            ["config", "scope"],
        )

        self.rate_limit_current_usage = Gauge(
            f"{namespace}_api_rate_limit_usage_percent",
            "Current rate limit usage percentage",
            ["config"],
        )

        self.rate_limit_latency = Histogram(
            f"{namespace}_api_rate_limit_check_seconds",
            "Rate limit check latency",
            ["config"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        )

    def add_config(self, config: RateLimitConfig):
        """Add a rate limit configuration.

        Args:
            config: Rate limit config
        """
        with self._lock:
            self._configs[config.name] = config
            self._limiters[config.name] = {}
            self._stats[config.name] = RateLimitStats(
                config_name=config.name,
                scope=config.scope,
            )

        logger.info(
            f"Added rate limit config: {config.name} "
            f"({config.limit}/{config.window_seconds}s, scope={config.scope.value})"
        )

    def check_rate_limit(
        self,
        config_name: str,
        key: str,
    ) -> RateLimitResult:
        """Check rate limit for a request.

        Args:
            config_name: Name of rate limit config
            key: Client/user/IP identifier

        Returns:
            RateLimitResult
        """
        start_time = time.time()

        with self._lock:
            config = self._configs.get(config_name)
            if not config or not config.enabled:
                return RateLimitResult(
                    allowed=True,
                    remaining=-1,
                    reset_at=datetime.now(),
                    limit=0,
                    config_name=config_name,
                )

            # Get or create limiter for this key
            if key not in self._limiters[config_name]:
                if config.strategy == LimitStrategy.TOKEN_BUCKET:
                    burst = config.burst_limit or config.limit
                    self._limiters[config_name][key] = TokenBucket(
                        capacity=burst,
                        refill_rate=config.limit / config.window_seconds,
                    )
                else:
                    self._limiters[config_name][key] = SlidingWindow(
                        limit=config.limit,
                        window_seconds=config.window_seconds,
                    )

            limiter = self._limiters[config_name][key]

        # Check the limiter
        if isinstance(limiter, TokenBucket):
            allowed = limiter.consume()
            remaining = int(limiter.get_tokens())
            wait_time = limiter.get_wait_time() if not allowed else 0
            reset_at = datetime.now() + timedelta(seconds=wait_time)
        else:
            allowed = limiter.allow(key)
            remaining = limiter.get_remaining(key)
            reset_time = limiter.get_reset_time(key)
            reset_at = datetime.fromtimestamp(reset_time)
            wait_time = max(0, reset_time - time.time()) if not allowed else 0

        # Apply penalty if violated
        if not allowed and config.penalty_seconds > 0:
            wait_time += config.penalty_seconds

        result = RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            limit=config.limit,
            retry_after_seconds=wait_time,
            config_name=config_name,
        )

        # Update stats
        with self._lock:
            stats = self._stats[config_name]
            stats.total_requests += 1
            if allowed:
                stats.allowed_requests += 1
            else:
                stats.rejected_requests += 1

            usage = config.limit - remaining
            stats.current_usage = usage
            stats.peak_usage = max(stats.peak_usage, usage)
            stats.timestamp = datetime.now()

        # Update metrics
        self.rate_limit_requests.labels(
            config=config_name,
            scope=config.scope.value,
            result="allowed" if allowed else "rejected",
        ).inc()

        self.rate_limit_remaining.labels(
            config=config_name,
            key=key[:32],  # Truncate for cardinality
        ).set(remaining)

        if not allowed:
            self.rate_limit_rejections.labels(
                config=config_name,
                scope=config.scope.value,
            ).inc()

            # Trigger callbacks
            for callback in self._violation_callbacks:
                try:
                    callback(config_name, key, result)
                except Exception as e:
                    logger.error(f"Rate limit callback error: {e}")

        usage_percent = (config.limit - remaining) / config.limit * 100
        self.rate_limit_current_usage.labels(config=config_name).set(usage_percent)

        elapsed = time.time() - start_time
        self.rate_limit_latency.labels(config=config_name).observe(elapsed)

        return result

    def on_violation(
        self,
        callback: Callable[[str, str, RateLimitResult], None],
    ):
        """Register callback for rate limit violations.

        Args:
            callback: Function(config_name, key, result)
        """
        self._violation_callbacks.append(callback)

    def get_stats(self, config_name: str) -> Optional[RateLimitStats]:
        """Get statistics for a config.

        Args:
            config_name: Config name

        Returns:
            RateLimitStats or None
        """
        with self._lock:
            return self._stats.get(config_name)

    def get_all_stats(self) -> Dict[str, RateLimitStats]:
        """Get all rate limit statistics.

        Returns:
            Dict of config name to stats
        """
        with self._lock:
            return dict(self._stats)

    def reset_limiter(self, config_name: str, key: str):
        """Reset limiter for a specific key.

        Args:
            config_name: Config name
            key: Client/user identifier
        """
        with self._lock:
            if config_name in self._limiters:
                self._limiters[config_name].pop(key, None)

    def update_config(self, config: RateLimitConfig):
        """Update a rate limit config (clears existing limiters).

        Args:
            config: New config
        """
        with self._lock:
            self._configs[config.name] = config
            self._limiters[config.name] = {}  # Clear limiters

        logger.info(f"Updated rate limit config: {config.name}")

    def disable_config(self, config_name: str):
        """Disable a rate limit config.

        Args:
            config_name: Config to disable
        """
        with self._lock:
            if config_name in self._configs:
                self._configs[config_name].enabled = False

    def enable_config(self, config_name: str):
        """Enable a rate limit config.

        Args:
            config_name: Config to enable
        """
        with self._lock:
            if config_name in self._configs:
                self._configs[config_name].enabled = True

    def get_summary(self) -> Dict[str, Any]:
        """Get rate limiting summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            configs = dict(self._configs)
            stats = dict(self._stats)

        total_requests = sum(s.total_requests for s in stats.values())
        total_rejections = sum(s.rejected_requests for s in stats.values())

        return {
            "active_configs": len([c for c in configs.values() if c.enabled]),
            "total_configs": len(configs),
            "total_requests": total_requests,
            "total_rejections": total_rejections,
            "overall_rejection_rate": total_rejections / total_requests if total_requests > 0 else 0,
            "configs": {
                name: {
                    "limit": configs[name].limit,
                    "window_seconds": configs[name].window_seconds,
                    "scope": configs[name].scope.value,
                    "rejection_rate": stats[name].rejection_rate if name in stats else 0,
                }
                for name in configs
            },
        }
