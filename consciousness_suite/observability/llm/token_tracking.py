"""Token Tracking for LLM Observability

Provides:
- Token counting with tiktoken
- Budget management
- Rate limiting
- Usage analytics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import logging

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    timestamp: float = field(default_factory=time.time)
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenTracker:
    """Track token usage across LLM calls.

    Usage:
        tracker = TokenTracker()

        # Track usage
        tracker.track("gpt-4", input_tokens=100, output_tokens=50)

        # Get statistics
        stats = tracker.get_stats()
        print(f"Total tokens: {stats['total_tokens']}")

        # Get usage by model
        by_model = tracker.usage_by_model()
    """

    def __init__(
        self,
        max_history: int = 10000,
        encoding_name: str = "cl100k_base",
    ):
        self._history: List[TokenUsage] = []
        self._max_history = max_history
        self._lock = threading.Lock()
        self._encoding = None
        self._encoding_cache: Dict[str, Any] = {}

        # Initialize tiktoken if available
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")

        # Aggregated counters
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._calls_count = 0
        self._by_model: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "calls": 0}
        )

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Optional model name for model-specific tokenizer

        Returns:
            Token count
        """
        if not text:
            return 0

        # Use tiktoken if available
        if self._encoding:
            try:
                return len(self._encoding.encode(text))
            except Exception:
                pass

        # Fallback: approximate based on characters
        # Average ~4 chars per token for English text
        return max(1, len(text) // 4)

    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
    ) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model name for accurate counting

        Returns:
            Total token count
        """
        # Get model-specific encoding if available
        encoding = self._get_encoding_for_model(model)

        total = 0
        for message in messages:
            # Count message overhead (~4 tokens per message for GPT models)
            total += 4

            role = message.get("role", "")
            content = message.get("content", "")

            if encoding:
                total += len(encoding.encode(role))
                total += len(encoding.encode(content))
            else:
                total += len(role) // 4 + 1
                total += len(content) // 4 + 1

        # Add conversation overhead
        total += 3

        return total

    def _get_encoding_for_model(self, model: str):
        """Get tiktoken encoding for a specific model."""
        if not TIKTOKEN_AVAILABLE:
            return None

        if model in self._encoding_cache:
            return self._encoding_cache[model]

        try:
            encoding = tiktoken.encoding_for_model(model)
            self._encoding_cache[model] = encoding
            return encoding
        except Exception:
            # Fall back to default encoding
            return self._encoding

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TokenUsage:
        """Track token usage for an LLM call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            metadata: Additional metadata

        Returns:
            TokenUsage record
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        with self._lock:
            # Add to history
            self._history.append(usage)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history // 2 :]

            # Update aggregates
            self._total_input += input_tokens
            self._total_output += output_tokens
            self._total_cost += cost_usd
            self._calls_count += 1

            # Update per-model stats
            self._by_model[model]["input"] += input_tokens
            self._by_model[model]["output"] += output_tokens
            self._by_model[model]["calls"] += 1

        return usage

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics.

        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            return {
                "total_input_tokens": self._total_input,
                "total_output_tokens": self._total_output,
                "total_tokens": self._total_input + self._total_output,
                "total_cost_usd": self._total_cost,
                "total_calls": self._calls_count,
                "avg_input_tokens": (
                    self._total_input / self._calls_count if self._calls_count > 0 else 0
                ),
                "avg_output_tokens": (
                    self._total_output / self._calls_count if self._calls_count > 0 else 0
                ),
                "history_size": len(self._history),
            }

    def usage_by_model(self) -> Dict[str, Dict[str, int]]:
        """Get token usage grouped by model.

        Returns:
            Dictionary mapping model names to usage stats
        """
        with self._lock:
            return dict(self._by_model)

    def usage_by_time(
        self,
        window: timedelta = timedelta(hours=1),
        bucket_size: timedelta = timedelta(minutes=5),
    ) -> List[Dict[str, Any]]:
        """Get token usage grouped by time buckets.

        Args:
            window: Time window to analyze
            bucket_size: Size of each time bucket

        Returns:
            List of usage buckets
        """
        now = time.time()
        window_start = now - window.total_seconds()
        bucket_seconds = bucket_size.total_seconds()

        buckets: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0, "calls": 0}
        )

        with self._lock:
            for usage in self._history:
                if usage.timestamp >= window_start:
                    bucket_idx = int((usage.timestamp - window_start) // bucket_seconds)
                    buckets[bucket_idx]["input"] += usage.input_tokens
                    buckets[bucket_idx]["output"] += usage.output_tokens
                    buckets[bucket_idx]["calls"] += 1

        # Convert to list with timestamps
        result = []
        for idx in sorted(buckets.keys()):
            bucket_start = window_start + idx * bucket_seconds
            result.append({
                "timestamp": datetime.fromtimestamp(bucket_start).isoformat(),
                "input_tokens": buckets[idx]["input"],
                "output_tokens": buckets[idx]["output"],
                "total_tokens": buckets[idx]["input"] + buckets[idx]["output"],
                "calls": buckets[idx]["calls"],
            })

        return result

    def recent_usage(self, count: int = 100) -> List[TokenUsage]:
        """Get recent token usage records.

        Args:
            count: Number of records to return

        Returns:
            List of recent TokenUsage records
        """
        with self._lock:
            return list(reversed(self._history[-count:]))

    def reset(self):
        """Reset all tracking data."""
        with self._lock:
            self._history.clear()
            self._total_input = 0
            self._total_output = 0
            self._total_cost = 0.0
            self._calls_count = 0
            self._by_model.clear()


class TokenBudget:
    """Token budget management with alerts.

    Usage:
        budget = TokenBudget(daily_limit=1_000_000)

        # Check before making a call
        if budget.can_use(estimated_tokens):
            # Make the call
            budget.use(actual_tokens)

        # Get remaining budget
        remaining = budget.remaining()
    """

    def __init__(
        self,
        daily_limit: int = 0,
        monthly_limit: int = 0,
        alert_threshold: float = 0.8,
        on_alert: Optional[Callable[[str, float], None]] = None,
    ):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.alert_threshold = alert_threshold
        self.on_alert = on_alert

        self._daily_used = 0
        self._monthly_used = 0
        self._last_daily_reset = datetime.now().date()
        self._last_monthly_reset = (datetime.now().year, datetime.now().month)
        self._lock = threading.Lock()
        self._alerts_sent: set = set()

    def _check_reset(self):
        """Check and reset counters if needed."""
        now = datetime.now()
        today = now.date()
        this_month = (now.year, now.month)

        if today != self._last_daily_reset:
            self._daily_used = 0
            self._last_daily_reset = today
            self._alerts_sent.discard("daily_80")
            self._alerts_sent.discard("daily_100")

        if this_month != self._last_monthly_reset:
            self._monthly_used = 0
            self._last_monthly_reset = this_month
            self._alerts_sent.discard("monthly_80")
            self._alerts_sent.discard("monthly_100")

    def can_use(self, tokens: int) -> bool:
        """Check if tokens can be used within budget.

        Args:
            tokens: Number of tokens to check

        Returns:
            True if within budget, False otherwise
        """
        with self._lock:
            self._check_reset()

            if self.daily_limit > 0:
                if self._daily_used + tokens > self.daily_limit:
                    return False

            if self.monthly_limit > 0:
                if self._monthly_used + tokens > self.monthly_limit:
                    return False

            return True

    def use(self, tokens: int) -> bool:
        """Record token usage.

        Args:
            tokens: Number of tokens used

        Returns:
            True if within budget, False if exceeded
        """
        with self._lock:
            self._check_reset()

            self._daily_used += tokens
            self._monthly_used += tokens

            # Check for alerts
            self._check_alerts()

            # Check if exceeded
            exceeded = False
            if self.daily_limit > 0 and self._daily_used > self.daily_limit:
                exceeded = True
            if self.monthly_limit > 0 and self._monthly_used > self.monthly_limit:
                exceeded = True

            return not exceeded

    def _check_alerts(self):
        """Check and send budget alerts."""
        if not self.on_alert:
            return

        # Daily alerts
        if self.daily_limit > 0:
            usage_pct = self._daily_used / self.daily_limit

            if usage_pct >= 1.0 and "daily_100" not in self._alerts_sent:
                self.on_alert("daily_budget_exceeded", usage_pct)
                self._alerts_sent.add("daily_100")
            elif usage_pct >= self.alert_threshold and "daily_80" not in self._alerts_sent:
                self.on_alert("daily_budget_warning", usage_pct)
                self._alerts_sent.add("daily_80")

        # Monthly alerts
        if self.monthly_limit > 0:
            usage_pct = self._monthly_used / self.monthly_limit

            if usage_pct >= 1.0 and "monthly_100" not in self._alerts_sent:
                self.on_alert("monthly_budget_exceeded", usage_pct)
                self._alerts_sent.add("monthly_100")
            elif usage_pct >= self.alert_threshold and "monthly_80" not in self._alerts_sent:
                self.on_alert("monthly_budget_warning", usage_pct)
                self._alerts_sent.add("monthly_80")

    def remaining(self) -> Dict[str, int]:
        """Get remaining budget.

        Returns:
            Dictionary with remaining daily and monthly tokens
        """
        with self._lock:
            self._check_reset()

            return {
                "daily": max(0, self.daily_limit - self._daily_used) if self.daily_limit > 0 else -1,
                "monthly": max(0, self.monthly_limit - self._monthly_used) if self.monthly_limit > 0 else -1,
            }

    def usage(self) -> Dict[str, Dict[str, Any]]:
        """Get current usage statistics.

        Returns:
            Dictionary with daily and monthly usage
        """
        with self._lock:
            self._check_reset()

            result = {}

            if self.daily_limit > 0:
                result["daily"] = {
                    "used": self._daily_used,
                    "limit": self.daily_limit,
                    "remaining": max(0, self.daily_limit - self._daily_used),
                    "percentage": (self._daily_used / self.daily_limit) * 100,
                }

            if self.monthly_limit > 0:
                result["monthly"] = {
                    "used": self._monthly_used,
                    "limit": self.monthly_limit,
                    "remaining": max(0, self.monthly_limit - self._monthly_used),
                    "percentage": (self._monthly_used / self.monthly_limit) * 100,
                }

            return result

    def reset(self, daily: bool = True, monthly: bool = True):
        """Reset budget counters.

        Args:
            daily: Reset daily counter
            monthly: Reset monthly counter
        """
        with self._lock:
            if daily:
                self._daily_used = 0
                self._alerts_sent.discard("daily_80")
                self._alerts_sent.discard("daily_100")
            if monthly:
                self._monthly_used = 0
                self._alerts_sent.discard("monthly_80")
                self._alerts_sent.discard("monthly_100")
