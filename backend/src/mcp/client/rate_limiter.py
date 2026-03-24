"""Rate limiting support using token bucket algorithm."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    calls_per_minute: int = 5
    calls_per_day: int = 500
    burst_limit: int = 3


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Implements a token bucket algorithm with support for:
    - Per-minute rate limiting
    - Per-day rate limiting
    - Burst allowance
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize the rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config or RateLimitConfig()

        # Token bucket state
        self._tokens = float(self.config.calls_per_minute)
        self._last_refill = time.monotonic()
        self._refill_rate = self.config.calls_per_minute / 60.0  # tokens per second

        # Daily tracking
        self._daily_calls = 0
        self._daily_reset_time = time.time()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.config.calls_per_minute,
            self._tokens + (elapsed * self._refill_rate),
        )
        self._last_refill = now

    def _check_daily_reset(self) -> None:
        """Reset daily counter if a day has passed."""
        now = time.time()
        if now - self._daily_reset_time >= 86400:  # 24 hours
            self._daily_calls = 0
            self._daily_reset_time = now

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens for an API call.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            True if tokens were acquired, False if rate limited.
        """
        async with self._lock:
            self._refill_tokens()
            self._check_daily_reset()

            # Check daily limit
            if self._daily_calls >= self.config.calls_per_day:
                logger.warning("Daily rate limit reached")
                return False

            # Check token bucket
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._daily_calls += 1
                return True

            return False

    async def wait_for_token(self, timeout: float = 60.0) -> bool:
        """Wait until a token is available.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if token was acquired, False if timeout.
        """
        start = time.monotonic()

        while True:
            if await self.acquire():
                return True

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return False

            # Calculate wait time until next token
            async with self._lock:
                tokens_needed = 1 - self._tokens
                wait_time = tokens_needed / self._refill_rate

            # Wait, but not longer than remaining timeout
            wait_time = min(wait_time, timeout - elapsed, 1.0)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def get_status(self) -> dict:
        """Get current rate limiter status.

        Returns:
            Dictionary with rate limiter state.
        """
        self._refill_tokens()
        self._check_daily_reset()

        return {
            "tokens_available": self._tokens,
            "max_tokens": self.config.calls_per_minute,
            "daily_calls_used": self._daily_calls,
            "daily_calls_limit": self.config.calls_per_day,
            "burst_limit": self.config.burst_limit,
        }

    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        self._tokens = float(self.config.calls_per_minute)
        self._last_refill = time.monotonic()
        self._daily_calls = 0
        self._daily_reset_time = time.time()


class MultiSourceRateLimiter:
    """Rate limiter for multiple data sources."""

    def __init__(self, configs: dict[str, RateLimitConfig] | None = None):
        """Initialize with per-source configurations.

        Args:
            configs: Dictionary mapping source names to their rate limit configs.
        """
        self._limiters: dict[str, RateLimiter] = {}
        self._default_config = RateLimitConfig()

        if configs:
            for source, config in configs.items():
                self._limiters[source] = RateLimiter(config)

    def get_limiter(self, source: str) -> RateLimiter:
        """Get or create a rate limiter for a source.

        Args:
            source: Source name.

        Returns:
            RateLimiter for the source.
        """
        if source not in self._limiters:
            self._limiters[source] = RateLimiter(self._default_config)
        return self._limiters[source]

    async def acquire(self, source: str, tokens: int = 1) -> bool:
        """Acquire tokens for a specific source.

        Args:
            source: Source name.
            tokens: Number of tokens.

        Returns:
            True if acquired, False if rate limited.
        """
        limiter = self.get_limiter(source)
        return await limiter.acquire(tokens)

    async def wait_for_token(self, source: str, timeout: float = 60.0) -> bool:
        """Wait for a token from a specific source.

        Args:
            source: Source name.
            timeout: Maximum wait time.

        Returns:
            True if acquired, False if timeout.
        """
        limiter = self.get_limiter(source)
        return await limiter.wait_for_token(timeout)

    def get_all_status(self) -> dict[str, dict]:
        """Get status for all sources.

        Returns:
            Dictionary mapping source names to their status.
        """
        return {
            source: limiter.get_status()
            for source, limiter in self._limiters.items()
        }
