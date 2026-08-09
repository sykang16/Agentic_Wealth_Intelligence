"""Unit tests for rate limiter."""

import asyncio
import time

import pytest

from backend.src.mcp.client.rate_limiter import (
    MultiSourceRateLimiter,
    RateLimitConfig,
    RateLimiter,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def limiter(self):
        """Create a rate limiter for testing."""
        config = RateLimitConfig(
            calls_per_minute=5,
            calls_per_day=100,
            burst_limit=3,
        )
        return RateLimiter(config)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful token acquisition."""
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_multiple(self, limiter):
        """Test acquiring multiple tokens."""
        results = []
        for _ in range(5):
            results.append(await limiter.acquire())

        assert all(results)  # All should succeed

    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self, limiter):
        """Test that acquiring beyond limit fails."""
        # Exhaust all tokens
        for _ in range(5):
            await limiter.acquire()

        # This should fail (no tokens left)
        result = await limiter.acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(
            calls_per_minute=60,  # 1 per second
            calls_per_day=1000,
        )
        limiter = RateLimiter(config)

        # Acquire all tokens
        for _ in range(60):
            await limiter.acquire()

        # Should fail immediately
        assert await limiter.acquire() is False

        # Wait for refill (a bit more than 1 second for 1 token)
        await asyncio.sleep(1.1)

        # Should succeed now
        assert await limiter.acquire() is True

    @pytest.mark.asyncio
    async def test_wait_for_token(self):
        """Test waiting for token with timeout."""
        config = RateLimitConfig(
            calls_per_minute=60,  # 1 per second
            calls_per_day=1000,
        )
        limiter = RateLimiter(config)

        # Exhaust tokens
        for _ in range(60):
            await limiter.acquire()

        # Wait for a token (should succeed within 2 seconds)
        result = await limiter.wait_for_token(timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_token_timeout(self):
        """Test waiting for token times out."""
        config = RateLimitConfig(
            calls_per_minute=1,  # Very slow refill
            calls_per_day=1000,
        )
        limiter = RateLimiter(config)

        # Exhaust the single token
        await limiter.acquire()

        # Wait should timeout
        result = await limiter.wait_for_token(timeout=0.1)
        assert result is False

    def test_get_status(self, limiter):
        """Test getting rate limiter status."""
        status = limiter.get_status()

        assert "tokens_available" in status
        assert "max_tokens" in status
        assert "daily_calls_used" in status
        assert "daily_calls_limit" in status
        assert status["max_tokens"] == 5
        assert status["daily_calls_limit"] == 100

    @pytest.mark.asyncio
    async def test_daily_limit(self):
        """Test daily call limit enforcement."""
        config = RateLimitConfig(
            calls_per_minute=1000,  # High per-minute limit
            calls_per_day=5,  # Low daily limit
        )
        limiter = RateLimiter(config)

        # Use up daily limit
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True

        # Should now fail due to daily limit
        result = await limiter.acquire()
        assert result is False

    def test_reset(self, limiter):
        """Test resetting the rate limiter."""
        # Use some tokens
        asyncio.run(limiter.acquire())
        asyncio.run(limiter.acquire())

        # Reset
        limiter.reset()

        status = limiter.get_status()
        assert status["tokens_available"] == 5
        assert status["daily_calls_used"] == 0


class TestMultiSourceRateLimiter:
    """Tests for MultiSourceRateLimiter class."""

    @pytest.fixture
    def multi_limiter(self):
        """Create a multi-source rate limiter."""
        configs = {
            "alpha_vantage": RateLimitConfig(calls_per_minute=5, calls_per_day=500),
            "news_api": RateLimitConfig(calls_per_minute=10, calls_per_day=100),
        }
        return MultiSourceRateLimiter(configs)

    @pytest.mark.asyncio
    async def test_acquire_from_source(self, multi_limiter):
        """Test acquiring from a specific source."""
        result = await multi_limiter.acquire("alpha_vantage")
        assert result is True

    @pytest.mark.asyncio
    async def test_sources_are_independent(self, multi_limiter):
        """Test that rate limits are independent per source."""
        # Exhaust alpha_vantage
        for _ in range(5):
            await multi_limiter.acquire("alpha_vantage")

        # alpha_vantage should be exhausted
        assert await multi_limiter.acquire("alpha_vantage") is False

        # But news_api should still work
        assert await multi_limiter.acquire("news_api") is True

    @pytest.mark.asyncio
    async def test_unknown_source_uses_default(self, multi_limiter):
        """Test that unknown sources use default config."""
        # This should create a new limiter with default config
        result = await multi_limiter.acquire("unknown_source")
        assert result is True

    def test_get_all_status(self, multi_limiter):
        """Test getting status for all sources."""
        status = multi_limiter.get_all_status()

        assert "alpha_vantage" in status
        assert "news_api" in status
        assert status["alpha_vantage"]["max_tokens"] == 5
        assert status["news_api"]["max_tokens"] == 10

    @pytest.mark.asyncio
    async def test_wait_for_token_specific_source(self, multi_limiter):
        """Test waiting for token from a specific source."""
        result = await multi_limiter.wait_for_token("alpha_vantage", timeout=1.0)
        assert result is True
