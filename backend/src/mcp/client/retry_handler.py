"""Retry logic with exponential backoff and jitter."""

import asyncio
import logging
import random
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryDecision(str, Enum):
    """Decision on whether to retry an operation."""

    RETRY = "retry"
    NO_RETRY = "no_retry"
    FATAL = "fatal"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter.

    Features:
    - Exponential backoff between retries
    - Random jitter to prevent thundering herd
    - Status-based retry decisions
    - Configurable max retries and delays
    """

    # HTTP status codes that should trigger a retry
    RETRYABLE_STATUS_CODES = {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    }

    # Status codes that should not be retried
    NON_RETRYABLE_STATUS_CODES = {
        400,  # Bad Request
        401,  # Unauthorized
        403,  # Forbidden
        404,  # Not Found
    }

    def __init__(self, config: RetryConfig | None = None):
        """Initialize the retry handler.

        Args:
            config: Retry configuration.
        """
        self.config = config or RetryConfig()

    def should_retry(
        self,
        attempt: int,
        status_code: int | None = None,
        exception: Exception | None = None,
    ) -> RetryDecision:
        """Determine if an operation should be retried.

        Args:
            attempt: Current attempt number (0-indexed).
            status_code: HTTP status code if applicable.
            exception: Exception that was raised if any.

        Returns:
            RetryDecision indicating whether to retry.
        """
        # Check max retries
        if attempt >= self.config.max_retries:
            logger.debug(f"Max retries ({self.config.max_retries}) reached")
            return RetryDecision.NO_RETRY

        # Check status code
        if status_code is not None:
            if status_code in self.NON_RETRYABLE_STATUS_CODES:
                return RetryDecision.FATAL
            if status_code in self.RETRYABLE_STATUS_CODES:
                return RetryDecision.RETRY

        # Check exception type
        if exception is not None:
            # Retry on connection errors
            exception_name = type(exception).__name__
            if exception_name in ("ConnectionError", "TimeoutError", "ConnectTimeout"):
                return RetryDecision.RETRY

            # Don't retry on validation errors
            if exception_name in ("ValidationError", "ValueError", "TypeError"):
                return RetryDecision.FATAL

        # Default to retry for unknown errors
        return RetryDecision.RETRY

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds before next retry.
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter = delay * self.config.jitter_factor * random.random()
        delay = delay + jitter

        return delay

    async def execute_with_retry(
        self,
        operation,
        *args,
        **kwargs,
    ):
        """Execute an async operation with retry logic.

        Args:
            operation: Async callable to execute.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result of the operation.

        Raises:
            Exception: The last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Extract status code if it's an HTTP error
                status_code = None
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                elif hasattr(e, "status_code"):
                    status_code = e.status_code

                # Check if we should retry
                decision = self.should_retry(attempt, status_code, e)

                if decision == RetryDecision.FATAL:
                    logger.error(f"Fatal error, not retrying: {e}")
                    raise

                if decision == RetryDecision.NO_RETRY:
                    logger.warning(f"Not retrying after {attempt + 1} attempts: {e}")
                    raise

                # Calculate and apply delay
                delay = self.get_delay(attempt)
                logger.info(f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.2f}s: {e}")
                await asyncio.sleep(delay)

        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception

    def execute_sync_with_retry(
        self,
        operation,
        *args,
        **kwargs,
    ):
        """Execute a sync operation with retry logic.

        Args:
            operation: Callable to execute.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result of the operation.

        Raises:
            Exception: The last exception if all retries fail.
        """
        import time

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Extract status code if it's an HTTP error
                status_code = None
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                elif hasattr(e, "status_code"):
                    status_code = e.status_code

                # Check if we should retry
                decision = self.should_retry(attempt, status_code, e)

                if decision == RetryDecision.FATAL:
                    logger.error(f"Fatal error, not retrying: {e}")
                    raise

                if decision == RetryDecision.NO_RETRY:
                    logger.warning(f"Not retrying after {attempt + 1} attempts: {e}")
                    raise

                # Calculate and apply delay
                delay = self.get_delay(attempt)
                logger.info(f"Retry {attempt + 1}/{self.config.max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)

        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception
