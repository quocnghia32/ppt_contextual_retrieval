"""
Rate limiting and token management utilities.
"""
import time
import asyncio
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import tiktoken
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


@dataclass
class RateLimitState:
    """Track rate limit state for API calls."""
    requests: deque = field(default_factory=lambda: deque(maxlen=100))
    tokens: deque = field(default_factory=lambda: deque(maxlen=100))
    max_requests_per_minute: int = 50
    max_tokens_per_minute: int = 100000


class RateLimiter:
    """
    Rate limiter for API calls with token tracking.

    Handles both request-based and token-based rate limiting.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 50,
        max_tokens_per_minute: int = 100000
    ):
        self.max_rpm = max_requests_per_minute
        self.max_tpm = max_tokens_per_minute
        self.states: Dict[str, RateLimitState] = {}
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _get_state(self, key: str = "default") -> RateLimitState:
        """Get or create rate limit state for a key."""
        if key not in self.states:
            self.states[key] = RateLimitState(
                max_requests_per_minute=self.max_rpm,
                max_tokens_per_minute=self.max_tpm
            )
        return self.states[key]

    def _clean_old_entries(self, state: RateLimitState):
        """Remove entries older than 1 minute."""
        now = time.time()
        one_minute_ago = now - 60

        # Clean requests
        while state.requests and state.requests[0] < one_minute_ago:
            state.requests.popleft()

        # Clean tokens
        while state.tokens and state.tokens[0][0] < one_minute_ago:
            state.tokens.popleft()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using word count * 1.3")
            return int(len(text.split()) * 1.3)

    async def wait_if_needed(
        self,
        key: str = "default",
        estimated_tokens: int = 0
    ):
        """
        Wait if rate limit would be exceeded.

        Args:
            key: Rate limit key (e.g., "openai")
            estimated_tokens: Estimated tokens for this request
        """
        state = self._get_state(key)
        self._clean_old_entries(state)

        now = time.time()

        # Check request limit
        if len(state.requests) >= state.max_requests_per_minute:
            oldest_request = state.requests[0]
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached for {key}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.1)
                self._clean_old_entries(state)

        # Check token limit
        if estimated_tokens > 0:
            current_tokens = sum(t[1] for t in state.tokens)
            if current_tokens + estimated_tokens > state.max_tokens_per_minute:
                # Wait until oldest tokens expire
                if state.tokens:
                    oldest_token_time = state.tokens[0][0]
                    wait_time = 60 - (now - oldest_token_time)
                    if wait_time > 0:
                        logger.info(
                            f"Token limit reached for {key}, waiting {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time + 0.1)
                        self._clean_old_entries(state)

        # Record this request
        state.requests.append(now)
        if estimated_tokens > 0:
            state.tokens.append((now, estimated_tokens))

    def get_stats(self, key: str = "default") -> Dict:
        """Get current rate limit stats."""
        state = self._get_state(key)
        self._clean_old_entries(state)

        return {
            "requests_in_last_minute": len(state.requests),
            "tokens_in_last_minute": sum(t[1] for t in state.tokens),
            "requests_remaining": state.max_requests_per_minute - len(state.requests),
            "tokens_remaining": state.max_tokens_per_minute - sum(t[1] for t in state.tokens)
        }


# Global rate limiter instance
rate_limiter = RateLimiter()


def with_retry(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 60
):
    """
    Decorator for retrying API calls with exponential backoff.

    Handles rate limit errors and temporary failures.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((
            Exception,  # Catch all for now, refine in production
        )),
        reraise=True
    )


class TokenBudget:
    """
    Manage token budget for operations.

    Useful for controlling costs in batch processing.
    """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def can_afford(self, text: str) -> bool:
        """Check if we can afford to process this text."""
        tokens = len(self.encoding.encode(text))
        return self.used_tokens + tokens <= self.max_tokens

    def consume(self, text: str) -> int:
        """
        Consume tokens from budget.

        Returns number of tokens consumed.
        """
        tokens = len(self.encoding.encode(text))
        self.used_tokens += tokens
        return tokens

    def remaining(self) -> int:
        """Get remaining token budget."""
        return self.max_tokens - self.used_tokens

    def reset(self):
        """Reset token budget."""
        self.used_tokens = 0


# Example usage
async def example_usage():
    """Example of rate limiter usage."""
    # Initialize rate limiter
    limiter = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=1000)

    # Before making API call
    text = "This is a test prompt"
    estimated_tokens = limiter.count_tokens(text)

    # Wait if needed
    await limiter.wait_if_needed(key="openai", estimated_tokens=estimated_tokens)

    # Make API call here
    # ...

    # Check stats
    stats = limiter.get_stats("openai")
    print(f"Rate limit stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())
