"""
Circuit Breaker Pattern — inspired by Netflix Hystrix and Microsoft's Polly library.

How it works (3 states):
  CLOSED   → Normal operation. Requests go through. Failures are counted.
  OPEN     → Too many failures. All requests are blocked immediately (fail fast).
  HALF_OPEN → Recovery probe. One request allowed through to test if service recovered.

Why it matters:
  Without a circuit breaker, if Gemini API goes down, your code keeps hammering it
  with retries — wasting time and burning through rate limits.
  With a circuit breaker, after 3 failures it stops trying for 60 seconds,
  then automatically tests recovery. This is production-grade resilience.

Used by: Netflix, Microsoft Azure SDK, Salesforce Apex retry policies.
"""

import time
from enum import Enum
from threading import Lock
from typing import Callable, Any, Optional
from config.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"         # Normal — requests pass through
    OPEN = "OPEN"             # Broken — requests blocked
    HALF_OPEN = "HALF_OPEN"   # Testing recovery — one request allowed


class CircuitBreakerOpenError(Exception):
    """
    Raised when a request is attempted while circuit is OPEN.
    Caller should handle this by using a fallback or returning a graceful error.
    """
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker for LLM API calls.

    Args:
        name: Identifier for this circuit (e.g., "gemini", "groq")
        failure_threshold: How many consecutive failures before OPEN
        recovery_timeout: Seconds to wait before attempting HALF_OPEN
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = Lock()  # Thread safety for concurrent requests

    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitState:
        """
        Internal state check — also handles automatic OPEN → HALF_OPEN transition
        when recovery_timeout has elapsed. Must be called with lock held.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        f"Circuit '{self.name}' transitioning OPEN → HALF_OPEN "
                        f"after {elapsed:.1f}s recovery wait"
                    )
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute func through the circuit breaker.

        If OPEN → raises CircuitBreakerOpenError immediately (fail fast).
        If CLOSED or HALF_OPEN → executes func, updates state based on result.

        Args:
            func: The callable to protect (e.g., API call function)
            *args, **kwargs: Passed through to func

        Returns:
            Whatever func returns on success

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN
            Exception: Re-raises whatever func raises (after recording failure)
        """
        with self._lock:
            current_state = self._get_state()

            if current_state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit '{self.name}' is OPEN — request blocked (fail fast)",
                    extra={"extra_fields": {"circuit": self.name, "state": "OPEN"}}
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service appears to be unavailable. "
                    f"Will retry after {self.recovery_timeout}s."
                )

        # Execute outside the lock so we don't block other threads during API call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except CircuitBreakerOpenError:
            raise  # Don't count circuit breaker errors as failures

        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        """Reset circuit to CLOSED on successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit '{self.name}' probe succeeded — transitioning HALF_OPEN → CLOSED"
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self, error: Exception) -> None:
        """Record failure and potentially OPEN the circuit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit '{self.name}' recorded failure "
                f"{self._failure_count}/{self.failure_threshold}: {type(error).__name__}: {error}",
                extra={
                    "extra_fields": {
                        "circuit": self.name,
                        "failure_count": self._failure_count,
                        "error_type": type(error).__name__,
                    }
                }
            )

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit '{self.name}' OPENED after {self._failure_count} failures. "
                    f"Blocking requests for {self.recovery_timeout}s.",
                    extra={"extra_fields": {"circuit": self.name, "state": "OPEN"}}
                )

    def get_status(self) -> dict:
        """Return current circuit status for health checks."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._get_state().value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Manually reset circuit to CLOSED — useful for testing."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")