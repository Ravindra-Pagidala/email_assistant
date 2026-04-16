"""
Base LLM Client — Abstract base class for all LLM integrations.

This is the equivalent of a Spring Boot abstract Service class.
Gemini and Groq clients both extend this — sharing retry logic,
timeout handling, and circuit breaker integration.

Design pattern: Template Method Pattern
  - Base class defines the algorithm skeleton (retry + circuit breaker)
  - Subclasses implement the actual API call (_call_api)
  - Shared logic stays in ONE place — DRY principle
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from config.logger import get_logger, ContextLogger
from config.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = get_logger(__name__)


@dataclass
class EmailGenerationRequest:
    """
    Input DTO for email generation.
    Equivalent to a Spring Boot @RequestBody DTO.
    """
    intent: str
    key_facts: List[str]
    tone: str
    correlation_id: Optional[str] = None


@dataclass
class EmailGenerationResponse:
    """
    Output DTO for email generation.
    Contains the generated email + metadata for evaluation pipeline.
    """
    email_text: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    @classmethod
    def failure(cls, model_name: str, error_message: str) -> "EmailGenerationResponse":
        """Factory method for failure responses — avoids None checks everywhere."""
        return cls(
            email_text="",
            model_name=model_name,
            success=False,
            error_message=error_message,
        )


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.

    Provides:
      - Exponential backoff retry logic
      - Circuit breaker integration
      - Timeout enforcement
      - Structured logging with correlation IDs
      - Consistent error handling

    Subclasses must implement:
      - _call_api(prompt: str) → str
      - health_check() → bool
      - model_name property
    """

    def __init__(
        self,
        max_retries: int,
        retry_delay_seconds: float,
        circuit_breaker: CircuitBreaker,
    ):
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._circuit_breaker = circuit_breaker

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier e.g. 'gemini-1.5-flash'"""
        pass

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Make the actual API call to the LLM.
        Must return the raw text response string.
        Subclasses implement provider-specific SDK calls here.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the model is reachable and responding.
        Called once at startup to catch misconfigured API keys early.
        """
        pass

    def generate_email(
        self,
        prompt: str,
        request: EmailGenerationRequest,
    ) -> EmailGenerationResponse:
        """
        Generate email with full retry + circuit breaker protection.

        Retry strategy: Exponential backoff
          Attempt 1 → fail → wait 2s
          Attempt 2 → fail → wait 4s
          Attempt 3 → fail → wait 8s
          → Return failure response

        Circuit breaker: After 3 consecutive failures → OPEN
          All subsequent calls blocked until recovery_timeout passes.

        Args:
            prompt: The fully constructed prompt string
            request: Original request DTO (for logging context)

        Returns:
            EmailGenerationResponse: Always returns — never raises to caller
        """
        ctx_logger = ContextLogger(logger, request.correlation_id)
        ctx_logger.info(
            f"Starting email generation",
            extra={
                "extra_fields": {
                    "model": self.model_name,
                    "tone": request.tone,
                    "intent_preview": request.intent[:50],
                }
            }
        )

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                start_time = time.time()

                ctx_logger.debug(f"Attempt {attempt}/{self._max_retries} — calling {self.model_name}")

                # Circuit breaker wraps the actual API call
                raw_response = self._circuit_breaker.call(self._call_api, prompt)

                latency_ms = (time.time() - start_time) * 1000

                # Validate response is not empty
                if not raw_response or not raw_response.strip():
                    raise ValueError(
                        f"{self.model_name} returned empty response on attempt {attempt}"
                    )

                ctx_logger.info(
                    f"Email generated successfully",
                    extra={
                        "extra_fields": {
                            "model": self.model_name,
                            "attempt": attempt,
                            "latency_ms": round(latency_ms, 2),
                            "response_length": len(raw_response),
                        }
                    }
                )

                return EmailGenerationResponse(
                    email_text=raw_response.strip(),
                    model_name=self.model_name,
                    latency_ms=latency_ms,
                    success=True,
                )

            except CircuitBreakerOpenError as e:
                # Circuit is OPEN — no point retrying, fail immediately
                ctx_logger.error(
                    f"Circuit breaker OPEN for {self.model_name} — aborting retries",
                    extra={"extra_fields": {"model": self.model_name}}
                )
                return EmailGenerationResponse.failure(
                    model_name=self.model_name,
                    error_message=f"Circuit breaker open: {str(e)}"
                )

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                ctx_logger.warning(
                    f"Attempt {attempt}/{self._max_retries} failed",
                    extra={
                        "extra_fields": {
                            "model": self.model_name,
                            "attempt": attempt,
                            "error_type": error_type,
                            "error": str(e)[:200],
                        }
                    }
                )

                if attempt < self._max_retries:
                    # Exponential backoff: 2s, 4s, 8s...
                    wait = self._retry_delay_seconds * (2 ** (attempt - 1))
                    ctx_logger.info(
                        f"Waiting {wait}s before retry {attempt + 1}",
                        extra={"extra_fields": {"wait_seconds": wait}}
                    )
                    time.sleep(wait)

        # All retries exhausted
        error_msg = f"All {self._max_retries} attempts failed. Last error: {str(last_error)}"
        ctx_logger.error(
            f"Email generation failed after all retries",
            extra={
                "extra_fields": {
                    "model": self.model_name,
                    "max_retries": self._max_retries,
                    "last_error": str(last_error),
                }
            }
        )

        return EmailGenerationResponse.failure(
            model_name=self.model_name,
            error_message=error_msg,
        )