"""
Google Gemini 1.5 Flash — LLM Client Implementation.

Extends BaseLLMClient — only implements the Gemini-specific API call.
All retry, circuit breaker, and logging logic lives in the base class.

Model: gemini-1.5-flash (free tier, no credit card needed)
SDK: google-generativeai
"""

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from config.settings import GeminiConfig
from config.circuit_breaker import CircuitBreaker
from config.logger import get_logger
from models.base_client import BaseLLMClient

logger = get_logger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Google Gemini 1.5 Flash client.

    Wraps the google-generativeai SDK with:
      - Timeout enforcement
      - Gemini-specific exception mapping
      - Safety settings configuration
    """

    def __init__(self, config: GeminiConfig):
        self._config = config

        # Initialize circuit breaker for Gemini
        circuit_breaker = CircuitBreaker(
            name="gemini",
            failure_threshold=config.max_retries,
            recovery_timeout=60.0,
        )

        super().__init__(
            max_retries=config.max_retries,
            retry_delay_seconds=config.retry_delay_seconds,
            circuit_breaker=circuit_breaker,
        )

        # Configure Gemini SDK with API key
        genai.configure(api_key=config.api_key)

        # Initialize the generative model
        self._model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config=genai.GenerationConfig(
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
            ),
        )

        logger.info(
            "GeminiClient initialized",
            extra={
                "extra_fields": {
                    "model": config.model_name,
                    "max_tokens": config.max_output_tokens,
                    "temperature": config.temperature,
                }
            }
        )

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _call_api(self, prompt: str) -> str:
        """
        Execute Gemini API call.
        Maps Gemini-specific exceptions to standard Python exceptions
        so the base class retry logic works uniformly.

        Args:
            prompt: Complete prompt string

        Returns:
            str: Generated text

        Raises:
            TimeoutError: On request timeout
            PermissionError: On invalid API key
            ConnectionError: On network issues
            RuntimeError: On content safety blocks or other API errors
        """
        try:
            response = self._model.generate_content(
                prompt,
                request_options={"timeout": self._config.timeout_seconds},
            )

            # Check for safety blocks — Gemini may refuse certain content
            if not response.candidates:
                raise RuntimeError(
                    "Gemini returned no candidates — possible safety filter block"
                )

            candidate = response.candidates[0]

            # Check finish reason for safety blocks
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)
                if "SAFETY" in finish_reason:
                    raise RuntimeError(
                        f"Gemini blocked response due to safety filters: {finish_reason}"
                    )

            # Extract text from response
            if not response.text:
                raise RuntimeError("Gemini response has no text content")

            return response.text

        except google_exceptions.DeadlineExceeded as e:
            raise TimeoutError(
                f"Gemini API timeout after {self._config.timeout_seconds}s: {e}"
            )

        except google_exceptions.PermissionDenied as e:
            raise PermissionError(
                f"Gemini API key invalid or unauthorized: {e}. "
                "Check your GEMINI_API_KEY in .env"
            )

        except google_exceptions.ResourceExhausted as e:
            raise ConnectionError(
                f"Gemini rate limit exceeded: {e}. "
                "Free tier allows 15 requests/minute."
            )

        except google_exceptions.ServiceUnavailable as e:
            raise ConnectionError(f"Gemini service unavailable: {e}")

        except (RuntimeError, ValueError):
            raise  # Re-raise our own exceptions unchanged

        except Exception as e:
            raise RuntimeError(f"Unexpected Gemini error: {type(e).__name__}: {e}")

    def health_check(self) -> bool:
        """
        Send a minimal test prompt to verify Gemini is reachable and key is valid.
        Called once at startup.

        Returns:
            bool: True if healthy, False if unreachable
        """
        logger.info("Running Gemini health check")
        try:
            response = self._model.generate_content(
                "Reply with only the word: OK",
                request_options={"timeout": 10},
            )
            is_healthy = bool(response.text and "OK" in response.text.upper())
            logger.info(
                f"Gemini health check {'PASSED' if is_healthy else 'FAILED'}",
                extra={"extra_fields": {"healthy": is_healthy}}
            )
            return is_healthy

        except Exception as e:
            logger.error(
                f"Gemini health check FAILED: {type(e).__name__}: {e}",
                extra={"extra_fields": {"healthy": False}}
            )
            return False