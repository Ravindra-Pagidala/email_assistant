"""
Groq (LLaMA 3 70B) — LLM Client Implementation.

Extends BaseLLMClient — only implements the Groq-specific API call.
Groq runs Meta's open-source LLaMA 3 70B on custom inference hardware.
This is what makes it free AND fast — Groq's LPU chips are extremely efficient.

Model: llama3-70b-8192 (free tier, no credit card needed)
SDK: groq (official Groq Python SDK)
"""

from groq import Groq, APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError
from config.settings import GroqConfig
from config.circuit_breaker import CircuitBreaker
from config.logger import get_logger
from models.base_client import BaseLLMClient

logger = get_logger(__name__)


class GroqClient(BaseLLMClient):
    """
    Groq LLaMA 3 70B client.

    Wraps the official Groq Python SDK with:
      - Timeout enforcement
      - Groq-specific exception mapping to standard exceptions
      - Chat completion format (Groq uses OpenAI-compatible messages API)
    """

    def __init__(self, config: GroqConfig):
        self._config = config

        # Initialize circuit breaker for Groq
        circuit_breaker = CircuitBreaker(
            name="groq",
            failure_threshold=config.max_retries,
            recovery_timeout=60.0,
        )

        super().__init__(
            max_retries=config.max_retries,
            retry_delay_seconds=config.retry_delay_seconds,
            circuit_breaker=circuit_breaker,
        )

        # Initialize Groq SDK client
        self._client = Groq(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

        logger.info(
            "GroqClient initialized",
            extra={
                "extra_fields": {
                    "model": config.model_name,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                }
            }
        )

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def _call_api(self, prompt: str) -> str:
        """
        Execute Groq API call using chat completions format.
        Groq uses OpenAI-compatible messages API — prompt goes as user message.

        Args:
            prompt: Complete prompt string

        Returns:
            str: Generated text

        Raises:
            TimeoutError: On request timeout
            PermissionError: On invalid API key
            ConnectionError: On network issues or rate limits
            RuntimeError: On empty response or unexpected errors
        """
        try:
            completion = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional email writing assistant. "
                            "Follow all instructions precisely and output only the requested email."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            # Validate response structure
            if not completion.choices:
                raise RuntimeError("Groq returned no choices in response")

            message = completion.choices[0].message
            if not message or not message.content:
                raise RuntimeError("Groq returned empty message content")

            content = message.content.strip()
            if not content:
                raise RuntimeError("Groq returned whitespace-only response")

            return content

        except APITimeoutError as e:
            raise TimeoutError(
                f"Groq API timeout after {self._config.timeout_seconds}s: {e}"
            )

        except AuthenticationError as e:
            raise PermissionError(
                f"Groq API key invalid: {e}. "
                "Check your GROQ_API_KEY in .env"
            )

        except RateLimitError as e:
            raise ConnectionError(
                f"Groq rate limit exceeded: {e}. "
                "Free tier allows 30 requests/minute."
            )

        except APIConnectionError as e:
            raise ConnectionError(f"Groq connection error: {e}")

        except (RuntimeError, ValueError):
            raise  # Re-raise our own exceptions unchanged

        except Exception as e:
            raise RuntimeError(f"Unexpected Groq error: {type(e).__name__}: {e}")

    def health_check(self) -> bool:
        """
        Send a minimal test message to verify Groq is reachable and key is valid.

        Returns:
            bool: True if healthy, False if unreachable
        """
        logger.info("Running Groq health check")
        try:
            completion = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=[{"role": "user", "content": "Reply with only the word: OK"}],
                max_tokens=10,
            )
            response_text = completion.choices[0].message.content or ""
            is_healthy = "OK" in response_text.upper()
            logger.info(
                f"Groq health check {'PASSED' if is_healthy else 'FAILED'}",
                extra={"extra_fields": {"healthy": is_healthy}}
            )
            return is_healthy

        except Exception as e:
            logger.error(
                f"Groq health check FAILED: {type(e).__name__}: {e}",
                extra={"extra_fields": {"healthy": False}}
            )
            return False