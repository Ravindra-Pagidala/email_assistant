"""
Email Generator Service — Orchestration Layer.

This is the equivalent of a Spring Boot @Service class.
It sits between the app entry point and the LLM clients.
It does NOT know about HTTP, CLI, or any delivery mechanism.
It ONLY knows: take inputs → build prompt → call LLM → return response.

Responsibilities:
  - Input validation (delegates to prompt module)
  - Prompt construction (delegates to advanced_prompt module)
  - LLM call orchestration (delegates to model clients)
  - Fallback handling (if primary model fails → try secondary)
"""

from typing import Optional
from models.base_client import BaseLLMClient, EmailGenerationRequest, EmailGenerationResponse
from prompts.advanced_prompt import build_prompt
from config.logger import get_logger, ContextLogger
import uuid

logger = get_logger(__name__)


class EmailGeneratorService:
    """
    Core business logic service for email generation.

    Supports two models — primary and optional fallback.
    If primary model fails (all retries + circuit breaker open),
    automatically falls back to secondary model.

    Like HubSpot's email generation service — they run primary + shadow
    models simultaneously and fall back transparently.

    Args:
        primary_client: Main LLM client to use (e.g., GeminiClient)
        fallback_client: Optional backup client (e.g., GroqClient)
    """

    def __init__(
        self,
        primary_client: BaseLLMClient,
        fallback_client: Optional[BaseLLMClient] = None,
    ):
        self._primary = primary_client
        self._fallback = fallback_client

        logger.info(
            "EmailGeneratorService initialized",
            extra={
                "extra_fields": {
                    "primary_model": primary_client.model_name,
                    "fallback_model": fallback_client.model_name if fallback_client else "none",
                }
            }
        )

    def generate(
        self,
        intent: str,
        key_facts: list,
        tone: str,
        correlation_id: Optional[str] = None,
    ) -> EmailGenerationResponse:
        """
        Generate a professional email from structured inputs.

        Flow:
          1. Validate inputs (via prompt module)
          2. Build advanced prompt
          3. Try primary model
          4. If primary fails → try fallback (if configured)
          5. Return response (success or failure)

        Args:
            intent: Purpose of the email
            key_facts: List of facts to include
            tone: Desired tone (formal/casual/urgent/empathetic)
            correlation_id: Optional request ID for tracing

        Returns:
            EmailGenerationResponse: Always returns, never raises
        """
        corr_id = correlation_id or str(uuid.uuid4())[:8]
        ctx_logger = ContextLogger(logger, corr_id)

        ctx_logger.info(
            "Email generation request received",
            extra={
                "extra_fields": {
                    "intent": intent[:80],
                    "tone": tone,
                    "facts_count": len(key_facts) if key_facts else 0,
                }
            }
        )

        # Step 1 + 2: Validate inputs and build prompt
        # ValueError raised here if inputs are invalid — caller should handle
        try:
            prompt = build_prompt(intent, key_facts, tone)
        except ValueError as e:
            ctx_logger.error(f"Input validation failed: {e}")
            return EmailGenerationResponse.failure(
                model_name=self._primary.model_name,
                error_message=f"Invalid input: {str(e)}"
            )

        # Step 3: Build request DTO
        request = EmailGenerationRequest(
            intent=intent,
            key_facts=key_facts,
            tone=tone,
            correlation_id=corr_id,
        )

        # Step 4: Try primary model
        ctx_logger.info(f"Trying primary model: {self._primary.model_name}")
        response = self._primary.generate_email(prompt, request)

        # Step 5: If primary failed and fallback is available, try fallback
        if not response.success and self._fallback is not None:
            ctx_logger.warning(
                f"Primary model failed — attempting fallback: {self._fallback.model_name}",
                extra={
                    "extra_fields": {
                        "primary_error": response.error_message,
                        "fallback_model": self._fallback.model_name,
                    }
                }
            )
            response = self._fallback.generate_email(prompt, request)

            if response.success:
                ctx_logger.info(
                    f"Fallback model succeeded",
                    extra={"extra_fields": {"fallback_model": self._fallback.model_name}}
                )
            else:
                ctx_logger.error(
                    "Both primary and fallback models failed",
                    extra={
                        "extra_fields": {
                            "primary_model": self._primary.model_name,
                            "fallback_model": self._fallback.model_name,
                        }
                    }
                )

        return response

    def health_check_all(self) -> dict:
        """
        Run health checks on all configured models.
        Returns a status dict for startup verification.

        Returns:
            dict: Health status for each model
        """
        status = {}

        logger.info("Running health checks on all LLM clients")

        status[self._primary.model_name] = self._primary.health_check()

        if self._fallback:
            status[self._fallback.model_name] = self._fallback.health_check()

        all_healthy = all(status.values())
        logger.info(
            f"Health check complete — {'ALL HEALTHY' if all_healthy else 'SOME UNHEALTHY'}",
            extra={"extra_fields": {"status": status}}
        )

        return status