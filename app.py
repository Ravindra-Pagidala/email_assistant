"""
Application Entry Point — app.py

Equivalent to Main.java in Spring Boot.
This is where:
  - Logging is initialized
  - Config is loaded and validated
  - LLM clients are instantiated
  - Health checks are run
  - EmailGeneratorService is wired together

Run this file directly to test a single email generation:
  python app.py

For the full evaluation pipeline, run:
  python evaluation/evaluator.py --model gemini
  python evaluation/evaluator.py --model groq
"""

import os
import sys

# Add project root to Python path so all imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.logger import setup_logging, get_logger
from config.settings import load_config
from models.gemini_client import GeminiClient
from models.groq_client import GroqClient
from models.email_generator_service import EmailGeneratorService


def create_app() -> EmailGeneratorService:
    """
    Application factory — initializes and wires all components.
    Returns a ready-to-use EmailGeneratorService.

    This pattern (factory function) makes it easy to:
      - Test with mock clients
      - Switch models without changing business logic
      - Initialize once and reuse across evaluation runs

    Returns:
        EmailGeneratorService: Fully initialized service

    Raises:
        EnvironmentError: If API keys are missing
        RuntimeError: If health checks fail
    """
    logger = get_logger(__name__)

    # ── 1. Load config ───────────────────────────────────────────
    config = load_config()

    # ── 2. Setup logging with config values ──────────────────────
    os.makedirs("logs", exist_ok=True)
    setup_logging(log_level=config.log_level, log_file=config.log_file)
    logger = get_logger(__name__)  # Re-get after setup

    logger.info("=" * 60)
    logger.info("Email Generation Assistant — Starting Up")
    logger.info("=" * 60)

    # ── 3. Ensure results directory exists ───────────────────────
    os.makedirs(config.results_dir, exist_ok=True)
    logger.info(f"Results directory ready: {config.results_dir}")

    # ── 4. Initialize LLM clients ─────────────────────────────────
    logger.info("Initializing LLM clients...")
    gemini_client = GeminiClient(config.gemini)
    groq_client = GroqClient(config.groq)

    # ── 5. Wire service with primary + fallback ───────────────────
    # Gemini = primary, Groq = fallback
    # If Gemini is down, Groq automatically takes over
    service = EmailGeneratorService(
        primary_client=gemini_client,
        fallback_client=groq_client,
    )

    # ── 6. Health checks ──────────────────────────────────────────
    logger.info("Running startup health checks...")
    health_status = service.health_check_all()

    unhealthy = [model for model, healthy in health_status.items() if not healthy]
    if unhealthy:
        logger.warning(
            f"Health check WARNING: {unhealthy} not responding. "
            "Continuing — may affect evaluation results."
        )
    else:
        logger.info("All models healthy — ready to generate emails")

    return service


def get_gemini_service() -> EmailGeneratorService:
    """
    Create a service using ONLY Gemini (no fallback).
    Used by evaluator when running Model A evaluation.
    """
    config = load_config()
    gemini_client = GeminiClient(config.gemini)
    return EmailGeneratorService(primary_client=gemini_client)


def get_groq_service() -> EmailGeneratorService:
    """
    Create a service using ONLY Groq/LLaMA (no fallback).
    Used by evaluator when running Model B evaluation.
    """
    config = load_config()
    groq_client = GroqClient(config.groq)
    return EmailGeneratorService(primary_client=groq_client)


# ── Manual test — run this file directly to test one email ──────
if __name__ == "__main__":
    # Initialize logging for standalone run
    os.makedirs("logs", exist_ok=True)
    setup_logging(log_level="INFO", log_file="logs/app.log")
    logger = get_logger(__name__)

    print("\n" + "=" * 60)
    print("  Email Generation Assistant — Manual Test")
    print("=" * 60 + "\n")

    try:
        service = create_app()

        # Test with a sample scenario
        response = service.generate(
            intent="Follow up after a job interview",
            key_facts=[
                "Interview was on April 14th, 2026",
                "Position: Senior AI Engineer",
                "Interviewer: Sarah Chen, Head of AI",
                "Discussed transformer architecture and prompt engineering",
                "Want to express strong continued interest",
            ],
            tone="formal",
        )

        if response.success:
            print("\n✅ EMAIL GENERATED SUCCESSFULLY")
            print(f"Model: {response.model_name}")
            print(f"Latency: {response.latency_ms:.0f}ms")
            print("\n" + "─" * 60)
            print(response.email_text)
            print("─" * 60)
        else:
            print(f"\n❌ EMAIL GENERATION FAILED")
            print(f"Error: {response.error_message}")

    except EnvironmentError as e:
        print(f"\n❌ CONFIGURATION ERROR:\n{e}")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)