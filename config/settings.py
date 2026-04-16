"""
Centralized configuration — equivalent to application.properties in Spring Boot.
All environment variables, model configs, retry settings, and timeouts live here.
Never hardcode values in business logic — always read from this module.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from config.logger import get_logger

logger = get_logger(__name__)

# Load .env file into environment variables at import time
load_dotenv()


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model_name: str =  "gemma-3-27b-it"
    max_output_tokens: int = 1024
    temperature: float = 0.7          # 0.0 = deterministic, 1.0 = creative
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0  # Base delay — doubles on each retry (exponential backoff)


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


@dataclass(frozen=True)
class AppConfig:
    gemini: GeminiConfig
    groq: GroqConfig
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    results_dir: str = "results"
    scenarios_path: str = "scenarios/test_scenarios.json"

    # Circuit breaker thresholds
    # If a model fails this many times consecutively → mark it as OPEN (disabled)
    circuit_breaker_failure_threshold: int = 3
    # After this many seconds → try again (HALF-OPEN state)
    circuit_breaker_recovery_seconds: int = 60


def load_config() -> AppConfig:
    """
    Load and validate all configuration from environment variables.
    Raises clear, actionable errors if required keys are missing.
    This is called ONCE at startup in app.py.

    Returns:
        AppConfig: Fully validated configuration object

    Raises:
        EnvironmentError: If any required API key is missing
    """
    logger.info("Loading application configuration from environment")

    gemini_key = _require_env("GEMINI_API_KEY")
    groq_key = _require_env("GROQ_API_KEY")

    gemini_config = GeminiConfig(api_key=gemini_key)
    groq_config = GroqConfig(api_key=groq_key)

    config = AppConfig(
        gemini=gemini_config,
        groq=groq_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/app.log"),
        results_dir=os.getenv("RESULTS_DIR", "results"),
        scenarios_path=os.getenv("SCENARIOS_PATH", "scenarios/test_scenarios.json"),
    )

    logger.info(
        "Configuration loaded successfully",
        extra={
            "extra_fields": {
                "gemini_model": config.gemini.model_name,
                "groq_model": config.groq.model_name,
                "log_level": config.log_level,
            }
        }
    )

    return config


def _require_env(key: str) -> str:
    """
    Read a required environment variable.
    Raises EnvironmentError with a helpful message if missing.
    Never returns None or empty string.

    Args:
        key: Environment variable name

    Returns:
        str: The value of the environment variable

    Raises:
        EnvironmentError: If variable is missing or empty
    """
    value: Optional[str] = os.getenv(key)

    if not value or value.strip() == "":
        raise EnvironmentError(
            f"\n\n❌ Missing required environment variable: '{key}'\n"
            f"   Steps to fix:\n"
            f"   1. Copy .env.example to .env\n"
            f"   2. Fill in your API key for '{key}'\n"
            f"   3. Get Gemini key free at: https://aistudio.google.com\n"
            f"   4. Get Groq key free at: https://console.groq.com\n"
        )

    return value.strip()