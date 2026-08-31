"""Optional, provider-neutral LLM access for AI-enriched RCA.

The application always has a deterministic no-LLM analysis path. This module
only constructs a model after Bedrock or Ollama is selected explicitly (or
``auto`` detects a configured provider).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import Config


LOGGER = logging.getLogger(__name__)
SUPPORTED_LLM_PROVIDERS = {"none", "auto", "bedrock", "ollama"}


class LLMProviderUnavailable(RuntimeError):
    """Raised when AI enrichment was requested without a usable provider."""


@dataclass(frozen=True)
class LLMProviderStatus:
    """Safe provider state for capability checks and dashboard display."""

    requested: str
    active: str
    configured: bool
    ready: bool
    model: str | None
    detail: str

    @property
    def display_name(self) -> str:
        return {
            "bedrock": "AWS Bedrock",
            "ollama": "Ollama",
            "none": "No LLM",
        }.get(self.active, self.active.title())


def _requested_provider(provider: str | None = None) -> str:
    return str(provider if provider is not None else Config.LLM_PROVIDER).strip().lower()


def _active_provider(requested: str) -> str:
    if requested != "auto":
        return requested
    if Config.has_aws_credentials():
        return "bedrock"
    if Config.OLLAMA_EXPLICITLY_CONFIGURED:
        return "ollama"
    return "none"


def get_llm_provider_status(provider: str | None = None) -> LLMProviderStatus:
    """Return configuration status without making a network or billable call."""

    requested = _requested_provider(provider)
    if requested not in SUPPORTED_LLM_PROVIDERS:
        return LLMProviderStatus(
            requested=requested,
            active="none",
            configured=False,
            ready=False,
            model=None,
            detail="Unsupported LLM_PROVIDER. Use none, auto, bedrock, or ollama.",
        )

    active = _active_provider(requested)
    if active == "none":
        detail = (
            "No provider was auto-detected; deterministic RCA remains available."
            if requested == "auto"
            else "AI enrichment is disabled; deterministic RCA remains available."
        )
        return LLMProviderStatus(requested, active, True, False, None, detail)

    if active == "bedrock":
        has_identity = Config.has_aws_credentials()
        credential_detail = (
            "A detectable AWS identity is configured."
            if has_identity
            else "No AWS identity was detected. Configure a profile, workload identity, or temporary credentials."
        )
        return LLMProviderStatus(
            requested,
            active,
            True,
            has_identity,
            Config.BEDROCK_MODEL_ID,
            credential_detail,
        )

    return LLMProviderStatus(
        requested,
        active,
        True,
        True,
        Config.OLLAMA_MODEL,
        "The configured Ollama endpoint is checked when analysis starts.",
    )


@lru_cache(maxsize=8)
def _build_llm(
    provider: str,
    model: str,
    region_or_url: str,
    timeout_seconds: int,
) -> BaseChatModel:
    if provider == "bedrock":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=model,
            region_name=region_or_url,
            model_kwargs={"temperature": 0.1, "max_tokens": 2000},
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - guarded by requirements
            raise LLMProviderUnavailable(
                "Ollama support is not installed. Install requirements.txt and restart the dashboard."
            ) from exc

        return ChatOllama(
            model=model,
            base_url=region_or_url,
            temperature=0.1,
            client_kwargs={"timeout": float(timeout_seconds)},
        )

    raise LLMProviderUnavailable("No LLM provider is active.")


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a cached LangChain chat model for the active provider."""

    status = get_llm_provider_status(provider)
    if not status.ready or status.active == "none" or not status.model:
        raise LLMProviderUnavailable(status.detail)

    endpoint = Config.AWS_REGION if status.active == "bedrock" else Config.OLLAMA_BASE_URL
    try:
        return _build_llm(
            status.active,
            status.model,
            endpoint,
            Config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
    except LLMProviderUnavailable:
        raise
    except Exception as exc:
        raise LLMProviderUnavailable(
            f"Could not initialize {status.display_name}: {exc.__class__.__name__}."
        ) from exc


def reset_llm_cache() -> None:
    """Clear cached clients after runtime configuration changes or in tests."""

    _build_llm.cache_clear()


MIN_DELAY_BETWEEN_CALLS = 2
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
_last_call_time = 0.0


def rate_limited_invoke(chain: Any, input_vars: dict, max_retries: int = MAX_RETRIES):
    """Invoke a chain with bounded throttling retries and no secret logging."""

    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_DELAY_BETWEEN_CALLS:
        time.sleep(MIN_DELAY_BETWEEN_CALLS - elapsed)

    for attempt in range(max_retries + 1):
        try:
            _last_call_time = time.monotonic()
            return chain.invoke(input_vars)
        except Exception as exc:
            message = str(exc).lower()
            throttled = "throttling" in message or "too many requests" in message or "429" in message
            if throttled and attempt < max_retries:
                wait_seconds = BACKOFF_FACTOR ** (attempt + 1)
                LOGGER.warning("LLM provider throttled; retrying in %s seconds", wait_seconds)
                time.sleep(wait_seconds)
                continue
            raise

    raise LLMProviderUnavailable("The LLM provider did not complete after bounded retries.")


__all__ = [
    "LLMProviderStatus",
    "LLMProviderUnavailable",
    "SUPPORTED_LLM_PROVIDERS",
    "get_llm",
    "get_llm_provider_status",
    "rate_limited_invoke",
    "reset_llm_cache",
]
