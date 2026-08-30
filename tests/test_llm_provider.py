from __future__ import annotations

import pytest

from config import Config
from src.utils import llm


def test_no_llm_provider_keeps_enrichment_disabled(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "none")

    status = llm.get_llm_provider_status()

    assert status.active == "none"
    assert not status.ready
    assert "deterministic RCA" in status.detail
    with pytest.raises(llm.LLMProviderUnavailable, match="deterministic RCA"):
        llm.get_llm()


def test_auto_provider_does_not_assume_ollama_is_running(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(Config, "OLLAMA_EXPLICITLY_CONFIGURED", False)
    monkeypatch.setattr(Config, "has_aws_credentials", classmethod(lambda cls: False))

    status = llm.get_llm_provider_status()

    assert status.active == "none"
    assert not status.ready


def test_explicit_ollama_provider_builds_configured_model(monkeypatch):
    sentinel = object()
    captured = {}
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(Config, "OLLAMA_MODEL", "qwen-test")
    monkeypatch.setattr(Config, "OLLAMA_BASE_URL", "http://ollama.test:11434")

    def fake_build(provider, model, endpoint, timeout):
        captured.update(provider=provider, model=model, endpoint=endpoint, timeout=timeout)
        return sentinel

    monkeypatch.setattr(llm, "_build_llm", fake_build)

    assert llm.get_llm() is sentinel
    assert captured == {
        "provider": "ollama",
        "model": "qwen-test",
        "endpoint": "http://ollama.test:11434",
        "timeout": Config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
    }


def test_bedrock_provider_requires_a_detectable_identity(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(Config, "has_aws_credentials", classmethod(lambda cls: False))

    status = llm.get_llm_provider_status()

    assert status.active == "bedrock"
    assert status.configured
    assert not status.ready
    assert "No AWS identity" in status.detail


def test_invalid_provider_is_actionable(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "not-a-provider")

    status = llm.get_llm_provider_status()

    assert not status.configured
    assert "none, auto, bedrock, or ollama" in status.detail
