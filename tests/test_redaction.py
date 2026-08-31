from __future__ import annotations

from src.utils.redaction import redact_sensitive_text


def test_redacts_common_ci_secret_formats():
    source = "\n".join(
        (
            "Authorization: Bearer bearer-value-that-must-not-leak",
            "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz123456",
            "AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF",
            "api_key='super-secret-value'",
            "https://build-user:build-password@example.test/artifact",
        )
    )

    redacted = redact_sensitive_text(source)

    assert "bearer-value-that-must-not-leak" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "AKIA1234567890ABCDEF" not in redacted
    assert "super-secret-value" not in redacted
    assert "build-password" not in redacted
    assert redacted.count("REDACTED") >= 5


def test_redaction_preserves_normal_error_evidence():
    source = "ModuleNotFoundError: No module named requests at tests/test_api.py:42"
    assert redact_sensitive_text(source) == source


def test_redaction_applies_length_limit_after_scrubbing():
    value = "token=abcdefghijk " + ("x" * 100)
    redacted = redact_sensitive_text(value, limit=25)
    assert "abcdefghijk" not in redacted
    assert len(redacted) == 25
