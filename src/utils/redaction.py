"""Best-effort secret redaction before external AI or search calls."""

from __future__ import annotations

import re


_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "[REDACTED_AWS_ACCESS_KEY]",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[REDACTED_JWT]",
    ),
    (
        re.compile(r"(?i)(authorization\s*:\s*(?:bearer|token|basic)\s+)[^\s]+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|client_secret|access_token|auth_token|token|"
            r"api[_-]?key)\b(\s*[:=]\s*)['\"]?[^\s,'\";]{4,}['\"]?"
        ),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(r"(?i)(https?://[^\s:/@]+:)[^\s/@]+(@)"),
        r"\1[REDACTED]\2",
    ),
)


def redact_sensitive_text(value: object, *, limit: int | None = None) -> str:
    """Return text with common credential formats replaced by markers."""

    redacted = str(value or "")
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted[:limit] if limit is not None else redacted
