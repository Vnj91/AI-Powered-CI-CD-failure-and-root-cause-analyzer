"""Map heterogeneous CI/CD evidence into a stable failure category taxonomy."""

from __future__ import annotations

import re
from enum import Enum
from typing import Mapping, Optional


class FailureCategory(str, Enum):
    DEPENDENCY = "dependency"
    TEST = "test"
    BUILD = "build"
    LINT = "lint"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    INFRASTRUCTURE = "infrastructure"
    CONFIGURATION = "configuration"
    SECRETS = "secrets"
    DEPLOYMENT = "deployment"
    NETWORK = "network"
    PERMISSIONS = "permissions"
    UNKNOWN = "unknown"


# Log parser ErrorCategory values and common synonyms → taxonomy
_LOG_PARSER_MAP = {
    "dependency": FailureCategory.DEPENDENCY,
    "syntax": FailureCategory.BUILD,
    "test_failure": FailureCategory.TEST,
    "configuration": FailureCategory.CONFIGURATION,
    "build": FailureCategory.BUILD,
    "runtime": FailureCategory.BUILD,
    "network": FailureCategory.NETWORK,
    "permission": FailureCategory.PERMISSIONS,
    "timeout": FailureCategory.INFRASTRUCTURE,
    "unknown": FailureCategory.UNKNOWN,
}

# Triage refined categories → taxonomy
_TRIAGE_MAP = {
    "missing_package": FailureCategory.DEPENDENCY,
    "version_conflict": FailureCategory.DEPENDENCY,
    "incompatible_dependency": FailureCategory.DEPENDENCY,
    "syntax_error": FailureCategory.BUILD,
    "type_error": FailureCategory.BUILD,
    "import_error": FailureCategory.DEPENDENCY,
    "assertion_failure": FailureCategory.TEST,
    "test_timeout": FailureCategory.TEST,
    "fixture_error": FailureCategory.TEST,
    "missing_env_var": FailureCategory.SECRETS,
    "invalid_config": FailureCategory.CONFIGURATION,
    "missing_file": FailureCategory.CONFIGURATION,
    "network_error": FailureCategory.NETWORK,
    "permission_denied": FailureCategory.PERMISSIONS,
    "resource_limit": FailureCategory.INFRASTRUCTURE,
    "unknown": FailureCategory.UNKNOWN,
}

# Keyword hints from logs, job names, or commit messages
_KEYWORD_RULES: list[tuple[tuple[str, ...], FailureCategory]] = [
    (("docker", "container", "dockerfile", "docker build"), FailureCategory.DOCKER),
    (("kubernetes", "k8s", "helm", "kubectl", "deployment.yaml"), FailureCategory.KUBERNETES),
    (("terraform", "pulumi", "cloudformation", "infrastructure"), FailureCategory.INFRASTRUCTURE),
    (("ruff", "flake8", "pylint", "eslint", "lint", "static analysis", "mypy"), FailureCategory.LINT),
    (("pytest", "unittest", "jest", "mvn test", "go test", "test failed"), FailureCategory.TEST),
    (("pip install", "npm install", "dependency", "requirements.txt", "package.json"), FailureCategory.DEPENDENCY),
    (("secret", "token", "credential", "api key", "env var"), FailureCategory.SECRETS),
    (("deploy", "release", "publish"), FailureCategory.DEPLOYMENT),
    (("permission denied", "403", "401", "unauthorized", "forbidden"), FailureCategory.PERMISSIONS),
    (("connection refused", "timeout", "dns", "network unreachable"), FailureCategory.NETWORK),
    (("compile", "build failed", "maven", "gradle", "cargo build"), FailureCategory.BUILD),
    (("config", "configuration", "invalid yaml", "invalid json"), FailureCategory.CONFIGURATION),
]


def _normalize(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def map_log_parser_category(category: Optional[str]) -> FailureCategory:
    normalized = _normalize(category)
    return _LOG_PARSER_MAP.get(normalized, FailureCategory.UNKNOWN)


def map_triage_category(category: Optional[str]) -> FailureCategory:
    normalized = _normalize(category)
    return _TRIAGE_MAP.get(normalized, FailureCategory.UNKNOWN) if normalized else FailureCategory.UNKNOWN


def map_from_keywords(*texts: Optional[str]) -> FailureCategory:
    combined = " ".join(_normalize(text) for text in texts if text)
    if not combined:
        return FailureCategory.UNKNOWN

    for keywords, category in _KEYWORD_RULES:
        if any(keyword in combined for keyword in keywords):
            return category
    return FailureCategory.UNKNOWN


def map_failure_category(
    *,
    log_parser_category: Optional[str] = None,
    triage_category: Optional[str] = None,
    failed_step: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    workflow_job_name: Optional[str] = None,
    changed_files: Optional[list[str]] = None,
) -> FailureCategory:
    """Combine available evidence into one taxonomy label."""

    candidates: list[FailureCategory] = []

    if log_parser_category:
        mapped = map_log_parser_category(log_parser_category)
        if mapped != FailureCategory.UNKNOWN:
            candidates.append(mapped)

    if triage_category:
        mapped = map_triage_category(triage_category)
        if mapped != FailureCategory.UNKNOWN:
            candidates.append(mapped)

    keyword_sources = [failed_step, error_type, error_message, workflow_job_name]
    if changed_files:
        keyword_sources.extend(changed_files[:10])

    keyword_category = map_from_keywords(*keyword_sources)
    if keyword_category != FailureCategory.UNKNOWN:
        candidates.append(keyword_category)

    if not candidates:
        return FailureCategory.UNKNOWN

    # Prefer explicit parser/triage labels over keyword inference
    if log_parser_category:
        mapped = map_log_parser_category(log_parser_category)
        if mapped != FailureCategory.UNKNOWN:
            return mapped

    if triage_category:
        mapped = map_triage_category(triage_category)
        if mapped != FailureCategory.UNKNOWN:
            return mapped

    return candidates[0]


def conclusion_to_actual_failure(conclusion: Optional[str]) -> Optional[int]:
    """Map GitHub Actions conclusion to binary failure label when definitive."""

    normalized = _normalize(conclusion)
    if not normalized:
        return None
    if normalized == "success":
        return 0
    if normalized in {"failure", "timed_out", "startup_failure", "action_required"}:
        return 1
    if normalized in {"cancelled", "skipped", "neutral", "stale"}:
        return None
    return None


def conclusion_to_outcome_label(conclusion: Optional[str]) -> str:
    """Human-readable outcome for feedback display."""

    normalized = _normalize(conclusion)
    if normalized == "success":
        return "success"
    if normalized in {"failure", "timed_out", "startup_failure", "action_required"}:
        return "failure"
    if normalized in {"cancelled", "skipped"}:
        return "cancelled"
    if normalized:
        return normalized
    return "unknown"
