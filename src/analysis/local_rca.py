"""Deterministic fallback analysis for uploaded or pasted CI logs.

The full product uses Bedrock and Tavily to enrich a failure.  This module is
deliberately credential-free: it turns parser evidence into conservative,
clearly labelled guidance so the dashboard remains useful during setup and
incident response when external services are unavailable.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from src.graph.state import DebuggingBrief, FixSuggestion
from src.prediction.category_mapper import FailureCategory, map_failure_category
from src.tools.log_parser import ParsedError


_GUIDANCE: dict[FailureCategory, dict[str, Any]] = {
    FailureCategory.DEPENDENCY: {
        "severity": "high",
        "cause": "The build environment could not resolve or load a required dependency.",
        "detail": (
            "The first concrete dependency error usually means the package is missing, the lock file is out of sync, "
            "or the CI runtime is using a different interpreter or registry configuration than development."
        ),
        "fixes": [
            ("Reproduce the install from a clean environment", "Use the CI runtime version and install only from the committed manifest or lock file.", ["Create a clean virtual environment or container.", "Run the exact dependency-install command from the failed job.", "Confirm the missing or conflicting package and version."]),
            ("Synchronize and pin dependencies", "Update the dependency manifest and lock file together, then verify that the required package is explicitly declared.", ["Add or correct the dependency constraint.", "Regenerate the lock file with the supported package manager.", "Commit both files and rerun the clean install."]),
            ("Validate the CI dependency cache", "A stale cache can preserve an incompatible environment even after the manifest changes.", ["Temporarily disable or invalidate the cache key.", "Key caches on the lock-file hash and runtime version.", "Rerun the job and restore caching only after a clean pass."]),
        ],
    },
    FailureCategory.TEST: {
        "severity": "medium",
        "cause": "A test assertion, fixture, or test-process condition failed.",
        "detail": (
            "The failing test and the first assertion are stronger root-cause signals than the final non-zero exit line. "
            "Reproduce that test in the same runtime before changing production code."
        ),
        "fixes": [
            ("Run the smallest failing test", "Reproduce the named test with verbose output and without parallel execution.", ["Copy the exact test identifier from the log.", "Run it with verbose output in the CI runtime.", "Compare inputs, fixtures, timezone, and environment variables with CI."]),
            ("Inspect the first assertion and its inputs", "Trace the earliest assertion failure back to the code or fixture that produced the unexpected value.", ["Record expected and actual values.", "Check the relevant change and fixture setup.", "Fix the implementation or correct an outdated expectation."]),
            ("Remove nondeterminism", "If the test passes locally, control clocks, randomness, network calls, ordering, and shared state.", ["Seed randomness and freeze time where appropriate.", "Mock external dependencies at a stable boundary.", "Run the test repeatedly before re-enabling parallel execution."]),
        ],
    },
    FailureCategory.LINT: {
        "severity": "medium",
        "cause": "A static-analysis or formatting rule rejected the submitted source.",
        "detail": "The first linter diagnostic normally identifies the exact file, line, rule, and remediation.",
        "fixes": [
            ("Run the repository linter locally", "Use the same command and tool version configured in CI.", ["Copy the linter command from the workflow.", "Run it against the reported file.", "Fix the first diagnostic before reviewing follow-on errors."]),
            ("Align tool configuration", "Confirm local and CI lint versions and configuration files are identical.", ["Pin the linter version.", "Check config discovery from the CI working directory.", "Remove conflicting editor-only overrides."]),
            ("Add a pre-commit check", "Catch the same diagnostics before code reaches CI.", ["Configure the repository lint command as a pre-commit hook.", "Run it across the repository once.", "Document the command for contributors."]),
        ],
    },
    FailureCategory.BUILD: {
        "severity": "high",
        "cause": "Compilation, import validation, or application packaging failed.",
        "detail": "Build logs often contain cascaded errors; the earliest compiler, syntax, or import diagnostic is usually the actionable one.",
        "fixes": [
            ("Fix the first build diagnostic", "Start at the earliest concrete error rather than the final exit-code message.", ["Locate the first file-and-line diagnostic.", "Reproduce the exact build command locally.", "Fix that error and rerun before addressing downstream messages."]),
            ("Match the CI toolchain", "Use the same runtime, compiler, build flags, and architecture as CI.", ["Confirm runtime and compiler versions.", "Compare environment variables and working directory.", "Pin versions in the project and CI configuration."]),
            ("Perform a clean rebuild", "Generated files or stale caches can make build behavior diverge.", ["Remove only documented build artifacts and caches.", "Restore dependencies from the lock file.", "Build again from a clean checkout."]),
        ],
    },
    FailureCategory.CONFIGURATION: {
        "severity": "high",
        "cause": "The job is missing a required file, value, or valid configuration.",
        "detail": "Configuration failures commonly come from different paths, defaults, or environment variables in CI.",
        "fixes": [
            ("Validate required configuration at startup", "Fail early with the exact missing key or invalid value.", ["List required values for the failed step.", "Check their names and scopes in CI.", "Add schema or startup validation."]),
            ("Compare CI and local paths", "Confirm the workflow working directory and case-sensitive file names match the repository.", ["Print the working directory and expected path without exposing secrets.", "Verify the file is committed and not ignored.", "Use repository-relative paths."]),
            ("Provide safe environment defaults", "Use defaults only for non-secret optional settings and document required overrides.", ["Separate required from optional settings.", "Keep secrets in the CI secret store.", "Add a credential-free configuration smoke test."]),
        ],
    },
    FailureCategory.SECRETS: {
        "severity": "critical",
        "cause": "A required credential or secret is missing, expired, or unavailable in this workflow context.",
        "detail": "Secret values must never be printed. Validate presence, scope, environment protection rules, and token lifetime instead.",
        "fixes": [
            ("Verify secret presence and scope", "Confirm the expected secret name exists in the repository or deployment environment.", ["Check the name without printing its value.", "Confirm the workflow/environment can access it.", "Review fork and protected-environment restrictions."]),
            ("Rotate or re-authorize the credential", "Replace expired or revoked credentials using least privilege.", ["Create a replacement with only required scopes.", "Update the secret store.", "Revoke the old credential after validation."]),
            ("Add a redacted preflight", "Detect missing credentials before the expensive part of the job.", ["Check only whether each value is non-empty.", "Never echo raw values.", "Return an actionable missing-secret name."]),
        ],
    },
    FailureCategory.NETWORK: {
        "severity": "medium",
        "cause": "The job could not reliably reach a required network service.",
        "detail": "DNS, TLS, rate limits, transient service errors, and incorrect endpoints should be distinguished before retries are added.",
        "fixes": [
            ("Identify the failing endpoint and status", "Capture the hostname, operation, status code, and timeout without logging credentials.", ["Check DNS and TLS from the runner.", "Confirm the endpoint and region.", "Review provider status and rate-limit headers."]),
            ("Use bounded retries", "Retry only transient failures with exponential backoff and jitter.", ["Classify retryable status codes.", "Set connection and request timeouts.", "Cap attempts and report the final cause."]),
            ("Remove hidden external dependencies", "Mock nonessential services in tests and make integration dependencies explicit.", ["Separate unit and integration jobs.", "Use a controlled test double where appropriate.", "Run live integration checks with scoped credentials."]),
        ],
    },
    FailureCategory.PERMISSIONS: {
        "severity": "high",
        "cause": "The runner identity lacks permission for a file, API, registry, or deployment target.",
        "detail": "Determine whether the denial is from filesystem ownership, GitHub token permissions, cloud IAM, or environment protection.",
        "fixes": [
            ("Identify the denied resource and identity", "Trace the first 401, 403, or permission-denied message to the acting principal.", ["Record the resource name without secrets.", "Confirm which identity the job uses.", "Check the effective permission at the failed step."]),
            ("Grant the smallest required permission", "Update the relevant workflow permission, filesystem ownership, or IAM action only.", ["Map the failed operation to one permission.", "Apply it to the narrowest resource.", "Retest and audit the resulting access."]),
            ("Test permissions before mutation", "Add a read-only preflight for the target service or path.", ["Use a list, describe, or dry-run operation.", "Return an actionable message on denial.", "Keep destructive operations behind protected environments."]),
        ],
    },
    FailureCategory.DOCKER: {
        "severity": "high",
        "cause": "The container image could not be built, started, or validated.",
        "detail": "The failing Dockerfile layer or container health log normally distinguishes build-context, dependency, permission, and startup problems.",
        "fixes": [
            ("Rebuild the failing layer with plain logs", "Disable cached output temporarily so the first failing Dockerfile command is visible.", ["Build with plain progress output.", "Inspect the failing layer and its input files.", "Keep the corrected layer deterministic."]),
            ("Validate build context and ownership", "Confirm required files are not excluded and the runtime user can read and write expected paths.", ["Review .dockerignore.", "Check COPY destinations and ownership.", "Run the image as its configured non-root user."]),
            ("Test startup and health independently", "Separate image construction from application startup diagnosis.", ["Run the image locally with the required environment.", "Inspect container logs.", "Call the configured health endpoint until healthy or timeout."]),
        ],
    },
}


def _generic_guidance() -> dict[str, Any]:
    return {
        "severity": "medium",
        "cause": "The log contains a non-zero or explicit error, but the parser cannot yet identify a specific root-cause family.",
        "detail": "Treat the first concrete diagnostic as primary and keep the raw context available for an AI-enriched or manual review.",
        "fixes": [
            ("Find the first concrete diagnostic", "Ignore final wrapper messages until the earliest error, exception, or failed command is understood.", ["Search upward from the final non-zero exit.", "Capture the failed step and first diagnostic.", "Rerun that command with verbose output."]),
            ("Reproduce in the CI runtime", "Use the same commit, runtime, working directory, and environment shape.", ["Start from a clean checkout.", "Run the exact workflow command.", "Compare only redacted configuration metadata."]),
            ("Escalate with a minimal evidence bundle", "Provide the failed command, first error, relevant stack frames, and recent changes.", ["Redact credentials and personal data.", "Include only the surrounding log lines.", "Use AI enrichment or a domain owner for the remaining diagnosis."]),
        ],
    }


def build_local_debugging_brief(
    error: ParsedError,
    repository: str | None = None,
) -> DebuggingBrief:
    """Build a conservative debugging brief from parser evidence only."""

    started = perf_counter()
    category = map_failure_category(
        log_parser_category=error.error_category.value if error.error_category else None,
        failed_step=error.failed_step,
        error_type=error.error_type,
        error_message=error.error_message,
    )
    guidance = _GUIDANCE.get(category, _generic_guidance())

    suggestions = [
        FixSuggestion(
            priority=index,
            title=title,
            description=description,
            implementation_steps=steps,
            confidence=max(0.55, 0.82 - ((index - 1) * 0.08)),
            source="deterministic_local_guidance",
        )
        for index, (title, description, steps) in enumerate(guidance["fixes"], start=1)
    ]

    affected_files = list(
        dict.fromkeys(frame.file for frame in error.stack_frames if frame.file)
    )[:8]
    affected_components = [error.failed_step] if error.failed_step else []
    confidence = 0.74 if category != FailureCategory.UNKNOWN else 0.45

    return DebuggingBrief(
        title=f"{error.error_type}: {error.error_message[:80]}",
        repository=repository,
        error_type=error.error_type,
        error_message=error.error_message,
        error_category=category.value,
        severity=str(guidance["severity"]),
        root_cause_summary=str(guidance["cause"]),
        root_cause_detailed=(
            f"{guidance['detail']} Evidence detected in this log: "
            f"{error.error_type}: {error.error_message}"
        ),
        affected_files=affected_files,
        affected_components=affected_components,
        fix_suggestions=suggestions,
        research_summary="Credential-free deterministic analysis; no log content was sent to an external service.",
        confidence_score=confidence,
        analysis_duration_seconds=perf_counter() - started,
    )
