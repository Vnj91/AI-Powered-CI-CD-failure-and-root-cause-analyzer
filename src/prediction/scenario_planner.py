"""Deterministic scenario planning for real GitHub Actions training-data generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Protocol

Outcome = Literal["success", "failure"]
FailureCategory = Literal["lint", "test", "build", "dependency", "docker"]

FAILURE_CATEGORIES: tuple[FailureCategory, ...] = (
    "lint",
    "test",
    "build",
    "dependency",
    "docker",
)

SUCCESS_CHANGE_TYPES: tuple[str, ...] = (
    "documentation",
    "python",
    "test",
    "configuration",
    "docker",
)

SUCCESS_CHANGE_DESCRIPTIONS: dict[str, str] = {
    "documentation": "documentation change",
    "python": "harmless Python change",
    "test": "harmless test change",
    "configuration": "harmless YAML/config change",
    "docker": "harmless Docker-related change",
}

ALLOWED_MODIFY_PREFIXES: tuple[str, ...] = (
    "docs/ml-training/",
    "tests/ml_training_generated/",
    "scripts/ml_training_generated/",
    ".github/ml-training-markers/",
)

ALLOWED_MODIFY_FILES: tuple[str, ...] = (
    ".dockerignore",
)

MAX_RUNS_DEFAULT = 50
BRANCH_PREFIX = "ml-data"

FORBIDDEN_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.env"),
    re.compile(r"(^|/)models/"),
    re.compile(r"(^|/)data/.*\.csv$"),
    re.compile(r"\.github/workflows/.*\.yml$"),
    re.compile(r"(^|/)requirements\.txt$"),
    re.compile(r"(^|/)Dockerfile$"),
    re.compile(r"(^|/)src/prediction/trainer\.py$"),
)

FORBIDDEN_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*rm\s+-rf", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*sudo\s+", re.MULTILINE | re.IGNORECASE),
    re.compile(r";\s*rm\s+", re.IGNORECASE),
    re.compile(r"\$\(", re.MULTILINE),
)


@dataclass(frozen=True)
class FileChange:
    path: str
    content: str
    mode: Literal["create", "append"] = "create"


@dataclass(frozen=True)
class TrainingScenario:
    scenario_id: str
    branch_name: str
    category: Optional[str]
    change_type: str
    files_to_modify: tuple[str, ...]
    file_changes: tuple[FileChange, ...]
    expected_outcome: Outcome
    description: str
    workflow_name: str
    workflow_file: str
    workflow_inputs: dict[str, str] = field(default_factory=dict)
    commit_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "expected_outcome": self.expected_outcome,
            "category": self.category,
            "change_type": self.change_type,
            "files_to_modify": list(self.files_to_modify),
            "description": self.description,
            "branch_name": self.branch_name,
            "workflow_name": self.workflow_name,
            "workflow_file": self.workflow_file,
            "workflow_inputs": dict(self.workflow_inputs),
            "commit_message": self.commit_message,
        }


class ScenarioPlanner(Protocol):
    """Interface for building training scenarios."""

    def plan(self, success_runs: int, failure_runs: int) -> list[TrainingScenario]:
        ...


class DeterministicScenarioPlanner:
    """Build scenarios without any external API credentials."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def plan(self, success_runs: int, failure_runs: int) -> list[TrainingScenario]:
        scenarios = build_training_scenarios(success_runs, failure_runs)
        if self.seed is None:
            return scenarios
        rng = random.Random(self.seed)
        ordered = list(scenarios)
        rng.shuffle(ordered)
        return ordered


class AIScenarioPlanner:
    """Optional AI-suggested scenarios validated against the allowlist before use."""

    def __init__(
        self,
        suggestions: list[dict[str, Any]],
        *,
        seed: Optional[int] = None,
    ):
        self.suggestions = suggestions
        self.seed = seed

    def plan(self, success_runs: int, failure_runs: int) -> list[TrainingScenario]:
        base = DeterministicScenarioPlanner(seed=self.seed).plan(success_runs, failure_runs)
        if not self.suggestions:
            return base
        return build_training_scenarios(
            success_runs,
            failure_runs,
            extra_ai_scenarios=self.suggestions,
        )


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_path_allowed(path: str) -> bool:
    normalized = _normalize_path(path)
    if any(pattern.search(normalized) for pattern in FORBIDDEN_PATH_PATTERNS):
        return False
    if normalized in ALLOWED_MODIFY_FILES:
        return True
    return any(normalized.startswith(prefix) for prefix in ALLOWED_MODIFY_PREFIXES)


def validate_file_changes(changes: Iterable[FileChange]) -> list[str]:
    errors: list[str] = []
    for change in changes:
        if not is_path_allowed(change.path):
            errors.append(f"Path not allowlisted: {change.path}")
        for pattern in FORBIDDEN_CONTENT_PATTERNS:
            if pattern.search(change.content):
                errors.append(f"Forbidden content pattern in {change.path}")
    return errors


def validate_ai_scenario(scenario: dict[str, Any]) -> list[str]:
    """Validate optional AI-suggested scenarios before any filesystem use."""

    errors: list[str] = []
    required = {"scenario_id", "category", "files_to_modify", "change_type", "expected_outcome"}
    missing = required - set(scenario)
    if missing:
        errors.append(f"Missing required AI scenario fields: {sorted(missing)}")
        return errors

    outcome = scenario.get("expected_outcome")
    if outcome not in {"success", "failure"}:
        errors.append(f"Invalid expected_outcome: {outcome}")

    category = scenario.get("category")
    if outcome == "failure" and category not in FAILURE_CATEGORIES:
        errors.append(f"Invalid failure category: {category}")

    files = scenario.get("files_to_modify") or []
    if not isinstance(files, list):
        errors.append("files_to_modify must be a list")
        return errors

    for path in files:
        if not isinstance(path, str) or not is_path_allowed(path):
            errors.append(f"AI suggested disallowed path: {path}")

    if scenario.get("shell_command") or scenario.get("command"):
        errors.append("AI scenarios must not include executable shell commands")

    return errors


def distribute_failure_categories(failure_runs: int) -> list[FailureCategory]:
    if failure_runs <= 0:
        return []
    categories: list[FailureCategory] = []
    for index in range(failure_runs):
        categories.append(FAILURE_CATEGORIES[index % len(FAILURE_CATEGORIES)])
    return categories


def _success_branch_name(index: int) -> str:
    return f"{BRANCH_PREFIX}/success-{index:03d}"


def _failure_branch_name(category: FailureCategory, index: int) -> str:
    return f"{BRANCH_PREFIX}/failure-{category}-{index:03d}"


def _build_success_changes(index: int, change_type: str) -> tuple[FileChange, ...]:
    marker = f"ml-training-success-{index:03d}"
    if change_type == "documentation":
        return (
            FileChange(
                path=f"docs/ml-training/{marker}.md",
                content=(
                    f"# ML training data marker\n\n"
                    f"Generated documentation-only success scenario {marker}.\n"
                    f"This file exists solely to create commit-level feature variation.\n"
                ),
            ),
        )
    if change_type == "test":
        return (
            FileChange(
                path=f"tests/ml_training_generated/test_{marker}.py",
                content=(
                    f'"""Harmless generated success test for {marker}."""\n\n\n'
                    f"def test_{marker}():\n"
                    f'    assert "{marker}" == "{marker}"\n'
                ),
            ),
        )
    if change_type == "python":
        return (
            FileChange(
                path=f"scripts/ml_training_generated/marker_{index:03d}.py",
                content=(
                    f'"""Harmless generated marker module for {marker}."""\n\n'
                    f"MARKER = {index!r}\n\n"
                    f"def describe() -> str:\n"
                    f'    return "ml-training-success-marker"\n'
                ),
            ),
        )
    if change_type == "configuration":
        return (
            FileChange(
                path=f".github/ml-training-markers/{marker}.yml",
                content=(
                    f"# Harmless YAML marker for ML training scenario {marker}\n"
                    f"marker: {marker}\n"
                    f"purpose: training-data-generation\n"
                ),
            ),
        )
    # docker
    return (
        FileChange(
            path=f"docs/ml-training/docker-notes-{index:03d}.md",
            content=f"# Docker-related training marker\n\nScenario {marker} documents a harmless docker note.\n",
        ),
        FileChange(
            path=".dockerignore",
            content=f"\n# ml-training marker {marker}\n",
            mode="append",
        ),
    )


def _build_failure_changes(index: int, category: FailureCategory) -> tuple[FileChange, ...]:
    marker = f"ml-training-failure-{category}-{index:03d}"
    return (
        FileChange(
            path=f"docs/ml-training/{marker}.md",
            content=(
                f"# Controlled failure marker\n\n"
                f"Branch for controlled {category} failure scenario {marker}.\n"
                f"The actual failure is produced by workflow_dispatch on test-failure.yml.\n"
            ),
        ),
    )


def _success_description(change_type: str, index: int) -> str:
    label = SUCCESS_CHANGE_DESCRIPTIONS.get(change_type, change_type)
    return f"Success scenario {index:03d}: {label} on isolated branch."


def _failure_description(category: FailureCategory, index: int) -> str:
    return (
        f"Controlled failure scenario {index:03d}: trigger test-failure.yml "
        f"with failure_type={category}."
    )


def build_training_scenarios(
    success_runs: int,
    failure_runs: int,
    *,
    extra_ai_scenarios: Optional[list[dict[str, Any]]] = None,
) -> list[TrainingScenario]:
    """Build a deterministic list of training scenarios."""

    if success_runs < 0 or failure_runs < 0:
        raise ValueError("success_runs and failure_runs must be non-negative")
    if success_runs + failure_runs == 0:
        raise ValueError("At least one scenario is required")

    scenarios: list[TrainingScenario] = []

    for index in range(1, success_runs + 1):
        change_type = SUCCESS_CHANGE_TYPES[(index - 1) % len(SUCCESS_CHANGE_TYPES)]
        branch = _success_branch_name(index)
        changes = _build_success_changes(index, change_type)
        errors = validate_file_changes(changes)
        if errors:
            raise ValueError(f"Invalid success scenario {branch}: {errors}")
        scenarios.append(
            TrainingScenario(
                scenario_id=f"success-{index:03d}",
                branch_name=branch,
                category=None,
                change_type=change_type,
                files_to_modify=tuple(change.path for change in changes),
                file_changes=changes,
                expected_outcome="success",
                description=_success_description(change_type, index),
                workflow_name="CI/CD Pipeline",
                workflow_file="ci.yml",
                commit_message=f"ml-training: add success scenario {branch} ({change_type})",
            )
        )

    failure_categories = distribute_failure_categories(failure_runs)
    category_counts: dict[str, int] = {cat: 0 for cat in FAILURE_CATEGORIES}
    for category in failure_categories:
        category_counts[category] += 1
        branch = _failure_branch_name(category, category_counts[category])
        changes = _build_failure_changes(category_counts[category], category)
        errors = validate_file_changes(changes)
        if errors:
            raise ValueError(f"Invalid failure scenario {branch}: {errors}")
        scenarios.append(
            TrainingScenario(
                scenario_id=f"failure-{category}-{category_counts[category]:03d}",
                branch_name=branch,
                category=category,
                change_type=f"controlled_{category}_failure",
                files_to_modify=tuple(change.path for change in changes),
                file_changes=changes,
                expected_outcome="failure",
                description=_failure_description(category, category_counts[category]),
                workflow_name="Controlled CI Failure Generator (Dev/Testing Only)",
                workflow_file="test-failure.yml",
                workflow_inputs={"failure_type": category},
                commit_message=f"ml-training: add controlled failure marker {branch} ({category})",
            )
        )

    if extra_ai_scenarios:
        for raw in extra_ai_scenarios:
            errors = validate_ai_scenario(raw)
            if errors:
                raise ValueError(f"Rejected AI scenario: {errors}")
            scenarios.append(
                TrainingScenario(
                    scenario_id=str(raw["scenario_id"]),
                    branch_name=f"{BRANCH_PREFIX}/ai-{raw['scenario_id']}",
                    category=str(raw.get("category")) if raw.get("expected_outcome") == "failure" else None,
                    change_type=str(raw["change_type"]),
                    files_to_modify=tuple(str(path) for path in raw["files_to_modify"]),
                    file_changes=tuple(
                        FileChange(path=str(path), content=f"# AI-described harmless marker for {raw['scenario_id']}\n")
                        for path in raw["files_to_modify"]
                    ),
                    expected_outcome=raw["expected_outcome"],
                    description=str(raw.get("description") or f"AI-described scenario {raw['scenario_id']}"),
                    workflow_name="CI/CD Pipeline" if raw["expected_outcome"] == "success" else "Controlled CI Failure Generator (Dev/Testing Only)",
                    workflow_file="ci.yml" if raw["expected_outcome"] == "success" else "test-failure.yml",
                    workflow_inputs={"failure_type": str(raw["category"])} if raw["expected_outcome"] == "failure" else {},
                    commit_message=f"ml-training: AI-described scenario {raw['scenario_id']}",
                )
            )

    return scenarios


def validate_run_count(total_runs: int, *, allow_large_run_set: bool = False) -> list[str]:
    warnings: list[str] = []
    if total_runs <= 0:
        raise ValueError("total runs must be positive")
    if total_runs > MAX_RUNS_DEFAULT and not allow_large_run_set:
        raise ValueError(
            f"Requested {total_runs} runs exceeds default maximum of {MAX_RUNS_DEFAULT}. "
            "Pass --allow-large-run-set to proceed."
        )
    if total_runs > MAX_RUNS_DEFAULT:
        warnings.append(
            f"WARNING: {total_runs} runs exceeds recommended maximum of {MAX_RUNS_DEFAULT}."
        )
    return warnings


def summarize_scenario_distribution(scenarios: list[TrainingScenario]) -> dict[str, Any]:
    failure_counts = {category: 0 for category in FAILURE_CATEGORIES}
    success_counts = {change_type: 0 for change_type in SUCCESS_CHANGE_TYPES}

    for scenario in scenarios:
        if scenario.expected_outcome == "failure" and scenario.category:
            failure_counts[scenario.category] += 1
        elif scenario.expected_outcome == "success":
            success_counts[scenario.change_type] += 1

    return {
        "total": len(scenarios),
        "success": sum(1 for scenario in scenarios if scenario.expected_outcome == "success"),
        "failure": sum(1 for scenario in scenarios if scenario.expected_outcome == "failure"),
        "failures_by_category": failure_counts,
        "success_by_change_type": success_counts,
    }
