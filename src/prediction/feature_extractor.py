"""Deterministic feature extraction for CI/CD failure prediction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .schemas import WorkflowRunRecord


class FailureFeatureExtractor:
    """Convert workflow run records into model-ready features."""

    EXTENSION_FEATURES = [
        "py",
        "yml",
        "yaml",
        "json",
        "tf",
        "hcl",
        "dockerfile",
        "sh",
        "md",
        "js",
        "ts",
        "java",
        "xml",
        "other",
    ]

    CORE_FEATURES = [
        "files_changed",
        "lines_added",
        "lines_deleted",
        "number_of_commits",
        "dependency_files_changed",
        "ci_workflow_files_changed",
        "docker_files_changed",
        "test_files_changed",
        "infrastructure_files_changed",
        "previous_run_status_success",
        "previous_run_status_failure",
        "previous_run_status_other",
        "previous_failure_count",
        "recent_failure_count",
        "recent_failure_rate",
        "previous_failures_same_workflow",
        "previous_failures_same_branch",
        "similar_previous_failures",
        "changed_extension_diversity",
    ]

    FEATURE_COLUMNS = CORE_FEATURES + [f"changed_ext_{name}_count" for name in EXTENSION_FEATURES]

    DEPENDENCY_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "Pipfile",
        "Pipfile.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Gemfile",
        "Gemfile.lock",
        "Cargo.toml",
        "Cargo.lock",
        "go.mod",
        "go.sum",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    }

    CI_PATH_HINTS = (".github/workflows", ".github/actions", ".circleci", ".gitlab-ci", "azure-pipelines")
    TEST_PATH_HINTS = ("/tests/", "_test.", "test_", "tests/")
    INFRA_PATH_HINTS = ("kubernetes", "k8s", "helm", "terraform", ".tf", ".tfvars")

    @classmethod
    def feature_columns(cls) -> list[str]:
        return list(cls.FEATURE_COLUMNS)

    @staticmethod
    def _ensure_record(record: WorkflowRunRecord | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(record, WorkflowRunRecord):
            return record.model_dump()
        return dict(record)

    @staticmethod
    def _load_changed_files(record: Mapping[str, Any]) -> list[str]:
        raw = record.get("changed_files_json") or "[]"
        if isinstance(raw, list):
            return [str(item) for item in raw]
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            data = json.loads(raw)
            return [str(item) for item in data] if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _extension_for_path(path: str) -> str:
        lowered = path.lower()
        name = Path(path).name.lower()
        if name == "dockerfile" or lowered.endswith("/dockerfile"):
            return "dockerfile"
        if name.endswith(".yaml"):
            return "yaml"
        if name.endswith(".yml"):
            return "yml"

        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix:
            if suffix in FailureFeatureExtractor.EXTENSION_FEATURES:
                return suffix
            return "other"
        return "other"

    @classmethod
    def _count_extensions(cls, files: list[str]) -> dict[str, int]:
        counts = Counter(cls._extension_for_path(path) for path in files)
        return {extension: int(counts.get(extension, 0)) for extension in cls.EXTENSION_FEATURES}

    @classmethod
    def _normalize_previous_status(cls, status: Any) -> tuple[int, int, int]:
        normalized = str(status).lower() if status is not None else "unknown"
        return (
            int(normalized == "success"),
            int(normalized == "failure"),
            int(normalized not in {"success", "failure"}),
        )

    @classmethod
    def to_feature_row(cls, record: WorkflowRunRecord | Mapping[str, Any]) -> dict[str, Any]:
        data = cls._ensure_record(record)
        files = cls._load_changed_files(data)

        previous_success, previous_failure, previous_other = cls._normalize_previous_status(
            data.get("previous_run_status")
        )

        extension_counts = cls._count_extensions(files)
        extension_diversity = sum(1 for value in extension_counts.values() if value > 0)

        row = {
            "files_changed": int(data.get("files_changed") or 0),
            "lines_added": int(data.get("lines_added") or 0),
            "lines_deleted": int(data.get("lines_deleted") or 0),
            "number_of_commits": int(data.get("number_of_commits") or 0),
            "dependency_files_changed": int(bool(data.get("dependency_files_changed", False))),
            "ci_workflow_files_changed": int(bool(data.get("ci_workflow_files_changed", False))),
            "docker_files_changed": int(bool(data.get("docker_files_changed", False))),
            "test_files_changed": int(bool(data.get("test_files_changed", False))),
            "infrastructure_files_changed": int(bool(data.get("infrastructure_files_changed", False))),
            "previous_run_status_success": previous_success,
            "previous_run_status_failure": previous_failure,
            "previous_run_status_other": previous_other,
            "previous_failure_count": int(data.get("previous_failure_count") or 0),
            "recent_failure_count": int(data.get("recent_failure_count") or 0),
            "recent_failure_rate": float(data.get("recent_failure_rate") or 0.0),
            "previous_failures_same_workflow": int(data.get("previous_failures_same_workflow") or 0),
            "previous_failures_same_branch": int(data.get("previous_failures_same_branch") or 0),
            "similar_previous_failures": int(data.get("similar_previous_failures") or 0),
            "changed_extension_diversity": int(extension_diversity),
        }

        for extension in cls.EXTENSION_FEATURES:
            row[f"changed_ext_{extension}_count"] = int(extension_counts.get(extension, 0))

        return row

    @classmethod
    def build_feature_frame(cls, records: list[WorkflowRunRecord | Mapping[str, Any]]) -> pd.DataFrame:
        rows = [cls.to_feature_row(record) for record in records]
        return pd.DataFrame(rows, columns=cls.feature_columns())

    @classmethod
    def prepare_training_frame(
        cls,
        dataset: pd.DataFrame,
        target_column: str = "actual_failure",
    ) -> tuple[pd.DataFrame, pd.Series]:
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        feature_frame = dataset.reindex(columns=cls.feature_columns(), fill_value=0)
        feature_frame = feature_frame.fillna(0)
        labels = dataset[target_column].fillna(0).astype(int)

        return feature_frame, labels
