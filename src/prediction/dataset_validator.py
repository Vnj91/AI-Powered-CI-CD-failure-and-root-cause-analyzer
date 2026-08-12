"""Validate historical datasets for leakage, ordering, and quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .feature_extractor import FailureFeatureExtractor


LEAKAGE_COLUMNS = {
    "actual_failure",
    "actual_category",
    "status",
    "conclusion",
    "duration_seconds",
}

FAILURE_CONCLUSIONS = {"failure", "cancelled", "timed_out", "startup_failure", "action_required"}


@dataclass
class DatasetValidationReport:
    total_runs: int = 0
    success_runs: int = 0
    failure_runs: int = 0
    other_runs: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    workflows: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    class_balance_failure_rate: float = 0.0
    missing_value_counts: dict[str, int] = field(default_factory=dict)
    duplicate_run_ids: int = 0
    leakage_columns_in_features: list[str] = field(default_factory=list)
    chronologically_ordered: bool = True
    feature_columns_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sufficient_for_training: bool = False
    sufficient_for_evaluation: bool = False
    cancelled_runs: int = 0
    skipped_runs: int = 0
    failures_by_workflow: dict[str, int] = field(default_factory=dict)
    failures_by_category: dict[str, int] = field(default_factory=dict)
    runs_by_branch: dict[str, int] = field(default_factory=dict)
    runs_by_week: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "success_runs": self.success_runs,
            "failure_runs": self.failure_runs,
            "other_runs": self.other_runs,
            "cancelled_runs": self.cancelled_runs,
            "skipped_runs": self.skipped_runs,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "workflows": self.workflows,
            "branches": self.branches,
            "class_balance_failure_rate": self.class_balance_failure_rate,
            "missing_value_counts": self.missing_value_counts,
            "duplicate_run_ids": self.duplicate_run_ids,
            "leakage_columns_in_features": self.leakage_columns_in_features,
            "chronologically_ordered": self.chronologically_ordered,
            "feature_columns_used": self.feature_columns_used,
            "failures_by_workflow": self.failures_by_workflow,
            "failures_by_category": self.failures_by_category,
            "runs_by_branch": self.runs_by_branch,
            "runs_by_week": self.runs_by_week,
            "warnings": self.warnings,
            "sufficient_for_training": self.sufficient_for_training,
            "sufficient_for_evaluation": self.sufficient_for_evaluation,
        }


def _is_chronological(dataset: pd.DataFrame) -> bool:
    if "timestamp" not in dataset.columns or dataset.empty:
        return True

    timestamps = pd.to_datetime(dataset["timestamp"], errors="coerce")
    sort_columns = ["timestamp"]
    if "run_id" in dataset.columns:
        sort_columns.append("run_id")

    ordered = dataset.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    compare_columns = ["timestamp"]
    if "run_id" in ordered.columns:
        compare_columns.append("run_id")

    return ordered[compare_columns].equals(dataset.reset_index(drop=True)[compare_columns])


def _load_dataset_frame(dataset: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a dataset from a path or return a copy of an in-memory frame."""

    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()

    path = Path(dataset)
    return pd.read_csv(path)


def validate_dataset(dataset: str | Path | pd.DataFrame) -> DatasetValidationReport:
    """Inspect a historical runs dataset before training or evaluation."""

    if not isinstance(dataset, pd.DataFrame):
        path = Path(dataset)
        if not path.exists():
            report = DatasetValidationReport()
            report.feature_columns_used = FailureFeatureExtractor.feature_columns()
            report.warnings.append(f"Dataset not found: {path}")
            return report

    frame = _load_dataset_frame(dataset)
    report = DatasetValidationReport()
    report.feature_columns_used = FailureFeatureExtractor.feature_columns()

    if frame.empty:
        report.warnings.append("Dataset is empty.")
        return report

    report.total_runs = len(frame)

    if "actual_failure" in frame.columns:
        labels = frame["actual_failure"].fillna(0).astype(int)
        report.failure_runs = int(labels.sum())
        report.success_runs = int((labels == 0).sum())
        report.class_balance_failure_rate = float(report.failure_runs / max(report.total_runs, 1))
    elif "conclusion" in frame.columns:
        conclusions = frame["conclusion"].astype(str).str.lower()
        report.failure_runs = int(conclusions.isin(FAILURE_CONCLUSIONS).sum())
        report.success_runs = int((conclusions == "success").sum())
        report.other_runs = report.total_runs - report.failure_runs - report.success_runs
        report.class_balance_failure_rate = float(report.failure_runs / max(report.total_runs, 1))
    else:
        report.warnings.append("No actual_failure or conclusion column found.")

    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        valid = timestamps.dropna()
        if not valid.empty:
            report.date_range_start = str(valid.min())
            report.date_range_end = str(valid.max())

    if "workflow_name" in frame.columns:
        report.workflows = sorted(frame["workflow_name"].dropna().astype(str).unique().tolist())
    if "branch" in frame.columns:
        report.branches = sorted(frame["branch"].dropna().astype(str).unique().tolist())
        report.runs_by_branch = (
            frame["branch"].fillna("unknown").astype(str).value_counts().sort_index().astype(int).to_dict()
        )

    if "conclusion" in frame.columns:
        conclusions = frame["conclusion"].astype(str).str.lower()
        report.cancelled_runs = int((conclusions == "cancelled").sum())
        report.skipped_runs = int((conclusions == "skipped").sum())
        if "actual_failure" not in frame.columns:
            report.other_runs = int(
                conclusions.isin({"cancelled", "skipped", "neutral", "stale"}).sum()
            )

    if "actual_failure" in frame.columns and "workflow_name" in frame.columns:
        failed = frame[frame["actual_failure"].fillna(0).astype(int) == 1]
        if not failed.empty:
            report.failures_by_workflow = (
                failed["workflow_name"].fillna("unknown").astype(str).value_counts().sort_index().astype(int).to_dict()
            )

    if "actual_category" in frame.columns and "actual_failure" in frame.columns:
        labeled = frame[
            (frame["actual_failure"].fillna(0).astype(int) == 1)
            & frame["actual_category"].notna()
            & (frame["actual_category"].astype(str).str.strip() != "")
        ]
        if not labeled.empty:
            report.failures_by_category = (
                labeled["actual_category"].astype(str).value_counts().sort_index().astype(int).to_dict()
            )

    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        valid_weeks = timestamps.dropna().dt.to_period("W").astype(str)
        if not valid_weeks.empty:
            report.runs_by_week = valid_weeks.value_counts().sort_index().astype(int).to_dict()

    report.missing_value_counts = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
        if frame[column].isna().any()
    }

    if "run_id" in frame.columns:
        report.duplicate_run_ids = int(frame["run_id"].duplicated().sum())

    report.leakage_columns_in_features = sorted(
        LEAKAGE_COLUMNS.intersection(set(FailureFeatureExtractor.feature_columns()))
    )
    if report.leakage_columns_in_features:
        report.warnings.append(
            f"Leakage risk: target/metadata columns appear in feature list: {report.leakage_columns_in_features}"
        )

    report.chronologically_ordered = _is_chronological(frame)
    if not report.chronologically_ordered:
        report.warnings.append("Dataset is not chronologically ordered by timestamp (+ run_id tie-breaker).")

    unique_classes = 0
    if "actual_failure" in frame.columns:
        unique_classes = frame["actual_failure"].dropna().nunique()

    report.sufficient_for_training = report.total_runs >= 20 and unique_classes >= 2
    report.sufficient_for_evaluation = report.total_runs >= 10 and unique_classes >= 2

    if report.total_runs < 20:
        report.warnings.append("Fewer than 20 runs — baseline model quality will be weak.")
    if unique_classes < 2:
        report.warnings.append("Only one outcome class present — cannot train a meaningful classifier.")

    return report
