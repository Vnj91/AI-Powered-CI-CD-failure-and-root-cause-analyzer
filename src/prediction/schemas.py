"""Pydantic schemas for prediction data, outputs, and history."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WorkflowRunRecord(BaseModel):
    """Flattened workflow run example used for training and inference."""

    repository: str
    workflow_name: Optional[str] = None
    workflow_id: Optional[int] = None
    run_id: int
    run_number: Optional[int] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    timestamp: datetime
    status: Optional[str] = None
    conclusion: Optional[str] = None
    duration_seconds: Optional[float] = None
    event: Optional[str] = None
    actor: Optional[str] = None
    commit_message: Optional[str] = None

    files_changed: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    number_of_commits: int = 0
    changed_files_json: str = "[]"
    changed_extensions_json: str = "{}"

    dependency_files_changed: bool = False
    ci_workflow_files_changed: bool = False
    docker_files_changed: bool = False
    test_files_changed: bool = False
    infrastructure_files_changed: bool = False

    previous_run_status: Optional[str] = None
    previous_failure_count: int = 0
    recent_failure_count: int = 0
    recent_failure_rate: float = 0.0
    previous_failures_same_workflow: int = 0
    previous_failures_same_branch: int = 0
    similar_previous_failures: int = 0

    actual_failure: int = 0
    actual_category: Optional[str] = None


class FeatureImportance(BaseModel):
    """Feature-level explanation for a prediction."""

    feature: str
    value: float = 0.0
    importance: float = 0.0
    contribution: float = 0.0


class FailurePrediction(BaseModel):
    """Prediction returned by the failure risk model."""

    model_available: bool = True
    failure_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    predicted_failure: Optional[bool] = None
    risk_level: str = "UNKNOWN"
    predicted_category: Optional[str] = None
    category_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_risk_factors: list[FeatureImportance] = Field(default_factory=list)
    feature_importances: list[FeatureImportance] = Field(default_factory=list)
    model_version: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Model evaluation summary."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: Optional[float] = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0
    time_split_at: Optional[datetime] = None


class PredictionHistoryRecord(BaseModel):
    """Audit record for a stored prediction."""

    prediction_id: str
    timestamp: datetime
    repository: str
    workflow: Optional[str] = None
    run_id: Optional[int] = None
    commit_sha: Optional[str] = None
    failure_probability: Optional[float] = None
    predicted_failure: Optional[bool] = None
    predicted_category: Optional[str] = None
    actual_failure: Optional[int] = None
    actual_category: Optional[str] = None
    model_version: Optional[str] = None
