"""Evaluation utilities for the failure prediction model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .feature_extractor import FailureFeatureExtractor
from .schemas import EvaluationMetrics


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    train_rows: int = 0,
    test_rows: int = 0,
    time_split_at: Optional[datetime] = None,
) -> EvaluationMetrics:
    """Compute the standard classification metrics the project needs."""

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            roc_auc = None

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=matrix,
        train_rows=train_rows,
        test_rows=test_rows,
        time_split_at=time_split_at,
    )


def evaluate_multiclass_classifier(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
    train_rows: int = 0,
    test_rows: int = 0,
    time_split_at: Optional[datetime] = None,
) -> EvaluationMetrics:
    """Compute macro-averaged metrics for a multiclass classifier."""

    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            roc_auc = None

    labels = sorted(set(list(y_true) + list(y_pred)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=matrix,
        train_rows=train_rows,
        test_rows=test_rows,
        time_split_at=time_split_at,
    )


def evaluate_saved_model(
    dataset_path: str | Path,
    model_path: str | Path,
    target_column: str = "actual_failure",
    test_fraction: float = 0.2,
) -> EvaluationMetrics:
    """Evaluate a persisted model against a time-ordered dataset."""

    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    dataset = pd.read_csv(dataset_path)

    if dataset.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    if "timestamp" not in dataset.columns:
        raise ValueError("Dataset must include a 'timestamp' column")

    timestamps = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=False)
    if timestamps.isna().any():
        raise ValueError("Dataset contains invalid timestamps")

    dataset = dataset.assign(_timestamp=timestamps)
    sort_columns = ["_timestamp"]
    if "run_id" in dataset.columns:
        sort_columns.append("run_id")
    dataset = dataset.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    features, labels = FailureFeatureExtractor.prepare_training_frame(dataset, target_column=target_column)

    split_index = max(1, int(len(dataset) * (1 - test_fraction)))
    split_index = min(split_index, len(dataset) - 1)

    X_test = features.iloc[split_index:].reset_index(drop=True)
    y_test = labels.iloc[split_index:].reset_index(drop=True)

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]

    predictions = pipeline.predict(X_test)
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]

    return evaluate_classifier(
        y_true=y_test,
        y_pred=predictions,
        y_proba=probabilities,
        train_rows=split_index,
        test_rows=len(X_test),
        time_split_at=dataset.loc[split_index, "_timestamp"].to_pydatetime(),
    )
