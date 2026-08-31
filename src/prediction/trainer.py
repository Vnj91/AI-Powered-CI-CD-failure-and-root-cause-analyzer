"""Training utilities for the baseline CI/CD failure predictor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .evaluator import evaluate_classifier, evaluate_multiclass_classifier
from .feature_extractor import FailureFeatureExtractor


def _atomic_joblib_dump(value: object, path: Path) -> None:
    """Replace a model artifact only after serialization completes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(value: object, path: Path) -> None:
    """Atomically replace model metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FailurePredictorTrainer:
    """Train and persist a baseline random forest failure predictor."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def _parse_timestamp_series(dataset: pd.DataFrame) -> pd.Series:
        if "timestamp" not in dataset.columns:
            raise ValueError("Dataset must include a 'timestamp' column")
        timestamps = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=False)
        if timestamps.isna().any():
            raise ValueError("Dataset contains invalid timestamps")
        return timestamps

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    @staticmethod
    def _select_temporal_split(labels: pd.Series, test_fraction: float) -> int:
        """Choose the nearest chronological split with usable classes on both sides."""

        desired = max(1, int(len(labels) * (1 - test_fraction)))
        desired = min(desired, len(labels) - 1)
        classes = set(labels.dropna().unique().tolist())
        candidates: list[int] = []
        for index in range(1, len(labels)):
            train_counts = labels.iloc[:index].value_counts()
            test_counts = labels.iloc[index:].value_counts()
            if (
                set(train_counts.index.tolist()) == classes
                and set(test_counts.index.tolist()) == classes
                and int(train_counts.min()) >= 2
                and int(test_counts.min()) >= 2
            ):
                candidates.append(index)

        if not candidates:
            raise ValueError(
                "Temporal training split must contain at least two classes in both "
                "the training prefix and holdout, with at least two examples per class. "
                "Collect more chronologically varied outcomes before training."
            )
        return min(candidates, key=lambda index: (abs(index - desired), -index))

    @staticmethod
    def _sort_time_ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
        sort_columns = ["_timestamp"]
        if "run_id" in frame.columns:
            sort_columns.append("run_id")
        return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    def _fit_and_evaluate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        timestamps: pd.Series,
        test_fraction: float,
    ) -> tuple[Pipeline, dict, int]:
        if labels.nunique() < 2:
            raise ValueError("Training data must contain at least two classes")

        split_index = self._select_temporal_split(labels, test_fraction)

        X_train = features.iloc[:split_index].reset_index(drop=True)
        y_train = labels.iloc[:split_index].reset_index(drop=True)
        X_test = features.iloc[split_index:].reset_index(drop=True)
        y_test = labels.iloc[split_index:].reset_index(drop=True)

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = evaluate_classifier(
            y_true=y_test,
            y_pred=predictions,
            y_proba=probabilities,
            train_rows=len(X_train),
            test_rows=len(X_test),
            time_split_at=timestamps.iloc[split_index].to_pydatetime(),
        )

        return pipeline, metrics.model_dump(), split_index

    def _train_failure_category_model(
        self,
        dataset: pd.DataFrame,
        timestamps: pd.Series,
        feature_columns: list[str],
        model_path: str | Path,
        metadata_path: str | Path,
        test_fraction: float,
    ) -> dict:
        if "actual_category" not in dataset.columns:
            return {
                "available": False,
                "trained": False,
                "reason": "Dataset does not include actual_category labels",
            }

        category_frame = dataset[
            (dataset["actual_failure"].fillna(0).astype(int) == 1)
            & dataset["actual_category"].notna()
            & (dataset["actual_category"].astype(str).str.strip() != "")
            & (dataset["actual_category"].astype(str).str.lower() != "unknown")
        ].copy()

        category_frame = self._sort_time_ordered_frame(category_frame)

        if category_frame.empty:
            return {
                "available": False,
                "trained": False,
                "reason": "No labeled failure categories available",
            }

        labels = category_frame["actual_category"].astype(str)
        if labels.nunique() < 2 or len(category_frame) < 10:
            return {
                "available": False,
                "trained": False,
                "reason": "Not enough labeled categories to train a reliable classifier",
                "rows": int(len(category_frame)),
                "unique_categories": int(labels.nunique()),
            }

        features, _ = FailureFeatureExtractor.prepare_training_frame(
            category_frame,
            target_column="actual_failure",
        )
        features = features.reindex(columns=feature_columns, fill_value=0).fillna(0)
        timestamps = pd.to_datetime(category_frame["_timestamp"], errors="coerce")

        split_index = max(1, int(len(features) * (1 - test_fraction)))
        split_index = min(split_index, len(features) - 1)

        X_train = features.iloc[:split_index].reset_index(drop=True)
        y_train = labels.iloc[:split_index].reset_index(drop=True)
        X_test = features.iloc[split_index:].reset_index(drop=True)
        y_test = labels.iloc[split_index:].reset_index(drop=True)

        all_categories = set(labels.unique().tolist())
        training_categories = set(y_train.unique().tolist())
        missing_categories = sorted(all_categories - training_categories)
        if len(training_categories) < 2 or missing_categories:
            reason = "Temporal category training split lacks required category classes"
            if missing_categories:
                reason += f": {', '.join(missing_categories)}"
            return {
                "available": False,
                "trained": False,
                "reason": reason,
                "rows": int(len(category_frame)),
                "unique_categories": int(len(all_categories)),
                "training_categories": sorted(training_categories),
                "missing_categories": missing_categories,
                "split_index": int(split_index),
            }

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        metrics = evaluate_multiclass_classifier(
            y_true=y_test,
            y_pred=predictions,
            y_score=probabilities,
            train_rows=len(X_train),
            test_rows=len(X_test),
            time_split_at=timestamps.iloc[split_index].to_pydatetime(),
        )

        artifact = {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "trained_at": datetime.now(UTC).isoformat(),
            "target_column": "actual_category",
            "metrics": metrics,
            "class_labels": sorted(labels.astype(str).unique().tolist()),
            "feature_version": 1,
            "random_state": self.random_state,
            "split_index": int(split_index),
        }

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        _atomic_joblib_dump(artifact, model_path)
        _atomic_write_json(artifact["metrics"], metadata_path)

        return {
            "available": True,
            "trained": True,
            "rows": int(len(category_frame)),
            "split_index": int(split_index),
            "metrics": metrics,
            "classes": artifact["class_labels"],
        }

    def train(
        self,
        dataset_path: str | Path,
        model_path: str | Path,
        metadata_path: Optional[str | Path] = None,
        category_model_path: Optional[str | Path] = None,
        category_metadata_path: Optional[str | Path] = None,
        target_column: str = "actual_failure",
        test_fraction: float = 0.2,
    ) -> dict:
        """Train a model from a stored CSV dataset and save the artifact."""

        dataset_path = Path(dataset_path)
        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_suffix(".json")

        dataset = pd.read_csv(dataset_path)
        if dataset.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")

        timestamps = self._parse_timestamp_series(dataset)
        dataset = self._sort_time_ordered_frame(dataset.assign(_timestamp=timestamps))

        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        raw_labels = pd.to_numeric(dataset[target_column], errors="coerce")
        if raw_labels.isna().any() or set(raw_labels.unique().tolist()) != {0, 1}:
            raise ValueError("Training target must contain only non-null binary labels 0 and 1")

        features, labels = FailureFeatureExtractor.prepare_training_frame(dataset, target_column=target_column)

        if labels.nunique() < 2:
            raise ValueError("Training data must contain both success and failure examples")

        pipeline, metrics, split_index = self._fit_and_evaluate(
            features=features,
            labels=labels,
            timestamps=dataset["_timestamp"],
            test_fraction=test_fraction,
        )

        artifact = {
            "pipeline": pipeline,
            "feature_columns": FailureFeatureExtractor.feature_columns(),
            "trained_at": datetime.now(UTC).isoformat(),
            "target_column": target_column,
            "metrics": metrics,
            "class_labels": [0, 1],
            "feature_version": 1,
            "random_state": self.random_state,
            "split_index": int(split_index),
        }

        _atomic_joblib_dump(artifact, model_path)
        _atomic_write_json(artifact["metrics"], metadata_path)

        category_model_path = (
            Path(category_model_path)
            if category_model_path
            else model_path.with_name("failure_category_predictor.joblib")
        )
        category_metadata_path = (
            Path(category_metadata_path)
            if category_metadata_path
            else metadata_path.with_name("failure_category_predictor_metadata.json")
        )

        category_artifact = self._train_failure_category_model(
            dataset=dataset,
            timestamps=dataset["_timestamp"],
            feature_columns=FailureFeatureExtractor.feature_columns(),
            model_path=category_model_path,
            metadata_path=category_metadata_path,
            test_fraction=test_fraction,
        )

        artifact["category_artifact"] = category_artifact
        return artifact
