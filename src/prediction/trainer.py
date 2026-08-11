"""Training utilities for the baseline CI/CD failure predictor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .evaluator import evaluate_classifier, evaluate_multiclass_classifier
from .feature_extractor import FailureFeatureExtractor


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

        split_index = max(1, int(len(features) * (1 - test_fraction)))
        split_index = min(split_index, len(features) - 1)

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

        features = category_frame.reindex(columns=feature_columns, fill_value=0).fillna(0)
        timestamps = pd.to_datetime(category_frame["_timestamp"], errors="coerce")

        split_index = max(1, int(len(features) * (1 - test_fraction)))
        split_index = min(split_index, len(features) - 1)

        X_train = features.iloc[:split_index].reset_index(drop=True)
        y_train = labels.iloc[:split_index].reset_index(drop=True)
        X_test = features.iloc[split_index:].reset_index(drop=True)
        y_test = labels.iloc[split_index:].reset_index(drop=True)

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
            "trained_at": datetime.utcnow().isoformat(),
            "target_column": "actual_category",
            "metrics": metrics,
            "class_labels": sorted(labels.astype(str).unique().tolist()),
            "feature_version": 1,
            "random_state": self.random_state,
        }

        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(artifact["metrics"], indent=2, default=str), encoding="utf-8")

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
            "trained_at": datetime.utcnow().isoformat(),
            "target_column": target_column,
            "metrics": metrics,
            "class_labels": [0, 1],
            "feature_version": 1,
            "random_state": self.random_state,
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_path)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(artifact["metrics"], indent=2, default=str), encoding="utf-8")

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
