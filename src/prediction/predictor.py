"""Prediction service for the CI/CD failure model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import pandas as pd

from .feature_extractor import FailureFeatureExtractor
from .schemas import FailurePrediction, FeatureImportance


class FailurePredictor:
    """Load a persisted model and generate failure-risk predictions."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: Optional[str | Path] = None,
        category_model_path: Optional[str | Path] = None,
        category_metadata_path: Optional[str | Path] = None,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.category_model_path = Path(category_model_path) if category_model_path else None
        self.category_metadata_path = Path(category_metadata_path) if category_metadata_path else None
        self.artifact: dict[str, Any] | None = None
        self.category_artifact: dict[str, Any] | None = None
        self.pipeline = None
        self.category_pipeline = None
        self.feature_columns = FailureFeatureExtractor.feature_columns()
        self.model_version = None
        self.category_model_version = None
        self.available = False
        self.category_available = False

        self.load()

    def load(self) -> bool:
        if not self.model_path.exists():
            self.available = False
            self.artifact = None
            self.pipeline = None
            return False

        try:
            self.artifact = joblib.load(self.model_path)
        except Exception:
            self.available = False
            self.artifact = None
            self.pipeline = None
            self.category_available = False
            self.category_pipeline = None
            return False

        self.pipeline = self.artifact.get("pipeline")
        self.feature_columns = list(self.artifact.get("feature_columns", self.feature_columns))
        self.model_version = str(self.artifact.get("trained_at", "unknown"))
        self.available = self.pipeline is not None

        if self.category_model_path and self.category_model_path.exists():
            try:
                self.category_artifact = joblib.load(self.category_model_path)
            except Exception:
                self.category_artifact = None
                self.category_pipeline = None
                self.category_available = False
            else:
                self.category_pipeline = self.category_artifact.get("pipeline")
                self.category_model_version = str(self.category_artifact.get("trained_at", "unknown"))
                self.category_available = self.category_pipeline is not None
        else:
            self.category_artifact = None
            self.category_pipeline = None
            self.category_model_version = None
            self.category_available = False

        return self.available

    def _align_features(self, features: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            frame = features.copy()
        else:
            frame = pd.DataFrame([dict(features)])

        return frame.reindex(columns=self.feature_columns, fill_value=0).fillna(0)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.75:
            return "HIGH"
        if probability >= 0.45:
            return "MEDIUM"
        return "LOW"

    def _feature_importances(self) -> list[float]:
        if not self.pipeline:
            return []

        estimator = self.pipeline.named_steps.get("model")
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return []

        return list(getattr(estimator, "feature_importances_"))

    def _predict_category(self, frame: pd.DataFrame) -> tuple[Optional[str], Optional[float]]:
        if not self.category_available:
            return None, None

        if self.category_pipeline is None:
            return None, None

        probabilities = self.category_pipeline.predict_proba(frame)[0]
        classes = list(getattr(self.category_pipeline.named_steps.get("model"), "classes_", []))
        if not classes:
            return None, None

        best_index = int(probabilities.argmax())
        return str(classes[best_index]), float(probabilities[best_index])

    def predict(self, features: Mapping[str, Any] | pd.DataFrame) -> FailurePrediction:
        if not self.available:
            return FailurePrediction(
                model_available=False,
                risk_level="UNKNOWN",
                warnings=["Failure predictor model is not trained or not available yet."],
            )

        frame = self._align_features(features)
        proba = float(self.pipeline.predict_proba(frame)[:, 1][0])
        predicted_failure = bool(proba >= 0.5)
        feature_importances = self._feature_importances()
        predicted_category, category_confidence = self._predict_category(frame)
        warnings: list[str] = []
        metrics = self.artifact.get("metrics", {}) if self.artifact else {}
        test_rows = int(metrics.get("test_rows", 0) or 0)
        f1 = float(metrics.get("f1", 0.0) or 0.0)
        if test_rows < 20:
            warnings.append(
                f"Experimental baseline: the chronological holdout contains only {test_rows} runs."
            )
        if f1 < 0.5:
            warnings.append(
                f"Weak holdout quality (F1 {f1:.1%}); treat this score as directional, not a release gate."
            )

        importance_rows: list[FeatureImportance] = []
        for index, feature_name in enumerate(self.feature_columns):
            value = float(frame.iloc[0][feature_name])
            importance = float(feature_importances[index]) if index < len(feature_importances) else 0.0
            contribution = importance * value if value > 0 else 0.0
            importance_rows.append(
                FeatureImportance(
                    feature=feature_name,
                    value=value,
                    importance=importance,
                    contribution=contribution,
                )
            )

        top_risk_factors = sorted(
            (item for item in importance_rows if item.contribution > 0),
            key=lambda item: item.contribution,
            reverse=True,
        )[:5]

        return FailurePrediction(
            model_available=True,
            failure_probability=proba,
            predicted_failure=predicted_failure,
            risk_level=self._risk_level(proba),
            predicted_category=predicted_category,
            category_confidence=category_confidence,
            top_risk_factors=top_risk_factors,
            feature_importances=importance_rows,
            model_version=self.model_version,
            warnings=warnings,
        )
