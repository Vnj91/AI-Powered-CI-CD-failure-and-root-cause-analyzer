"""High-level orchestration for model loading, prediction, and history."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .history_store import PredictionHistoryStore
from .predictor import FailurePredictor
from .schemas import FailurePrediction


class FailurePredictionService:
    """Convenience service that keeps prediction logic out of the UI."""

    def __init__(
        self,
        model_path: str | Path,
        history_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
        category_model_path: Optional[str | Path] = None,
        category_metadata_path: Optional[str | Path] = None,
    ):
        self.predictor = FailurePredictor(
            model_path=model_path,
            metadata_path=metadata_path,
            category_model_path=category_model_path,
            category_metadata_path=category_metadata_path,
        )
        self.history_store = PredictionHistoryStore(history_path) if history_path else None

    @property
    def model_available(self) -> bool:
        return self.predictor.available

    def predict(self, features: Mapping[str, Any]) -> FailurePrediction:
        return self.predictor.predict(features)

    def predict_and_record(
        self,
        repository: str,
        workflow: Optional[str],
        run_id: Optional[int],
        commit_sha: Optional[str],
        features: Mapping[str, Any],
        actual_failure: Optional[int] = None,
        actual_category: Optional[str] = None,
    ) -> tuple[FailurePrediction, Optional[str]]:
        prediction = self.predict(features)

        if not self.history_store:
            return prediction, None

        prediction_id = self.history_store.append_prediction(
            repository=repository,
            workflow=workflow,
            run_id=run_id,
            commit_sha=commit_sha,
            prediction=prediction,
            actual_failure=actual_failure,
            actual_category=actual_category,
        )
        return prediction, prediction_id

    def record_actual_outcome(
        self,
        prediction_id: str,
        actual_failure: int,
        actual_category: Optional[str] = None,
    ) -> bool:
        """Update a previously stored prediction with the final CI outcome."""

        if not self.history_store:
            return False

        return self.history_store.update_actual_outcome(
            prediction_id=prediction_id,
            actual_failure=actual_failure,
            actual_category=actual_category,
        )
