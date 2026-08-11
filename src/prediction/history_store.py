"""Local CSV persistence for prediction history and feedback loops."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .schemas import PredictionHistoryRecord, FailurePrediction


class PredictionHistoryStore:
    """Persist predictions and their eventual outcomes in a CSV file."""

    def __init__(self, history_path: str | Path):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.history_path)

    def _save(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.history_path, index=False)

    def append_prediction(
        self,
        repository: str,
        workflow: Optional[str],
        run_id: Optional[int],
        commit_sha: Optional[str],
        prediction: FailurePrediction,
        actual_failure: Optional[int] = None,
        actual_category: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        prediction_id = str(uuid.uuid4())
        record = PredictionHistoryRecord(
            prediction_id=prediction_id,
            timestamp=timestamp or datetime.utcnow(),
            repository=repository,
            workflow=workflow,
            run_id=run_id,
            commit_sha=commit_sha,
            failure_probability=prediction.failure_probability,
            predicted_failure=prediction.predicted_failure,
            predicted_category=prediction.predicted_category,
            actual_failure=actual_failure,
            actual_category=actual_category,
            model_version=prediction.model_version,
        )

        frame = self._load()
        if not frame.empty:
            for column in ["actual_category", "predicted_category", "repository", "workflow", "commit_sha", "model_version"]:
                if column in frame.columns:
                    frame[column] = frame[column].astype(object)
        updated = pd.concat([frame, pd.DataFrame([record.model_dump()])], ignore_index=True)
        self._save(updated)
        return prediction_id

    def update_actual_outcome(
        self,
        prediction_id: str,
        actual_failure: int,
        actual_category: Optional[str] = None,
    ) -> bool:
        frame = self._load()
        if frame.empty or "prediction_id" not in frame.columns:
            return False

        if "actual_category" not in frame.columns:
            frame["actual_category"] = pd.Series([None] * len(frame), dtype=object)
        else:
            frame["actual_category"] = frame["actual_category"].astype(object)

        mask = frame["prediction_id"].astype(str) == str(prediction_id)
        if not mask.any():
            return False

        frame.loc[mask, "actual_failure"] = int(actual_failure)
        if actual_category is not None:
            frame.loc[mask, "actual_category"] = actual_category

        self._save(frame)
        return True
