"""Local CSV persistence for prediction history and feedback loops."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import Lock, RLock
from typing import Optional

import pandas as pd

from .category_mapper import conclusion_to_actual_failure, conclusion_to_outcome_label
from .schemas import PredictionHistoryRecord, FailurePrediction


_LOCKS_GUARD = Lock()
_HISTORY_LOCKS: dict[Path, RLock] = {}


def _lock_for(path: Path) -> RLock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        return _HISTORY_LOCKS.setdefault(resolved, RLock())


def _synchronized(method):
    """Serialize read-modify-write transactions for one history path."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class PredictionHistoryStore:
    """Persist predictions and their eventual outcomes in a CSV file."""

    HISTORY_COLUMNS = [
        "prediction_id",
        "timestamp",
        "repository",
        "workflow",
        "run_id",
        "commit_sha",
        "failure_probability",
        "predicted_failure",
        "predicted_category",
        "actual_failure",
        "actual_category",
        "actual_conclusion",
        "feedback_recorded_at",
        "model_version",
    ]

    def __init__(self, history_path: str | Path):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock_for(self.history_path)

    def _load(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=self.HISTORY_COLUMNS)
        frame = pd.read_csv(self.history_path)
        for column in self.HISTORY_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return self._coerce_object_columns(frame)

    def _save(self, frame: pd.DataFrame) -> None:
        temporary = self.history_path.with_name(
            f".{self.history_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            frame.to_csv(temporary, index=False)
            temporary.replace(self.history_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _coerce_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
        for column in [
            "actual_category",
            "predicted_category",
            "repository",
            "workflow",
            "commit_sha",
            "model_version",
            "actual_conclusion",
            "feedback_recorded_at",
            "timestamp",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].astype(object)
        return frame

    def find_by_run_id(self, run_id: int) -> Optional[pd.Series]:
        frame = self._load()
        if frame.empty or "run_id" not in frame.columns:
            return None
        matches = frame[frame["run_id"].astype("Int64") == int(run_id)]
        if matches.empty:
            return None
        return matches.iloc[-1]

    def find_pending_by_commit(
        self,
        repository: str,
        commit_sha: str,
        workflow: Optional[str] = None,
    ) -> Optional[pd.Series]:
        frame = self._load()
        if frame.empty:
            return None

        mask = (
            frame["repository"].astype(str) == repository
        ) & (
            frame["commit_sha"].astype(str).str.startswith(str(commit_sha)[:7])
        )
        if workflow:
            mask &= frame["workflow"].astype(str) == workflow

        pending = frame[mask & frame["actual_failure"].isna()]
        if pending.empty:
            pending = frame[mask & frame["actual_conclusion"].isna()]
        if pending.empty:
            return None
        return pending.iloc[-1]

    @_synchronized
    def append_prediction(
        self,
        repository: str,
        workflow: Optional[str],
        run_id: Optional[int],
        commit_sha: Optional[str],
        prediction: FailurePrediction,
        actual_failure: Optional[int] = None,
        actual_category: Optional[str] = None,
        actual_conclusion: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        frame = self._load()

        if run_id is not None and not frame.empty and "run_id" in frame.columns:
            existing = frame[frame["run_id"].astype("Int64") == int(run_id)]
            if not existing.empty:
                return str(existing.iloc[-1]["prediction_id"])

        prediction_id = str(uuid.uuid4())
        record = PredictionHistoryRecord(
            prediction_id=prediction_id,
            timestamp=timestamp or datetime.now(UTC),
            repository=repository,
            workflow=workflow,
            run_id=run_id,
            commit_sha=commit_sha,
            failure_probability=prediction.failure_probability,
            predicted_failure=prediction.predicted_failure,
            predicted_category=prediction.predicted_category,
            actual_failure=actual_failure,
            actual_category=actual_category,
            actual_conclusion=actual_conclusion,
            feedback_recorded_at=datetime.now(UTC) if actual_failure is not None else None,
            model_version=prediction.model_version,
        )

        frame = self._coerce_object_columns(frame)
        updated = pd.concat([frame, pd.DataFrame([record.model_dump()])], ignore_index=True)
        self._save(updated)
        return prediction_id

    @_synchronized
    def update_actual_outcome(
        self,
        prediction_id: str,
        actual_failure: int,
        actual_category: Optional[str] = None,
        actual_conclusion: Optional[str] = None,
    ) -> bool:
        frame = self._load()
        if frame.empty or "prediction_id" not in frame.columns:
            return False

        frame = self._coerce_object_columns(frame)
        mask = frame["prediction_id"].astype(str) == str(prediction_id)
        if not mask.any():
            return False

        frame.loc[mask, "actual_failure"] = int(actual_failure)
        if actual_category is not None:
            frame.loc[mask, "actual_category"] = actual_category
        if actual_conclusion is not None:
            frame.loc[mask, "actual_conclusion"] = actual_conclusion
        frame.loc[mask, "feedback_recorded_at"] = datetime.now(UTC).isoformat()

        self._save(frame)
        return True

    @_synchronized
    def record_outcome_by_run_id(
        self,
        run_id: int,
        conclusion: Optional[str],
        actual_category: Optional[str] = None,
    ) -> tuple[bool, Optional[str], str]:
        """Idempotently record workflow outcome using run_id as the stable key."""

        frame = self._load()
        if frame.empty:
            return False, None, "no_history"

        frame = self._coerce_object_columns(frame)
        mask = frame["run_id"].astype("Int64") == int(run_id)
        if not mask.any():
            return False, None, "prediction_not_found"

        row = frame.loc[mask].iloc[-1]
        prediction_id = str(row["prediction_id"])

        if pd.notna(row.get("actual_failure")) and pd.notna(row.get("actual_conclusion")):
            return True, prediction_id, "already_recorded"

        actual_failure = conclusion_to_actual_failure(conclusion)
        outcome_label = conclusion_to_outcome_label(conclusion)

        if actual_failure is None:
            frame.loc[mask, "actual_conclusion"] = outcome_label
            frame.loc[mask, "feedback_recorded_at"] = datetime.now(UTC).isoformat()
            self._save(frame)
            return True, prediction_id, "recorded_non_binary_outcome"

        frame.loc[mask, "actual_failure"] = int(actual_failure)
        frame.loc[mask, "actual_conclusion"] = outcome_label
        if actual_category is not None:
            frame.loc[mask, "actual_category"] = actual_category
        frame.loc[mask, "feedback_recorded_at"] = datetime.now(UTC).isoformat()
        self._save(frame)
        return True, prediction_id, "recorded"

    @_synchronized
    def record_outcome_for_commit(
        self,
        repository: str,
        commit_sha: str,
        conclusion: Optional[str],
        run_id: Optional[int] = None,
        workflow: Optional[str] = None,
        actual_category: Optional[str] = None,
    ) -> tuple[bool, Optional[str], str]:
        """Record outcome for the latest pending prediction matching repository + commit."""

        if run_id is not None:
            result = self.record_outcome_by_run_id(run_id, conclusion, actual_category)
            if result[0]:
                return result

        pending = self.find_pending_by_commit(repository, commit_sha, workflow)
        if pending is None:
            return False, None, "prediction_not_found"

        prediction_id = str(pending["prediction_id"])
        actual_failure = conclusion_to_actual_failure(conclusion)
        outcome_label = conclusion_to_outcome_label(conclusion)

        if actual_failure is None:
            frame = self._load()
            frame = self._coerce_object_columns(frame)
            mask = frame["prediction_id"].astype(str) == prediction_id
            frame.loc[mask, "actual_conclusion"] = outcome_label
            if run_id is not None:
                frame.loc[mask, "run_id"] = int(run_id)
            frame.loc[mask, "feedback_recorded_at"] = datetime.now(UTC).isoformat()
            self._save(frame)
            return True, prediction_id, "recorded_non_binary_outcome"

        updated = self.update_actual_outcome(
            prediction_id,
            actual_failure=actual_failure,
            actual_category=actual_category,
            actual_conclusion=outcome_label,
        )
        if updated and run_id is not None:
            frame = self._load()
            mask = frame["prediction_id"].astype(str) == prediction_id
            frame.loc[mask, "run_id"] = int(run_id)
            self._save(frame)

        return updated, prediction_id, "recorded" if updated else "update_failed"

    def recent_feedback_summary(self, limit: int = 10) -> pd.DataFrame:
        frame = self._load()
        if frame.empty:
            return frame
        scored = frame[frame["actual_failure"].notna()].copy()
        if scored.empty:
            return scored
        scored["prediction_correct"] = (
            scored["predicted_failure"].astype(bool) == scored["actual_failure"].astype(int).astype(bool)
        )
        return scored.tail(limit)
