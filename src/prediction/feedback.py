"""Close the loop between predictions and actual CI outcomes."""

from __future__ import annotations

from typing import Optional

from github.WorkflowRun import WorkflowRun

from .category_mapper import conclusion_to_outcome_label, map_failure_category
from .history_store import PredictionHistoryStore
from .schemas import FailurePrediction


class PredictionFeedbackService:
    """Record actual workflow outcomes against prior predictions."""

    def __init__(self, history_store: PredictionHistoryStore):
        self.history_store = history_store

    @staticmethod
    def outcome_from_run(run: WorkflowRun) -> tuple[Optional[str], Optional[int]]:
        conclusion = getattr(run, "conclusion", None)
        return conclusion, getattr(run, "id", None)

    def record_from_workflow_run(
        self,
        repository: str,
        run: WorkflowRun,
        actual_category: Optional[str] = None,
    ) -> tuple[bool, Optional[str], str]:
        conclusion = getattr(run, "conclusion", None)
        run_id = getattr(run, "id", None)
        commit_sha = getattr(run, "head_sha", None)
        workflow_name = getattr(getattr(run, "workflow", None), "name", None) or getattr(run, "name", None)

        if run_id is not None:
            success, prediction_id, status = self.history_store.record_outcome_by_run_id(
                run_id=int(run_id),
                conclusion=conclusion,
                actual_category=actual_category,
            )
            if success:
                return success, prediction_id, status

        if commit_sha:
            return self.history_store.record_outcome_for_commit(
                repository=repository,
                commit_sha=str(commit_sha),
                conclusion=conclusion,
                run_id=int(run_id) if run_id is not None else None,
                workflow=workflow_name,
                actual_category=actual_category,
            )

        return False, None, "missing_identifiers"

    def record_manual_outcome(
        self,
        prediction_id: str,
        conclusion: str,
        actual_category: Optional[str] = None,
    ) -> bool:
        from .category_mapper import conclusion_to_actual_failure

        actual_failure = conclusion_to_actual_failure(conclusion)
        if actual_failure is None:
            frame = self.history_store._load()
            mask = frame["prediction_id"].astype(str) == str(prediction_id)
            if not mask.any():
                return False
            frame.loc[mask, "actual_conclusion"] = conclusion_to_outcome_label(conclusion)
            self.history_store._save(frame)
            return True

        return self.history_store.update_actual_outcome(
            prediction_id=prediction_id,
            actual_failure=actual_failure,
            actual_category=actual_category,
            actual_conclusion=conclusion_to_outcome_label(conclusion),
        )

    def feedback_accuracy_summary(self) -> dict:
        frame = self.history_store.recent_feedback_summary(limit=1000)
        if frame.empty:
            return {"recorded": 0, "correct": 0, "accuracy": None}

        correct = int(frame["prediction_correct"].sum())
        total = len(frame)
        return {
            "recorded": total,
            "correct": correct,
            "accuracy": float(correct / total) if total else None,
        }
