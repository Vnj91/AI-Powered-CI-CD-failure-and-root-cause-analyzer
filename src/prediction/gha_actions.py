"""GitHub Actions helpers for pre-CI prediction and feedback recording."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from config import Config
from src.prediction.data_collector import HistoricalRunCollector
from src.prediction.feature_extractor import FailureFeatureExtractor
from src.prediction.feedback import PredictionFeedbackService
from src.prediction.history_store import PredictionHistoryStore
from src.prediction.predictor import FailurePredictor
from src.prediction.service import FailurePredictionService


MODEL_ARTIFACT_NAME = Config.MODEL_ARTIFACT_NAME
PREDICTION_HISTORY_ARTIFACT_NAME = Config.PREDICTION_HISTORY_ARTIFACT_NAME


def run_pre_ci_prediction() -> int:
    """Run leakage-safe pre-CI prediction. Always exits 0 unless unexpected crash."""

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")
    ref = os.environ.get("GITHUB_REF", "")
    branch = ref.split("/")[-1] if ref else "main"
    run_id_raw = os.environ.get("GITHUB_RUN_ID")
    run_id = int(run_id_raw) if run_id_raw else None
    token = os.environ.get("GITHUB_ACCESS_TOKEN")
    model_path = Config.PREDICTOR_MODEL_PATH

    print("CI Failure Prediction")
    print("---------------------")

    if not repo or not sha:
        print("Prediction skipped — missing repository context.")
        return 0

    if not token:
        print("Prediction model unavailable — GITHUB_ACCESS_TOKEN not configured.")
        return 0

    if not model_path.exists():
        print("Prediction model unavailable — skipping prediction.")
        print("Run the 'Train Failure Predictor' workflow to publish a model artifact.")
        return 0

    try:
        predictor = FailurePredictor(
            model_path,
            metadata_path=Config.PREDICTOR_METADATA_PATH,
            category_model_path=Config.CATEGORY_MODEL_PATH,
            category_metadata_path=Config.CATEGORY_METADATA_PATH,
        )
    except Exception as exc:
        print(f"Prediction model unavailable — failed to load artifact: {exc.__class__.__name__}")
        return 0

    if not predictor.available:
        print("Prediction model unavailable — skipping prediction.")
        if predictor.artifact is None:
            print("The model file exists but could not be loaded.")
        return 0

    try:
        collector = HistoricalRunCollector(token=token)
        features = collector.build_prediction_features(
            repository=repo,
            commit_sha=sha,
            branch=branch,
        )
    except Exception as exc:
        print(f"Prediction skipped — could not build features: {exc.__class__.__name__}")
        return 0

    feature_row = FailureFeatureExtractor.to_feature_row(features)
    Config.ensure_directories()
    service = FailurePredictionService(
        model_path=model_path,
        history_path=Config.PREDICTION_HISTORY_PATH,
        metadata_path=Config.PREDICTOR_METADATA_PATH,
        category_model_path=Config.CATEGORY_MODEL_PATH,
        category_metadata_path=Config.CATEGORY_METADATA_PATH,
    )
    prediction, prediction_id = service.predict_and_record(
        repository=repo,
        workflow="Pre-CI Failure Prediction",
        run_id=run_id,
        commit_sha=sha,
        features=feature_row,
    )

    if not prediction.model_available:
        print("Prediction model unavailable — skipping prediction.")
        return 0

    probability = prediction.failure_probability or 0.0
    outcome = "FAILURE" if prediction.predicted_failure else "SUCCESS"
    print(f"Probability: {probability:.0%}")
    print(f"Risk: {prediction.risk_level}")
    print(f"Predicted outcome: {outcome}")
    if prediction.predicted_category:
        suffix = (
            f" ({prediction.category_confidence:.0%})"
            if prediction.category_confidence is not None
            else ""
        )
        print(f"Predicted category: {prediction.predicted_category}{suffix}")
    if prediction.top_risk_factors:
        print("Risk factors associated with this prediction:")
        for factor in prediction.top_risk_factors[:5]:
            print(f"- {factor.feature}: value={factor.value:.2f}, importance={factor.importance:.4f}")
    if prediction_id:
        print(f"Prediction recorded: {prediction_id}")
    return 0


def run_workflow_feedback() -> int:
    """Record actual workflow outcome against pending predictions."""

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")
    run_id_raw = os.environ.get("GITHUB_RUN_ID")
    conclusion = os.environ.get("GITHUB_CONCLUSION", "")

    if not repo or not sha or not run_id_raw:
        print("Feedback skipped — missing workflow context.")
        return 0

    run_id = int(run_id_raw)
    history_path = Path(Config.PREDICTION_HISTORY_PATH)
    if not history_path.exists():
        print("No prediction history file yet — nothing to update.")
        return 0

    store = PredictionHistoryStore(history_path)
    service = PredictionFeedbackService(store)

    ok, prediction_id, status = store.record_outcome_by_run_id(run_id=run_id, conclusion=conclusion)
    if ok and status != "prediction_not_found":
        print(f"Feedback recorded by run_id: prediction_id={prediction_id} status={status}")
        return 0

    ok, prediction_id, status = store.record_outcome_for_commit(
        repository=repo,
        commit_sha=sha,
        conclusion=conclusion,
        run_id=run_id,
    )
    print(f"Feedback by commit: updated={ok} prediction_id={prediction_id} status={status}")
    if ok:
        summary = service.feedback_accuracy_summary()
        print(f"Feedback summary: recorded={summary['recorded']} correct={summary['correct']}")
    return 0


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "predict":
        raise SystemExit(run_pre_ci_prediction())
    if command == "feedback":
        raise SystemExit(run_workflow_feedback())
    print("Usage: python -m src.prediction.gha_actions [predict|feedback]")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
