"""Command line interface for the prediction layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from config import Config
from .data_collector import HistoricalRunCollector
from .dataset_validator import validate_dataset
from .evaluator import evaluate_saved_model
from .feature_extractor import FailureFeatureExtractor
from .feedback import PredictionFeedbackService
from .history_store import PredictionHistoryStore
from .predictor import FailurePredictor
from .trainer import FailurePredictorTrainer


def _print_prediction(prediction):
    print("Failure probability:", f"{(prediction.failure_probability or 0.0):.0%}")
    print("Predicted outcome:", "FAILURE" if prediction.predicted_failure else "SUCCESS")
    print("Risk level:", prediction.risk_level)
    if prediction.predicted_category:
        if prediction.category_confidence is not None:
            print("Predicted category:", f"{prediction.predicted_category} ({prediction.category_confidence:.0%})")
        else:
            print("Predicted category:", prediction.predicted_category)
    if prediction.top_risk_factors:
        print("Top risk factors associated with this prediction:")
        for factor in prediction.top_risk_factors:
            print(f" - {factor.feature}: importance={factor.importance:.4f}, value={factor.value}")
    if prediction.warnings:
        print("Warnings:")
        for warning in prediction.warnings:
            print(f" - {warning}")


def _print_feature_importance(model_path: Path, top_n: int = 10) -> None:
    if not model_path.exists():
        return
    artifact = joblib.load(model_path)
    pipeline = artifact.get("pipeline")
    columns = artifact.get("feature_columns", [])
    if pipeline is None:
        return
    estimator = pipeline.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return
    importances = list(estimator.feature_importances_)
    ranked = sorted(
        zip(columns, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    print("Top feature importances:")
    for name, value in ranked:
        print(f" - {name}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CI/CD failure prediction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect workflow history into CSV")
    collect_parser.add_argument("repository", help="owner/repo")
    collect_parser.add_argument("--output", default=str(Config.HISTORICAL_DATASET_PATH))
    collect_parser.add_argument("--limit", type=int, default=None)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a historical dataset for quality/leakage")
    inspect_parser.add_argument("--dataset", default=str(Config.HISTORICAL_DATASET_PATH))

    train_parser = subparsers.add_parser("train", help="Train the failure predictor")
    train_parser.add_argument("--dataset", default=str(Config.HISTORICAL_DATASET_PATH))
    train_parser.add_argument("--model", default=str(Config.PREDICTOR_MODEL_PATH))
    train_parser.add_argument("--metadata", default=str(Config.PREDICTOR_METADATA_PATH))

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained predictor")
    eval_parser.add_argument("--dataset", default=str(Config.HISTORICAL_DATASET_PATH))
    eval_parser.add_argument("--model", default=str(Config.PREDICTOR_MODEL_PATH))

    predict_parser = subparsers.add_parser("predict", help="Predict failure risk for a commit")
    predict_parser.add_argument("repository", help="owner/repo")
    predict_parser.add_argument("commit_sha", help="Commit SHA to score")
    predict_parser.add_argument("--branch", default=None)
    predict_parser.add_argument("--workflow", default=None)
    predict_parser.add_argument("--model", default=str(Config.PREDICTOR_MODEL_PATH))
    predict_parser.add_argument("--category-model", default=str(Config.CATEGORY_MODEL_PATH))

    feedback_parser = subparsers.add_parser("feedback", help="Record actual outcome for a pending prediction")
    feedback_parser.add_argument("repository", help="owner/repo")
    feedback_parser.add_argument("commit_sha", help="Commit SHA")
    feedback_parser.add_argument("--conclusion", required=True, help="GitHub Actions conclusion")
    feedback_parser.add_argument("--run-id", type=int, default=None)
    feedback_parser.add_argument("--workflow", default=None)
    feedback_parser.add_argument("--category", default=None)
    feedback_parser.add_argument("--history", default=str(Config.PREDICTION_HISTORY_PATH))

    args = parser.parse_args()

    if args.command == "collect":
        collector = HistoricalRunCollector()
        output = collector.collect_to_csv(args.repository, args.output, limit=args.limit)
        print(f"Saved historical dataset to {output}")
        report = validate_dataset(output)
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    if args.command == "inspect":
        report = validate_dataset(args.dataset)
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    if args.command == "train":
        report = validate_dataset(args.dataset)
        if not report.sufficient_for_training:
            print("Insufficient real historical data for meaningful model training.")
            print(json.dumps(report.to_dict(), indent=2, default=str))
            sys.exit(1)
        trainer = FailurePredictorTrainer()
        artifact = trainer.train(
            args.dataset,
            args.model,
            args.metadata,
            category_model_path=Config.CATEGORY_MODEL_PATH,
            category_metadata_path=Config.CATEGORY_METADATA_PATH,
        )
        print("Trained model successfully")
        print(f"Metrics: {artifact['metrics']}")
        if artifact.get("category_artifact", {}).get("trained"):
            print(f"Category model trained: {artifact['category_artifact']['classes']}")
        else:
            print(f"Category model skipped: {artifact.get('category_artifact', {}).get('reason', 'unknown')}")
        return

    if args.command == "evaluate":
        report = validate_dataset(args.dataset)
        if not report.sufficient_for_evaluation:
            print("Insufficient real historical data for meaningful model evaluation.")
            print(json.dumps(report.to_dict(), indent=2, default=str))
            sys.exit(1)
        if not Path(args.model).exists():
            print(f"Model not found: {args.model}")
            sys.exit(1)
        metrics = evaluate_saved_model(args.dataset, args.model)
        print(metrics.model_dump_json(indent=2, default=str))
        print(json.dumps({
            "class_distribution": {
                "success": report.success_runs,
                "failure": report.failure_runs,
                "other": report.other_runs,
                "failure_rate": report.class_balance_failure_rate,
            }
        }, indent=2))
        _print_feature_importance(Path(args.model))
        return

    if args.command == "predict":
        if not Path(args.model).exists():
            print("Prediction model unavailable — train a model first.")
            return
        collector = HistoricalRunCollector()
        predictor = FailurePredictor(args.model, category_model_path=args.category_model)
        features = collector.build_prediction_features(
            repository=args.repository,
            commit_sha=args.commit_sha,
            branch=args.branch,
            workflow_name=args.workflow,
        )
        feature_row = FailureFeatureExtractor.to_feature_row(features)
        prediction = predictor.predict(feature_row)
        _print_prediction(prediction)
        return

    if args.command == "feedback":
        store = PredictionHistoryStore(args.history)
        service = PredictionFeedbackService(store)
        updated, prediction_id, status = store.record_outcome_for_commit(
            repository=args.repository,
            commit_sha=args.commit_sha,
            conclusion=args.conclusion,
            run_id=args.run_id,
            workflow=args.workflow,
            actual_category=args.category,
        )
        print(json.dumps({"updated": updated, "prediction_id": prediction_id, "status": status}, indent=2))
        print(json.dumps(service.feedback_accuracy_summary(), indent=2))


if __name__ == "__main__":
    main()
