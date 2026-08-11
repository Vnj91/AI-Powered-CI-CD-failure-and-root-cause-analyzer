"""Command line interface for the prediction layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Config
from .data_collector import HistoricalRunCollector
from .evaluator import evaluate_saved_model
from .feature_extractor import FailureFeatureExtractor
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
        print("Top risk factors:")
        for factor in prediction.top_risk_factors:
            print(f" - {factor.feature}: importance={factor.importance:.4f}, value={factor.value}")
    if prediction.warnings:
        print("Warnings:")
        for warning in prediction.warnings:
            print(f" - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CI/CD failure prediction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect workflow history into CSV")
    collect_parser.add_argument("repository", help="owner/repo")
    collect_parser.add_argument("--output", default=str(Config.HISTORICAL_DATASET_PATH))
    collect_parser.add_argument("--limit", type=int, default=None)

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

    args = parser.parse_args()

    if args.command == "collect":
        collector = HistoricalRunCollector()
        output = collector.collect_to_csv(args.repository, args.output, limit=args.limit)
        print(f"Saved historical dataset to {output}")
        return

    if args.command == "train":
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
        metrics = evaluate_saved_model(args.dataset, args.model)
        print(metrics.model_dump_json(indent=2, default=str))
        return

    if args.command == "predict":
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


if __name__ == "__main__":
    main()
