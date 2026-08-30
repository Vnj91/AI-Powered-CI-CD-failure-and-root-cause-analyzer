"""Tests for GitHub Actions helpers and hardened predictor behavior."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import pytest

from config import Config
from src.prediction.gha_actions import (
    _prediction_branch,
    _target_workflow_name,
    run_pre_ci_prediction,
    run_workflow_feedback,
)
from src.prediction.history_store import PredictionHistoryStore
from src.prediction.predictor import FailurePredictor
from src.prediction.schemas import FailurePrediction
from tests.test_prediction import make_training_dataset


def test_predictor_rejects_corrupt_model_file(tmp_path: Path):
    corrupt_path = tmp_path / "failure_predictor.joblib"
    corrupt_path.write_bytes(b"not-a-valid-joblib-artifact")

    predictor = FailurePredictor(corrupt_path)

    assert predictor.available is False
    assert predictor.pipeline is None


def test_predictor_probability_bounds(tmp_path: Path):
    dataset = make_training_dataset(rows=30)
    dataset_path = tmp_path / "historical_runs.csv"
    model_path = tmp_path / "failure_predictor.joblib"
    metadata_path = tmp_path / "failure_predictor_metadata.json"
    dataset.to_csv(dataset_path, index=False)

    from src.prediction.trainer import FailurePredictorTrainer

    FailurePredictorTrainer(random_state=3).train(dataset_path, model_path, metadata_path)

    predictor = FailurePredictor(model_path)
    row = dataset.iloc[0].to_dict()
    prediction = predictor.predict(row)

    assert prediction.failure_probability is not None
    assert 0.0 <= prediction.failure_probability <= 1.0


def test_gha_predict_exits_zero_without_model(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "abc123def456")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "dummy")
    monkeypatch.setattr(Config, "PREDICTOR_MODEL_PATH", tmp_path / "missing.joblib")

    assert run_pre_ci_prediction() == 0


def test_prediction_branch_preserves_slash_delimited_head_ref(monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature/risk-dashboard")
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)

    assert _prediction_branch() == "feature/risk-dashboard"


def test_prediction_branch_prefers_pull_request_head_ref(monkeypatch):
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    monkeypatch.setenv("GITHUB_HEAD_REF", "feature/from-pr")

    assert _prediction_branch() == "feature/from-pr"


def test_target_workflow_defaults_to_application_pipeline(monkeypatch):
    monkeypatch.delenv("TARGET_WORKFLOW_NAME", raising=False)

    assert _target_workflow_name() == "CI/CD Pipeline"


def test_target_workflow_supports_explicit_pipeline_name(monkeypatch):
    monkeypatch.setenv("TARGET_WORKFLOW_NAME", "Release validation")

    assert _target_workflow_name() == "Release validation"


def test_gha_predict_exits_zero_with_corrupt_model(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "failure_predictor.joblib"
    model_path.write_bytes(b"corrupt")

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "abc123def456")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_ACCESS_TOKEN", "dummy")
    monkeypatch.setattr(Config, "PREDICTOR_MODEL_PATH", model_path)

    assert run_pre_ci_prediction() == 0


def test_gha_feedback_exits_zero_without_history(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.setenv("GITHUB_CONCLUSION", "success")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Config, "PREDICTION_HISTORY_PATH", Path("/tmp/nonexistent_prediction_history.csv"))
        assert run_workflow_feedback() == 0


def test_feedback_cancelled_and_skipped_not_success(tmp_path: Path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.5,
        predicted_failure=False,
        risk_level="MEDIUM",
        model_version="test",
    )
    store.append_prediction("owner/repo", "CI/CD Pipeline", 42, "deadbeef", prediction)

    ok_cancel, _, status_cancel = store.record_outcome_by_run_id(42, "cancelled")
    frame = store._load()
    assert ok_cancel
    assert status_cancel == "recorded_non_binary_outcome"
    assert pd.isna(frame.loc[0, "actual_failure"])

    store.append_prediction("owner/repo", "CI/CD Pipeline", 43, "cafebabe", prediction)
    ok_skip, _, status_skip = store.record_outcome_by_run_id(43, "skipped")
    frame = store._load()
    assert ok_skip
    assert status_skip == "recorded_non_binary_outcome"
    assert pd.isna(frame.loc[frame["run_id"] == 43, "actual_failure"].iloc[0])


def test_gha_feedback_records_by_commit(tmp_path: Path, monkeypatch):
    history_path = tmp_path / "prediction_history.csv"
    store = PredictionHistoryStore(history_path)
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.7,
        predicted_failure=True,
        risk_level="HIGH",
        model_version="test",
    )
    store.append_prediction(
        "owner/repo",
        "Pre-CI Failure Prediction",
        100,
        "abc1234567890",
        prediction,
    )

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "abc1234567890")
    monkeypatch.setenv("GITHUB_RUN_ID", "200")
    monkeypatch.setenv("GITHUB_CONCLUSION", "failure")
    monkeypatch.setenv("GITHUB_WORKFLOW_NAME", "CI/CD Pipeline")
    monkeypatch.setattr(Config, "PREDICTION_HISTORY_PATH", history_path)

    assert run_workflow_feedback() == 0

    frame = store._load()
    assert int(frame.loc[0, "actual_failure"]) == 1
    assert int(frame.loc[0, "run_id"]) == 200
