from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.prediction.category_mapper import (
    FailureCategory,
    conclusion_to_actual_failure,
    conclusion_to_outcome_label,
    map_failure_category,
)
from src.prediction.dataset_validator import LEAKAGE_COLUMNS, validate_dataset
from src.prediction.feature_extractor import FailureFeatureExtractor
from src.prediction.feedback import PredictionFeedbackService
from src.prediction.history_store import PredictionHistoryStore
from src.prediction.schemas import FailurePrediction
from src.prediction.data_collector import HistoricalRunCollector
from src.tools.commit_analyzer import CommitAnalyzer
from src.tools.log_parser import ErrorCategory, ParsedError


def test_category_mapper_maps_log_parser_dependency():
    category = map_failure_category(log_parser_category="dependency")
    assert category == FailureCategory.DEPENDENCY


def test_category_mapper_maps_docker_keywords():
    category = map_failure_category(
        error_message="docker build failed",
        failed_step="Docker Build",
    )
    assert category == FailureCategory.DOCKER


def test_category_taxonomy_has_thirteen_values():
    from src.prediction.category_mapper import FailureCategory

    assert len(FailureCategory) == 13
    assert FailureCategory.UNKNOWN.value == "unknown"


def test_category_mapper_unknown_without_evidence():
    category = map_failure_category()
    assert category == FailureCategory.UNKNOWN


def test_conclusion_mapping_does_not_treat_unknown_as_success():
    assert conclusion_to_actual_failure("success") == 0
    assert conclusion_to_actual_failure("failure") == 1
    assert conclusion_to_actual_failure("cancelled") is None
    assert conclusion_to_actual_failure(None) is None
    assert conclusion_to_outcome_label("cancelled") == "cancelled"


def test_dataset_validator_accepts_pathlib_path(tmp_path):
    dataset_path = tmp_path / "historical_runs.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "actual_failure": [0, 1],
            "run_id": [1, 2],
            "workflow_name": ["CI/CD Pipeline", "CI/CD Pipeline"],
            "branch": ["main", "main"],
        }
    ).to_csv(dataset_path, index=False)

    report = validate_dataset(dataset_path)

    assert report.total_runs == 2
    assert report.success_runs == 1
    assert report.failure_runs == 1
    assert report.leakage_columns_in_features == []


def test_dataset_validator_reports_breakdowns():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-08", "2026-01-09"],
            "actual_failure": [0, 1, 0],
            "run_id": [1, 2, 3],
            "workflow_name": ["CI/CD Pipeline", "Controlled CI Failure Generator (Dev/Testing Only)", "CI/CD Pipeline"],
            "branch": ["main", "main", "develop"],
            "conclusion": ["success", "failure", "success"],
            "actual_category": [None, "test", None],
        }
    )
    report = validate_dataset(frame)
    assert report.total_runs == 3
    assert report.failures_by_workflow["Controlled CI Failure Generator (Dev/Testing Only)"] == 1
    assert report.failures_by_category["test"] == 1
    assert report.runs_by_branch["main"] == 2
    assert report.runs_by_branch["develop"] == 1
    assert len(report.runs_by_week) >= 1


def test_dataset_validator_detects_leakage_free_features():
    report = validate_dataset(pd.DataFrame({"actual_failure": [0, 1], "timestamp": ["2026-01-01", "2026-01-02"]}))
    assert report.leakage_columns_in_features == []
    assert LEAKAGE_COLUMNS.isdisjoint(set(FailureFeatureExtractor.feature_columns()))


def test_dataset_validator_warns_on_single_class():
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(5)],
            "actual_failure": [0, 0, 0, 0, 0],
            "run_id": [1, 2, 3, 4, 5],
        }
    )
    report = validate_dataset(frame)
    assert report.sufficient_for_training is False
    assert any("one outcome class" in warning for warning in report.warnings)


def test_dataset_validator_rejects_non_binary_failure_labels():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="D"),
            "actual_failure": [0, 1] * 9 + [0, 2],
            "run_id": list(range(20)),
        }
    )

    report = validate_dataset(frame)

    assert report.sufficient_for_training is False
    assert report.other_runs == 1
    assert any("binary labels" in warning for warning in report.warnings)


def test_feedback_idempotent_by_run_id(tmp_path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.7,
        predicted_failure=True,
        risk_level="HIGH",
        model_version="test",
    )
    store.append_prediction("owner/repo", "CI/CD Pipeline", 123, "abc123", prediction)
    ok1, pid1, status1 = store.record_outcome_by_run_id(123, "success")
    ok2, pid2, status2 = store.record_outcome_by_run_id(123, "success")
    assert ok1 and ok2
    assert pid1 == pid2
    assert status1 == "recorded"
    assert status2 == "already_recorded"
    frame = pd.read_csv(tmp_path / "history.csv")
    assert len(frame) == 1
    assert int(frame.loc[0, "actual_failure"]) == 0


def test_feedback_service_accuracy_summary(tmp_path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    service = PredictionFeedbackService(store)
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.8,
        predicted_failure=True,
        risk_level="HIGH",
        model_version="test",
    )
    pid = store.append_prediction("owner/repo", "ci", 1, "sha1", prediction)
    store.update_actual_outcome(pid, actual_failure=1, actual_conclusion="failure")
    summary = service.feedback_accuracy_summary()
    assert summary["recorded"] == 1
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1.0


def test_commit_analyzer_scores_dependency_manifest():
    analyzer = CommitAnalyzer()
    parsed = ParsedError(
        error_type="ModuleNotFoundError",
        error_message="No module named requests",
        error_category=ErrorCategory.DEPENDENCY,
        failed_step="Install dependencies",
    )
    result = analyzer.analyze_failed_run(
        commit_sha="abc1234567890",
        changed_files=["requirements.txt", "app.py"],
        commit_message="update deps",
        parsed_error=parsed,
        error_category="dependency",
        failed_step="Install dependencies",
    )
    assert result.commit_sha == "abc1234567890"
    assert result.confidence in {"medium", "high", "low"}
    assert any("Dependency" in item for item in result.evidence)


def test_commit_analyzer_ranks_multiple_commits():
    analyzer = CommitAnalyzer()
    parsed = ParsedError(
        error_type="SyntaxError",
        error_message='File "app.py", line 10',
        error_category=ErrorCategory.SYNTAX,
        failed_step="Run tests",
    )
    result = analyzer.analyze_commit_list(
        commits=[
            {"sha": "aaa", "message": "docs", "files": ["README.md"]},
            {"sha": "bbb", "message": "fix app", "files": ["app.py"]},
        ],
        parsed_error=parsed,
        error_category="build",
    )
    assert result.commit_sha == "bbb"
    assert result.ranked_candidates[0]["commit_sha"] == "bbb"


def test_history_store_pending_commit_lookup(tmp_path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.4,
        predicted_failure=False,
        risk_level="LOW",
        model_version="test",
    )
    store.append_prediction("owner/repo", "ci", None, "deadbeef", prediction)
    pending = store.find_pending_by_commit("owner/repo", "deadbeef")
    assert pending is not None
    ok, pid, status = store.record_outcome_for_commit("owner/repo", "deadbeef", "failure")
    assert ok and pid and status == "recorded"


def test_history_store_refuses_ambiguous_abbreviated_sha(tmp_path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.4,
        predicted_failure=False,
        risk_level="LOW",
        model_version="test",
    )
    store.append_prediction("owner/repo", "ci", None, "deadbee" + "1" * 33, prediction)
    store.append_prediction("owner/repo", "ci", None, "deadbee" + "2" * 33, prediction)

    assert store.find_pending_by_commit("owner/repo", "deadbee", "ci") is None


def test_history_store_prefers_exact_sha_over_prefix_collision(tmp_path):
    store = PredictionHistoryStore(tmp_path / "history.csv")
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.4,
        predicted_failure=False,
        risk_level="LOW",
        model_version="test",
    )
    exact_sha = "deadbee" + "1" * 33
    expected = store.append_prediction("owner/repo", "ci", None, exact_sha, prediction)
    store.append_prediction("owner/repo", "ci", None, "deadbee" + "2" * 33, prediction)

    pending = store.find_pending_by_commit("owner/repo", exact_sha, "ci")

    assert pending is not None
    assert pending["prediction_id"] == expected


def test_temporal_ordering_uses_run_id_tiebreaker():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "run_id": [3, 1, 2],
            "actual_failure": [0, 1, 0],
        }
    )
    ordered = frame.sort_values(["timestamp", "run_id"], kind="mergesort").reset_index(drop=True)
    assert list(ordered["run_id"]) == [1, 3, 2]


def test_missing_github_token_uses_anonymous_public_client():
    collector = HistoricalRunCollector(token="")

    assert collector.token is None
    assert collector.github is not None


def test_train_cli_refuses_insufficient_data(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    dataset = tmp_path / "tiny.csv"
    dataset.write_text("timestamp,actual_failure,run_id\n2026-01-01,0,1\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.prediction.cli", "train", "--dataset", str(dataset)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Insufficient real historical data" in (result.stdout + result.stderr)
