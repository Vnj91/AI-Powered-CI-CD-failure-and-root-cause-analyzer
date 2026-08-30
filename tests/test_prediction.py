from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from src.prediction import (
    FailureFeatureExtractor,
    FailurePrediction,
    FailurePredictor,
    FailurePredictorTrainer,
    FailurePredictionService,
    HistoricalRunCollector,
    PredictionHistoryStore,
)
from src.prediction.evaluator import evaluate_saved_model
from src.graph.workflow import run_analysis


def make_training_dataset(rows: int = 30) -> pd.DataFrame:
    feature_columns = FailureFeatureExtractor.feature_columns()
    records = []
    start = datetime(2026, 1, 1)

    for index in range(rows):
        record = {column: 0 for column in feature_columns}
        record.update(
            {
                "timestamp": start + timedelta(days=index),
                "actual_failure": int(index % 2 == 0),
                "files_changed": index % 5,
                "lines_added": 5 + index,
                "lines_deleted": index % 3,
                "dependency_files_changed": int(index % 3 == 0),
                "ci_workflow_files_changed": int(index % 7 == 0),
                "previous_failure_count": index // 3,
                "recent_failure_count": index // 4,
                "recent_failure_rate": min(index / max(rows, 1), 1.0),
                "changed_ext_py_count": int(index % 4 == 0),
                "changed_ext_yml_count": int(index % 6 == 0),
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def make_category_training_dataset(rows: int = 40) -> pd.DataFrame:
    feature_columns = FailureFeatureExtractor.feature_columns()
    records = []
    start = datetime(2026, 2, 1)
    categories = ["dependency", "build", "test"]

    for index in range(rows):
        record = {column: 0 for column in feature_columns}
        failure = int(index % 2 == 0)
        category = categories[index % len(categories)] if failure else None
        record.update(
            {
                "timestamp": start + timedelta(days=index),
                "actual_failure": failure,
                "actual_category": category,
                "files_changed": (index % 4) + 1,
                "lines_added": 10 + index,
                "lines_deleted": index % 5,
                "dependency_files_changed": int(category == "dependency"),
                "ci_workflow_files_changed": int(index % 6 == 0),
                "docker_files_changed": int(index % 7 == 0),
                "test_files_changed": int(category == "test"),
                "previous_failure_count": index // 4,
                "recent_failure_count": index // 5,
                "recent_failure_rate": min(index / max(rows, 1), 1.0),
                "changed_ext_py_count": int(index % 3 == 0),
                "changed_ext_yml_count": int(index % 4 == 0),
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def test_feature_extraction_handles_missing_values():
    feature_row = FailureFeatureExtractor.to_feature_row({"files_changed": None, "changed_files_json": "[]"})

    assert feature_row["files_changed"] == 0
    assert feature_row["lines_added"] == 0
    assert feature_row["changed_ext_py_count"] == 0


def test_prepare_training_frame_materializes_collector_shaped_rows():
    dataset = pd.DataFrame(
        [
            {
                "actual_failure": 1,
                "previous_run_status": "failure",
                "changed_files_json": '["src/main.py", ".github/workflows/ci.yml", "Dockerfile"]',
                "files_changed": 3,
                "lines_added": 12,
                "lines_deleted": 4,
                "dependency_files_changed": False,
                "ci_workflow_files_changed": True,
                "docker_files_changed": True,
            }
        ]
    )

    features, labels = FailureFeatureExtractor.prepare_training_frame(dataset)

    assert list(features.columns) == FailureFeatureExtractor.feature_columns()
    assert features.loc[0, "previous_run_status_failure"] == 1
    assert features.loc[0, "previous_run_status_success"] == 0
    assert features.loc[0, "changed_ext_py_count"] == 1
    assert features.loc[0, "changed_ext_yml_count"] == 1
    assert features.loc[0, "changed_ext_dockerfile_count"] == 1
    assert features.loc[0, "changed_extension_diversity"] == 3
    assert labels.tolist() == [1]


def test_prepare_training_frame_preserves_explicit_derived_features():
    dataset = pd.DataFrame(
        [
            {
                "actual_failure": 0,
                "previous_run_status": "failure",
                "changed_files_json": '["src/main.py"]',
                "previous_run_status_failure": 0,
                "previous_run_status_success": 1,
                "changed_ext_py_count": 7,
                "changed_extension_diversity": 9,
            }
        ]
    )

    features, _ = FailureFeatureExtractor.prepare_training_frame(dataset)

    assert features.loc[0, "previous_run_status_failure"] == 0
    assert features.loc[0, "previous_run_status_success"] == 1
    assert features.loc[0, "changed_ext_py_count"] == 7
    assert features.loc[0, "changed_extension_diversity"] == 9


def test_history_store_appends_and_updates(tmp_path: Path):
    store = PredictionHistoryStore(tmp_path / "prediction_history.csv")

    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.8,
        predicted_failure=True,
        risk_level="HIGH",
        model_version="test",
    )

    prediction_id = store.append_prediction(
        repository="owner/repo",
        workflow="build",
        run_id=1,
        commit_sha="abc123",
        prediction=prediction,
    )

    assert prediction_id
    assert store.update_actual_outcome(prediction_id, actual_failure=1, actual_category="dependency")

    frame = pd.read_csv(tmp_path / "prediction_history.csv")
    assert len(frame) == 1
    assert int(frame.loc[0, "actual_failure"]) == 1


def test_history_store_preserves_concurrent_predictions(tmp_path: Path):
    history_path = tmp_path / "prediction_history.csv"
    prediction = FailurePrediction(
        model_available=True,
        failure_probability=0.6,
        predicted_failure=True,
        risk_level="MEDIUM",
        model_version="concurrency-test",
    )

    def append(index: int) -> str:
        store = PredictionHistoryStore(history_path)
        return store.append_prediction(
            repository="owner/repo",
            workflow="CI/CD Pipeline",
            run_id=index,
            commit_sha=f"sha-{index}",
            prediction=prediction,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        prediction_ids = list(executor.map(append, range(24)))

    frame = pd.read_csv(history_path)
    assert len(frame) == 24
    assert frame["run_id"].nunique() == 24
    assert len(set(prediction_ids)) == 24


def test_prediction_service_does_not_record_when_model_is_unavailable(tmp_path: Path):
    history_path = tmp_path / "prediction_history.csv"
    service = FailurePredictionService(
        model_path=tmp_path / "missing_model.joblib",
        history_path=history_path,
    )

    prediction, prediction_id = service.predict_and_record(
        repository="owner/repo",
        workflow="CI/CD Pipeline",
        run_id=123,
        commit_sha="abc123",
        features={"files_changed": 1, "changed_files_json": '["src/main.py"]'},
    )

    assert prediction.model_available is False
    assert prediction_id is None
    assert not history_path.exists()


def test_trainer_predictor_and_evaluator_round_trip(tmp_path: Path):
    dataset = make_training_dataset()
    dataset_path = tmp_path / "historical_runs.csv"
    model_path = tmp_path / "failure_predictor.joblib"
    metadata_path = tmp_path / "failure_predictor_metadata.json"
    dataset.to_csv(dataset_path, index=False)

    trainer = FailurePredictorTrainer(random_state=7)
    artifact = trainer.train(dataset_path, model_path, metadata_path)

    assert model_path.exists()
    assert artifact["metrics"]["train_rows"] > 0

    predictor = FailurePredictor(model_path)
    prediction = predictor.predict(FailureFeatureExtractor.to_feature_row(dataset.iloc[0].to_dict()))

    assert prediction.model_available is True
    assert prediction.failure_probability is not None
    assert prediction.risk_level in {"LOW", "MEDIUM", "HIGH"}

    service = FailurePredictionService(model_path, history_path=tmp_path / "history.csv")
    assert service.model_available is True
    service_prediction = service.predict(FailureFeatureExtractor.to_feature_row(dataset.iloc[1].to_dict()))
    assert service_prediction.model_available is True

    metrics = evaluate_saved_model(dataset_path, model_path)
    assert metrics.train_rows > 0
    assert metrics.test_rows > 0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_trainer_rejects_one_class_temporal_training_prefix(tmp_path: Path):
    dataset = make_training_dataset(rows=20)
    dataset["actual_failure"] = [0] * 16 + [1] * 4
    dataset_path = tmp_path / "historical_runs.csv"
    model_path = tmp_path / "failure_predictor.joblib"
    dataset.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Temporal training split must contain at least two classes"):
        FailurePredictorTrainer(random_state=13).train(dataset_path, model_path)

    assert not model_path.exists()


def test_trainer_moves_temporal_split_to_keep_both_holdout_classes():
    labels = pd.Series([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1])

    split_index = FailurePredictorTrainer._select_temporal_split(labels, test_fraction=0.2)

    assert split_index == 6
    assert set(labels.iloc[:split_index]) == {0, 1}
    assert set(labels.iloc[split_index:]) == {0, 1}


def test_trainer_rejects_non_binary_target_values(tmp_path: Path):
    dataset = make_training_dataset(rows=20)
    dataset.loc[dataset.index[-1], "actual_failure"] = 2
    dataset_path = tmp_path / "historical_runs.csv"
    dataset.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="binary labels 0 and 1"):
        FailurePredictorTrainer().train(dataset_path, tmp_path / "model.joblib")


def test_category_model_trains_and_predicts(tmp_path: Path):
    dataset = make_category_training_dataset()
    dataset_path = tmp_path / "historical_runs.csv"
    model_path = tmp_path / "failure_predictor.joblib"
    metadata_path = tmp_path / "failure_predictor_metadata.json"
    dataset.to_csv(dataset_path, index=False)

    trainer = FailurePredictorTrainer(random_state=11)
    artifact = trainer.train(dataset_path, model_path, metadata_path)

    assert artifact["category_artifact"]["trained"] is True

    category_model_path = tmp_path / "failure_category_predictor.joblib"
    assert category_model_path.exists()

    predictor = FailurePredictor(model_path, category_model_path=category_model_path)
    prediction = predictor.predict(FailureFeatureExtractor.to_feature_row(dataset.iloc[0].to_dict()))

    assert prediction.predicted_category in {"dependency", "build", "test"}
    assert prediction.category_confidence is not None


def test_category_model_skips_when_temporal_prefix_lacks_a_category(tmp_path: Path):
    dataset = make_category_training_dataset(rows=40)
    failure_indices = dataset.index[dataset["actual_failure"] == 1].tolist()
    dataset.loc[failure_indices[:8], "actual_category"] = "dependency"
    dataset.loc[failure_indices[8:16], "actual_category"] = "build"
    dataset.loc[failure_indices[16:], "actual_category"] = "test"

    dataset_path = tmp_path / "historical_runs.csv"
    model_path = tmp_path / "failure_predictor.joblib"
    metadata_path = tmp_path / "failure_predictor_metadata.json"
    dataset.to_csv(dataset_path, index=False)

    artifact = FailurePredictorTrainer(random_state=17).train(
        dataset_path,
        model_path,
        metadata_path,
    )

    category_artifact = artifact["category_artifact"]
    assert category_artifact["trained"] is False
    assert category_artifact["missing_categories"] == ["test"]
    assert "Temporal category training split" in category_artifact["reason"]
    assert model_path.exists()
    assert not (tmp_path / "failure_category_predictor.joblib").exists()


def test_collector_builds_prediction_features_from_fakes(monkeypatch):
    class FakeCommitStats:
        additions = 12
        deletions = 4

    @dataclass
    class FakeFile:
        filename: str

    class FakeCommit:
        files = [FakeFile("requirements.txt"), FakeFile(".github/workflows/ci.yml"), FakeFile("app.py")]
        stats = FakeCommitStats()
        parents = []
        commit = type("obj", (), {"message": "update deps"})()
        author = type("obj", (), {"login": "alice"})()

    class FakeRun:
        def __init__(self, sha: str, branch: str, conclusion: str, created_at: datetime):
            self.head_sha = sha
            self.head_branch = branch
            self.conclusion = conclusion
            self.created_at = created_at
            self.workflow = type("obj", (), {"name": "build"})()
            self.name = "build"
            self.workflow_id = 123
            self.id = 456
            self.run_number = 10
            self.status = "completed"
            self.event = "push"
            self.actor = type("obj", (), {"login": "alice"})()

    class FakeRepo:
        def __init__(self):
            self.runs = [
                FakeRun("sha-old", "main", "success", datetime(2026, 1, 1)),
                FakeRun("sha-new", "main", "failure", datetime(2026, 1, 2)),
            ]

        def get_commit(self, sha):
            return FakeCommit()

        def compare(self, base, head):
            return type("obj", (), {"total_commits": 2, "files": FakeCommit.files})()

        def get_workflow_runs(self, status="completed"):
            return self.runs

    collector = HistoricalRunCollector(token="dummy")
    monkeypatch.setattr(collector, "_repo", lambda repository: FakeRepo())

    features = collector.build_prediction_features("owner/repo", commit_sha="sha-new", branch="main", workflow_name="build")

    assert features["files_changed"] == 3
    assert features["dependency_files_changed"] is True
    assert features["ci_workflow_files_changed"] is True
    assert features["recent_failure_count"] >= 0


def test_original_rca_workflow_smoke(monkeypatch):
    from src.graph import workflow as workflow_module
    from src.tools.log_parser import ParsedError, LogParseResult, ErrorCategory
    from src.agents.triage_agent import Severity, RefinedErrorCategory, TriageResult
    from src.agents.research_agent import ResearchResult, SolutionCandidate
    from src.graph.state import DebuggingBrief, FixSuggestion

    fake_log_path = Path("/tmp/fake_build_log.txt")
    fake_log_path.write_text("Run tests\nModuleNotFoundError: No module named example\n", encoding="utf-8")

    def fake_fetch_failed_build_logs(repo_name):
        return fake_log_path

    def fake_parse_log_file(file_path):
        primary_error = ParsedError(
            error_type="ModuleNotFoundError",
            error_message="No module named example",
            error_category=ErrorCategory.DEPENDENCY,
            failed_step="Run tests",
            exit_code=1,
            stack_trace=["Traceback (most recent call last):"],
            raw_error_block="No module named example",
        )
        return LogParseResult(
            success=True,
            errors=[primary_error],
            primary_error=primary_error,
            total_lines=2,
            error_count=1,
            summary="dependency failure",
        )

    class FakeTriageAgent:
        def analyze(self, error):
            return TriageResult(
                severity=Severity.HIGH,
                severity_reasoning="Critical dependency failure",
                root_cause="Missing dependency",
                root_cause_detailed="The package is not installed.",
                error_category_refined=RefinedErrorCategory.MISSING_PACCKAGE,
                affected_files=["requirements.txt"],
                affected_components=["build"],
                immediate_suggestions=["Install dependency"],
                requires_research=False,
                research_queries=["missing dependency fix"],
                confidence_score=0.9,
            )

    class FakeResearchAgent:
        def __init__(self, repo_name=None):
            self.repo_name = repo_name

        def research(self, triage_result, parsed_error):
            return ResearchResult(
                error_summary="ModuleNotFoundError: No module named example",
                research_completed=True,
                web_searches_performed=0,
                web_findings=["Install the missing package"],
                relevant_urls=["https://example.com/fix"],
                repo_analyzed=self.repo_name,
                relevant_files=["requirements.txt"],
                code_observations=["Dependency declaration missing"],
                solutions=[
                    SolutionCandidate(
                        title="Install package",
                        description="Add the dependency to the manifest.",
                        steps=["Update requirements.txt"],
                        source="test",
                        confidence=0.95,
                    )
                ],
                primary_recommendation="Add dependency",
                raw_llm_response="{}",
            )

    class FakeSynthesisAgent:
        def synthesize(self, parsed_error, triage_result, research_result, repo_name):
            return DebuggingBrief(
                title="Dependency failure",
                repository=repo_name,
                error_type=parsed_error.error_type,
                error_message=parsed_error.error_message,
                error_category=triage_result.error_category_refined.value,
                severity=triage_result.severity.value,
                root_cause_summary=triage_result.root_cause,
                root_cause_detailed=triage_result.root_cause_detailed,
                affected_files=triage_result.affected_files,
                affected_components=triage_result.affected_components,
                fix_suggestions=[
                    FixSuggestion(
                        priority=1,
                        title="Install dependency",
                        description="Add the missing dependency.",
                        implementation_steps=["Update manifest"],
                        confidence=0.95,
                    )
                ],
                relevant_links=research_result.relevant_urls,
                confidence_score=0.9,
            )

    monkeypatch.setattr(workflow_module, "fetch_failed_build_logs", fake_fetch_failed_build_logs)
    monkeypatch.setattr(workflow_module, "parse_log_file", fake_parse_log_file)
    monkeypatch.setattr(workflow_module, "TriageAgent", FakeTriageAgent)
    monkeypatch.setattr(workflow_module, "ResearchAgent", FakeResearchAgent)
    monkeypatch.setattr(workflow_module, "SynthesisAgent", FakeSynthesisAgent)
    monkeypatch.setattr(workflow_module, "DELAY_BETWEEN_LLM_CALLS", 0)

    result = run_analysis("owner/repo")

    assert result.debugging_brief is not None
    assert result.debugging_brief.error_type == "ModuleNotFoundError"
    assert result.current_phase.value == "completed"
