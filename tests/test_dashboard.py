from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

from config import Config
from src.integrations import github_automation
from src.integrations.github_automation import (
    GitHubAutomatedPrediction,
    GitHubChangedFile,
    GitHubCommitSnapshot,
    GitHubConnectionStatus,
    GitHubFailedRunContext,
    GitHubPredictionInput,
    GitHubWorkflowLogBundle,
    GitHubWorkflowLogFile,
    GitHubWorkflowRunSummary,
)
from src.prediction.schemas import FailurePrediction, FeatureImportance


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
REPOSITORY = "Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer"


def _isolate_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Config, "HISTORICAL_DATASET_PATH", tmp_path / "historical_runs.csv")
    monkeypatch.setattr(Config, "PREDICTION_HISTORY_PATH", tmp_path / "prediction_history.csv")
    monkeypatch.setattr(Config, "PREDICTOR_MODEL_PATH", tmp_path / "failure_predictor.joblib")
    monkeypatch.setattr(Config, "PREDICTOR_METADATA_PATH", tmp_path / "failure_predictor_metadata.json")
    monkeypatch.setattr(Config, "CATEGORY_MODEL_PATH", tmp_path / "failure_category_predictor.joblib")
    monkeypatch.setattr(Config, "CATEGORY_METADATA_PATH", tmp_path / "failure_category_predictor_metadata.json")
    monkeypatch.setattr(Config, "ANALYSIS_COOLDOWN_SECONDS", 0)
    monkeypatch.setattr(Config, "GITHUB_ACCESS_TOKEN", None)
    monkeypatch.setattr(Config, "APP_PASSWORD", None)


def _button(app: AppTest, key: str):
    return next(button for button in app.button if button.key == key)


def _radio(app: AppTest, label: str):
    return next(radio for radio in app.radio if radio.label == label)


def _sample_commit() -> GitHubCommitSnapshot:
    return GitHubCommitSnapshot(
        sha="a" * 40,
        short_sha="aaaaaaa",
        message="Automate GitHub change intelligence",
        author_login="octocat",
        committed_at=datetime(2026, 8, 30, 9, 15, tzinfo=UTC),
        html_url="https://github.com/example/repo/commit/" + "a" * 40,
        additions=84,
        deletions=11,
        total_changes=95,
        files=[
            GitHubChangedFile(
                filename="app.py",
                status="modified",
                additions=84,
                deletions=11,
                changes=95,
            )
        ],
    )


def _sample_run() -> GitHubWorkflowRunSummary:
    return GitHubWorkflowRunSummary(
        id=8123,
        run_number=42,
        workflow_name="CI/CD Pipeline",
        status="completed",
        conclusion="failure",
        branch="main",
        head_sha="a" * 40,
        created_at=datetime(2026, 8, 30, 9, 20, tzinfo=UTC),
        html_url="https://github.com/example/repo/actions/runs/8123",
    )


class FakeGitHubAutomationService:
    tokens: list[str | None] = []
    connect_calls: list[str] = []
    prediction_calls: list[dict] = []
    failure_calls: list[dict] = []

    def __init__(self, token=None, repository=None):
        self.token = token
        self.repository = repository
        self.__class__.tokens.append(token)

    @classmethod
    def reset(cls) -> None:
        cls.tokens = []
        cls.connect_calls = []
        cls.prediction_calls = []
        cls.failure_calls = []

    def connect_repository(self, repository):
        self.__class__.connect_calls.append(repository)
        return GitHubConnectionStatus(
            connected=True,
            authenticated=bool(self.token),
            account_login="octocat" if self.token else None,
            repository=repository,
            repository_accessible=True,
            private=False,
            default_branch="main",
            html_url="https://github.com/example/repo",
        )

    def get_recent_commits(self, repository=None, branch=None, limit=10):
        return [_sample_commit()]

    def get_workflow_runs(self, repository=None, limit=20, status=None):
        return [_sample_run()]

    def predict_latest_change(
        self,
        prediction_service,
        repository=None,
        branch=None,
        workflow_name=None,
        record=True,
    ):
        self.__class__.prediction_calls.append(
            {"repository": repository, "workflow_name": workflow_name, "record": record}
        )
        prediction = FailurePrediction(
            model_available=True,
            failure_probability=0.73,
            predicted_failure=True,
            risk_level="HIGH",
            predicted_category="test_failure",
            category_confidence=0.81,
            model_version="test-model",
            top_risk_factors=[
                FeatureImportance(feature="test_files_changed", value=1, importance=0.4, contribution=0.4)
            ],
            feature_importances=[],
        )
        prediction_input = GitHubPredictionInput(
            repository=repository or REPOSITORY,
            workflow_name=workflow_name or "CI/CD Pipeline",
            branch="main",
            commit=_sample_commit(),
            workflow_run_id=8123,
            workflow_status="completed",
            workflow_conclusion="failure",
        )
        return GitHubAutomatedPrediction(
            input=prediction_input,
            prediction=prediction,
            mode="retrospective",
            is_retrospective=True,
            prediction_recorded=False,
            record_reason="The target workflow already completed.",
        )

    def latest_failed_run_context(
        self,
        repository=None,
        *,
        workflow_name=None,
        include_system_workflows=False,
        include_logs=True,
        max_log_bytes=2 * 1024 * 1024,
    ):
        self.__class__.failure_calls.append(
            {
                "repository": repository,
                "workflow_name": workflow_name,
                "include_system_workflows": include_system_workflows,
            }
        )
        content = """Run pytest -q
FAILED tests/test_api.py::test_health - AssertionError: expected 200
##[error]Process completed with exit code 1.
"""
        return GitHubFailedRunContext(
            run=_sample_run(),
            logs=GitHubWorkflowLogBundle(
                repository=repository or REPOSITORY,
                run_id=8123,
                files=[GitHubWorkflowLogFile(name="test.log", content=content, size_bytes=len(content))],
                combined_text=content,
                byte_count=len(content),
            ),
        )


class FakePredictionService:
    def __init__(self, *args, **kwargs):
        self.model_available = True
        self.predictor = SimpleNamespace(category_available=False)

    def predict(self, features):
        return FailurePrediction(
            model_available=True,
            failure_probability=0.42,
            predicted_failure=False,
            risk_level="MEDIUM",
            model_version="test-model",
        )

    def predict_and_record(self, **kwargs):
        return self.predict(kwargs["features"]), "prediction-test-id"


class FakeHistoryCollector:
    tokens: list[str | None] = []
    prediction_features: list[dict] = []

    def __init__(self, token=None, *args, **kwargs):
        self.__class__.tokens.append(token)

    def collect_repository_runs(self, repository, limit=None):
        rows = []
        started = datetime(2026, 8, 1, tzinfo=UTC)
        for index in range(24):
            failed = int(index % 3 == 0)
            rows.append(
                {
                    "repository": repository,
                    "timestamp": (started + timedelta(days=index)).isoformat(),
                    "workflow_name": "CI/CD Pipeline",
                    "run_id": index + 1,
                    "branch": "main",
                    "commit_sha": f"{index:040x}",
                    "conclusion": "failure" if failed else "success",
                    "actual_failure": failed,
                    "actual_category": "test_failure" if failed else None,
                    "lines_added": 20 + index * 10,
                    "lines_deleted": index,
                    "files_changed": 2 + index % 4,
                }
            )
        return pd.DataFrame(rows)

    def build_prediction_features(self, repository, commit_sha, branch=None, workflow_name=None):
        call = {
            "repository": repository,
            "commit_sha": commit_sha,
            "branch": branch,
            "workflow_name": workflow_name,
        }
        self.__class__.prediction_features.append(call)
        return {
            "files_changed": 2,
            "lines_added": 20,
            "lines_deleted": 3,
            "number_of_commits": 1,
            "changed_files_json": '["app.py"]',
        }


class FakeTrainer:
    calls: list[tuple] = []

    def train(self, *args):
        self.__class__.calls.append(args)
        return {
            "metrics": {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.7,
                "f1": 0.72,
                "test_rows": 8,
            }
        }


def _patch_github_service(monkeypatch) -> None:
    FakeGitHubAutomationService.reset()
    monkeypatch.setattr(github_automation, "GitHubAutomationService", FakeGitHubAutomationService)


def _patch_prediction_service(monkeypatch) -> None:
    import src.prediction

    monkeypatch.setattr(src.prediction, "FailurePredictionService", FakePredictionService)


def _connect(app: AppTest, token: str | None = None) -> AppTest:
    if token is not None:
        app.text_input(key="github_session_token").set_value(token)
    _button(app, "github_connect_refresh").click()
    return app.run()


def test_dashboard_cold_start_without_secrets_or_model(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert len(app.tabs) == 10  # seven top-level tabs plus the three model sub-tabs
    assert any("Predict earlier. Diagnose automatically." in item.value for item in app.markdown)
    assert any("Prediction is inactive" in item.value for item in app.warning)
    assert app.text_input(key="github_session_token").value == ""
    assert not _button(app, "github_connect_refresh").disabled
    assert _button(app, "github_predict_latest").disabled
    assert _button(app, "github_analyze_latest_failure").disabled
    assert _button(app, "github_sync_history").disabled


def test_dashboard_analyzes_a_pasted_log_as_advanced_fallback(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    _radio(app, "Analysis source").set_value("Advanced: paste or upload log")
    app.run()
    log = """##[group]Run pytest -q
FAILED tests/test_api.py::test_health - AssertionError: expected 200
##[error]Process completed with exit code 1.
"""
    app.text_area(key="manual_log_content").set_value(log)
    _button(app, "analyze_local").click()
    app.run()

    assert not app.exception
    assert any("Detected" in item.value for item in app.success)
    assert any("PytestFailure" in item.value for item in app.code)
    assert any(button.label == "Download debugging brief" for button in app.download_button)


def test_dashboard_password_gate_hides_operational_controls(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(Config, "APP_PASSWORD", "correct-horse-battery-staple")

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert len(app.tabs) == 0
    assert any("Protected operations console" in item.value for item in app.markdown)
    assert app.text_input[0].label == "Dashboard password"


def test_dashboard_password_unlock_handles_rejection_and_success(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    monkeypatch.setattr(Config, "APP_PASSWORD", "correct-password")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app.text_input(key="dashboard_password").set_value("wrong-password")
    _button(app, "dashboard_unlock").click()
    app.run()
    assert any("password is not valid" in item.value for item in app.error)

    app.text_input(key="dashboard_password").set_value("correct-password")
    _button(app, "dashboard_unlock").click()
    app.run()
    assert not app.exception
    assert len(app.tabs) == 10


def test_invalid_repository_never_constructs_an_invalid_service(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app.text_input(key="repository_name").set_value("not-a-repository")
    app.run()

    assert not app.exception
    assert _button(app, "github_connect_refresh").disabled
    assert any("owner/repository" in item.value for item in app.error)


def test_repository_allowlist_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(Config, "ALLOWED_REPOSITORIES", ("owner/allowed",))

    assert Config.repository_is_allowed("Owner/Allowed")
    assert not Config.repository_is_allowed("owner/other")


def test_session_token_connect_refresh_and_disconnect(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app = _connect(app, token="github_pat_test_secret")

    assert not app.exception
    assert app.session_state["github_snapshot"]["status"]["authenticated"] is True
    assert FakeGitHubAutomationService.tokens[-1] == "github_pat_test_secret"
    assert FakeGitHubAutomationService.connect_calls == [REPOSITORY]
    visible_text = " ".join(
        str(item.value)
        for collection in (app.markdown, app.caption, app.text, app.success, app.info)
        for item in collection
    )
    assert "github_pat_test_secret" not in visible_text
    assert _button(app, "github_connect_refresh").label == "Refresh GitHub activity"

    _button(app, "github_connect_refresh").click()
    app.run()
    assert len(FakeGitHubAutomationService.connect_calls) == 2

    _button(app, "github_disconnect").click()
    app.run()
    assert "github_snapshot" not in app.session_state
    assert app.text_input(key="github_session_token").value == ""
    assert _button(app, "github_disconnect").disabled


def test_latest_change_prediction_is_retrospective_and_display_only(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    _patch_prediction_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = _connect(app, token="test-token")

    assert not _button(app, "github_predict_latest").disabled
    _button(app, "github_predict_latest").click()
    app.run()

    assert not app.exception
    assert FakeGitHubAutomationService.prediction_calls[-1] == {
        "repository": REPOSITORY,
        "workflow_name": "CI/CD Pipeline",
        "record": True,
    }
    assert any(metric.label == "Failure probability" and metric.value == "73.0%" for metric in app.metric)
    assert any("Retrospective display-only score" in item.value for item in app.info)
    assert any("Automate GitHub change intelligence" in item.value for item in app.markdown)


def test_risk_lab_manual_snapshot_button_scores_without_feedback(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    _patch_prediction_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    _button(app, "score_manual_snapshot").click()
    app.run()

    assert not app.exception
    assert any(metric.label == "Failure probability" and metric.value == "42.0%" for metric in app.metric)
    assert any("Scenario risk score" in item.value for item in app.subheader)
    assert "prediction_id" not in app.session_state


def test_risk_lab_exact_commit_works_anonymously_for_public_repo(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    _patch_prediction_service(monkeypatch)
    import src.prediction

    FakeHistoryCollector.tokens = []
    FakeHistoryCollector.prediction_features = []
    monkeypatch.setattr(src.prediction, "HistoricalRunCollector", FakeHistoryCollector)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    _radio(app, "Prediction input").set_value("GitHub commit")
    app.run()
    app.text_input(key="prediction_sha").set_value("b" * 40)
    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "I confirm this commit has not completed the target CI workflow."
    ).check()
    app.run()

    assert not _button(app, "predict_manual_commit").disabled
    _button(app, "predict_manual_commit").click()
    app.run()

    assert not app.exception
    assert FakeHistoryCollector.tokens[-1] is None
    assert FakeHistoryCollector.prediction_features[-1]["commit_sha"] == "b" * 40
    assert any("Feedback record: prediction-test-id" in item.value for item in app.caption)


def test_automatic_failed_run_analysis_uses_selected_workflow(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = _connect(app, token="test-token")

    assert not _button(app, "github_analyze_latest_failure").disabled
    _button(app, "github_analyze_latest_failure").click()
    app.run()

    assert not app.exception
    assert FakeGitHubAutomationService.failure_calls[-1] == {
        "repository": REPOSITORY,
        "workflow_name": "CI/CD Pipeline",
        "include_system_workflows": False,
    }
    assert any("logs were fetched and analyzed automatically" in item.value for item in app.success)
    assert any("PytestFailure" in item.value for item in app.code)


def test_automatic_logs_require_actions_read_but_public_metadata_does_not(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = _connect(app)

    assert app.session_state["github_snapshot"]["status"]["authenticated"] is False
    assert _button(app, "github_analyze_latest_failure").disabled
    assert not _button(app, "github_sync_history").disabled
    assert any("Actions: read permission" in item.value for item in app.caption)


def test_sync_history_persists_real_rows_and_unlocks_charts(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    import src.prediction

    FakeHistoryCollector.tokens = []
    monkeypatch.setattr(src.prediction, "HistoricalRunCollector", FakeHistoryCollector)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = _connect(app, token="session-history-token")

    _button(app, "github_sync_history").click()
    app.run()

    assert not app.exception
    dataset = pd.read_csv(Config.HISTORICAL_DATASET_PATH)
    assert len(dataset) == 24
    assert set(dataset["actual_failure"]) == {0, 1}
    assert FakeHistoryCollector.tokens[-1] == "session-history-token"
    assert len(app.get("vega_lite_chart")) >= 3


def test_advanced_history_collection_works_anonymously(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    import src.prediction

    FakeHistoryCollector.tokens = []
    monkeypatch.setattr(src.prediction, "HistoricalRunCollector", FakeHistoryCollector)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not _button(app, "collect_history_advanced").disabled
    _button(app, "collect_history_advanced").click()
    app.run()

    assert not app.exception
    assert len(pd.read_csv(Config.HISTORICAL_DATASET_PATH)) == 24
    assert FakeHistoryCollector.tokens[-1] is None


def test_dataset_upload_and_training_buttons_activate_artifacts(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    import src.prediction

    FakeTrainer.calls = []
    monkeypatch.setattr(src.prediction, "FailurePredictorTrainer", FakeTrainer)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    minimal_csv = b"timestamp,actual_failure\n2026-08-01T00:00:00Z,0\n2026-08-02T00:00:00Z,1\n"
    app.file_uploader(key="dataset_upload").upload("history.csv", minimal_csv, "text/csv")
    app.run()
    assert not _button(app, "activate_uploaded_dataset").disabled
    _button(app, "activate_uploaded_dataset").click()
    app.run()
    assert len(pd.read_csv(Config.HISTORICAL_DATASET_PATH)) == 2

    training_frame = FakeHistoryCollector().collect_repository_runs(REPOSITORY)
    training_frame.to_csv(Config.HISTORICAL_DATASET_PATH, index=False)
    app.run()
    assert not _button(app, "train_model").disabled
    _button(app, "train_model").click()
    app.run()

    assert not app.exception
    assert FakeTrainer.calls
    assert any(metric.label == "Accuracy" and metric.value == "80.0%" for metric in app.metric)


def test_ai_enriched_rca_button_uses_configured_backend(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    monkeypatch.setattr(Config, "GITHUB_ACCESS_TOKEN", "deployment-token")
    monkeypatch.setattr(Config, "TAVILY_API_KEY", "tavily-key")
    monkeypatch.setattr(Config, "has_aws_credentials", classmethod(lambda cls: True))

    from src.analysis import build_local_debugging_brief
    from src.graph import workflow as graph_workflow
    from src.tools.log_parser import parse_log_content

    parsed = parse_log_content("FAILED tests/test_api.py::test_health - AssertionError: expected 200")
    brief = build_local_debugging_brief(parsed.primary_error, REPOSITORY)
    monkeypatch.setattr(
        graph_workflow,
        "run_analysis",
        lambda repository: SimpleNamespace(debugging_brief=brief, error_message=None),
    )
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    _radio(app, "Analysis source").set_value("AI-enriched GitHub RCA")
    app.run()

    assert not _button(app, "ai_enriched_rca").disabled
    _button(app, "ai_enriched_rca").click()
    app.run()

    assert not app.exception
    assert any("AI analysis completed" in item.value for item in app.success)


def test_all_workflows_selector_includes_system_workflows(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app = _connect(app, token="test-token")

    app.selectbox(key="automation_workflow").set_value("All workflows")
    _button(app, "github_analyze_latest_failure").click()
    app.run()

    assert FakeGitHubAutomationService.failure_calls[-1]["workflow_name"] is None
    assert FakeGitHubAutomationService.failure_calls[-1]["include_system_workflows"] is True


def test_manual_fallback_load_and_clear_buttons_are_functional(monkeypatch, tmp_path):
    _isolate_runtime(monkeypatch, tmp_path)
    _patch_github_service(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    _radio(app, "Analysis source").set_value("Advanced: paste or upload log")
    app.run()

    next(button for button in app.button if button.label == "Load sample log").click()
    app.run()
    assert "ModuleNotFoundError" in app.text_area(key="manual_log_content").value

    next(button for button in app.button if button.label == "Clear log").click()
    app.run()
    assert app.text_area(key="manual_log_content").value == ""

    _button(app, "analyze_local").click()
    app.run()
    assert any("non-empty CI log" in item.value for item in app.error)
