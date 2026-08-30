from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.integrations.github_automation import (
    GitHubAuthenticationRequired,
    GitHubAutomationService,
)
from src.prediction.schemas import FailurePrediction


def _commit(sha: str = "abcdef1234567890"):
    changed_file = SimpleNamespace(
        filename="src/application.py",
        status="modified",
        additions=17,
        deletions=4,
        changes=21,
        previous_filename=None,
        blob_url="https://github.test/blob",
        raw_url="https://github.test/raw",
        patch="+print('safe')",
    )
    git_author = SimpleNamespace(name="Developer", date=datetime(2026, 8, 30, tzinfo=UTC))
    return SimpleNamespace(
        sha=sha,
        commit=SimpleNamespace(message="Improve application", author=git_author, committer=git_author),
        author=SimpleNamespace(login="developer"),
        html_url=f"https://github.test/commit/{sha}",
        stats=SimpleNamespace(additions=17, deletions=4, total=21),
        files=[changed_file],
    )


def _run(
    run_id: int = 44,
    *,
    sha: str = "abcdef1234567890",
    status: str = "completed",
    conclusion: str | None = "success",
    workflow_name: str = "build",
):
    return SimpleNamespace(
        id=run_id,
        run_number=7,
        run_attempt=1,
        workflow=SimpleNamespace(name=workflow_name),
        name=workflow_name,
        display_title="Build application",
        status=status,
        conclusion=conclusion,
        event="push",
        head_branch="main",
        head_sha=sha,
        actor=SimpleNamespace(login="developer"),
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        updated_at=datetime(2026, 8, 30, tzinfo=UTC),
        html_url=f"https://github.test/actions/runs/{run_id}",
        logs_url=f"https://api.github.test/runs/{run_id}/logs",
    )


class FakeRepo:
    full_name = "owner/repo"
    owner = SimpleNamespace(login="owner")
    private = False
    visibility = "public"
    default_branch = "main"
    html_url = "https://github.test/owner/repo"
    description = "Test repository"
    updated_at = datetime(2026, 8, 30, tzinfo=UTC)
    permissions = SimpleNamespace(raw_data={"pull": True, "push": False})

    def __init__(self, commits=None, runs=None, workflows=None):
        self.commits = commits or [_commit()]
        self.runs = runs or []
        self.workflows = workflows or [
            SimpleNamespace(
                id=1,
                name="CI/CD Pipeline",
                path=".github/workflows/ci.yml",
                state="active",
                html_url="https://github.test/actions/workflows/ci.yml",
            )
        ]

    def get_commits(self, sha=None):
        assert sha == "main"
        return self.commits

    def get_commit(self, sha):
        return next(commit for commit in self.commits if commit.sha == sha)

    def get_workflow_runs(self, status=None, head_sha=None):
        rows = self.runs
        if status:
            rows = [run for run in rows if run.conclusion == status]
        if head_sha:
            rows = [run for run in rows if run.head_sha == head_sha]
        return rows

    def get_workflow_run(self, run_id):
        return next(run for run in self.runs if run.id == run_id)

    def get_workflows(self):
        return self.workflows


class FakeGithub:
    def __init__(self, repo, login="octocat"):
        self.repo = repo
        self.user = SimpleNamespace(login=login, get_repos=lambda **kwargs: [repo])

    def get_repo(self, name):
        assert name == "owner/repo"
        return self.repo

    def get_user(self):
        return self.user

    def get_rate_limit(self):
        core = SimpleNamespace(remaining=4999, limit=5000, reset=datetime(2026, 8, 30, tzinfo=UTC))
        return SimpleNamespace(core=core)


class FakeCollector:
    def __init__(self, token):
        self.received_token = token
        self.github = None

    def build_prediction_features(self, repository, commit_sha, branch=None, workflow_name=None):
        return {
            "repository": repository,
            "commit_sha": commit_sha,
            "branch": branch,
            "workflow_name": workflow_name or "unknown",
            "files_changed": 1,
            "lines_added": 17,
            "lines_deleted": 4,
            "number_of_commits": 1,
            "changed_files_json": '["src/application.py"]',
            "previous_run_status": "success",
            "recent_failure_count": 1,
            "recent_failure_rate": 0.1,
        }


def test_anonymous_connection_and_snapshot_use_real_repository_metadata():
    repo = FakeRepo(runs=[_run()])
    service = GitHubAutomationService(token="", repository="owner/repo", github_client=FakeGithub(repo))

    snapshot = service.get_snapshot(commit_limit=1, run_limit=1)

    assert snapshot.connection.connected
    assert snapshot.connection.repository_accessible
    assert not snapshot.connection.authenticated
    assert snapshot.connection.default_branch == "main"
    assert snapshot.commits[0].files[0].filename == "src/application.py"
    assert snapshot.commits[0].additions == 17
    assert snapshot.workflows[0].name == "CI/CD Pipeline"
    assert snapshot.workflow_runs[0].head_sha == "abcdef1234567890"


def test_workflow_discovery_reads_definitions_not_only_recent_run_names():
    repo = FakeRepo(
        runs=[_run(workflow_name="Legacy Run Name")],
        workflows=[
            SimpleNamespace(
                id=91,
                name="Current Pipeline",
                path=".github/workflows/current.yml",
                state="active",
                html_url="https://github.test/actions/workflows/current.yml",
            )
        ],
    )
    service = GitHubAutomationService(token="", repository="owner/repo", github_client=FakeGithub(repo))

    workflows = service.get_workflows()

    assert [workflow.name for workflow in workflows] == ["Current Pipeline"]
    assert workflows[0].path == ".github/workflows/current.yml"


def test_authenticated_status_reports_account_but_never_serializes_token():
    token = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    service = GitHubAutomationService(
        token=token,
        repository="owner/repo",
        github_client=FakeGithub(FakeRepo()),
    )

    status = service.verify_connection()

    assert status.authenticated
    assert status.account_login == "octocat"
    assert token not in repr(service)
    assert token not in status.model_dump_json()


def test_latest_commit_context_reuses_collector_features():
    repo = FakeRepo(runs=[])
    service = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
        collector_factory=FakeCollector,
    )

    context = service.get_latest_commit_context(workflow_name="build")

    assert context.commit.sha == "abcdef1234567890"
    assert context.raw_features["lines_added"] == 17
    assert context.model_features["files_changed"] == 1
    assert context.model_features["changed_ext_py_count"] == 1
    assert context.workflow_run_id is None


class FakePredictionService:
    def __init__(self):
        self.record_calls = 0
        self.predict_calls = 0

    @staticmethod
    def _prediction():
        return FailurePrediction(
            model_available=True,
            failure_probability=0.61,
            predicted_failure=True,
            risk_level="MEDIUM",
        )

    def predict_and_record(self, **kwargs):
        self.record_calls += 1
        return self._prediction(), "prediction-1"

    def predict(self, features):
        self.predict_calls += 1
        return self._prediction()


def test_completed_workflow_prediction_is_retrospective_and_not_recorded():
    repo = FakeRepo(runs=[_run(status="completed", conclusion="success")])
    github = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
        collector_factory=FakeCollector,
    )
    prediction_service = FakePredictionService()

    result = github.predict_latest_change(prediction_service, workflow_name="build")

    assert result.is_retrospective
    assert result.mode == "retrospective"
    assert not result.prediction_recorded
    assert result.input.workflow_status == "completed"
    assert prediction_service.record_calls == 0
    assert prediction_service.predict_calls == 1


def test_in_progress_workflow_prediction_can_be_recorded():
    repo = FakeRepo(runs=[_run(status="in_progress", conclusion=None)])
    github = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
        collector_factory=FakeCollector,
    )
    prediction_service = FakePredictionService()

    result = github.predict_latest_change(prediction_service, workflow_name="build")

    assert not result.is_retrospective
    assert result.mode == "active_run"
    assert result.prediction_recorded
    assert result.prediction_id == "prediction-1"
    assert prediction_service.record_calls == 1


def _zip_payload(content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("build/1_test.txt", content)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def iter_content(self, chunk_size=65536):
        yield self.content


def test_anonymous_public_log_attempt_is_bounded_and_redacted():
    secret = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    payload = _zip_payload(f"test failed token={secret}")
    captured_headers = {}

    def http_get(url, **kwargs):
        captured_headers.update(kwargs["headers"])
        return FakeResponse(200, payload)

    repo = FakeRepo(runs=[_run(conclusion="failure")])
    service = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
        http_get=http_get,
    )

    logs = service.get_workflow_logs(44)

    assert "Authorization" not in captured_headers
    assert secret not in logs.combined_text
    assert "[REDACTED" in logs.combined_text
    assert logs.files[0].name == "build/1_test.txt"


def test_log_403_returns_actionable_authentication_error():
    repo = FakeRepo(runs=[_run(conclusion="failure")])
    service = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
        http_get=lambda *args, **kwargs: FakeResponse(403),
    )

    with pytest.raises(GitHubAuthenticationRequired, match="Actions read permission"):
        service.get_workflow_logs(44)


def test_list_repositories_requires_authentication():
    service = GitHubAutomationService(token="", github_client=FakeGithub(FakeRepo()))

    with pytest.raises(GitHubAuthenticationRequired):
        service.list_repositories()


def test_latest_failure_excludes_system_automation_and_accepts_workflow_selector():
    training_failure = _run(run_id=50, conclusion="failure", workflow_name="Train Failure Predictor")
    application_failure = _run(run_id=49, conclusion="failure", workflow_name="CI/CD Pipeline")
    repo = FakeRepo(runs=[training_failure, application_failure])
    service = GitHubAutomationService(
        token="",
        repository="owner/repo",
        github_client=FakeGithub(repo),
    )

    default_result = service.get_latest_failed_run()
    selected_result = service.get_latest_failed_run(workflow_name="Train Failure Predictor")

    assert default_result is not None
    assert default_result.id == 49
    assert selected_result is not None
    assert selected_result.id == 50
