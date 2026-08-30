from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

from src.tools import github_loader
from src.tools.commit_analyzer import analyze_culprit_for_repository
from src.tools.log_parser import ErrorCategory, classify_error, parse_log_content


def test_latest_failed_run_uses_github_failure_filter(monkeypatch):
    newer_success = SimpleNamespace(id=23, conclusion="success")
    failed_run = SimpleNamespace(id=22, conclusion="failure")

    class FakeRepo:
        def get_workflow_runs(self, status):
            assert status == "failure"
            all_runs_newest_first = [newer_success, failed_run]
            return [run for run in all_runs_newest_first if run.conclusion == status]

    fake_client = SimpleNamespace(get_repo=lambda repository: FakeRepo())
    monkeypatch.setattr(github_loader, "get_github_client", lambda: fake_client)

    result = github_loader.get_latest_failed_workflow_run("owner/repo")

    assert result is failed_run


def test_fetch_failed_logs_does_not_treat_newer_success_as_healthy(monkeypatch, tmp_path):
    failed_run = SimpleNamespace(id=22, conclusion="failure")
    downloaded_path = tmp_path / "build_log_22.txt"

    monkeypatch.setattr(github_loader, "get_latest_failed_workflow_run", lambda repository: failed_run)
    monkeypatch.setattr(github_loader, "download_worflow_logs", lambda run: downloaded_path)

    assert github_loader.fetch_failed_build_logs("owner/repo") == downloaded_path


def test_downloaded_logs_use_run_specific_filenames(monkeypatch, tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as zip_file:
        zip_file.writestr("job/1_step.txt", "Process completed with exit code 1.")

    response = SimpleNamespace(status_code=200, content=archive.getvalue(), text="")

    def fake_get(url, *, headers, stream, timeout):
        assert stream is True
        assert timeout == 60
        return response

    monkeypatch.setattr(github_loader, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(github_loader.requests, "get", fake_get)

    first = github_loader.download_worflow_logs(SimpleNamespace(id=101, logs_url="https://example.test/101"))
    second = github_loader.download_worflow_logs(SimpleNamespace(id=202, logs_url="https://example.test/202"))

    assert first.name == "build_log_101.txt"
    assert second.name == "build_log_202.txt"
    assert first != second
    assert first.exists() and second.exists()


def test_culprit_analysis_uses_the_ingested_run_id(monkeypatch):
    failed_run = SimpleNamespace(id=42, head_sha="abc123def456")
    commit = SimpleNamespace(
        files=[SimpleNamespace(filename="requirements.txt")],
        commit=SimpleNamespace(message="Update dependencies"),
        author=SimpleNamespace(login="developer"),
    )

    class FakeRepo:
        def get_workflow_run(self, run_id):
            assert run_id == 42
            return failed_run

        def get_commit(self, sha):
            assert sha == failed_run.head_sha
            return commit

    fake_client = SimpleNamespace(get_repo=lambda repository: FakeRepo())
    monkeypatch.setattr(github_loader, "get_github_client", lambda: fake_client)

    result = analyze_culprit_for_repository("owner/repo", workflow_run_id=42)

    assert result.commit_sha == failed_run.head_sha
    assert "Dependency manifest changed" in result.evidence


def test_parser_detects_pytest_failed_summary():
    result = parse_log_content(
        """2026-01-01T00:00:00.000Z ##[group]Run pytest -q
2026-01-01T00:00:01.000Z FAILED tests/test_api.py::test_health - AssertionError
2026-01-01T00:00:02.000Z ##[error]Process completed with exit code 1.
"""
    )

    assert result.primary_error is not None
    assert result.primary_error.error_type == "PytestFailure"
    assert result.primary_error.error_category == ErrorCategory.TEST_FAILURE
    assert result.primary_error.failed_step == "pytest -q"


def test_parser_detects_bare_pytest_assertion_error():
    result = parse_log_content(
        """##[group]Run pytest -q
E   AssertionError
##[error]Process completed with exit code 1.
"""
    )

    assert result.primary_error is not None
    assert result.primary_error.error_type == "AssertionError"
    assert result.primary_error.error_message == "Assertion failed"
    assert result.primary_error.error_category == ErrorCategory.TEST_FAILURE


def test_parser_strips_ansi_colours_before_matching_errors():
    result = parse_log_content("\x1b[31mModuleNotFoundError: No module named requests\x1b[0m\n")

    assert result.primary_error is not None
    assert result.primary_error.error_type == "ModuleNotFoundError"
    assert result.primary_error.error_category == ErrorCategory.DEPENDENCY


def test_parser_preserves_exit_code_only_failure():
    result = parse_log_content(
        """##[group]Run custom-build-command
Process completed with exit code 17.
"""
    )

    assert result.primary_error is not None
    assert result.primary_error.error_type == "ProcessExitError"
    assert result.primary_error.exit_code == 17
    assert result.primary_error.failed_step == "custom-build-command"


def test_timeout_errors_have_timeout_category():
    assert classify_error("TimeoutError", "Operation timed out") == ErrorCategory.TIMEOUT
