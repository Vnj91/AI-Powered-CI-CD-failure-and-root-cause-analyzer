from __future__ import annotations

import json
import subprocess
import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.prediction.scenario_planner import build_training_scenarios
from src.prediction.training_data_generator import (
    GeneratedRunRecord,
    GeneratedRunStore,
    GhCliClient,
    GitWorkspace,
    TrainingDataGenerator,
    build_plan,
    format_execution_plan,
    summarize_generation,
    write_run_plan,
)


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], subprocess.CompletedProcess[str]] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def run(self, command, **kwargs):
        self.calls.append(list(command))
        key = tuple(command)
        if key in self.responses:
            return self.responses[key]
        if command[:2] == ["gh", "auth"]:
            return subprocess.CompletedProcess(command, 0, stdout="Logged in", stderr="")
        if command[:2] == ["gh", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="gh version 2.0.0", stderr="")
        if command[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "fetch"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="main\n", stderr="")
        if command[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "add"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "commit"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc1234567890\n", stderr="")
        if command[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "workflow", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")


class LiveWorkflowRunner(FakeRunner):
    """Fully local command double for dispatch, discovery, and polling tests."""

    def __init__(self, conclusion: str = "success", responses=None):
        super().__init__(responses)
        self.conclusion = conclusion

    def run(self, command, **kwargs):
        if command[:3] == ["gh", "run", "list"]:
            self.calls.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout='[{"databaseId": 123}]', stderr="")
        if command[:3] == ["gh", "run", "view"]:
            self.calls.append(list(command))
            payload = {
                "status": "completed",
                "conclusion": self.conclusion,
                "headSha": "actual-sha",
                "workflowName": "CI/CD Pipeline",
                "updatedAt": "2026-01-01T00:10:00Z",
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        return super().run(command, **kwargs)


def _generator(tmp_path: Path, runner: FakeRunner) -> TrainingDataGenerator:
    return TrainingDataGenerator(
        repo="owner/repo",
        repo_root=tmp_path,
        gh_client=GhCliClient("owner/repo", runner=runner, dispatch_delay_seconds=0),
        git_workspace=GitWorkspace(tmp_path, runner=runner),
        store=GeneratedRunStore(tmp_path / "generated_runs.json"),
        poll_interval=1,
        timeout=5,
    )


def _load_generation_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_training_data.py"
    spec = importlib.util.spec_from_file_location("generate_training_data_script", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_plan_returns_warnings_for_large_run_set():
    scenarios, warnings, distribution = build_plan(25, 30, allow_large_run_set=True)
    assert len(scenarios) == 55
    assert warnings
    assert distribution["total"] == 55


def test_format_execution_plan_matches_phase_one_output():
    _, _, distribution = build_plan(1, 1)
    text = format_execution_plan(
        total_runs=2,
        success_runs=1,
        failure_runs=1,
        distribution=distribution,
        dry_run=True,
    )
    assert "Training Data Generation Plan" in text
    assert "NO GITHUB ACTIONS WERE DISPATCHED." in text
    assert "dry-run planning phase" in text


def test_write_run_plan_rejects_token_like_content(tmp_path: Path):
    from dataclasses import replace

    scenarios = build_training_scenarios(success_runs=0, failure_runs=1)
    _, _, distribution = build_plan(0, 1)
    scenarios = [replace(scenarios[0], description="contains ghp_secretshouldneverbestored")]
    with pytest.raises(RuntimeError, match="token"):
        write_run_plan(tmp_path / "generated_run_plan.json", scenarios=scenarios, distribution=distribution)


def test_write_run_plan_persists_both_outcomes(tmp_path: Path):
    scenarios, _, distribution = build_plan(2, 3)
    path = write_run_plan(tmp_path / "generated_run_plan.json", scenarios=scenarios, distribution=distribution, seed=11)
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcomes = {item["expected_outcome"] for item in payload["scenarios"]}
    assert outcomes == {"success", "failure"}
    assert payload["seed"] == 11


def test_generated_run_store_round_trip(tmp_path: Path):
    store = GeneratedRunStore(tmp_path / "generated_runs.json")
    record = GeneratedRunRecord(
        run_id=123,
        workflow="CI/CD Pipeline",
        branch="ml-data/success-001",
        commit_sha="abc123",
        expected_category=None,
        expected_outcome="success",
        actual_conclusion="success",
        scenario_id="success-001",
        created_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
    )
    store.save([record])
    loaded = store.load()
    assert loaded[0].run_id == 123


def test_generated_run_metadata_keeps_expected_and_actual_outcomes_separate(tmp_path: Path):
    store = GeneratedRunStore(tmp_path / "generated_runs.json")
    store.save([
        GeneratedRunRecord(
            run_id=91, workflow="CI/CD Pipeline", branch="ml-data/success-001", commit_sha="abc",
            expected_category=None, expected_outcome="success", actual_conclusion="timed_out",
            scenario_id="success-001", created_at="2026-01-01T00:00:00Z",
        )
    ])
    payload = json.loads((tmp_path / "generated_runs.json").read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert set(run) >= {"run_id", "workflow", "branch", "commit_sha", "scenario_id", "expected_outcome", "expected_category", "actual_conclusion", "created_at", "completed_at"}
    assert run["expected_outcome"] == "success"
    assert run["actual_conclusion"] == "timed_out"


def test_generated_run_store_rejects_token_like_content(tmp_path: Path):
    store = GeneratedRunStore(tmp_path / "generated_runs.json")
    record = GeneratedRunRecord(
        run_id=1,
        workflow="CI/CD Pipeline",
        branch="ml-data/success-001",
        commit_sha="ghp_secretshouldneverbestored",
        expected_category=None,
        expected_outcome="success",
        actual_conclusion="success",
        scenario_id="success-001",
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(RuntimeError, match="token"):
        store.save([record])


def test_dry_run_does_not_call_git_push(tmp_path: Path):
    runner = FakeRunner()
    generator = TrainingDataGenerator(
        repo="owner/repo",
        repo_root=tmp_path,
        gh_client=GhCliClient("owner/repo", runner=runner, dispatch_delay_seconds=0),
        git_workspace=GitWorkspace(tmp_path, runner=runner),
        store=GeneratedRunStore(tmp_path / "generated_runs.json"),
        poll_interval=1,
    )
    scenarios = build_training_scenarios(success_runs=1, failure_runs=1)
    progress = generator.generate(scenarios, dry_run=True)
    assert progress.completed == 2
    assert not any(call[:2] == ["git", "push"] for call in runner.calls)
    assert not any(call[:3] == ["gh", "workflow", "run"] for call in runner.calls)


def test_auth_failure_halts_execution_before_git_changes(tmp_path: Path):
    runner = FakeRunner({
        ("gh", "auth", "status"): subprocess.CompletedProcess(
            ["gh", "auth", "status"], 1, stdout="", stderr="not logged in"
        )
    })
    generator = _generator(tmp_path, runner)
    with pytest.raises(RuntimeError, match="authentication required"):
        generator.generate(build_training_scenarios(1, 0))
    assert not any(call[:2] == ["git", "checkout"] for call in runner.calls)
    assert not any(call[:3] == ["gh", "workflow", "run"] for call in runner.calls)


def test_command_wrappers_capture_text_when_they_inspect_output(tmp_path: Path):
    class RecordingRunner:
        def __init__(self):
            self.calls = []

        def run(self, command, **kwargs):
            self.calls.append((list(command), kwargs))
            stdout = "" if command[:3] == ["git", "status", "--porcelain"] else "main\n"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    runner = RecordingRunner()
    GitWorkspace(tmp_path, runner=runner).ensure_clean_base()
    GhCliClient("owner/repo", runner=runner).ensure_authenticated()
    for _, kwargs in runner.calls:
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True


def test_run_discovery_retries_transient_error_and_empty_result(monkeypatch):
    class DelayedDiscoveryRunner:
        def __init__(self):
            self.attempts = 0

        def run(self, command, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.CalledProcessError(1, command)
            if self.attempts == 2:
                return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout='[{"databaseId": 456}]', stderr="")

    runner = DelayedDiscoveryRunner()
    monkeypatch.setattr("src.prediction.training_data_generator.time.sleep", lambda _: None)
    run_id = GhCliClient("owner/repo", runner=runner).find_latest_run_id(
        "ci.yml", "ml-data/success-011", max_attempts=3, retry_delay_seconds=0
    )
    assert run_id == 456
    assert runner.attempts == 3


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled", "skipped", "timed_out"])
def test_dispatch_and_poll_records_actual_github_conclusion(tmp_path: Path, conclusion: str):
    runner = LiveWorkflowRunner(conclusion)
    generator = _generator(tmp_path, runner)
    progress = generator.generate(build_training_scenarios(1, 0))
    assert progress.completed == 1
    assert progress.records[0].expected_outcome == "success"
    assert progress.records[0].actual_conclusion == conclusion
    assert any(call[:3] == ["gh", "workflow", "run"] for call in runner.calls)
    assert any(call[:3] == ["gh", "run", "view"] for call in runner.calls)
    assert ["git", "push", "--force-with-lease", "-u", "origin", "ml-data/success-001"] in runner.calls


def test_completed_scenarios_are_skipped_when_resuming(tmp_path: Path):
    runner = LiveWorkflowRunner()
    store = GeneratedRunStore(tmp_path / "generated_runs.json")
    store.save([
        GeneratedRunRecord(
            run_id=123, workflow="CI/CD Pipeline", branch="ml-data/success-001", commit_sha="abc",
            expected_category=None, expected_outcome="success", actual_conclusion="success",
            scenario_id="success-001", created_at="2026-01-01T00:00:00Z",
        )
    ])
    progress = _generator(tmp_path, runner).generate(build_training_scenarios(1, 0))
    assert progress.completed == 0
    assert not any(call[:3] == ["gh", "workflow", "run"] for call in runner.calls)


def test_cleanup_targets_only_generator_branch_records(tmp_path: Path):
    runner = LiveWorkflowRunner()
    store = GeneratedRunStore(tmp_path / "generated_runs.json")
    records = [
        GeneratedRunRecord(1, "CI", "ml-data/success-001", "a", None, "success", "success", "one", "now"),
        GeneratedRunRecord(2, "CI", "main", "b", None, "success", "success", "two", "now"),
        GeneratedRunRecord(3, "CI", "feature/developer-work", "c", None, "success", "success", "three", "now"),
    ]
    store.save(records)
    _generator(tmp_path, runner).generate([], cleanup=True)
    delete_calls = [call for call in runner.calls if call[:3] == ["gh", "api", "-X"]]
    assert len(delete_calls) == 1
    assert delete_calls[0][-1].endswith("heads/ml-data/success-001")


def test_rate_limit_stops_gracefully_and_preserves_prior_progress(tmp_path: Path):
    runner = LiveWorkflowRunner(responses={
        ("gh", "workflow", "run", "ci.yml", "--repo", "owner/repo", "--ref", "ml-data/success-001"):
            subprocess.CompletedProcess([], 1, stdout="API rate limit exceeded", stderr="")
    })
    progress = _generator(tmp_path, runner).generate(build_training_scenarios(1, 0))
    assert progress.stopped_early is True
    assert "rate limit" in (progress.stop_reason or "").lower()
    assert (tmp_path / "generated_runs.json").exists()


def test_original_branch_is_restored_when_execution_errors(tmp_path: Path):
    class DispatchFailureRunner(LiveWorkflowRunner):
        def run(self, command, **kwargs):
            if command[:3] == ["gh", "workflow", "run"]:
                self.calls.append(list(command))
                raise RuntimeError("simulated dispatch failure")
            return super().run(command, **kwargs)

    runner = DispatchFailureRunner()
    with pytest.raises(RuntimeError, match="simulated dispatch failure"):
        _generator(tmp_path, runner).generate(build_training_scenarios(1, 0))
    checkout_calls = [call for call in runner.calls if call[:2] == ["git", "checkout"]]
    assert checkout_calls[-1] == ["git", "checkout", "main"]


def test_check_environment_reports_missing_configuration_without_mutation(tmp_path: Path):
    class DiagnosticRunner:
        def __init__(self):
            self.calls = []

        def run(self, command, **kwargs):
            self.calls.append(command)
            if command == ["git", "config", "user.name"]:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            if command == ["git", "config", "user.email"]:
                return subprocess.CompletedProcess(command, 0, stdout="dev@example.test\n", stderr="")
            if command == ["gh", "--version"]:
                return subprocess.CompletedProcess(command, 0, stdout="gh version test", stderr="")
            if command == ["gh", "auth", "status"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="not logged in")
            if command == ["git", "status", "--porcelain"]:
                return subprocess.CompletedProcess(command, 0, stdout=" M app.py\n", stderr="")
            raise AssertionError(command)

    module = _load_generation_script()
    runner = DiagnosticRunner()
    failures = module.check_environment(tmp_path, runner)
    assert "git user.name: not configured" in failures
    assert "GitHub CLI authentication: check failed" in failures
    assert "clean working tree: uncommitted changes detected" in failures
    assert runner.calls == [
        ["git", "config", "user.name"], ["git", "config", "user.email"], ["gh", "--version"],
        ["gh", "auth", "status"], ["git", "status", "--porcelain"],
    ]


def test_summarize_generation_reports_outcomes():
    records = [
        GeneratedRunRecord(
            run_id=1,
            workflow="CI/CD Pipeline",
            branch="ml-data/success-001",
            commit_sha="abc",
            expected_category=None,
            expected_outcome="success",
            actual_conclusion="success",
            scenario_id="success-001",
            created_at="2026-01-01T00:00:00Z",
        ),
        GeneratedRunRecord(
            run_id=2,
            workflow="Controlled CI Failure Generator (Dev/Testing Only)",
            branch="ml-data/failure-lint-001",
            commit_sha="def",
            expected_category="lint",
            expected_outcome="failure",
            actual_conclusion="failure",
            scenario_id="failure-lint-001",
            created_at="2026-01-01T00:00:00Z",
        ),
    ]
    progress = MagicMock(requested=2, completed=2, records=records, stopped_early=False, stop_reason=None)
    summary = summarize_generation(progress)
    assert "Success: 1" in summary
    assert "Failure: 1" in summary
    assert "lint: 1" in summary


def test_generate_training_data_script_dry_run_never_dispatches(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/generate_training_data.py"),
            "--dry-run",
            "--runs",
            "30",
            "--success-runs",
            "15",
            "--failure-runs",
            "15",
            "--seed",
            "42",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Training Data Generation Plan" in result.stdout
    assert "NO GITHUB ACTIONS WERE DISPATCHED." in result.stdout
    assert "lint: 3" in result.stdout


def test_generate_training_data_script_phase_one_blocks_execution():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/generate_training_data.py"),
            "--runs",
            "2",
            "--success-runs",
            "1",
            "--failure-runs",
            "1",
            "--no-write-plan",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NO GITHUB ACTIONS WERE DISPATCHED." in result.stdout


def test_generate_training_data_script_rejects_over_limit_without_flag():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/generate_training_data.py"),
            "--dry-run",
            "--runs",
            "51",
            "--success-runs",
            "26",
            "--failure-runs",
            "25",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "50" in result.stderr or "50" in result.stdout
