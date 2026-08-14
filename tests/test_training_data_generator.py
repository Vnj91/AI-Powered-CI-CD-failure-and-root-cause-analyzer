from __future__ import annotations

import json
import subprocess
import sys
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
        if command[:2] == ["git", "checkout"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "add"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["git", "commit"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc1234567890\n", stderr="")
        if command[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "workflow", "run"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")


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
