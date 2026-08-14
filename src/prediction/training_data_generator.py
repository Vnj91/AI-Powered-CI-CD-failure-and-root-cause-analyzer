"""Orchestrate real GitHub Actions runs for ML training-data generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from config import Config
from src.prediction.dataset_validator import validate_dataset
from src.prediction.scenario_planner import (
    DeterministicScenarioPlanner,
    TrainingScenario,
    summarize_scenario_distribution,
    validate_run_count,
)


class CommandRunner(Protocol):
    def run(
        self,
        command: list[str],
        *,
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
        cwd: Optional[str | Path] = None,
    ) -> subprocess.CompletedProcess[str]:
        ...


@dataclass
class GeneratedRunRecord:
    run_id: Optional[int]
    workflow: str
    branch: str
    commit_sha: Optional[str]
    expected_category: Optional[str]
    expected_outcome: str
    actual_conclusion: Optional[str]
    scenario_id: str
    created_at: str
    completed_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationProgress:
    requested: int = 0
    completed: int = 0
    records: list[GeneratedRunRecord] = field(default_factory=list)
    branches_created: list[str] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: Optional[str] = None


class RateLimitError(RuntimeError):
    pass


class GhCliClient:
    """Thin wrapper around GitHub CLI for testability."""

    def __init__(
        self,
        repo: str,
        runner: CommandRunner | None = None,
        dispatch_delay_seconds: float = 8.0,
    ):
        self.repo = repo
        self.runner = runner or subprocess
        self.dispatch_delay_seconds = dispatch_delay_seconds
        self._last_dispatch_at: float = 0.0

    def _run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        full_command = list(command)
        result = self.runner.run(full_command, **kwargs)
        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        if "rate limit" in combined.lower() or "API rate limit exceeded" in combined:
            raise RateLimitError(combined.strip())
        return result

    def ensure_authenticated(self) -> None:
        result = self._run(["gh", "auth", "status"], check=False)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "GitHub CLI is not authenticated.").strip()
            raise RuntimeError(
                "GitHub CLI authentication required. Run `gh auth login` and retry.\n"
                f"{message}"
            )

    def ensure_gh_available(self) -> None:
        result = self._run(["gh", "--version"], check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "GitHub CLI (`gh`) is not available. Install it from https://cli.github.com/ and retry."
            )

    def dispatch_workflow(
        self,
        workflow_file: str,
        ref: str,
        inputs: Optional[dict[str, str]] = None,
    ) -> None:
        elapsed = time.monotonic() - self._last_dispatch_at
        if elapsed < self.dispatch_delay_seconds:
            time.sleep(self.dispatch_delay_seconds - elapsed)

        command = [
            "gh",
            "workflow",
            "run",
            workflow_file,
            "--repo",
            self.repo,
            "--ref",
            ref,
        ]
        if inputs:
            for key, value in inputs.items():
                command.extend(["-f", f"{key}={value}"])
        self._run(command, check=True)
        self._last_dispatch_at = time.monotonic()
        time.sleep(2)

    def find_latest_run_id(self, workflow_file: str, branch: str) -> int:
        result = self._run(
            [
                "gh",
                "run",
                "list",
                "--repo",
                self.repo,
                "--workflow",
                workflow_file,
                "--branch",
                branch,
                "--limit",
                "1",
                "--json",
                "databaseId,headSha,createdAt",
            ],
            check=True,
        )
        payload = json.loads(result.stdout or "[]")
        if not payload:
            raise RuntimeError(f"No workflow run found for {workflow_file} on {branch}")
        return int(payload[0]["databaseId"])

    def wait_for_run(self, run_id: int, poll_interval: int) -> dict[str, Any]:
        while True:
            result = self._run(
                [
                    "gh",
                    "run",
                    "view",
                    str(run_id),
                    "--repo",
                    self.repo,
                    "--json",
                    "status,conclusion,headSha,workflowName,createdAt,updatedAt,url",
                ],
                check=True,
            )
            payload = json.loads(result.stdout or "{}")
            status = payload.get("status")
            if status == "completed":
                return payload
            time.sleep(poll_interval)

    def delete_branch(self, branch: str) -> None:
        owner, name = self.repo.split("/", 1)
        ref = f"heads/{branch}"
        self._run(
            [
                "gh",
                "api",
                "-X",
                "DELETE",
                f"repos/{owner}/{name}/git/refs/{ref}",
            ],
            check=False,
        )


class GitWorkspace:
    """Apply scenario file changes locally and push isolated branches."""

    def __init__(self, repo_root: Path, runner: CommandRunner | None = None):
        self.repo_root = repo_root
        self.runner = runner or subprocess

    def _run(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.runner.run(command, cwd=str(self.repo_root), **kwargs)

    def ensure_clean_base(self, base_branch: str = "main") -> None:
        self._run(["git", "fetch", "origin", base_branch], check=True)
        status = self._run(["git", "status", "--porcelain"], check=True)
        if status.stdout.strip():
            raise RuntimeError(
                "Working tree is not clean. Commit or stash local changes before generating training data."
            )

    def create_branch(self, branch: str, base_branch: str = "main") -> None:
        self._run(["git", "checkout", "-B", branch, f"origin/{base_branch}"], check=True)

    def apply_changes(self, scenario: TrainingScenario) -> str:
        for change in scenario.file_changes:
            path = self.repo_root / change.path
            path.parent.mkdir(parents=True, exist_ok=True)
            if change.mode == "append" and path.exists():
                existing = path.read_text(encoding="utf-8")
                if change.content.strip() not in existing:
                    path.write_text(existing + change.content, encoding="utf-8")
            else:
                path.write_text(change.content, encoding="utf-8")

        paths = [change.path for change in scenario.file_changes]
        self._run(["git", "add", *paths], check=True)
        commit = self._run(
            ["git", "commit", "-m", scenario.commit_message],
            check=True,
        )
        rev = self._run(["git", "rev-parse", "HEAD"], check=True)
        sha = rev.stdout.strip()
        push = self._run(["git", "push", "-u", "origin", scenario.branch_name], check=True)
        _ = commit, push
        return sha

    def return_to_base(self, base_branch: str = "main") -> None:
        self._run(["git", "checkout", base_branch], check=False)


class GeneratedRunStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[GeneratedRunRecord]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [GeneratedRunRecord(**item) for item in payload.get("runs", [])]

    def save(self, records: list[GeneratedRunRecord]) -> None:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": [record.to_dict() for record in records],
        }
        text = json.dumps(payload, indent=2)
        if "ghp_" in text or "github_pat_" in text:
            raise RuntimeError("Refusing to write token-like content to generated run metadata.")
        self.path.write_text(text, encoding="utf-8")


class TrainingDataGenerator:
    CI_WORKFLOW_FILE = "ci.yml"
    FAILURE_WORKFLOW_FILE = "test-failure.yml"

    def __init__(
        self,
        repo: str,
        repo_root: Path,
        gh_client: GhCliClient,
        git_workspace: GitWorkspace,
        store: GeneratedRunStore,
        poll_interval: int = 10,
    ):
        self.repo = repo
        self.repo_root = repo_root
        self.gh = gh_client
        self.git = git_workspace
        self.store = store
        self.poll_interval = poll_interval

    def execute_scenario(self, scenario: TrainingScenario, *, dry_run: bool = False) -> GeneratedRunRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        if dry_run:
            return GeneratedRunRecord(
                run_id=None,
                workflow=scenario.workflow_name,
                branch=scenario.branch_name,
                commit_sha=None,
                expected_category=scenario.category,
                expected_outcome=scenario.expected_outcome,
                actual_conclusion=None,
                scenario_id=scenario.scenario_id,
                created_at=created_at,
            )

        self.git.create_branch(scenario.branch_name)
        commit_sha = self.git.apply_changes(scenario)
        self.gh.dispatch_workflow(
            scenario.workflow_file,
            ref=scenario.branch_name,
            inputs=scenario.workflow_inputs or None,
        )
        run_id = self.gh.find_latest_run_id(scenario.workflow_file, scenario.branch_name)
        run_payload = self.gh.wait_for_run(run_id, self.poll_interval)
        actual_conclusion = run_payload.get("conclusion")
        completed_at = run_payload.get("updatedAt")

        return GeneratedRunRecord(
            run_id=run_id,
            workflow=str(run_payload.get("workflowName") or scenario.workflow_name),
            branch=scenario.branch_name,
            commit_sha=str(run_payload.get("headSha") or commit_sha),
            expected_category=scenario.category,
            expected_outcome=scenario.expected_outcome,
            actual_conclusion=actual_conclusion,
            scenario_id=scenario.scenario_id,
            created_at=created_at,
            completed_at=completed_at,
        )

    def generate(
        self,
        scenarios: list[TrainingScenario],
        *,
        dry_run: bool = False,
        cleanup: bool = False,
    ) -> GenerationProgress:
        progress = GenerationProgress(requested=len(scenarios))
        existing = self.store.load()
        records = list(existing)

        if not dry_run:
            self.gh.ensure_gh_available()
            self.gh.ensure_authenticated()
            self.git.ensure_clean_base()

        try:
            for scenario in scenarios:
                try:
                    record = self.execute_scenario(scenario, dry_run=dry_run)
                except RateLimitError as exc:
                    progress.stopped_early = True
                    progress.stop_reason = str(exc)
                    break

                records.append(record)
                progress.records.append(record)
                progress.completed += 1
                if not dry_run:
                    progress.branches_created.append(scenario.branch_name)

            if not dry_run:
                self.store.save(records)
                self.git.return_to_base()
                if cleanup:
                    for branch in progress.branches_created:
                        self.gh.delete_branch(branch)
        except Exception:
            if not dry_run:
                self.store.save(records)
                self.git.return_to_base()
            raise

        progress.records = records[-len(scenarios) :] if dry_run else progress.records
        return progress

    def collect_dataset(self, output_path: Path) -> Path:
        if not os.environ.get("GITHUB_ACCESS_TOKEN"):
            token_result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                check=False,
            )
            if token_result.returncode == 0 and token_result.stdout.strip():
                os.environ["GITHUB_ACCESS_TOKEN"] = token_result.stdout.strip()
            else:
                raise RuntimeError(
                    "GITHUB_ACCESS_TOKEN is not set and `gh auth token` is unavailable. "
                    "Authenticate with `gh auth login` or export GITHUB_ACCESS_TOKEN before --collect."
                )

        command = [
            sys.executable,
            "-m",
            "src.prediction.cli",
            "collect",
            self.repo,
            "--output",
            str(output_path),
        ]
        subprocess.run(command, cwd=str(self.repo_root), check=True)
        return output_path

    def inspect_dataset(self, dataset_path: Path):
        return validate_dataset(dataset_path)

    def train_if_ready(self, dataset_path: Path) -> int:
        report = self.inspect_dataset(dataset_path)
        if not report.sufficient_for_training:
            print("Training refused — insufficient real historical data.")
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 1

        train_cmd = [
            sys.executable,
            "-m",
            "src.prediction.cli",
            "train",
            "--dataset",
            str(dataset_path),
        ]
        train = subprocess.run(train_cmd, cwd=str(self.repo_root))
        if train.returncode != 0:
            return train.returncode

        eval_cmd = [
            sys.executable,
            "-m",
            "src.prediction.cli",
            "evaluate",
            "--dataset",
            str(dataset_path),
        ]
        evaluate = subprocess.run(eval_cmd, cwd=str(self.repo_root))
        return evaluate.returncode


def summarize_generation(
    progress: GenerationProgress,
    dataset_report: Optional[Any] = None,
) -> str:
    records = progress.records
    success = sum(1 for record in records if record.actual_conclusion == "success")
    failure = sum(1 for record in records if record.actual_conclusion == "failure")
    cancelled = sum(1 for record in records if record.actual_conclusion == "cancelled")
    skipped = sum(1 for record in records if record.actual_conclusion == "skipped")
    pending = sum(1 for record in records if record.actual_conclusion in (None, ""))

    category_counts: dict[str, int] = {}
    for record in records:
        if record.expected_outcome == "failure" and record.expected_category:
            key = record.expected_category
            if record.actual_conclusion == "failure":
                category_counts[key] = category_counts.get(key, 0) + 1

    lines = [
        "Generated runs:",
        f"  Requested: {progress.requested}",
        f"  Completed: {progress.completed}",
        "",
        "Outcomes:",
        f"  Success: {success}",
        f"  Failure: {failure}",
        f"  Cancelled: {cancelled}",
        f"  Skipped: {skipped}",
    ]
    if pending:
        lines.append(f"  Pending/unknown: {pending}")
    if progress.stopped_early:
        lines.extend(["", f"Stopped early: {progress.stop_reason}"])
    if category_counts:
        lines.extend(["", "Failure categories (actual failures):"])
        for key in sorted(category_counts):
            lines.append(f"  {key}: {category_counts[key]}")

    if dataset_report is not None:
        lines.extend(
            [
                "",
                "Dataset:",
                f"  Total historical runs: {dataset_report.total_runs}",
                f"  Success: {dataset_report.success_runs}",
                f"  Failure: {dataset_report.failure_runs}",
                "",
                "Training readiness:",
                f"  >=20 runs: {'YES' if dataset_report.total_runs >= 20 else 'NO'}",
                f"  >=2 outcome classes: {'YES' if dataset_report.success_runs > 0 and dataset_report.failure_runs > 0 else 'NO'}",
                f"  Sufficient for training: {'YES' if dataset_report.sufficient_for_training else 'NO'}",
            ]
        )
    return "\n".join(lines)


def build_plan(
    success_runs: int,
    failure_runs: int,
    *,
    seed: Optional[int] = None,
    allow_large_run_set: bool = False,
) -> tuple[list[TrainingScenario], list[str], dict[str, Any]]:
    total = success_runs + failure_runs
    warnings = validate_run_count(total, allow_large_run_set=allow_large_run_set)
    planner = DeterministicScenarioPlanner(seed=seed)
    scenarios = planner.plan(success_runs, failure_runs)
    distribution = summarize_scenario_distribution(scenarios)
    return scenarios, warnings, distribution


def format_execution_plan(
    *,
    total_runs: int,
    success_runs: int,
    failure_runs: int,
    distribution: dict[str, Any],
    dry_run: bool,
    seed: Optional[int] = None,
) -> str:
    lines = [
        "Training Data Generation Plan",
        "",
        f"Total requested: {total_runs}",
        f"Success: {success_runs}",
        f"Failure: {failure_runs}",
    ]
    if seed is not None:
        lines.append(f"Seed: {seed}")
    lines.extend(
        [
            "",
            "Failures:",
        ]
    )
    for category, count in distribution["failures_by_category"].items():
        if count:
            lines.append(f"  {category}: {count}")
    lines.extend(["", "Success scenarios:"])
    for change_type, count in distribution["success_by_change_type"].items():
        if count:
            label = change_type
            lines.append(f"  {label}: {count}")
    lines.append("")
    if dry_run:
        lines.extend(
            [
                "NO GITHUB ACTIONS WERE DISPATCHED.",
                "This is a dry-run planning phase.",
            ]
        )
    else:
        lines.append("NO GITHUB ACTIONS WERE DISPATCHED.")
        lines.append("Phase 1 supports planning only — workflow execution is not enabled yet.")
    return "\n".join(lines)


def write_run_plan(
    path: Path,
    *,
    scenarios: list[TrainingScenario],
    distribution: dict[str, Any],
    seed: Optional[int] = None,
) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "distribution": distribution,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    text = json.dumps(payload, indent=2)
    if "ghp_" in text or "github_pat_" in text:
        raise RuntimeError("Refusing to write token-like content to generated run plan.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
