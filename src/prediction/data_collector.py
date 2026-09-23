"""Historical GitHub Actions data collection for model training."""

from __future__ import annotations

import io
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional
import zipfile

import pandas as pd
import requests
from github import Auth, Github, GithubException
from github.WorkflowRun import WorkflowRun

from ..constants import GITHUB_ACCESS_TOKEN
from ..tools.log_parser import parse_log_content
from .category_mapper import map_failure_category
from .feature_extractor import FailureFeatureExtractor
from .schemas import WorkflowRunRecord


FAILURE_CONCLUSIONS = {
    "failure",
    "timed_out",
    "startup_failure",
    "action_required",
}
SUCCESS_CONCLUSIONS = {"success"}
SYSTEM_WORKFLOW_NAMES = {
    "pre-ci failure prediction",
    "prediction feedback",
    "train failure predictor",
    "controlled ci failure generator (dev/testing only)",
}


@dataclass
class _HistoryState:
    """Running history used to compute leakage-safe features."""

    recent_outcomes: deque[int]
    branch_outcomes: dict[str, deque[int]]
    workflow_outcomes: dict[str, deque[int]]
    branch_failures: defaultdict[str, int]
    workflow_failures: defaultdict[str, int]


class HistoricalRunCollector:
    """Collect workflow runs and flatten them into a trainable dataset."""

    def __init__(self, token: Optional[str] = None, recent_window: int = 10):
        # If a token argument was explicitly passed (even an empty string), respect it.
        # An explicit empty string means 'no token' (anonymous client) for tests
        # and for deployments that want credential-free reads.
        if token is not None:
            self.token = token or None
        else:
            self.token = GITHUB_ACCESS_TOKEN or None
        self.recent_window = recent_window
        self.github = Github(auth=Auth.Token(self.token)) if self.token else Github()
        self._file_feature_cache: dict[str, dict[str, object]] = {}

    def _repo(self, repository: str):
        try:
            return self.github.get_repo(repository)
        except GithubException as exc:
            raise ValueError(f"Could not access repository '{repository}': {exc}") from exc

    @staticmethod
    def _is_failure(conclusion: Optional[str]) -> int:
        if not conclusion:
            return 0
        return int(conclusion.lower() in FAILURE_CONCLUSIONS)

    @staticmethod
    def _workflow_name(run: WorkflowRun) -> str:
        return str(
            getattr(getattr(run, "workflow", None), "name", None)
            or getattr(run, "name", None)
            or "unknown"
        )

    @classmethod
    def _is_trainable_run(cls, run: WorkflowRun, *, include_system_workflows: bool = False) -> bool:
        """Keep only definitive target outcomes used by the risk model."""

        conclusion = str(getattr(run, "conclusion", None) or "").lower()
        if conclusion not in FAILURE_CONCLUSIONS | SUCCESS_CONCLUSIONS:
            return False
        if include_system_workflows:
            return True
        return cls._workflow_name(run).strip().lower() not in SYSTEM_WORKFLOW_NAMES

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _sort_runs_by_time(runs: list[WorkflowRun]) -> list[WorkflowRun]:
        return sorted(
            runs,
            key=lambda run: (
                getattr(run, "created_at", datetime.min) or datetime.min,
                HistoricalRunCollector._safe_int(getattr(run, "id", 0)),
            ),
        )

    def _compute_file_features(self, repo, run: WorkflowRun) -> dict[str, object]:
        commit_sha = str(getattr(run, "head_sha", "") or "")
        if commit_sha and commit_sha in self._file_feature_cache:
            return dict(self._file_feature_cache[commit_sha])

        changed_files: list[str] = []
        files_changed = 0
        lines_added = 0
        lines_deleted = 0
        number_of_commits = 0

        try:
            commit = repo.get_commit(run.head_sha)
            files = getattr(commit, "files", []) or []
            changed_files = [file.filename for file in files if getattr(file, "filename", None)]
            files_changed = len(changed_files)
            lines_added = self._safe_int(getattr(commit.stats, "additions", 0))
            lines_deleted = self._safe_int(getattr(commit.stats, "deletions", 0))

            # A workflow run is tied to one head commit. The commit endpoint
            # already returns its changed files and stats, so a second compare
            # request adds cost without changing this per-run feature.
            number_of_commits = 1
        except Exception:
            pass

        lowered = [path.lower() for path in changed_files]

        dependency_files_changed = any(
            Path(path).name in FailureFeatureExtractor.DEPENDENCY_FILES for path in changed_files
        )
        ci_workflow_files_changed = any(
            any(hint in path for hint in FailureFeatureExtractor.CI_PATH_HINTS) for path in lowered
        )
        docker_files_changed = any("dockerfile" in Path(path).name.lower() or "dockerfile" in path for path in lowered)
        test_files_changed = any(any(hint in path for hint in FailureFeatureExtractor.TEST_PATH_HINTS) for path in lowered)
        infrastructure_files_changed = any(
            any(hint in path for hint in FailureFeatureExtractor.INFRA_PATH_HINTS) for path in lowered
        )

        changed_extensions = defaultdict(int)
        for path in changed_files:
            extension = FailureFeatureExtractor._extension_for_path(path)
            changed_extensions[extension] += 1

        features = {
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "number_of_commits": number_of_commits,
            "changed_files_json": json.dumps(changed_files),
            "changed_extensions_json": json.dumps(dict(changed_extensions)),
            "dependency_files_changed": dependency_files_changed,
            "ci_workflow_files_changed": ci_workflow_files_changed,
            "docker_files_changed": docker_files_changed,
            "test_files_changed": test_files_changed,
            "infrastructure_files_changed": infrastructure_files_changed,
        }
        if commit_sha:
            self._file_feature_cache[commit_sha] = dict(features)
        return features

    def _compute_history_features(self, state: _HistoryState, branch: str, workflow: str) -> dict[str, object]:
        recent_failure_count = sum(state.recent_outcomes)
        recent_failure_rate = recent_failure_count / len(state.recent_outcomes) if state.recent_outcomes else 0.0

        branch_history = state.branch_outcomes[branch]
        workflow_history = state.workflow_outcomes[workflow]

        previous_run_status = "success"
        if branch_history:
            previous_run_status = "failure" if branch_history[-1] == 1 else "success"

        return {
            "previous_run_status": previous_run_status if branch_history else "unknown",
            "previous_failure_count": int(sum(state.recent_outcomes)),
            "recent_failure_count": int(recent_failure_count),
            "recent_failure_rate": float(recent_failure_rate),
            "previous_failures_same_workflow": int(state.workflow_failures[workflow]),
            "previous_failures_same_branch": int(state.branch_failures[branch]),
            "similar_previous_failures": int(state.workflow_failures[workflow] + state.branch_failures[branch]),
        }

    @staticmethod
    def _duration_seconds(run: WorkflowRun) -> Optional[float]:
        started = getattr(run, "created_at", None)
        updated = getattr(run, "updated_at", None)
        if not started or not updated:
            return None
        try:
            delta = updated - started
            return max(delta.total_seconds(), 0.0)
        except Exception:
            return None

    def _extract_actual_category(self, repo, run: WorkflowRun) -> Optional[str]:
        """Best-effort label extraction from failed workflow logs."""

        if not self._is_failure(getattr(run, "conclusion", None)):
            return None
        if not self.token:
            # Public metadata collection works anonymously, but Actions log
            # archives require authenticated access. Category labels remain
            # optional for the binary failure model.
            return None

        logs_url = getattr(run, "logs_url", None)
        if not logs_url:
            return None

        try:
            response = requests.get(logs_url, headers={"Authorization": f"token {self.token}"}, timeout=60)
            if response.status_code != 200:
                return None

            combined_logs: list[str] = []
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    for filename in sorted(zip_file.namelist()):
                        with zip_file.open(filename) as log_file:
                            content = log_file.read().decode("utf-8", errors="replace")
                            combined_logs.append(content)
            except zipfile.BadZipFile:
                combined_logs.append(response.text)

            parsed = parse_log_content("\n".join(combined_logs))
            if parsed.primary_error and parsed.primary_error.error_category:
                mapped = map_failure_category(
                    log_parser_category=parsed.primary_error.error_category.value,
                    error_type=parsed.primary_error.error_type,
                    error_message=parsed.primary_error.error_message,
                    failed_step=parsed.primary_error.failed_step,
                )
                return mapped.value
        except Exception:
            return None

        return None

    def collect_repository_runs(
        self,
        repository: str,
        limit: Optional[int] = None,
        status: str = "completed",
        include_system_workflows: bool = False,
    ) -> pd.DataFrame:
        """Collect historical workflow runs as a leakage-safe dataset."""

        repo = self._repo(repository)
        runs = [
            run
            for run in self._sort_runs_by_time(list(repo.get_workflow_runs(status=status)))
            if self._is_trainable_run(run, include_system_workflows=include_system_workflows)
        ]

        if limit is not None:
            runs = runs[-limit:]

        history = _HistoryState(
            recent_outcomes=deque(maxlen=self.recent_window),
            branch_outcomes=defaultdict(lambda: deque(maxlen=self.recent_window)),
            workflow_outcomes=defaultdict(lambda: deque(maxlen=self.recent_window)),
            branch_failures=defaultdict(int),
            workflow_failures=defaultdict(int),
        )

        records: list[dict[str, object]] = []

        for run in runs:
            branch = str(getattr(run, "head_branch", "unknown") or "unknown")
            workflow_name = self._workflow_name(run)

            file_features = self._compute_file_features(repo, run)
            history_features = self._compute_history_features(history, branch, workflow_name)
            actual_failure = self._is_failure(getattr(run, "conclusion", None))
            actual_category = self._extract_actual_category(repo, run)

            record = WorkflowRunRecord(
                repository=repository,
                workflow_name=workflow_name,
                workflow_id=self._safe_int(getattr(run, "workflow_id", None)),
                run_id=self._safe_int(getattr(run, "id", None)),
                run_number=self._safe_int(getattr(run, "run_number", None)),
                branch=branch,
                commit_sha=str(getattr(run, "head_sha", None) or ""),
                timestamp=getattr(run, "created_at", datetime.now(UTC)) or datetime.now(UTC),
                status=str(getattr(run, "status", None) or ""),
                conclusion=str(getattr(run, "conclusion", None) or ""),
                duration_seconds=self._duration_seconds(run),
                event=str(getattr(run, "event", None) or ""),
                actor=str(getattr(getattr(run, "actor", None), "login", None) or ""),
                commit_message=str(getattr(getattr(run, "head_commit", None), "message", None) or ""),
                actual_failure=actual_failure,
                actual_category=actual_category,
                **file_features,
                **history_features,
            )

            records.append(record.model_dump())

            history.recent_outcomes.append(actual_failure)
            history.branch_outcomes[branch].append(actual_failure)
            history.workflow_outcomes[workflow_name].append(actual_failure)
            if actual_failure:
                history.branch_failures[branch] += 1
                history.workflow_failures[workflow_name] += 1

        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            # Ensure one row per workflow run id: when duplicate run_id values
            # appear (e.g., due to repeated API reads or inconsistent repo
            # behaviors), keep the most recent record by timestamp so the
            # collector output is idempotent and deterministic.
            if "run_id" in frame.columns:
                frame = (
                    frame.sort_values(["timestamp", "run_id"], kind="mergesort")
                    .drop_duplicates(subset=["run_id"], keep="last")
                    .sort_values("timestamp", kind="mergesort")
                    .reset_index(drop=True)
                )
        return frame

    def build_prediction_features(
        self,
        repository: str,
        commit_sha: str,
        branch: Optional[str] = None,
        workflow_name: Optional[str] = None,
    ) -> dict[str, object]:
        """Build a leakage-safe feature row for a commit that has not finished yet."""

        repo = self._repo(repository)
        try:
            commit = repo.get_commit(commit_sha)
        except GithubException as exc:
            raise ValueError(f"Could not load commit '{commit_sha}' for '{repository}': {exc}") from exc

        branch_name = branch or "unknown"
        workflow_filter = workflow_name or "unknown"

        recent_runs = [
            run
            for run in self._sort_runs_by_time(list(repo.get_workflow_runs(status="completed")))
            if self._is_trainable_run(run)
        ]

        history = _HistoryState(
            recent_outcomes=deque(maxlen=self.recent_window),
            branch_outcomes=defaultdict(lambda: deque(maxlen=self.recent_window)),
            workflow_outcomes=defaultdict(lambda: deque(maxlen=self.recent_window)),
            branch_failures=defaultdict(int),
            workflow_failures=defaultdict(int),
        )

        for run in recent_runs:
            run_branch = str(getattr(run, "head_branch", "unknown") or "unknown")
            run_workflow = self._workflow_name(run)
            if getattr(run, "head_sha", None) == commit_sha:
                break

            actual_failure = self._is_failure(getattr(run, "conclusion", None))
            history.recent_outcomes.append(actual_failure)
            history.branch_outcomes[run_branch].append(actual_failure)
            history.workflow_outcomes[run_workflow].append(actual_failure)
            if actual_failure:
                history.branch_failures[run_branch] += 1
                history.workflow_failures[run_workflow] += 1

        class _CommitRun:
            head_sha = commit_sha
            head_branch = branch_name

        file_features = self._compute_file_features(repo, _CommitRun())
        history_features = self._compute_history_features(history, branch_name, workflow_filter)

        return {
            "repository": repository,
            "workflow_name": workflow_filter,
            "workflow_id": None,
            "run_id": None,
            "run_number": None,
            "branch": branch_name,
            "commit_sha": commit_sha,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "in_progress",
            "conclusion": None,
            "duration_seconds": None,
            "event": "prediction",
            "actor": getattr(getattr(commit, "author", None), "login", None) or "unknown",
            "commit_message": getattr(getattr(commit, "commit", None), "message", None) or "",
            **file_features,
            **history_features,
            "actual_failure": 0,
            "actual_category": None,
        }

    def collect_to_csv(
        self,
        repository: str,
        output_path: str | Path,
        limit: Optional[int] = None,
        status: str = "completed",
        include_system_workflows: bool = False,
    ) -> Path:
        """Collect a repository's workflow history into a CSV dataset."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = self.collect_repository_runs(
            repository=repository,
            limit=limit,
            status=status,
            include_system_workflows=include_system_workflows,
        )
        frame.to_csv(output_path, index=False)
        return output_path
