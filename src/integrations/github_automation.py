"""GitHub-backed automation for repository changes, Actions, and predictions.

The service deliberately keeps authentication credentials out of all returned
models, logs, prediction history, and filesystem artifacts.  A token supplied
by the caller is held only for the lifetime of this in-memory service instance.
Public repository metadata continues to work through GitHub's anonymous API.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from collections.abc import Callable, Iterable
from datetime import datetime
from itertools import islice
from typing import Any, Optional, TYPE_CHECKING
from pathlib import Path
import logging

import requests
from github import Auth, Github, GithubException
from pydantic import BaseModel, Field

from config import Config
from src.prediction.data_collector import HistoricalRunCollector, SYSTEM_WORKFLOW_NAMES
from src.prediction.feature_extractor import FailureFeatureExtractor
from src.prediction.schemas import FailurePrediction
from src.utils.redaction import redact_sensitive_text

if TYPE_CHECKING:
    from src.prediction.service import FailurePredictionService


DEFAULT_LOG_LIMIT_BYTES = 2 * 1024 * 1024
DEFAULT_PATCH_LIMIT = 4_000
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubAutomationError(RuntimeError):
    """Safe, user-facing failure from the GitHub automation boundary."""


class GitHubAuthenticationRequired(GitHubAutomationError):
    """Raised when GitHub does not permit an operation anonymously."""


class GitHubChangedFile(BaseModel):
    """A real file change reported by GitHub's commit API."""

    filename: str
    status: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    previous_filename: Optional[str] = None
    blob_url: Optional[str] = None
    raw_url: Optional[str] = None
    patch: Optional[str] = None


class GitHubCommitSnapshot(BaseModel):
    """Commit metadata and its changed files."""

    sha: str
    short_sha: str
    message: str = ""
    author_login: Optional[str] = None
    author_name: Optional[str] = None
    committed_at: Optional[datetime] = None
    html_url: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    total_changes: int = 0
    files: list[GitHubChangedFile] = Field(default_factory=list)

    @property
    def changed_files(self) -> list[str]:
        """Return filenames in a form convenient for charts and tables."""

        return [item.filename for item in self.files]


class GitHubRepositorySummary(BaseModel):
    """Repository identity safe to expose in the dashboard."""

    full_name: str
    owner_login: Optional[str] = None
    private: bool = False
    visibility: Optional[str] = None
    default_branch: str = "main"
    html_url: Optional[str] = None
    description: Optional[str] = None
    updated_at: Optional[datetime] = None
    permissions: dict[str, bool] = Field(default_factory=dict)


class GitHubConnectionStatus(BaseModel):
    """Authenticated account and selected-repository connection state."""

    connected: bool = False
    authenticated: bool = False
    account_login: Optional[str] = None
    repository: Optional[str] = None
    repository_accessible: bool = False
    private: Optional[bool] = None
    visibility: Optional[str] = None
    default_branch: Optional[str] = None
    html_url: Optional[str] = None
    permissions: dict[str, bool] = Field(default_factory=dict)
    rate_limit_remaining: Optional[int] = None
    rate_limit_total: Optional[int] = None
    rate_limit_reset_at: Optional[datetime] = None
    error: Optional[str] = None


class GitHubWorkflowRunSummary(BaseModel):
    """Dashboard-safe GitHub Actions run metadata."""

    id: int
    run_number: Optional[int] = None
    run_attempt: Optional[int] = None
    workflow_name: str = "unknown"
    display_title: Optional[str] = None
    status: Optional[str] = None
    conclusion: Optional[str] = None
    event: Optional[str] = None
    branch: Optional[str] = None
    head_sha: Optional[str] = None
    actor: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    html_url: Optional[str] = None


class GitHubWorkflowSummary(BaseModel):
    """An actual Actions workflow configured in the selected repository."""

    id: int
    name: str
    path: Optional[str] = None
    state: Optional[str] = None
    html_url: Optional[str] = None


class GitHubWorkflowStepSummary(BaseModel):
    """A job step shown without requiring a raw log upload."""

    number: Optional[int] = None
    name: str = "unknown"
    status: Optional[str] = None
    conclusion: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class GitHubWorkflowJobSummary(BaseModel):
    """GitHub Actions job and step status."""

    id: int
    name: str = "unknown"
    status: Optional[str] = None
    conclusion: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    html_url: Optional[str] = None
    steps: list[GitHubWorkflowStepSummary] = Field(default_factory=list)


class GitHubWorkflowLogFile(BaseModel):
    """One redacted log file from an Actions archive."""

    name: str
    content: str
    size_bytes: int = 0


class GitHubWorkflowLogBundle(BaseModel):
    """Bounded, in-memory workflow logs ready for automatic RCA."""

    repository: str
    run_id: int
    files: list[GitHubWorkflowLogFile] = Field(default_factory=list)
    combined_text: str = ""
    byte_count: int = 0
    redacted: bool = True


class GitHubPredictionInput(BaseModel):
    """Latest real application change transformed into predictor inputs."""

    repository: str
    workflow_name: str
    branch: str
    commit: GitHubCommitSnapshot
    workflow_run_id: Optional[int] = None
    workflow_status: Optional[str] = None
    workflow_conclusion: Optional[str] = None
    raw_features: dict[str, Any] = Field(default_factory=dict)
    model_features: dict[str, Any] = Field(default_factory=dict)


class GitHubAutomatedPrediction(BaseModel):
    """Prediction result tied to the exact GitHub commit evaluated."""

    input: GitHubPredictionInput
    prediction: FailurePrediction
    prediction_id: Optional[str] = None
    mode: str = "pre_ci"
    is_retrospective: bool = False
    prediction_recorded: bool = False
    record_reason: Optional[str] = None


class GitHubFailedRunContext(BaseModel):
    """Latest failed workflow data needed for automatic root-cause analysis."""

    run: Optional[GitHubWorkflowRunSummary] = None
    jobs: list[GitHubWorkflowJobSummary] = Field(default_factory=list)
    jobs_error: Optional[str] = None
    logs: Optional[GitHubWorkflowLogBundle] = None
    logs_error: Optional[str] = None


class GitHubAutomationSnapshot(BaseModel):
    """Repository state for one dashboard refresh."""

    connection: GitHubConnectionStatus
    commits: list[GitHubCommitSnapshot] = Field(default_factory=list)
    workflows: list[GitHubWorkflowSummary] = Field(default_factory=list)
    workflow_runs: list[GitHubWorkflowRunSummary] = Field(default_factory=list)


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_error(value: object) -> str:
    return redact_sensitive_text(value, limit=500)


def _limited(items: Iterable[Any], limit: int) -> list[Any]:
    return list(islice(items, max(int(limit), 0)))


class GitHubAutomationService:
    """Read GitHub state and turn recent changes into prediction/RCA inputs.

    ``token`` is never written to disk or included in any Pydantic model.  When
    omitted, ``GITHUB_ACCESS_TOKEN`` is read at construction time; an absent
    token creates an anonymous client suitable for public repositories.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        repository: Optional[str] = None,
        *,
        github_client: Optional[Github] = None,
        http_get: Callable[..., Any] = requests.get,
        collector_factory: Callable[[Optional[str]], HistoricalRunCollector] = HistoricalRunCollector,
    ):
        configured_token = token if token is not None else os.getenv("GITHUB_ACCESS_TOKEN")
        self.__token = str(configured_token).strip() if configured_token else None
        self.authenticated = bool(self.__token)
        self.github = github_client or (
            Github(auth=Auth.Token(self.__token), retry=0, timeout=20)
            if self.__token
            else Github(retry=0, timeout=20)
        )
        self.repository = self._validate_repository(repository) if repository else None
        self._http_get = http_get
        self._collector_factory = collector_factory

    def __repr__(self) -> str:
        return (
            f"GitHubAutomationService(authenticated={self.authenticated!r}, "
            f"repository={self.repository!r})"
        )

    @staticmethod
    def _validate_repository(repository: str) -> str:
        normalized = str(repository or "").strip()
        if not REPOSITORY_PATTERN.fullmatch(normalized):
            raise GitHubAutomationError("Repository must use the 'owner/repository' format.")
        if not Config.repository_is_allowed(normalized):
            raise GitHubAutomationError(f"Repository '{normalized}' is not in the configured allowlist.")
        return normalized

    def _repository_name(self, repository: Optional[str]) -> str:
        candidate = repository or self.repository
        if not candidate:
            raise GitHubAutomationError("Select a GitHub repository before using this operation.")
        return self._validate_repository(candidate)

    def _repo(self, repository: Optional[str] = None):
        name = self._repository_name(repository)
        try:
            return self.github.get_repo(name)
        except GithubException as exc:
            if getattr(exc, "status", None) in {401, 403}:
                raise GitHubAuthenticationRequired(
                    f"GitHub denied access to '{name}'. Connect an account with repository access."
                ) from exc
            if getattr(exc, "status", None) == 404:
                raise GitHubAutomationError(
                    f"Repository '{name}' was not found or is not visible to the connected account."
                ) from exc
            raise GitHubAutomationError(f"Could not access GitHub repository '{name}': {_safe_error(exc)}") from exc
        except Exception as exc:
            raise GitHubAutomationError(f"Could not access GitHub repository '{name}': {_safe_error(exc)}") from exc

    @staticmethod
    def _permissions(repo: Any) -> dict[str, bool]:
        permissions = getattr(repo, "permissions", None)
        if permissions is None:
            return {}
        raw = getattr(permissions, "raw_data", permissions)
        if not isinstance(raw, dict):
            return {}
        return {str(key): bool(value) for key, value in raw.items()}

    @classmethod
    def _repository_summary(cls, repo: Any) -> GitHubRepositorySummary:
        full_name = str(getattr(repo, "full_name", None) or getattr(repo, "name", "unknown"))
        return GitHubRepositorySummary(
            full_name=full_name,
            owner_login=getattr(getattr(repo, "owner", None), "login", None),
            private=bool(getattr(repo, "private", False)),
            visibility=getattr(repo, "visibility", None),
            default_branch=str(getattr(repo, "default_branch", None) or "main"),
            html_url=getattr(repo, "html_url", None),
            description=getattr(repo, "description", None),
            updated_at=getattr(repo, "updated_at", None),
            permissions=cls._permissions(repo),
        )

    def _rate_limit(self) -> tuple[Optional[int], Optional[int], Optional[datetime]]:
        try:
            limit = self.github.get_rate_limit().core
            return (
                _safe_int(getattr(limit, "remaining", None)),
                _safe_int(getattr(limit, "limit", None)),
                getattr(limit, "reset", None),
            )
        except Exception:
            return None, None, None

    def verify_connection(self, repository: Optional[str] = None) -> GitHubConnectionStatus:
        """Verify account authentication and optional repository access."""

        name = repository or self.repository
        account_login: Optional[str] = None
        account_error: Optional[str] = None
        if self.authenticated:
            try:
                account_login = str(self.github.get_user().login)
            except Exception as exc:
                account_error = f"GitHub authentication failed: {_safe_error(exc)}"

        remaining, total, reset = self._rate_limit()
        status = GitHubConnectionStatus(
            connected=not bool(account_error),
            authenticated=self.authenticated and not bool(account_error),
            account_login=account_login,
            repository=name,
            rate_limit_remaining=remaining,
            rate_limit_total=total,
            rate_limit_reset_at=reset,
            error=account_error,
        )
        if not name or account_error:
            return status

        try:
            repo = self._repo(name)
        except GitHubAutomationError as exc:
            status.connected = False
            status.error = _safe_error(exc)
            return status

        summary = self._repository_summary(repo)
        status.connected = True
        status.repository = summary.full_name
        status.repository_accessible = True
        status.private = summary.private
        status.visibility = summary.visibility
        status.default_branch = summary.default_branch
        status.html_url = summary.html_url
        status.permissions = summary.permissions
        return status

    connection_status = verify_connection

    def connect_repository(self, repository: str) -> GitHubConnectionStatus:
        """Select a repository after confirming it is accessible."""

        normalized = self._validate_repository(repository)
        status = self.verify_connection(normalized)
        if status.repository_accessible:
            self.repository = normalized
        return status

    def list_repositories(self, limit: int = 30) -> list[GitHubRepositorySummary]:
        """List repositories visible to the authenticated account."""

        if not self.authenticated:
            raise GitHubAuthenticationRequired(
                "Repository discovery requires a connected GitHub account; public repositories can still be entered by name."
            )
        try:
            repos = self.github.get_user().get_repos(sort="updated", direction="desc")
            return [self._repository_summary(repo) for repo in _limited(repos, min(max(limit, 0), 100))]
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not list GitHub repositories: {_safe_error(exc)}") from exc

    @staticmethod
    def _changed_file(file: Any) -> GitHubChangedFile:
        patch = getattr(file, "patch", None)
        return GitHubChangedFile(
            filename=str(getattr(file, "filename", None) or "unknown"),
            status=getattr(file, "status", None),
            additions=_safe_int(getattr(file, "additions", None)),
            deletions=_safe_int(getattr(file, "deletions", None)),
            changes=_safe_int(getattr(file, "changes", None)),
            previous_filename=getattr(file, "previous_filename", None),
            blob_url=getattr(file, "blob_url", None),
            raw_url=getattr(file, "raw_url", None),
            patch=redact_sensitive_text(patch, limit=DEFAULT_PATCH_LIMIT) if patch else None,
        )

    @classmethod
    def _commit_snapshot(cls, commit: Any) -> GitHubCommitSnapshot:
        git_commit = getattr(commit, "commit", None)
        author = getattr(git_commit, "author", None)
        committer = getattr(git_commit, "committer", None)
        stats = getattr(commit, "stats", None)
        files = [cls._changed_file(item) for item in (getattr(commit, "files", None) or [])]
        sha = str(getattr(commit, "sha", None) or "")
        return GitHubCommitSnapshot(
            sha=sha,
            short_sha=sha[:7],
            message=str(getattr(git_commit, "message", None) or ""),
            author_login=getattr(getattr(commit, "author", None), "login", None),
            author_name=getattr(author, "name", None),
            committed_at=getattr(author, "date", None) or getattr(committer, "date", None),
            html_url=getattr(commit, "html_url", None),
            additions=_safe_int(getattr(stats, "additions", None)),
            deletions=_safe_int(getattr(stats, "deletions", None)),
            total_changes=_safe_int(getattr(stats, "total", None)),
            files=files,
        )

    def get_commit_changes(
        self,
        commit_sha: str,
        repository: Optional[str] = None,
    ) -> GitHubCommitSnapshot:
        """Fetch an exact commit and its changed files from GitHub."""

        repo = self._repo(repository)
        try:
            return self._commit_snapshot(repo.get_commit(str(commit_sha).strip()))
        except GithubException as exc:
            raise GitHubAutomationError(
                f"Could not load commit '{str(commit_sha)[:12]}': {_safe_error(exc)}"
            ) from exc

    def get_recent_commits(
        self,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 10,
    ) -> list[GitHubCommitSnapshot]:
        """Fetch recent commits and their real changed-file details."""

        repo = self._repo(repository)
        selected_branch = branch or getattr(repo, "default_branch", None) or "main"
        try:
            commits = repo.get_commits(sha=selected_branch)
            return [self._commit_snapshot(commit) for commit in _limited(commits, min(max(limit, 0), 100))]
        except GithubException as exc:
            raise GitHubAutomationError(
                f"Could not load recent commits for branch '{selected_branch}': {_safe_error(exc)}"
            ) from exc

    @staticmethod
    def _workflow_name(run: Any) -> str:
        return str(
            getattr(getattr(run, "workflow", None), "name", None)
            or getattr(run, "name", None)
            or "unknown"
        )

    @classmethod
    def _run_summary(cls, run: Any) -> GitHubWorkflowRunSummary:
        return GitHubWorkflowRunSummary(
            id=_safe_int(getattr(run, "id", None)),
            run_number=_safe_int(getattr(run, "run_number", None)) or None,
            run_attempt=_safe_int(getattr(run, "run_attempt", None)) or None,
            workflow_name=cls._workflow_name(run),
            display_title=getattr(run, "display_title", None),
            status=getattr(run, "status", None),
            conclusion=getattr(run, "conclusion", None),
            event=getattr(run, "event", None),
            branch=getattr(run, "head_branch", None),
            head_sha=getattr(run, "head_sha", None),
            actor=getattr(getattr(run, "actor", None), "login", None),
            created_at=getattr(run, "created_at", None),
            updated_at=getattr(run, "updated_at", None),
            html_url=getattr(run, "html_url", None),
        )

    def get_workflow_runs(
        self,
        repository: Optional[str] = None,
        limit: int = 20,
        status: Optional[str] = None,
        workflow_name: Optional[str] = None,
        *,
        include_system_workflows: bool = True,
    ) -> list[GitHubWorkflowRunSummary]:
        """Fetch recent Actions workflow runs without requiring log uploads."""

        repo = self._repo(repository)
        try:
            target_limit = min(max(limit, 0), 100)
            if target_limit == 0:
                return []
            runs = repo.get_workflow_runs(status=status) if status else repo.get_workflow_runs()
            expected_name = str(workflow_name or "").strip().casefold()
            selected: list[GitHubWorkflowRunSummary] = []
            for run in _limited(runs, 100):
                current_name = self._workflow_name(run).strip().casefold()
                if expected_name and current_name != expected_name:
                    continue
                if not expected_name and not include_system_workflows and current_name in SYSTEM_WORKFLOW_NAMES:
                    continue
                selected.append(self._run_summary(run))
                if len(selected) >= target_limit:
                    break
            return selected
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not load GitHub Actions runs: {_safe_error(exc)}") from exc

    def get_workflows(
        self,
        repository: Optional[str] = None,
        limit: int = 100,
    ) -> list[GitHubWorkflowSummary]:
        """Discover repository workflows instead of inferring them from run history."""

        repo = self._repo(repository)
        try:
            workflows = repo.get_workflows()
            summaries = []
            for workflow in _limited(workflows, min(max(limit, 0), 100)):
                summaries.append(
                    GitHubWorkflowSummary(
                        id=_safe_int(getattr(workflow, "id", None)),
                        name=str(getattr(workflow, "name", None) or "unknown"),
                        path=getattr(workflow, "path", None),
                        state=getattr(workflow, "state", None),
                        html_url=getattr(workflow, "html_url", None),
                    )
                )
            return summaries
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not discover GitHub Actions workflows: {_safe_error(exc)}") from exc

    def get_latest_failed_run(
        self,
        repository: Optional[str] = None,
        workflow_name: Optional[str] = None,
        *,
        include_system_workflows: bool = False,
    ) -> Optional[GitHubWorkflowRunSummary]:
        """Return the most recent completed failure, if one exists."""

        runs = self.get_workflow_runs(
            repository=repository,
            limit=1,
            status="failure",
            workflow_name=workflow_name,
            include_system_workflows=include_system_workflows,
        )
        return runs[0] if runs else None

    @staticmethod
    def _job_summary(job: Any) -> GitHubWorkflowJobSummary:
        steps = [
            GitHubWorkflowStepSummary(
                number=_safe_int(getattr(step, "number", None)) or None,
                name=str(getattr(step, "name", None) or "unknown"),
                status=getattr(step, "status", None),
                conclusion=getattr(step, "conclusion", None),
                started_at=getattr(step, "started_at", None),
                completed_at=getattr(step, "completed_at", None),
            )
            for step in (getattr(job, "steps", None) or [])
        ]
        return GitHubWorkflowJobSummary(
            id=_safe_int(getattr(job, "id", None)),
            name=str(getattr(job, "name", None) or "unknown"),
            status=getattr(job, "status", None),
            conclusion=getattr(job, "conclusion", None),
            started_at=getattr(job, "started_at", None),
            completed_at=getattr(job, "completed_at", None),
            html_url=getattr(job, "html_url", None),
            steps=steps,
        )

    def get_workflow_jobs(
        self,
        run_id: int,
        repository: Optional[str] = None,
    ) -> list[GitHubWorkflowJobSummary]:
        """Fetch jobs and steps for an exact Actions run."""

        repo = self._repo(repository)
        try:
            run = repo.get_workflow_run(int(run_id))
            return [self._job_summary(job) for job in run.jobs()]
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not load jobs for run {int(run_id)}: {_safe_error(exc)}") from exc

    def get_workflow_run(
        self,
        run_id: int,
        repository: Optional[str] = None,
    ) -> GitHubWorkflowRunSummary:
        """Fetch one exact workflow run by its stable GitHub run ID."""

        repo = self._repo(repository)
        try:
            return self._run_summary(repo.get_workflow_run(int(run_id)))
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not load workflow run {int(run_id)}: {_safe_error(exc)}") from exc

    def _download_bytes(self, url: str, max_bytes: int) -> bytes:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.__token:
            headers["Authorization"] = f"Bearer {self.__token}"
        try:
            response = self._http_get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        except requests.RequestException as exc:
            raise GitHubAutomationError(f"Could not download workflow logs: {_safe_error(exc)}") from exc

        if int(getattr(response, "status_code", 0)) != 200:
            status_code = int(getattr(response, "status_code", 0))
            if status_code in {401, 403}:
                raise GitHubAuthenticationRequired(
                    "GitHub denied access to these workflow logs. Reconnect with Actions read permission."
                )
            if status_code == 410:
                raise GitHubAutomationError("GitHub no longer retains logs for this workflow run.")
            raise GitHubAutomationError(f"GitHub returned HTTP {status_code} while downloading workflow logs.")

        chunks: list[bytes] = []
        byte_count = 0
        if hasattr(response, "iter_content"):
            iterator = response.iter_content(chunk_size=64 * 1024)
        else:
            iterator = [getattr(response, "content", b"")]
        for chunk in iterator:
            if not chunk:
                continue
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise GitHubAutomationError(
                    f"Workflow log archive exceeds the {max_bytes:,}-byte safety limit."
                )
            chunks.append(bytes(chunk))
        return b"".join(chunks)

    @staticmethod
    def _decode_log_archive(payload: bytes, max_bytes: int) -> list[GitHubWorkflowLogFile]:
        log_files: list[GitHubWorkflowLogFile] = []
        total_uncompressed = 0
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                truncated = False
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    if info.is_dir():
                        continue
                    file_size = int(info.file_size)
                    if total_uncompressed + file_size > max_bytes:
                        # Gracefully truncate this file's content to fit remaining budget
                        remaining = max_bytes - total_uncompressed
                        if remaining <= 0:
                            truncated = True
                            break
                        try:
                            with archive.open(info) as fh:
                                part = fh.read(remaining)
                        except Exception:
                            part = b""
                        content = part.decode("utf-8", errors="replace")
                        redacted = redact_sensitive_text(content)
                        log_files.append(
                            GitHubWorkflowLogFile(
                                name=f"{info.filename}.part",
                                content=redacted,
                                size_bytes=len(redacted.encode("utf-8")),
                            )
                        )
                        total_uncompressed += len(part)
                        truncated = True
                        break
                    total_uncompressed += file_size
                    content = archive.read(info).decode("utf-8", errors="replace")
                    redacted = redact_sensitive_text(content)
                    log_files.append(
                        GitHubWorkflowLogFile(
                            name=info.filename,
                            content=redacted,
                            size_bytes=len(redacted.encode("utf-8")),
                        )
                    )
                if truncated:
                    # Add a small notice file indicating truncation occurred
                    notice = (
                        "Workflow logs were truncated because the expanded archive exceeded the safety limit. "
                        f"Showing the first {max_bytes:,} bytes of uncompressed log data."
                    )
                    log_files.append(
                        GitHubWorkflowLogFile(name="TRUNCATED_NOTICE", content=notice, size_bytes=len(notice.encode("utf-8")))
                    )
        except zipfile.BadZipFile:
            content = redact_sensitive_text(payload.decode("utf-8", errors="replace"))
            if len(content.encode("utf-8")) > max_bytes:
                # Truncate raw text payload instead of raising
                truncated_content = content.encode("utf-8")[:max_bytes].decode("utf-8", errors="replace")
                notice = (
                    "Workflow logs were truncated because the expanded archive exceeded the safety limit. "
                    f"Showing the first {max_bytes:,} bytes of uncompressed log data."
                )
                log_files.append(
                    GitHubWorkflowLogFile(name="workflow.log", content=truncated_content, size_bytes=len(truncated_content.encode("utf-8")))
                )
                log_files.append(
                    GitHubWorkflowLogFile(name="TRUNCATED_NOTICE", content=notice, size_bytes=len(notice.encode("utf-8")))
                )
            
        return log_files

    def get_workflow_logs(
        self,
        run_id: int,
        repository: Optional[str] = None,
        max_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
    ) -> GitHubWorkflowLogBundle:
        """Download, bound, and redact Actions logs entirely in memory."""

        if max_bytes < 1:
            raise GitHubAutomationError("Workflow log size limit must be positive.")

        name = self._repository_name(repository)
        repo = self._repo(name)
        try:
            run = repo.get_workflow_run(int(run_id))
            logs_url = str(getattr(run, "logs_url", None) or "")
        except GithubException as exc:
            raise GitHubAutomationError(f"Could not load workflow run {int(run_id)}: {_safe_error(exc)}") from exc
        if not logs_url:
            raise GitHubAutomationError(f"Workflow run {int(run_id)} does not expose a log archive URL.")

        payload = self._download_bytes(logs_url, max_bytes=max_bytes)
        files = self._decode_log_archive(payload, max_bytes=max_bytes)
        combined = "\n\n".join(
            f"{'=' * 72}\nLOG FILE: {item.name}\n{'=' * 72}\n{item.content}" for item in files
        )
        return GitHubWorkflowLogBundle(
            repository=name,
            run_id=int(run_id),
            files=files,
            combined_text=combined,
            byte_count=sum(item.size_bytes for item in files),
        )

    def latest_failed_run_context(
        self,
        repository: Optional[str] = None,
        *,
        workflow_name: Optional[str] = None,
        include_system_workflows: bool = False,
        include_logs: bool = True,
        max_log_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
    ) -> GitHubFailedRunContext:
        """Load the latest failure, its jobs, and logs for one-click RCA."""

        run = self.get_latest_failed_run(
            repository,
            workflow_name=workflow_name,
            include_system_workflows=include_system_workflows,
        )
        if run is None:
            return GitHubFailedRunContext()

        jobs: list[GitHubWorkflowJobSummary] = []
        jobs_error: Optional[str] = None
        try:
            jobs = self.get_workflow_jobs(run.id, repository=repository)
        except GitHubAutomationError as exc:
            jobs_error = _safe_error(exc)
        logs: Optional[GitHubWorkflowLogBundle] = None
        logs_error: Optional[str] = None
        if include_logs:
            try:
                logs = self.get_workflow_logs(run.id, repository=repository, max_bytes=max_log_bytes)
            except GitHubAutomationError as exc:
                logs_error = _safe_error(exc)
        return GitHubFailedRunContext(
            run=run,
            jobs=jobs,
            jobs_error=jobs_error,
            logs=logs,
            logs_error=logs_error,
        )

    def _matching_workflow_run(
        self,
        repo: Any,
        commit_sha: str,
        workflow_name: Optional[str],
    ) -> Optional[GitHubWorkflowRunSummary]:
        """Find the newest target workflow run tied to one exact commit."""

        try:
            try:
                candidate_runs = repo.get_workflow_runs(head_sha=commit_sha)
            except TypeError:
                # Lightweight test doubles and older PyGithub versions may not
                # expose the REST ``head_sha`` filter. Filter a bounded page in
                # memory while preserving the same behavior.
                candidate_runs = repo.get_workflow_runs()
            expected_name = str(workflow_name or "").strip().casefold()
            for run in _limited(candidate_runs, 100):
                if str(getattr(run, "head_sha", None) or "") != commit_sha:
                    continue
                current_name = self._workflow_name(run).strip().casefold()
                if expected_name and current_name != expected_name:
                    continue
                if not expected_name and current_name in SYSTEM_WORKFLOW_NAMES:
                    continue
                return self._run_summary(run)
        except GithubException as exc:
            raise GitHubAutomationError(
                f"Could not determine workflow status for commit '{commit_sha[:12]}': {_safe_error(exc)}"
            ) from exc
        return None

    def get_latest_commit_context(
        self,
        repository: Optional[str] = None,
        workflow_name: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> GitHubPredictionInput:
        """Build model-ready inputs automatically from the latest real commit."""

        name = self._repository_name(repository)
        repo = self._repo(name)
        selected_branch = branch or getattr(repo, "default_branch", None) or "main"
        try:
            commits = _limited(repo.get_commits(sha=selected_branch), 1)
        except GithubException as exc:
            raise GitHubAutomationError(
                f"Could not load the latest commit for branch '{selected_branch}': {_safe_error(exc)}"
            ) from exc
        if not commits:
            raise GitHubAutomationError(f"Branch '{selected_branch}' does not contain any commits.")

        commit = self._commit_snapshot(commits[0])
        matching_run = self._matching_workflow_run(repo, commit.sha, workflow_name)
        collector = self._collector_factory(self.__token)
        # Reuse the same client/repository permissions and avoid a second auth
        # handshake.  The collector remains the canonical source of leakage-safe
        # historical features used during model training.
        collector.github = self.github
        try:
            raw_features = collector.build_prediction_features(
                repository=name,
                commit_sha=commit.sha,
                branch=selected_branch,
                workflow_name=workflow_name,
            )
        except ValueError as exc:
            raise GitHubAutomationError(_safe_error(exc)) from exc
        model_features = FailureFeatureExtractor.to_feature_row(raw_features)
        return GitHubPredictionInput(
            repository=name,
            workflow_name=str(workflow_name or raw_features.get("workflow_name") or "unknown"),
            branch=selected_branch,
            commit=commit,
            workflow_run_id=matching_run.id if matching_run else None,
            workflow_status=matching_run.status if matching_run else None,
            workflow_conclusion=matching_run.conclusion if matching_run else None,
            raw_features=raw_features,
            model_features=model_features,
        )

    build_latest_prediction_input = get_latest_commit_context

    def predict_latest_change(
        self,
        prediction_service: "FailurePredictionService",
        repository: Optional[str] = None,
        workflow_name: Optional[str] = None,
        branch: Optional[str] = None,
        *,
        record: bool = True,
    ) -> GitHubAutomatedPrediction:
        """Predict failure risk for the latest commit without manual inputs."""

        prediction_input = self.get_latest_commit_context(
            repository=repository,
            workflow_name=workflow_name,
            branch=branch,
        )
        workflow_completed = (
            str(prediction_input.workflow_status or "").casefold() == "completed"
            or bool(prediction_input.workflow_conclusion)
        )
        should_record = bool(record and not workflow_completed)
        if should_record:
            prediction, prediction_id = prediction_service.predict_and_record(
                repository=prediction_input.repository,
                workflow=prediction_input.workflow_name,
                run_id=None,
                commit_sha=prediction_input.commit.sha,
                features=prediction_input.raw_features,
            )
        else:
            prediction = prediction_service.predict(prediction_input.raw_features)
            prediction_id = None
        if workflow_completed:
            mode = "retrospective"
            record_reason = "The target workflow already completed; this score is display-only and was not added to feedback."
        elif prediction_input.workflow_status:
            mode = "active_run"
            record_reason = None if should_record else "Prediction recording was disabled by the caller."
        else:
            mode = "pre_ci"
            record_reason = None if should_record else "Prediction recording was disabled by the caller."
        return GitHubAutomatedPrediction(
            input=prediction_input,
            prediction=prediction,
            prediction_id=prediction_id,
            mode=mode,
            is_retrospective=workflow_completed,
            prediction_recorded=bool(prediction_id) if should_record else False,
            record_reason=record_reason,
        )

    def get_snapshot(
        self,
        repository: Optional[str] = None,
        *,
        commit_limit: int = 10,
        run_limit: int = 20,
        branch: Optional[str] = None,
        workflow_name: Optional[str] = None,
        include_system_workflows: bool = False,
    ) -> GitHubAutomationSnapshot:
        """Load the connection, recent changes, and Actions runs in one call."""

        name = self._repository_name(repository)
        connection = self.verify_connection(name)
        if not connection.repository_accessible:
            return GitHubAutomationSnapshot(connection=connection)
        return GitHubAutomationSnapshot(
            connection=connection,
            commits=self.get_recent_commits(name, branch=branch, limit=commit_limit),
            workflows=self.get_workflows(name),
            workflow_runs=self.get_workflow_runs(
                name,
                limit=run_limit,
                workflow_name=workflow_name,
                include_system_workflows=include_system_workflows,
            ),
        )

    def download_latest_model_artifact(
        self,
        repository: Optional[str],
        artifact_name: str,
        dest_dir: str | Path,
        expected_files: Optional[list[str]] = None,
        max_runs: int = 20,
    ) -> bool:
        """Attempt to download the latest successful training workflow's named artifact.

        Returns True when at least one expected file was extracted into dest_dir.
        This uses the Actions REST API and requires an authenticated token.
        """

        if not self.__token:
            raise GitHubAutomationError("Authenticated GitHub token required to download artifacts.")

        name = self._repository_name(repository)
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        if expected_files is None:
            expected_files = [
                "failure_predictor.joblib",
                "failure_predictor_metadata.json",
                "failure_category_predictor.joblib",
                "failure_category_predictor_metadata.json",
            ]

        session = requests.Session()
        # Use the legacy 'token' scheme which GitHub expects in many API flows,
        # and include a sensible Accept/User-Agent to avoid API surprises.
        session.headers.update(
            {
                "Authorization": f"token {self.__token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "CI-CD-Root-Cause-Analyzer",
            }
        )

        api = f"https://api.github.com/repos/{name}/actions/workflows/train-model.yml/runs?status=success&per_page={int(max_runs)}"
        try:
            resp = session.get(api, timeout=30)
            resp.raise_for_status()
            runs = resp.json().get("workflow_runs", [])
        except Exception as exc:
            logger = logging.getLogger("github_automation")
            logger.debug("LIST_RUNS_FAILED: repo=%s url=%s error=%s", name, api, _safe_error(exc))
            raise GitHubAutomationError(f"Could not list training workflow runs: {_safe_error(exc)}") from exc

        for run in runs:
            run_id = int(run.get("id") or 0)
            if not run_id:
                continue
            artifacts_api = f"https://api.github.com/repos/{name}/actions/runs/{run_id}/artifacts"
            try:
                aresp = session.get(artifacts_api, timeout=30)
                aresp.raise_for_status()
                artifacts = aresp.json().get("artifacts", [])
            except Exception:
                logger = logging.getLogger("github_automation")
                logger.debug("ARTIFACTS_LIST_FAILED: run_id=%s url=%s", run_id, artifacts_api)
                continue

            for art in artifacts:
                if str(art.get("name") or "") != artifact_name:
                    continue
                archive_url = art.get("archive_download_url")
                if not archive_url:
                    continue
                try:
                    down = session.get(archive_url, stream=True, timeout=120)
                    status_code = getattr(down, "status_code", None)
                    if status_code is None or int(status_code) != 200:
                        logger = logging.getLogger("github_automation")
                        logger.debug(
                            "ARTIFACT_DOWNLOAD_FAILED: run_id=%s archive_url=%s status=%s",
                            run_id,
                            archive_url,
                            status_code,
                        )
                        raise GitHubAutomationError(f"Failed to download artifact archive: HTTP {status_code}")
                except Exception as exc:
                    raise GitHubAutomationError(f"Failed to download artifact archive: {_safe_error(exc)}") from exc

                # Extract only expected files to the destination directory.
                extracted_any = False
                try:
                    with zipfile.ZipFile(io.BytesIO(down.content)) as archive:
                        for info in archive.infolist():
                            member_name = info.filename
                            base = Path(member_name).name
                            if base in expected_files:
                                target = dest / base
                                with archive.open(info) as source, open(target, "wb") as out:
                                    out.write(source.read())
                                extracted_any = True
                except zipfile.BadZipFile:
                    # Some artifact uploads may be single files; try saving raw content
                    for fname in expected_files:
                        if fname in (art.get("name") or ""):
                            target = dest / fname
                            target.write_bytes(down.content)
                            extracted_any = True
                # Record which files were extracted for diagnosis
                logger = logging.getLogger("github_automation")
                if extracted_any:
                    files = [p.name for p in dest.iterdir() if p.is_file()]
                    logger.debug("EXTRACTED: run_id=%s files=%s", run_id, files)
                else:
                    logger.debug("NO_EXPECTED_FILES_IN_ARTIFACT: run_id=%s artifact_name=%s", run_id, art.get("name"))

                if extracted_any:
                    return True

        return False


__all__ = [
    "GitHubAuthenticationRequired",
    "GitHubAutomatedPrediction",
    "GitHubAutomationError",
    "GitHubAutomationService",
    "GitHubAutomationSnapshot",
    "GitHubChangedFile",
    "GitHubCommitSnapshot",
    "GitHubConnectionStatus",
    "GitHubFailedRunContext",
    "GitHubPredictionInput",
    "GitHubRepositorySummary",
    "GitHubWorkflowJobSummary",
    "GitHubWorkflowLogBundle",
    "GitHubWorkflowLogFile",
    "GitHubWorkflowRunSummary",
    "GitHubWorkflowSummary",
    "GitHubWorkflowStepSummary",
]
