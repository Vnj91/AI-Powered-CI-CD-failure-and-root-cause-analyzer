"""Deterministic likely-culprit commit analysis for failed CI runs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..tools.log_parser import ParsedError
from ..prediction.feature_extractor import FailureFeatureExtractor


class CulpritCommitResult(BaseModel):
    """Ranked culprit commit analysis."""

    commit_sha: Optional[str] = None
    short_sha: Optional[str] = None
    commit_message: Optional[str] = None
    author: Optional[str] = None
    confidence: str = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    changed_files: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    ranked_candidates: list[dict] = Field(default_factory=list)


@dataclass
class _CommitCandidate:
    sha: str
    message: str
    author: str
    files: list[str]
    score: float
    evidence: list[str] = field(default_factory=list)


class CommitAnalyzer:
    """Score commits/changes against failure context using deterministic signals."""

    DEPENDENCY_FILES = FailureFeatureExtractor.DEPENDENCY_FILES
    CI_HINTS = FailureFeatureExtractor.CI_PATH_HINTS
    TEST_HINTS = FailureFeatureExtractor.TEST_PATH_HINTS
    INFRA_HINTS = FailureFeatureExtractor.INFRA_PATH_HINTS

    ERROR_FILE_PATTERN = re.compile(r'File "([^"]+)"')
    MODULE_PATTERN = re.compile(r"(?:module|package|import)\s+['\"]?([\w./-]+)")

    def analyze_failed_run(
        self,
        commit_sha: str,
        changed_files: list[str],
        commit_message: str = "",
        author: str = "",
        parsed_error: Optional[ParsedError] = None,
        error_category: Optional[str] = None,
        failed_step: Optional[str] = None,
    ) -> CulpritCommitResult:
        """Score a single failed-run commit against the observed error context."""

        candidate = _CommitCandidate(
            sha=commit_sha,
            message=commit_message or "",
            author=author or "",
            files=list(changed_files or []),
            score=0.0,
        )
        self._score_candidate(
            candidate,
            parsed_error=parsed_error,
            error_category=error_category,
            failed_step=failed_step,
        )

        confidence, confidence_score = self._confidence_from_score(candidate.score, len(candidate.evidence))

        return CulpritCommitResult(
            commit_sha=candidate.sha,
            short_sha=candidate.sha[:7] if candidate.sha else None,
            commit_message=candidate.message,
            author=candidate.author,
            confidence=confidence,
            confidence_score=confidence_score,
            changed_files=candidate.files,
            evidence=candidate.evidence,
            ranked_candidates=[
                {
                    "commit_sha": candidate.sha,
                    "score": candidate.score,
                    "evidence": candidate.evidence,
                }
            ],
        )

    def analyze_commit_list(
        self,
        commits: list[dict],
        parsed_error: Optional[ParsedError] = None,
        error_category: Optional[str] = None,
        failed_step: Optional[str] = None,
    ) -> CulpritCommitResult:
        """Rank multiple commits when a workflow span includes several changes."""

        candidates: list[_CommitCandidate] = []
        for commit in commits:
            candidate = _CommitCandidate(
                sha=str(commit.get("sha", "")),
                message=str(commit.get("message", "")),
                author=str(commit.get("author", "")),
                files=list(commit.get("files", []) or []),
                score=0.0,
            )
            self._score_candidate(
                candidate,
                parsed_error=parsed_error,
                error_category=error_category,
                failed_step=failed_step,
            )
            candidates.append(candidate)

        if not candidates:
            return CulpritCommitResult()

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        best = ranked[0]
        confidence, confidence_score = self._confidence_from_score(best.score, len(best.evidence))

        return CulpritCommitResult(
            commit_sha=best.sha,
            short_sha=best.sha[:7] if best.sha else None,
            commit_message=best.message,
            author=best.author,
            confidence=confidence,
            confidence_score=confidence_score,
            changed_files=best.files,
            evidence=best.evidence,
            ranked_candidates=[
                {"commit_sha": item.sha, "score": item.score, "evidence": item.evidence[:5]}
                for item in ranked[:5]
            ],
        )

    def _score_candidate(
        self,
        candidate: _CommitCandidate,
        parsed_error: Optional[ParsedError],
        error_category: Optional[str],
        failed_step: Optional[str],
    ) -> None:
        lowered_files = [path.lower() for path in candidate.files]
        message = candidate.message.lower()
        category = (error_category or "").lower()
        step = (failed_step or "").lower()

        if any(Path(path).name in self.DEPENDENCY_FILES for path in candidate.files):
            candidate.score += 2.0
            candidate.evidence.append("Dependency manifest changed")

        if any(any(hint in path for hint in self.CI_HINTS) for path in lowered_files):
            candidate.score += 2.5
            candidate.evidence.append("CI workflow file changed")

        if any("dockerfile" in path or path.endswith("dockerfile") for path in lowered_files):
            candidate.score += 2.0
            candidate.evidence.append("Docker-related file changed")

        if any(any(hint in path for hint in self.TEST_HINTS) for path in lowered_files):
            candidate.score += 1.5
            candidate.evidence.append("Test file changed")

        if any(any(hint in path for hint in self.INFRA_HINTS) for path in lowered_files):
            candidate.score += 1.5
            candidate.evidence.append("Infrastructure file changed")

        if "dependency" in category and any(Path(f).name in self.DEPENDENCY_FILES for f in candidate.files):
            candidate.score += 2.0
            candidate.evidence.append("Failure category is dependency-related and dependency files changed")

        if "test" in category and any(any(h in path for h in self.TEST_HINTS) for path in lowered_files):
            candidate.score += 2.0
            candidate.evidence.append("Failure category is test-related and test files changed")

        if parsed_error:
            self._score_error_overlap(candidate, parsed_error)

        if step and any(token in message for token in step.split() if len(token) > 3):
            candidate.score += 0.5
            candidate.evidence.append(f"Commit message references failed step context: {failed_step}")

        if candidate.files:
            candidate.evidence.append(f"{len(candidate.files)} file(s) changed in candidate commit")

    def _score_error_overlap(self, candidate: _CommitCandidate, parsed_error: ParsedError) -> None:
        error_text = f"{parsed_error.error_type} {parsed_error.error_message}"
        if parsed_error.stack_trace:
            error_text += " " + " ".join(parsed_error.stack_trace[:5])

        referenced_paths = set(self.ERROR_FILE_PATTERN.findall(error_text))
        for token in self.MODULE_PATTERN.findall(error_text):
            referenced_paths.add(token.replace(".", "/"))

        for path in candidate.files:
            normalized = path.lower()
            basename = Path(path).name.lower()
            for ref in referenced_paths:
                ref_lower = ref.lower()
                if ref_lower in normalized or basename in ref_lower or normalized.endswith(ref_lower):
                    candidate.score += 3.0
                    candidate.evidence.append(f"Changed file overlaps error reference: {path}")
                    break

        if parsed_error.error_category and parsed_error.error_category.value == "dependency":
            if any(Path(f).name in self.DEPENDENCY_FILES for f in candidate.files):
                candidate.score += 1.5
                candidate.evidence.append("Dependency error with manifest change")

    @staticmethod
    def _confidence_from_score(score: float, evidence_count: int) -> tuple[str, float]:
        if score >= 6.0 and evidence_count >= 3:
            return "high", min(0.95, 0.55 + score * 0.05)
        if score >= 3.0 and evidence_count >= 2:
            return "medium", min(0.75, 0.35 + score * 0.05)
        if score > 0:
            return "low", min(0.5, 0.15 + score * 0.05)
        return "unknown", 0.0


def analyze_culprit_for_repository(
    repo_name: str,
    parsed_error: Optional[ParsedError] = None,
    error_category: Optional[str] = None,
    failed_step: Optional[str] = None,
    workflow_run_id: Optional[int] = None,
    github_token: Optional[str] = None,
) -> CulpritCommitResult:
    """Fetch failed-run commit context from GitHub and score likely culprit.

    When ingestion supplies a run ID, use that exact run so a new failure
    arriving mid-analysis cannot change the commit being evaluated.
    """

    # This enhancement is invoked after authenticated Actions-log retrieval in
    # the dashboard. Avoid an unexpected anonymous network call in offline or
    # test-only graph executions.
    from config import Config

    if not github_token and not Config.GITHUB_ACCESS_TOKEN:
        return CulpritCommitResult()

    try:
        from ..integrations.github_automation import GitHubAutomationService

        automation = GitHubAutomationService(token=github_token, repository=repo_name)
        failed_run = (
            automation.get_workflow_run(int(workflow_run_id), repository=repo_name)
            if workflow_run_id is not None
            else automation.get_latest_failed_run(repo_name, include_system_workflows=True)
        )
        if failed_run is None or not failed_run.head_sha:
            return CulpritCommitResult()

        commit = automation.get_commit_changes(failed_run.head_sha, repository=repo_name)

        analyzer = CommitAnalyzer()
        return analyzer.analyze_failed_run(
            commit_sha=commit.sha,
            changed_files=commit.changed_files,
            commit_message=commit.message,
            author=commit.author_login or commit.author_name or "",
            parsed_error=parsed_error,
            error_category=error_category,
            failed_step=failed_step,
        )
    except Exception:
        return CulpritCommitResult()
