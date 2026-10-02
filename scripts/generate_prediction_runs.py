#!/usr/bin/env python3
"""Generate real GitHub Actions CI prediction observations safely.

This script intentionally creates harmless marker-file commits on the main branch,
which triggers the existing CI/CD Pipeline. Each successful run is expected to
create a prediction snapshot and later be finalized by the feedback workflow.

It does not fabricate prediction history rows or modify the training model,
RCA logic, or dashboard behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MARKER_PATH = PROJECT_ROOT / "data" / "prediction_run_marker.txt"
DEFAULT_REPO = "Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer"
ALLOWED_PATHS = {
    ".tools/",
    "report.md",
    "reportdata.md",
    "data/generated_run_plan.json",
    "data/generated_runs.json",
    "data/historical_runs.csv",
    "data/prediction_history.csv",
    "data/prediction_run_marker.txt",
}


def _run_command(args: Sequence[str], *, check: bool = True, capture: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _normalize_status(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _normalize_status_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _is_allowed_status_path(path: str) -> bool:
    normalized = _normalize_status_path(path)
    if not normalized:
        return False
    if normalized in ALLOWED_PATHS:
        return True
    if normalized == ".tools" or normalized.startswith(".tools/"):
        return True
    return False


def _status_is_clean(status_lines: Sequence[str]) -> bool:
    if not status_lines:
        return True

    remaining_paths: list[str] = []

    for raw_line in status_lines:
        line = raw_line.strip()
        if not line:
            continue

        if len(line) < 3:
            remaining_paths.append(line)
            continue

        raw_path = line[3:].strip()

        if raw_path == "data/prediction_run_marker.txt":
            continue

        path = _normalize_status_path(raw_path)
        if not path:
            remaining_paths.append(line)
            continue

        if path == ".tools" or path.startswith(".tools/"):
            continue

        if path in ALLOWED_PATHS:
            continue

        if path == "data" or path.startswith("data/"):
            if path == "data":
                continue
            rel = path.removeprefix("data/")
            allowed_data = {
                "generated_run_plan.json",
                "generated_runs.json",
                "historical_runs.csv",
                "prediction_history.csv",
                "prediction_run_marker.txt",
            }

            if rel in allowed_data:
                continue

            if path == "data":
                data_dir = PROJECT_ROOT / "data"
                if data_dir.is_dir():
                    actual_files = {
                        str(p.relative_to(data_dir)).replace("\\", "/")
                        for p in data_dir.rglob("*")
                        if p.is_file()
                    }
                    if actual_files <= allowed_data:
                        continue

        if path == "report.md" or path == "reportdata.md":
            continue

        remaining_paths.append(path)

    return not remaining_paths


def _ensure_git_available() -> None:
    try:
        _run_command(["git", "--version"], check=True)
        _run_command(["gh", "--version"], check=True)
    except Exception as exc:  # pragma: no cover - environment validation
        raise RuntimeError(f"Required tools are not available: {exc}") from exc


def _require_clean_worktree() -> None:
    status = _normalize_status(_run_command(["git", "status", "--short", "--untracked-files=all"]).stdout)
    if status and not _status_is_clean(status):
        raise RuntimeError(
            "Refusing to run: working tree is not clean. "
            "Please commit or stash unrelated changes before generating CI prediction runs."
        )


def _repo_name() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo.strip()
    result = _run_command(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], check=True)
    return result.stdout.strip()


def _default_branch() -> str:
    result = _run_command(["git", "branch", "--show-current"], check=True)
    current = result.stdout.strip()
    if current:
        return current
    result = _run_command(["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"], check=True)
    return result.stdout.strip() or "main"


def _checkout_main_branch() -> None:
    branch = _default_branch()
    if branch != "main":
        try:
            _run_command(["git", "checkout", branch], check=True)
        except RuntimeError:
            _run_command(["git", "checkout", "main"], check=False)
    else:
        _run_command(["git", "checkout", "main"], check=False)


def _ensure_marker_file_exists() -> None:
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MARKER_PATH.exists():
        MARKER_PATH.write_text("# prediction-run-marker\n", encoding="utf-8")


def _verify_marker_only_change() -> None:
    status = _normalize_status(_run_command(["git", "status", "--short", "--untracked-files=all"]).stdout)
    if not status:
        return
    if not _status_is_clean(status):
        raise RuntimeError(
            "Refusing to proceed: only the dedicated marker file can be modified before a run. "
            f"Detected: {status}"
        )


def _read_prediction_history_count() -> int:
    history_path = PROJECT_ROOT / "data" / "prediction_history.csv"
    if not history_path.exists():
        return 0
    try:
        import pandas as pd

        frame = pd.read_csv(history_path)
        return int(len(frame))
    except Exception:
        return 0


def _gh_run_status(run_id: int) -> dict[str, str]:
    result = _run_command(
        ["gh", "run", "view", str(run_id), "--repo", _repo_name(), "--json", "status,conclusion,displayTitle,workflowName,headBranch,headSha,url"],
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"status": "unknown", "conclusion": "unknown", "displayTitle": "", "workflowName": "", "headBranch": "", "headSha": "", "url": ""}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "unknown", "conclusion": "unknown", "displayTitle": "", "workflowName": "", "headBranch": "", "headSha": "", "url": ""}
    return {
        "status": payload.get("status", "unknown"),
        "conclusion": payload.get("conclusion") or "pending",
        "displayTitle": payload.get("displayTitle", ""),
        "workflowName": payload.get("workflowName", ""),
        "headBranch": payload.get("headBranch", ""),
        "headSha": payload.get("headSha", ""),
        "url": payload.get("url", ""),
    }


def _wait_for_ci_run(run_id: int, *, poll_interval: int = 15, timeout_seconds: int = 1800) -> dict[str, str]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _gh_run_status(run_id)
        status = str(state.get("status", "unknown")).lower()
        conclusion = str(state.get("conclusion", "pending")).lower()
        if status in {"completed", "success", "failure", "cancelled", "timed_out", "startup_failure"}:
            if status == "completed":
                return state
            if conclusion in {"success", "failure", "cancelled", "timed_out", "startup_failure", "action_required"}:
                return state
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for CI run {run_id} to complete.")


def _wait_for_feedback_workflow(run_id: int, *, poll_interval: int = 15, timeout_seconds: int = 1800) -> bool:
    repo = _repo_name()
    current_sha = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _run_command(
            [
                "gh",
                "run",
                "list",
                "--repo",
                repo,
                "--workflow",
                "feedback.yml",
                "--limit",
                "20",
                "--json",
                "databaseId,headSha,status,conclusion,workflowName,displayTitle",
            ],
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = []
            matches = [
                item for item in payload if str(item.get("headSha", "")).startswith(current_sha[:7])
            ]
            if matches:
                latest = sorted(matches, key=lambda item: int(item.get("databaseId", 0)))[-1]
                state = _gh_run_status(int(latest["databaseId"]))
                if str(state.get("status", "")).lower() == "completed":
                    return True
        time.sleep(poll_interval)
    return False


def _run_ci_and_wait(commit_sha: str, *, poll_interval: int, timeout_seconds: int) -> tuple[int, dict[str, str]]:
    repo = _repo_name()
    result = _run_command(["gh", "run", "list", "--repo", repo, "--workflow", "CI/CD Pipeline", "--limit", "1", "--json", "databaseId"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Unable to query GitHub CI runs.")

    # Trigger the workflow by pushing the commit to the main branch.
    _run_command(["git", "push", "origin", "HEAD:main"], check=True)

    run_list = _run_command([
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        "CI/CD Pipeline",
        "--branch",
        "main",
        "--limit",
        "5",
        "--json",
        "databaseId,headSha,status,conclusion",
    ], check=True)
    try:
        runs = json.loads(run_list.stdout) if run_list.stdout.strip() else []
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse GitHub run list response.") from exc

    matching = [entry for entry in runs if str(entry.get("headSha", "")).startswith(commit_sha[:7])]
    if not matching:
        raise RuntimeError(f"No CI run found for commit {commit_sha[:12]}")

    run_id = int(matching[0]["databaseId"])
    state = _wait_for_ci_run(run_id, poll_interval=poll_interval, timeout_seconds=timeout_seconds)
    if str(state.get("conclusion", "")).lower() not in {"success", "neutral", "skipped"}:
        raise RuntimeError(
            f"CI run {run_id} concluded as '{state.get('conclusion')}' instead of success. "
            "Stopping immediately."
        )
    return run_id, state


def _create_prediction_commit(counter: int, *, dry_run: bool) -> str:
    _ensure_marker_file_exists()
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    stamp = f"{counter}-{timestamp}"
    marker = MARKER_PATH.read_text(encoding="utf-8")
    marker += f"{stamp}: prediction generation run {counter}\n"
    if dry_run:
        print(f"[dry-run] would append marker: {stamp}")
        return stamp
    MARKER_PATH.write_text(marker, encoding="utf-8")
    _run_command(["git", "add", str(MARKER_PATH.relative_to(PROJECT_ROOT))], check=True)
    _run_command(["git", "commit", "-m", f"chore: prediction run marker {stamp}"], check=True)
    commit_sha = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
    return commit_sha


def _validate_only_marker_changed() -> None:
    status = _normalize_status(_run_command(["git", "status", "--short"]).stdout)
    if not status:
        return
    if not _status_is_clean(status):
        raise RuntimeError(
            "Refusing to proceed: repository state is not limited to the dedicated marker file. "
            f"Detected: {status}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a short sequence of harmless GitHub CI prediction observations using the "
            "existing prediction → CI → feedback flow."
        )
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of genuine CI runs to generate")
    parser.add_argument("--interval", type=int, default=30, help="Seconds to wait between runs")
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without creating commits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer.")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0.")

    _ensure_git_available()
    _require_clean_worktree()
    _checkout_main_branch()
    _ensure_marker_file_exists()
    _validate_only_marker_changed()

    print(f"Repository: {_repo_name()}")
    print(f"Dry run: {args.dry_run}")
    print(f"Target run count: {args.runs}")

    if args.dry_run:
        for index in range(1, args.runs + 1):
            print(f"[dry-run] run {index}/{args.runs}: would create harmless marker commit on main and wait for CI + feedback")
        return 0

    for index in range(1, args.runs + 1):
        _validate_only_marker_changed()
        teaser = f"prediction run marker {index}"
        print(f"\nRun {index}/{args.runs}: preparing commit")
        MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MARKER_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ')} {teaser}\n")
        _run_command(["git", "add", "data/prediction_run_marker.txt"], check=True)
        _run_command(["git", "commit", "-m", f"chore: {teaser}"], check=True)
        commit_sha = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
        print(f"- Commit: {commit_sha}")
        run_id, run_state = _run_ci_and_wait(commit_sha, poll_interval=15, timeout_seconds=1800)
        print(f"- CI run: {run_id} status={run_state.get('status')} conclusion={run_state.get('conclusion')}")
        feedback_ok = _wait_for_feedback_workflow(run_id, poll_interval=15, timeout_seconds=1800)
        print(f"- Feedback workflow: {'completed' if feedback_ok else 'not completed yet'}")
        history_count = _read_prediction_history_count()
        print(f"- Prediction history records: {history_count}")
        if index < args.runs and args.interval:
            print(f"- Waiting {args.interval}s before next run...")
            time.sleep(args.interval)

    print("\nCompleted requested prediction-generation sequence.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
