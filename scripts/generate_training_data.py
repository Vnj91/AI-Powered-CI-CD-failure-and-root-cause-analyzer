#!/usr/bin/env python3
# ruff: noqa: E402
"""Plan or explicitly execute real GitHub Actions training-data scenarios."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.prediction.training_data_generator import (
    GeneratedRunStore,
    GhCliClient,
    GitWorkspace,
    TrainingDataGenerator,
    build_plan,
    format_execution_plan,
    summarize_generation,
    write_run_plan,
)

DEFAULT_REPO = "Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real GitHub Actions training data safely.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target repository in owner/name format")
    parser.add_argument("--runs", type=int, default=30, help="Total runs to plan")
    parser.add_argument("--success-runs", type=int, default=15, help="Successful CI runs")
    parser.add_argument("--failure-runs", type=int, default=15, help="Controlled failure runs")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic scenario ordering")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; never interacts with GitHub")
    parser.add_argument("--execute", action="store_true", help="Create branches and dispatch real workflows")
    parser.add_argument("--check-env", action="store_true", help="Verify local Git/GitHub execution prerequisites")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between GitHub status checks")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds to wait per workflow")
    parser.add_argument("--cleanup", action="store_true", help="Delete only generator-created branches after execution")
    parser.add_argument("--collect", action="store_true", help="Collect actual GitHub history and inspect it")
    parser.add_argument("--train", action="store_true", help="Train/evaluate only if validation permits it")
    parser.add_argument("--plan-path", default=str(Config.GENERATED_RUN_PLAN_PATH))
    parser.add_argument("--no-write-plan", action="store_true")
    parser.add_argument("--allow-large-run-set", action="store_true")
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    if args.success_runs + args.failure_runs != args.runs:
        raise ValueError("--runs must equal --success-runs + --failure-runs.")
    if args.execute and args.dry_run:
        raise ValueError("Choose either --dry-run or --execute, not both.")
    if (args.collect or args.train) and not args.execute:
        raise ValueError("--collect and --train require explicit --execute.")
    if args.poll_interval <= 0 or args.timeout <= 0:
        raise ValueError("--poll-interval and --timeout must be positive.")


def check_environment(repo_root: Path = PROJECT_ROOT, runner=subprocess) -> list[str]:
    """Run non-mutating local prerequisites checks before any live execution."""

    checks = (
        ("git user.name", ["git", "config", "user.name"]),
        ("git user.email", ["git", "config", "user.email"]),
        ("GitHub CLI", ["gh", "--version"]),
        ("GitHub CLI authentication", ["gh", "auth", "status"]),
        ("clean working tree", ["git", "status", "--porcelain"]),
    )
    failures: list[str] = []
    for label, command in checks:
        try:
            result = runner.run(command, cwd=str(repo_root), capture_output=True, text=True, check=False)
        except FileNotFoundError:
            failures.append(f"{label}: required command is not installed")
            continue
        if result.returncode != 0:
            failures.append(f"{label}: check failed")
        elif command[:2] == ["git", "status"] and result.stdout.strip():
            failures.append("clean working tree: uncommitted changes detected")
        elif command[:2] == ["git", "config"] and not result.stdout.strip():
            failures.append(f"{label}: not configured")
    return failures


def _print_environment_check(repo_root: Path = PROJECT_ROOT) -> int:
    failures = check_environment(repo_root)
    if failures:
        print("Local pre-flight checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Local pre-flight checks passed.")
    return 0


def main() -> int:
    args = parse_args()
    _validate(args)
    if args.check_env:
        return _print_environment_check()
    if args.cleanup and not args.execute:
        if _print_environment_check():
            return 1
        generator = TrainingDataGenerator(
            repo=args.repo, repo_root=PROJECT_ROOT, gh_client=GhCliClient(args.repo),
            git_workspace=GitWorkspace(PROJECT_ROOT), store=GeneratedRunStore(Config.GENERATED_RUNS_PATH),
            poll_interval=args.poll_interval, timeout=args.timeout,
        )
        generator.generate([], cleanup=True)
        return 0
    scenarios, warnings, distribution = build_plan(
        args.success_runs, args.failure_runs, seed=args.seed, allow_large_run_set=args.allow_large_run_set
    )
    for warning in warnings:
        print(warning)
    planning_only = not args.execute
    print(format_execution_plan(total_runs=args.runs, success_runs=args.success_runs,
          failure_runs=args.failure_runs, distribution=distribution, dry_run=planning_only, seed=args.seed))
    if not args.no_write_plan and not args.dry_run:
        path = write_run_plan(Path(args.plan_path), scenarios=scenarios, distribution=distribution, seed=args.seed)
        print(f"\nPlan written to: {path}")
    if planning_only:
        return 0

    preflight_exit = _print_environment_check()
    if preflight_exit:
        return preflight_exit

    generator = TrainingDataGenerator(
        repo=args.repo, repo_root=PROJECT_ROOT, gh_client=GhCliClient(args.repo),
        git_workspace=GitWorkspace(PROJECT_ROOT), store=GeneratedRunStore(Config.GENERATED_RUNS_PATH),
        poll_interval=args.poll_interval, timeout=args.timeout,
    )
    progress = generator.generate(scenarios, cleanup=args.cleanup)
    report = None
    if args.collect:
        generator.collect_dataset(Config.HISTORICAL_DATASET_PATH)
        report = generator.inspect_dataset(Config.HISTORICAL_DATASET_PATH)
    training_exit = 0
    trained = False
    evaluated = False
    if args.train:
        if report is None:
            report = generator.inspect_dataset(Config.HISTORICAL_DATASET_PATH)
        if not report.sufficient_for_training:
            print("Training readiness: insufficient — " + "; ".join(report.warnings))
            training_exit = 1
        else:
            training_exit = generator.train_if_ready(Config.HISTORICAL_DATASET_PATH)
            trained = training_exit == 0
            evaluated = training_exit == 0
    print("\n" + summarize_generation(progress, report))
    print(f"\nModel:\n  trained: {'YES' if trained else 'NO'}")
    print(f"\nEvaluation:\n  performed: {'YES' if evaluated else 'NO'}")
    return training_exit or (1 if progress.stopped_early else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
