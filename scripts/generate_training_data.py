#!/usr/bin/env python3
# ruff: noqa: E402
"""Plan real GitHub Actions training-data scenarios for the failure predictor.

Phase 1 is planning-only: no branches, commits, or workflow dispatches are performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.prediction.training_data_generator import (
    build_plan,
    format_execution_plan,
    write_run_plan,
)

DEFAULT_REPO = "Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer"
PHASE_1_PLANNING_ONLY = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan GitHub Actions training-data scenarios (Phase 1: planning only).",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Target repository in owner/name format (reserved for future execution phases)",
    )
    parser.add_argument("--runs", type=int, default=30, help="Total runs to plan")
    parser.add_argument("--success-runs", type=int, default=15, help="Successful CI runs to plan")
    parser.add_argument("--failure-runs", type=int, default=15, help="Controlled failure runs to plan")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for deterministic scenario ordering")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing execution artifacts")
    parser.add_argument(
        "--plan-path",
        default=str(Config.DATA_DIR / "generated_run_plan.json"),
        help="Output path for the generated run plan (gitignored)",
    )
    parser.add_argument(
        "--no-write-plan",
        action="store_true",
        help="Do not write data/generated_run_plan.json",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Reserved for a future execution phase (ignored in Phase 1)",
    )
    parser.add_argument(
        "--allow-large-run-set",
        action="store_true",
        help="Allow more than 50 planned runs",
    )
    return parser.parse_args()


def _validate_counts(args: argparse.Namespace) -> None:
    if args.success_runs + args.failure_runs != args.runs:
        raise SystemExit(
            f"--runs ({args.runs}) must equal --success-runs ({args.success_runs}) + "
            f"--failure-runs ({args.failure_runs})."
        )


def main() -> int:
    args = parse_args()
    _validate_counts(args)

    if args.cleanup and PHASE_1_PLANNING_ONLY:
        print("Note: --cleanup is ignored in Phase 1 (planning only).")

    scenarios, warnings, distribution = build_plan(
        args.success_runs,
        args.failure_runs,
        seed=args.seed,
        allow_large_run_set=args.allow_large_run_set,
    )
    for warning in warnings:
        print(warning)

    plan_text = format_execution_plan(
        total_runs=args.runs,
        success_runs=args.success_runs,
        failure_runs=args.failure_runs,
        distribution=distribution,
        dry_run=args.dry_run or PHASE_1_PLANNING_ONLY,
        seed=args.seed,
    )
    print(plan_text)

    if not args.no_write_plan and not args.dry_run:
        plan_path = write_run_plan(
            Path(args.plan_path),
            scenarios=scenarios,
            distribution=distribution,
            seed=args.seed,
        )
        print(f"\nPlan written to: {plan_path}")

    if PHASE_1_PLANNING_ONLY:
        return 0

    raise SystemExit("Workflow execution is not enabled in Phase 1.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
