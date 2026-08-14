from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prediction.scenario_planner import (
    AIScenarioPlanner,
    FAILURE_CATEGORIES,
    FileChange,
    MAX_RUNS_DEFAULT,
    SUCCESS_CHANGE_TYPES,
    DeterministicScenarioPlanner,
    build_training_scenarios,
    distribute_failure_categories,
    is_path_allowed,
    summarize_scenario_distribution,
    validate_ai_scenario,
    validate_file_changes,
    validate_run_count,
)


def test_build_training_scenarios_counts():
    scenarios = build_training_scenarios(success_runs=4, failure_runs=5)
    assert len(scenarios) == 9
    assert sum(1 for scenario in scenarios if scenario.expected_outcome == "success") == 4
    assert sum(1 for scenario in scenarios if scenario.expected_outcome == "failure") == 5


def test_failure_distribution_is_balanced():
    categories = distribute_failure_categories(15)
    assert len(categories) == 15
    for category in FAILURE_CATEGORIES:
        assert categories.count(category) == 3


def test_success_scenarios_use_five_supported_change_types():
    scenarios = build_training_scenarios(success_runs=10, failure_runs=0)
    change_types = {scenario.change_type for scenario in scenarios}
    assert change_types.issubset(set(SUCCESS_CHANGE_TYPES))
    assert len(SUCCESS_CHANGE_TYPES) == 5


def test_scenario_schema_includes_description_and_required_fields():
    scenarios = build_training_scenarios(success_runs=1, failure_runs=1)
    for scenario in scenarios:
        payload = scenario.to_dict()
        assert payload["scenario_id"]
        assert payload["expected_outcome"] in {"success", "failure"}
        assert payload["change_type"]
        assert isinstance(payload["files_to_modify"], list)
        assert payload["description"]


def test_no_duplicate_scenario_ids():
    scenarios = build_training_scenarios(success_runs=5, failure_runs=10)
    ids = [scenario.scenario_id for scenario in scenarios]
    assert len(ids) == len(set(ids))


def test_branch_names_are_isolated_and_marked():
    scenarios = build_training_scenarios(success_runs=2, failure_runs=2)
    branches = [scenario.branch_name for scenario in scenarios]
    assert branches[0].startswith("ml-data/success-")
    assert branches[-1].startswith("ml-data/failure-")


def test_disallowed_paths_rejected():
    assert is_path_allowed("docs/ml-training/note.md") is True
    assert is_path_allowed(".github/workflows/ci.yml") is False
    assert is_path_allowed("data/historical_runs.csv") is False
    assert is_path_allowed(".env") is False


def test_validate_file_changes_rejects_forbidden_shell():
    changes = (
        FileChange(path="docs/ml-training/evil.md", content="rm -rf /\n"),
    )
    errors = validate_file_changes(changes)
    assert errors


def test_validate_ai_scenario_requires_allowlisted_paths():
    errors = validate_ai_scenario(
        {
            "scenario_id": "ai-1",
            "category": "lint",
            "files_to_modify": [".github/workflows/ci.yml"],
            "change_type": "documentation",
            "expected_outcome": "failure",
        }
    )
    assert any("disallowed path" in error for error in errors)


def test_validate_ai_scenario_rejects_shell_commands():
    errors = validate_ai_scenario(
        {
            "scenario_id": "ai-2",
            "category": "test",
            "files_to_modify": ["docs/ml-training/ai-2.md"],
            "change_type": "documentation",
            "expected_outcome": "failure",
            "shell_command": "rm -rf /",
        }
    )
    assert any("shell" in error.lower() for error in errors)


def test_invalid_failure_category_rejected_by_planner():
    errors = validate_ai_scenario(
        {
            "scenario_id": "bad",
            "category": "kubernetes",
            "files_to_modify": ["docs/ml-training/bad.md"],
            "change_type": "documentation",
            "expected_outcome": "failure",
        }
    )
    assert any("Invalid failure category" in error for error in errors)


def test_maximum_run_guard_requires_explicit_opt_in():
    with pytest.raises(ValueError, match=str(MAX_RUNS_DEFAULT)):
        validate_run_count(MAX_RUNS_DEFAULT + 1, allow_large_run_set=False)

    warnings = validate_run_count(MAX_RUNS_DEFAULT + 1, allow_large_run_set=True)
    assert warnings and "WARNING" in warnings[0]


def test_deterministic_scenario_generation_without_seed():
    first = build_training_scenarios(success_runs=3, failure_runs=3)
    second = build_training_scenarios(success_runs=3, failure_runs=3)
    assert [scenario.to_dict() for scenario in first] == [scenario.to_dict() for scenario in second]


def test_seed_changes_order_but_is_repeatable():
    first = DeterministicScenarioPlanner(seed=42).plan(3, 3)
    second = DeterministicScenarioPlanner(seed=42).plan(3, 3)
    unseeded = DeterministicScenarioPlanner().plan(3, 3)
    assert [scenario.scenario_id for scenario in first] == [scenario.scenario_id for scenario in second]
    assert [scenario.scenario_id for scenario in first] != [scenario.scenario_id for scenario in unseeded]


def test_distribution_summary_contains_both_outcomes():
    scenarios = build_training_scenarios(success_runs=4, failure_runs=6)
    distribution = summarize_scenario_distribution(scenarios)
    assert distribution["success"] == 4
    assert distribution["failure"] == 6
    assert sum(distribution["failures_by_category"].values()) == 6


def test_ai_scenario_planner_validates_suggestions():
    planner = AIScenarioPlanner(
        suggestions=[
            {
                "scenario_id": "ai-success-001",
                "category": "lint",
                "files_to_modify": ["docs/ml-training/ai-success-001.md"],
                "change_type": "documentation",
                "expected_outcome": "success",
                "description": "Validated AI suggestion",
            }
        ],
        seed=7,
    )
    scenarios = planner.plan(1, 1)
    assert len(scenarios) == 3
    assert any(scenario.scenario_id == "ai-success-001" for scenario in scenarios)
