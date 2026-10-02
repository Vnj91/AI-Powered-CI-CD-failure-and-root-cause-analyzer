from __future__ import annotations

from pathlib import Path

import pytest

import scripts.generate_prediction_runs as generator
from scripts.generate_prediction_runs import _normalize_status, _status_is_clean


ALLOWED_DATA_FILES = {
    "data/generated_run_plan.json",
    "data/generated_runs.json",
    "data/historical_runs.csv",
    "data/prediction_history.csv",
    "data/prediction_run_marker.txt",
}


def test_normalize_status_handles_blank_and_dirty_outputs():
    assert _normalize_status("\n") == []
    assert _normalize_status(" M data/prediction_run_marker.txt\n") == ["M data/prediction_run_marker.txt"]
    assert _normalize_status(" M data/prediction_run_marker.txt\n?? report.md\n") == [
        "M data/prediction_run_marker.txt",
        "?? report.md",
    ]


def test_status_is_clean_ignores_only_allowed_paths():
    status = [
        "?? .tools/capture_pytest.py",
        "?? data/prediction_run_marker.txt",
        "?? report.md",
        "?? reportdata.md",
    ]
    assert _status_is_clean(status) is True

    for allowed in list(ALLOWED_DATA_FILES):
        assert _status_is_clean([f"?? {allowed}"]) is True

    assert _status_is_clean(["?? data/unexpected.txt"]) is False
    assert _status_is_clean(["?? other.txt"]) is False
    assert _status_is_clean([" M scripts/generate_prediction_runs.py"]) is False

    bad_data = ["?? data/unexpected.txt", "?? data/prediction_run_marker.txt"]
    assert _status_is_clean(bad_data) is False
