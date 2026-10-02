from __future__ import annotations

from pathlib import Path

import pytest

import scripts.generate_prediction_runs as generator
from scripts.generate_prediction_runs import _normalize_status, _status_is_clean


def test_normalize_status_handles_blank_and_dirty_outputs():
    assert _normalize_status("\n") == []
    assert _normalize_status(" M data/prediction_run_marker.txt\n") == ["M data/prediction_run_marker.txt"]
    assert _normalize_status(" M data/prediction_run_marker.txt\n?? report.md\n") == [
        "M data/prediction_run_marker.txt",
        "?? report.md",
    ]


def test_status_is_clean_ignores_only_allowed_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "prediction_run_marker.txt").write_text("marker\n", encoding="utf-8")

    assert _status_is_clean([" M data/prediction_run_marker.txt"]) is True
    assert _status_is_clean(["?? .tools/"]) is True
    assert _status_is_clean(["?? report.md"]) is True
    assert _status_is_clean(["?? reportdata.md"]) is True
    assert _status_is_clean(["?? data/", "?? data/prediction_run_marker.txt"]) is True

    mixed = ["M data/prediction_run_marker.txt", "?? report.md", "?? notes.txt"]
    assert _status_is_clean(mixed) is False

    other = ["M src/prediction/history_store.py"]
    assert _status_is_clean(other) is False

    untracked = ["?? temp.log"]
    assert _status_is_clean(untracked) is False

    bad_data_dir = tmp_path / "data"
    (bad_data_dir / "other.txt").write_text("nope\n", encoding="utf-8")
    assert _status_is_clean(["?? data/", "?? data/other.txt"]) is False
