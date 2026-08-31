from __future__ import annotations

from types import SimpleNamespace

from src.prediction.data_collector import HistoricalRunCollector


def _run(name: str, conclusion: str):
    return SimpleNamespace(
        name=name,
        workflow=SimpleNamespace(name=name),
        conclusion=conclusion,
    )


def test_collector_excludes_product_automation_workflows_by_default():
    assert HistoricalRunCollector._is_trainable_run(_run("CI/CD Pipeline", "success"))
    assert HistoricalRunCollector._is_trainable_run(
        _run("Controlled CI Failure Generator (Dev/Testing Only)", "failure")
    )
    assert not HistoricalRunCollector._is_trainable_run(
        _run("Pre-CI Failure Prediction", "success")
    )
    assert not HistoricalRunCollector._is_trainable_run(
        _run("Prediction Feedback", "success")
    )
    assert not HistoricalRunCollector._is_trainable_run(
        _run("Train Failure Predictor", "success")
    )


def test_collector_excludes_non_binary_workflow_outcomes():
    for conclusion in ("cancelled", "skipped", "neutral", "stale", None):
        assert not HistoricalRunCollector._is_trainable_run(
            _run("CI/CD Pipeline", conclusion)
        )


def test_system_workflows_can_be_explicitly_included_for_audit_exports():
    assert HistoricalRunCollector._is_trainable_run(
        _run("Prediction Feedback", "success"),
        include_system_workflows=True,
    )
