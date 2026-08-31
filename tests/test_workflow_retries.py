from __future__ import annotations

import pytest

from config import Config
from src.graph.state import WorkflowPhase, create_initial_state
from src.graph.workflow import (
    MAX_FAILURES,
    hybrid_decide,
    ingest_node,
    parse_node,
    run_enriched_analysis,
    supervisor_node,
)
from src.utils.llm import LLMProviderUnavailable


def test_retry_counts_are_isolated_per_graph_state():
    first = create_initial_state("owner/first")
    second = create_initial_state("owner/second")
    first.failure_counts["ingest"] = MAX_FAILURES

    assert hybrid_decide(first) == "FINISH"
    assert hybrid_decide(second) == "ingest"
    assert second.failure_counts == {}


def test_retry_exhaustion_marks_workflow_failed():
    state = create_initial_state("owner/repo")
    state.failure_counts["ingest"] = MAX_FAILURES
    state.error_message = "GitHub unavailable"

    update = supervisor_node(state)

    assert update["next_action"] == "FINISH"
    assert update["current_phase"] == WorkflowPhase.FAILED
    assert update["completed_at"] is not None
    assert update["error_message"] == "GitHub unavailable"


def test_no_failed_build_is_terminal_without_retries(monkeypatch):
    state = create_initial_state("owner/repo")
    monkeypatch.setattr("src.graph.workflow.fetch_failed_build_logs", lambda repository: None)

    update = ingest_node(state)

    assert update["current_phase"] == WorkflowPhase.FAILED
    assert update["completed_at"] is not None
    assert update["error_message"] == "No failed builds found"


def test_ingest_records_the_exact_failed_run_id(monkeypatch, tmp_path):
    state = create_initial_state("owner/repo")
    log_path = tmp_path / "build_log_987654.txt"
    log_path.write_text("Process completed with exit code 1.\n", encoding="utf-8")
    monkeypatch.setattr("src.graph.workflow.fetch_failed_build_logs", lambda repository: log_path)

    update = ingest_node(state)

    assert update["workflow_run_id"] == 987654
    assert update["log_file_path"] == str(log_path)


def test_parse_node_accepts_bounded_in_memory_github_logs():
    state = create_initial_state("owner/repo").model_copy(
        update={
            "raw_log_content": "ModuleNotFoundError: No module named 'example'\n",
            "current_phase": WorkflowPhase.PARSING,
        }
    )

    update = parse_node(state)

    assert update["primary_error"].error_type == "ModuleNotFoundError"
    assert update["current_phase"] == WorkflowPhase.TRIAGING


def test_enriched_analysis_fails_fast_when_no_provider_is_configured(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "none")

    with pytest.raises(LLMProviderUnavailable, match="deterministic RCA"):
        run_enriched_analysis("owner/repo", "Process completed with exit code 1.")
