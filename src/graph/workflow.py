"""
workflow.py - Hybrid Supervisor Pattern (Minimal LLM calls)
"""

import re
import time
from datetime import datetime
from typing import Literal
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state import (
    GraphState, 
    WorkflowPhase,
    create_initial_state,
)
from src.tools.github_loader import fetch_failed_build_logs
from src.tools.log_parser import parse_log_content, parse_log_file
from src.tools.commit_analyzer import analyze_culprit_for_repository
from src.prediction.category_mapper import map_failure_category
from src.agents.triage_agent import TriageAgent
from src.agents.research_agent import ResearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.utils.llm import LLMProviderUnavailable, get_llm_provider_status
from config import Config

load_dotenv()

MAX_FAILURES = 3
DELAY_BETWEEN_LLM_CALLS = 5  # seconds
_RUN_LOG_FILENAME = re.compile(r"^build_log_(\d+)\.txt$")


def _workflow_run_id_from_log_path(log_path: Path) -> int | None:
    """Recover the run ID embedded by the run-specific log downloader."""

    match = _RUN_LOG_FILENAME.fullmatch(log_path.name)
    return int(match.group(1)) if match else None


def _current_step(state: GraphState) -> str | None:
    """Return the next incomplete step, or ``None`` for terminal state."""

    if state.current_phase in {WorkflowPhase.COMPLETED, WorkflowPhase.FAILED}:
        return None
    if state.raw_log_content is None:
        return "ingest"
    if state.primary_error is None:
        return "parse"
    if state.triage_result is None:
        return "triage"
    if state.research_result is None:
        return "research"
    if state.debugging_brief is None:
        return "synthesize"
    return None


def _updated_failure_counts(state: GraphState, step: str, *, failed: bool) -> tuple[dict[str, int], int]:
    """Copy and update a retry counter without sharing state across runs."""

    counts = dict(state.failure_counts)
    count = counts.get(step, 0) + 1 if failed else 0
    counts[step] = count
    return counts, count


def hybrid_decide(state: GraphState) -> str:
    """
    Hybrid supervisor: Logic for obvious cases, no LLM needed.
    This reduces LLM calls from 9+ to just 3 (the agents themselves).
    """
    current_step = _current_step(state)
    if current_step is None:
        return "FINISH"
    
    # Check failure count
    if state.failure_counts.get(current_step, 0) >= MAX_FAILURES:
        print(f"[SUPERVISOR] {current_step} failed {MAX_FAILURES} times. Giving up.")
        return "FINISH"
    
    print(f"[SUPERVISOR] Decision: {current_step}")
    return current_step


def supervisor_node(state: GraphState) -> dict:
    """Supervisor using logic-based decisions."""
    decision = hybrid_decide(state)
    update = {
        "next_action": decision,
        "messages": [f"Supervisor: {decision}"]
    }

    # A FINISH decision for a non-terminal state means that the current step
    # exhausted its retry budget. Make the terminal outcome explicit instead
    # of returning a stale intermediate phase.
    if decision == "FINISH" and state.current_phase not in {
        WorkflowPhase.COMPLETED,
        WorkflowPhase.FAILED,
    }:
        current_step = _current_step(state) or "workflow"
        update.update({
            "current_phase": WorkflowPhase.FAILED,
            "completed_at": datetime.now(),
            "error_message": state.error_message or f"{current_step} failed after {MAX_FAILURES} attempts",
            "messages": [f"Supervisor: {current_step} exhausted retry budget"],
        })

    return update


def ingest_node(state: GraphState) -> dict:
    """Fetch build logs."""
    print("\n[INGEST] Fetching build logs...")
    
    try:
        log_path = fetch_failed_build_logs(state.repo_name)
        
        if log_path is None:
            counts, _ = _updated_failure_counts(state, "ingest", failed=True)
            return {
                "error_message": "No failed builds found",
                "current_phase": WorkflowPhase.FAILED,
                "completed_at": datetime.now(),
                "failure_counts": counts,
                "messages": ["Ingest: No failed builds"]
            }
        
        log_content = log_path.read_text(encoding='utf-8', errors='replace')
        workflow_run_id = _workflow_run_id_from_log_path(log_path)
        counts, _ = _updated_failure_counts(state, "ingest", failed=False)
        
        return {
            "raw_log_content": log_content,
            "log_file_path": str(log_path),
            "workflow_run_id": workflow_run_id,
            "current_phase": WorkflowPhase.PARSING,
            "error_message": None,
            "failure_counts": counts,
            "messages": [f"Ingest: OK ({len(log_content)} chars)"]
        }
    except Exception as e:
        counts, _ = _updated_failure_counts(state, "ingest", failed=True)
        return {
            "error_message": str(e),
            "failure_counts": counts,
            "messages": [f"Ingest error: {e}"],
        }


def parse_node(state: GraphState) -> dict:
    """Parse logs."""
    print("\n[PARSE] Parsing logs...")
    
    try:
        if state.log_file_path:
            result = parse_log_file(state.log_file_path)
        elif state.raw_log_content is not None:
            result = parse_log_content(state.raw_log_content)
        else:
            raise ValueError("No workflow log content is available to parse.")
        
        if not result.primary_error:
            counts, _ = _updated_failure_counts(state, "parse", failed=True)
            return {
                "error_message": "No errors in logs",
                "failure_counts": counts,
                "messages": ["Parse: No errors found"]
            }
        
        counts, _ = _updated_failure_counts(state, "parse", failed=False)
        return {
            "parse_result": result,
            "primary_error": result.primary_error,
            "current_phase": WorkflowPhase.TRIAGING,
            "error_message": None,
            "failure_counts": counts,
            "messages": [f"Parse: Found {result.error_count} error(s)"]
        }
    except Exception as e:
        counts, _ = _updated_failure_counts(state, "parse", failed=True)
        return {
            "error_message": str(e),
            "failure_counts": counts,
            "messages": [f"Parse error: {e}"],
        }


def triage_node(state: GraphState) -> dict:
    """Triage with delay and error handling."""
    print("\n[TRIAGE] Analyzing error...")
    print(f"[Rate Limit] Waiting {DELAY_BETWEEN_LLM_CALLS}s before LLM call...")
    time.sleep(DELAY_BETWEEN_LLM_CALLS)
    
    try:
        agent = TriageAgent()
        result = agent.analyze(state.primary_error)
        counts, _ = _updated_failure_counts(state, "triage", failed=False)
        
        return {
            "triage_result": result,
            "current_phase": WorkflowPhase.RESEARCHING,
            "error_message": None,
            "failure_counts": counts,
            "messages": [f"Triage: {result.severity.value}"]
        }
    except Exception as e:
        counts, count = _updated_failure_counts(state, "triage", failed=True)
        print(f"[TRIAGE] Failed (attempt {count}/{MAX_FAILURES}): {e}")
        return {
            "error_message": str(e),
            "failure_counts": counts,
            "messages": [f"Triage error: {e}"],
        }


def research_node(state: GraphState, *, github_token: str | None = None) -> dict:
    """Research with delay and error handling."""
    print("\n[RESEARCH] Finding solutions...")
    print(f"[Rate Limit] Waiting {DELAY_BETWEEN_LLM_CALLS}s before LLM call...")
    time.sleep(DELAY_BETWEEN_LLM_CALLS)
    
    try:
        agent = (
            ResearchAgent(repo_name=state.repo_name, github_token=github_token)
            if github_token
            else ResearchAgent(repo_name=state.repo_name)
        )
        result = agent.research(state.triage_result, state.primary_error)
        counts, _ = _updated_failure_counts(state, "research", failed=False)
        
        return {
            "research_result": result,
            "current_phase": WorkflowPhase.SYNTHESIZING,
            "error_message": None,
            "failure_counts": counts,
            "messages": [f"Research: {len(result.solutions)} solutions"]
        }
    except Exception as e:
        counts, count = _updated_failure_counts(state, "research", failed=True)
        print(f"[RESEARCH] Failed (attempt {count}/{MAX_FAILURES}): {e}")
        return {
            "error_message": str(e),
            "failure_counts": counts,
            "messages": [f"Research error: {e}"],
        }


def synthesize_node(state: GraphState, *, github_token: str | None = None) -> dict:
    """Synthesize with delay and error handling."""
    print("\n[SYNTHESIZE] Creating debugging brief...")
    print(f"[Rate Limit] Waiting {DELAY_BETWEEN_LLM_CALLS}s before LLM call...")
    time.sleep(DELAY_BETWEEN_LLM_CALLS)
    
    try:
        agent = SynthesisAgent()
        brief = agent.synthesize(
            state.primary_error,
            state.triage_result,
            state.research_result,
            state.repo_name
        )
        brief.workflow_run_id = state.workflow_run_id

        mapped_category = map_failure_category(
            log_parser_category=state.primary_error.error_category.value if state.primary_error else None,
            triage_category=state.triage_result.error_category_refined.value if state.triage_result else None,
            failed_step=state.primary_error.failed_step if state.primary_error else None,
            error_type=state.primary_error.error_type if state.primary_error else None,
            error_message=state.primary_error.error_message if state.primary_error else None,
        )
        brief.error_category = mapped_category.value

        # Culprit ranking is an optional, deterministic enhancement. A GitHub
        # metadata error must not discard an otherwise complete AI brief.
        try:
            culprit = analyze_culprit_for_repository(
                repo_name=state.repo_name,
                parsed_error=state.primary_error,
                error_category=mapped_category.value,
                failed_step=state.primary_error.failed_step if state.primary_error else None,
                workflow_run_id=state.workflow_run_id,
                github_token=github_token,
            )
            if culprit.commit_sha:
                brief.likely_culprit_sha = culprit.commit_sha
                brief.likely_culprit_message = culprit.commit_message
                brief.likely_culprit_confidence = culprit.confidence
                brief.likely_culprit_evidence = culprit.evidence
        except Exception:
            pass

        counts, _ = _updated_failure_counts(state, "synthesize", failed=False)
        
        return {
            "debugging_brief": brief,
            "current_phase": WorkflowPhase.COMPLETED,
            "completed_at": datetime.now(),
            "error_message": None,
            "failure_counts": counts,
            "messages": [f"Synthesize: {len(brief.fix_suggestions)} fixes"]
        }
    except Exception as e:
        counts, count = _updated_failure_counts(state, "synthesize", failed=True)
        print(f"[SYNTHESIZE] Failed (attempt {count}/{MAX_FAILURES}): {e}")
        return {
            "error_message": str(e),
            "failure_counts": counts,
            "messages": [f"Synthesize error: {e}"],
        }


def route_from_supervisor(state: GraphState) -> Literal[
    "ingest", "parse", "triage", "research", "synthesize", "__end__"
]:
    """Route based on supervisor decision."""
    decision = state.next_action
    
    if decision == "FINISH":
        return "__end__"
    elif decision in ["ingest", "parse", "triage", "research", "synthesize"]:
        return decision
    return "__end__"


def create_workflow(*, github_token: str | None = None) -> StateGraph:
    """Create the workflow graph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("parse", parse_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("research", lambda state: research_node(state, github_token=github_token))
    workflow.add_node("synthesize", lambda state: synthesize_node(state, github_token=github_token))
    
    workflow.add_edge(START, "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "ingest": "ingest",
            "parse": "parse",
            "triage": "triage",
            "research": "research",
            "synthesize": "synthesize",
            "__end__": END
        }
    )
    
    workflow.add_edge("ingest", "supervisor")
    workflow.add_edge("parse", "supervisor")
    workflow.add_edge("triage", "supervisor")
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("synthesize", "supervisor")
    
    return workflow.compile()


def run_analysis(repo_name: str) -> GraphState:
    """Run analysis."""
    print("\n" + "="*60)
    print("CI/CD ROOT CAUSE ANALYZER")
    print("="*60)
    print(f"Repository: {repo_name}")
    print(f"Max retries per step: {MAX_FAILURES}")
    print(f"Delay between LLM calls: {DELAY_BETWEEN_LLM_CALLS}s")
    print("="*60)
    
    initial_state = create_initial_state(repo_name)
    workflow = create_workflow()
    
    final_state = workflow.invoke(initial_state)
    
    if isinstance(final_state, dict):
        final_state = GraphState(**final_state)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    return final_state


def run_enriched_analysis(
    repo_name: str,
    raw_log_content: str,
    *,
    workflow_run_id: int | None = None,
    github_token: str | None = None,
) -> GraphState:
    """Run provider-backed enrichment on already-fetched, in-memory logs.

    The dashboard uses this entry point after the bounded GitHub automation
    service downloads and redacts a log archive. It avoids writing raw logs to
    disk and does not place authentication tokens in graph state.
    """

    if not str(raw_log_content or "").strip():
        raise ValueError("GitHub returned no readable workflow log content.")
    provider = get_llm_provider_status()
    if not provider.ready:
        raise LLMProviderUnavailable(provider.detail)

    initial_state = create_initial_state(repo_name).model_copy(
        update={
            "raw_log_content": raw_log_content,
            "workflow_run_id": workflow_run_id,
            "current_phase": WorkflowPhase.PARSING,
            "messages": [f"Workflow initialized from GitHub run {workflow_run_id or 'unknown'}"],
        }
    )
    final_state = create_workflow(github_token=github_token).invoke(initial_state)
    return GraphState(**final_state) if isinstance(final_state, dict) else final_state


if __name__ == "__main__":
    TEST_REPO = Config.DEFAULT_TEST_REPO
    
    try:
        final_state = run_analysis(TEST_REPO)
        
        if final_state.debugging_brief:
            output_dir = Config.OUTPUT_DIR
            output_dir.mkdir(exist_ok=True)
            
            md_path = output_dir / "debugging_brief.md"
            md_path.write_text(
                final_state.debugging_brief.to_markdown(), 
                encoding='utf-8'
            )
            print(f"\nSaved: {md_path}")
        else:
            print("\nNo brief generated.")
            if final_state.error_message:
                print(f"Error: {final_state.error_message}")
            
    except Exception as e:
        print(f"\nFailed: {e}")
