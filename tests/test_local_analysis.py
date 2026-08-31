from __future__ import annotations

from src.analysis import build_local_debugging_brief
from src.tools.log_parser import parse_log_content


def test_local_analysis_builds_three_actionable_dependency_fixes():
    parsed = parse_log_content(
        """##[group]Run pytest -q
Traceback (most recent call last):
  File "/workspace/tests/test_api.py", line 42, in test_health
    import requests_mock
ModuleNotFoundError: No module named 'requests_mock'
##[error]Process completed with exit code 1.
"""
    )

    brief = build_local_debugging_brief(parsed.primary_error, "owner/repo")

    assert brief.repository == "owner/repo"
    assert brief.error_category == "dependency"
    assert len(brief.fix_suggestions) == 3
    assert all(fix.implementation_steps for fix in brief.fix_suggestions)
    assert all(fix.source == "deterministic_local_guidance" for fix in brief.fix_suggestions)
    assert "no log content was sent" in (brief.research_summary or "")


def test_local_analysis_handles_exit_code_only_evidence_conservatively():
    parsed = parse_log_content(
        """##[group]Run ./build.sh
Process completed with exit code 17.
"""
    )

    brief = build_local_debugging_brief(parsed.primary_error)

    assert brief.error_type == "ProcessExitError"
    assert brief.error_category == "unknown"
    assert brief.confidence_score < 0.5
    assert "first concrete diagnostic" in brief.root_cause_detailed.lower()


def test_local_debugging_brief_exports_markdown():
    parsed = parse_log_content("TimeoutError: deployment timed out\n")
    brief = build_local_debugging_brief(parsed.primary_error)

    markdown = brief.to_markdown()

    assert "CI/CD Debugging Brief" in markdown
    assert "Fix #1" in markdown
    assert "TimeoutError" in markdown
