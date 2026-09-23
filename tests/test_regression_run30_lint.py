from src.tools.log_parser import parse_log_content
from src.tools.log_parser import ErrorCategory


def test_run30_ruff_failure_detected():
    # Snippet from run #30 Lint & Static Checks job where ruff reports many errors
    snippet = (
        "2026-09-23T12:47:18.6417382Z Found 3404 errors.\n"
        "2026-09-23T12:47:18.6417613Z [*] 94 fixable with the `--fix` option (570 hidden fixes can be enabled with the `--unsafe-fixes` option).\n"
        "2026-09-23T12:47:18.6426761Z ##[error]Process completed with exit code 1.\n"
    )

    result = parse_log_content(snippet)
    assert result.primary_error is not None
    # Ensure we capture a ProcessExitError fallback if no python exception
    assert result.primary_error.error_type in ("ProcessExitError", "Error", "GitHubActionsError")
    # But category should reflect lint when message mentions ruff/fixable
    assert result.primary_error.error_category == ErrorCategory.LINT
