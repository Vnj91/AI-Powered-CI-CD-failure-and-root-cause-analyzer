import io
import zipfile
from src.integrations.github_automation import GitHubAutomationService, DEFAULT_LOG_LIMIT_BYTES
from src.tools.log_parser import parse_log_content
from config import Config


def make_large_log(head: str, tail: str, total_size: int) -> bytes:
    # Construct a bytes object where head at start and tail near the end
    head_b = head.encode('utf-8')
    tail_b = tail.encode('utf-8')
    if len(head_b) + len(tail_b) >= total_size:
        # Just concatenate
        return head_b + tail_b
    middle_size = total_size - (len(head_b) + len(tail_b))
    middle = b"x" * middle_size
    return head_b + middle + tail_b


def test_truncation_preserves_tail(tmp_path):
    # Prepare a synthetic zip archive with one very large log file whose
    # final lines contain a lint-like failure summary.
    head = "Header: Starting lint run\nAll good until here\n"
    tail = "Found 3404 errors.\n94 fixable with the `--fix` option.\n##[error]Process completed with exit code 1.\n"
    total_uncompressed = DEFAULT_LOG_LIMIT_BYTES + 600_000  # force truncation
    content_bytes = make_large_log(head, tail, total_uncompressed)

    # Create an in-memory zip
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("8_Lint & Static Checks/5_Run ruff.txt", content_bytes)
    archive_bytes.seek(0)

    # Use the service's decode logic by calling the private _decode_log_archive
    # We instantiate a service and call the staticmethod
    service = GitHubAutomationService(token=Config.GITHUB_ACCESS_TOKEN, repository=Config.DEFAULT_TEST_REPO)
    payload = archive_bytes.getvalue()
    files = service._decode_log_archive(payload, max_bytes=DEFAULT_LOG_LIMIT_BYTES)

    # Ensure we stayed under the limit
    total = sum(f.size_bytes for f in files)
    assert total <= DEFAULT_LOG_LIMIT_BYTES

    # Confirm truncation notice present
    names = [f.name for f in files]
    assert any("TRUNCATED_NOTICE" == n for n in names)

    # Find the truncated lint file and ensure tail snippet is present
    lint_file = None
    for f in files:
        if 'ruff' in f.name.lower() or 'lint' in f.name.lower():
            lint_file = f
            break
    assert lint_file is not None
    content = lint_file.content

    # The tail summary must be present in the preserved content
    assert 'Found 3404 errors' in content or 'Process completed with exit code 1' in content

    # Finally, parse the combined text to ensure parser can detect the lint
    combined = "\n\n".join(f"{'='*72}\nLOG FILE: {fi.name}\n{'='*72}\n{fi.content}" for fi in files)
    parsed = parse_log_content(combined)
    assert parsed.primary_error is not None
    assert parsed.primary_error.error_category.value == 'lint'
    assert parsed.primary_error.failed_step is not None or parsed.primary_error.exit_code == 1
