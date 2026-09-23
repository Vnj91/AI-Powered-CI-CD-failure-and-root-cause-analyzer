Summary of fixes: idempotent ingestion and analytics filtering

Root cause
- Duplicate `run_id` rows could appear when the activation boundary overwrote or combined datasets without deterministic deduplication.
- Analytics used the raw dataset which included system/test workflows, skewing production failure rates.

Changes implemented
- `app._atomic_save_csv` now merges on-disk dataset with incoming frame and deduplicates by `run_id`, keeping the most recent row by `timestamp` before atomically replacing the file.
- `HistoricalRunCollector.collect_repository_runs` now deduplicates its produced DataFrame by `run_id` (keep latest `timestamp`) to ensure the collector itself produces idempotent output.
- Added tests:
  - `tests/test_collect_idempotent.py` verifies `_atomic_save_csv` merging and idempotency.
  - `tests/test_collector_no_duplicates.py` verifies `HistoricalRunCollector.collect_repository_runs` dedup behavior on duplicate run inputs.

Verification performed
- Ran focused tests: the two new tests passed.
- Ran full test suite: `pytest` showed `146 passed`.
- Ran dataset validation and atomic save on `data/historical_runs.csv`: the dataset had 30 rows and 0 duplicate `run_id`s before and after (no changes required for this dataset).

Notes and recommendations
- The dedup policy keeps the most recent `timestamp` for a `run_id`. If you prefer to keep the earliest or prefer identical-row dedup only, we can change the policy accordingly.
- Analytics filtering to exclude `SYSTEM_WORKFLOW_NAMES` was planned; if you want I can implement explicit analytics view changes in `app.render_analytics` to surface "Raw runs" vs "Production runs (system workflows excluded)".

Next steps (optional)
- Commit changes and open a PR.
- Implement UI analytics filtering and add tests for analytics exclusion.
- Add a changelog entry and version bump.
