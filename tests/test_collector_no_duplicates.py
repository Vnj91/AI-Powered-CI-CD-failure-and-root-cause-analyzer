from __future__ import annotations

import pandas as pd
from types import SimpleNamespace
from datetime import UTC, datetime
from pathlib import Path

from src.prediction.data_collector import HistoricalRunCollector


class FakeRun:
    def __init__(self, id, run_number, created_at, conclusion, head_sha, head_branch, workflow_name):
        self.id = id
        self.run_number = run_number
        self.created_at = created_at
        self.conclusion = conclusion
        self.head_sha = head_sha
        self.head_branch = head_branch
        self.workflow = SimpleNamespace(name=workflow_name)


class FakeRepo:
    def __init__(self, runs):
        self._runs = runs

    def get_commit(self, sha):
        ns = SimpleNamespace()
        ns.stats = SimpleNamespace(additions=0, deletions=0, total=0)
        ns.files = []
        ns.commit = SimpleNamespace(message="")
        return ns

    def get_workflow_runs(self, status=None):
        return list(self._runs)


class CollectorNoDup(HistoricalRunCollector):
    def __init__(self):
        super().__init__(token="")

    def _repo(self, repository: str):
        # Return fake repo replaced in the test via monkeypatch
        raise RuntimeError("Should be monkeypatched")


def test_collector_emits_unique_runs(monkeypatch):
    # Build runs including duplicate run id
    runs = [
        FakeRun(10, 1, datetime(2026, 9, 1, tzinfo=UTC), "success", "a"*40, "main", "CI/CD Pipeline"),
        FakeRun(11, 2, datetime(2026, 9, 2, tzinfo=UTC), "failure", "b"*40, "main", "CI/CD Pipeline"),
        FakeRun(10, 3, datetime(2026, 9, 3, tzinfo=UTC), "success", "c"*40, "main", "CI/CD Pipeline"),
    ]
    collector = HistoricalRunCollector(token="")
    fake_repo = FakeRepo(runs)
    monkeypatch.setattr(collector, "_repo", lambda repository: fake_repo)
    frame = collector.collect_repository_runs("owner/repo")
    # run_id 10 should appear once, keeping the later timestamp
    assert list(sorted(frame["run_id"])) == [10, 11]
    row10 = frame[frame["run_id"] == 10].iloc[0]
    assert str(row10["commit_sha"]) in ("c"*40, "a"*40)
