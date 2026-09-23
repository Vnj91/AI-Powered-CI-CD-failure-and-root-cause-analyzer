from __future__ import annotations

import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from app import _atomic_save_csv


def make_frame(run_ids):
    rows = []
    for i, rid in enumerate(run_ids):
        rows.append({
            "run_id": int(rid),
            "timestamp": f"2026-09-01 0{i}:00:00+00:00",
            "actual_failure": 1 if i % 2 == 0 else 0,
        })
    return pd.DataFrame(rows)


def test_atomic_save_merges_and_dedups():
    with TemporaryDirectory() as td:
        path = Path(td) / "historical_runs.csv"
        f1 = make_frame([1, 2, 3])
        _atomic_save_csv(f1, path)
        assert path.exists()
        df1 = pd.read_csv(path)
        assert len(df1) == 3

        # Save overlapping frame with duplicate run_id 2 and new run_id 4
        f2 = make_frame([2, 4])
        # Update timestamp for run_id 2 to be later to ensure keep last
        f2.loc[f2["run_id"] == 2, "timestamp"] = "2026-09-02 00:00:00+00:00"
        _atomic_save_csv(f2, path)
        df2 = pd.read_csv(path)
        # Expect run_ids 1,3,2,4 -> but only unique run_id rows
        assert set(df2["run_id"]) == {1, 2, 3, 4}
        # run_id 2 should have the later timestamp
        row2 = df2[df2["run_id"] == 2].iloc[0]
        assert "2026-09-02" in row2["timestamp"]

        # Re-run save with same incoming rows should not increase counts
        _atomic_save_csv(f2, path)
        df3 = pd.read_csv(path)
        assert len(df3) == len(df2)
