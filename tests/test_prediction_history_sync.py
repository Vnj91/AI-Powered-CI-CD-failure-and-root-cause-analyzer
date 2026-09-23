import io
import zipfile
import pandas as pd
from src.integrations.github_automation import GitHubAutomationService
from src.prediction.history_store import save_prediction_history, load_prediction_history
from config import Config


def make_artifact_bytes(csv_bytes: bytes) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('data/prediction_history.csv', csv_bytes)
    return bio.getvalue()


def test_prediction_history_download_and_merge(tmp_path, monkeypatch):
    # Existing local history
    existing = pd.DataFrame([
        {'prediction_id': 'a', 'failure_probability': 0.1},
        {'prediction_id': 'b', 'failure_probability': 0.2},
    ])
    save_prediction_history(existing, tmp_path / 'history.csv')

    # Downloaded finalized history with updated 'b' and new 'c'
    downloaded = pd.DataFrame([
        {'prediction_id': 'b', 'failure_probability': 0.9},
        {'prediction_id': 'c', 'failure_probability': 0.8},
    ])
    csv_bytes = downloaded.to_csv(index=False).encode('utf-8')
    artifact = make_artifact_bytes(csv_bytes)

    # monkeypatch GitHubAutomationService.download_latest_prediction_history to write the artifact to dest
    def fake_download(self, repository, dest_path, artifact_name='prediction-history', expected_member='data/prediction_history.csv', max_runs=40):
        Path(dest_path).write_bytes(downloaded.to_csv(index=False).encode('utf-8'))
        return True

    from pathlib import Path
    monkeypatch.setattr(GitHubAutomationService, 'download_latest_prediction_history', fake_download)

    # Call attempt function and verify merge
    # Ensure the dashboard uses the test file as the canonical prediction history
    from config import Config
    Config.PREDICTION_HISTORY_PATH = tmp_path / 'history.csv'

    from app import _attempt_download_prediction_history
    success = _attempt_download_prediction_history('owner/repo', 'fake-token')
    assert success
    merged = load_prediction_history(Config.PREDICTION_HISTORY_PATH)
    assert 'a' in set(merged['prediction_id'])
    assert 'b' in set(merged['prediction_id'])
    assert 'c' in set(merged['prediction_id'])
    # b must be updated to 0.9 from downloaded
    assert float(merged[merged['prediction_id']=='b']['failure_probability'].iloc[0]) == 0.9
