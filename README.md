# CI/CD Root Cause Analyzer

This project now has two layers:

1. CI/CD failure risk prediction before a workflow finishes.
2. Existing root cause analysis after a failure occurs.

The RCA pipeline is unchanged in spirit: GitHub Actions logs are fetched, parsed, triaged, researched, and synthesized into a debugging brief.

## What Changed

- Added historical workflow-run collection for supervised learning.
- Added deterministic feature extraction for workflow and commit metadata.
- Added a baseline Random Forest failure predictor with saved model artifacts.
- Added an optional failure-category classifier when enough labeled historical failures exist.
- Added prediction history storage so predictions can be compared with actual outcomes later.
- Extended the Streamlit UI with a separate failure-risk prediction panel.

## Architecture

### Existing RCA flow

GitHub Repository -> Failed GitHub Actions Run -> Fetch Logs -> Parse Error -> Triage -> Research -> Synthesis

### New prediction flow

Historical GitHub Actions runs -> Feature extraction -> Random Forest model -> Failure probability + risk factors

### Combined flow

Commit / PR -> Prediction -> GitHub Actions -> If failed, RCA -> Fix suggestions

## New Modules

- `src/prediction/data_collector.py`
- `src/prediction/feature_extractor.py`
- `src/prediction/trainer.py`
- `src/prediction/predictor.py`
- `src/prediction/evaluator.py`
- `src/prediction/history_store.py`
- `src/prediction/service.py`
- `src/prediction/cli.py`

## Tech Stack

- Python 3.11
- LangGraph
- Claude 3.5 Sonnet via AWS Bedrock
- Tavily API
- PyGithub
- Streamlit
- pandas / numpy / scikit-learn / joblib

## Setup

1. Copy `.env.example` to `.env`.
2. Set `GITHUB_ACCESS_TOKEN`, `TAVILY_API_KEY`, and AWS credentials.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the UI:

```bash
streamlit run app.py
```

## Collect Historical Data

Use the prediction CLI to collect workflow history into a CSV dataset.

```bash
python3 -m src.prediction.cli collect owner/repo --output data/historical_runs.csv
```

This dataset is one training example per workflow run.

## Train the Failure Model

```bash
python3 -m src.prediction.cli train --dataset data/historical_runs.csv --model models/failure_predictor.joblib --metadata models/failure_predictor_metadata.json
```

The trainer uses a time-based split so newer runs are held out for validation.
If enough labeled failures exist, it also writes `models/failure_category_predictor.joblib` and a matching metadata file.

## Evaluate the Model

```bash
python3 -m src.prediction.cli evaluate --dataset data/historical_runs.csv --model models/failure_predictor.joblib
```

The evaluator reports accuracy, precision, recall, F1, ROC-AUC when available, and a confusion matrix.

## Run Prediction

```bash
python3 -m src.prediction.cli predict owner/repo <commit_sha> --branch main --workflow build
```

If branch or workflow are omitted, the app falls back to the latest available workflow run when possible.

If the category model was trained, the UI and CLI will also show a predicted failure category with confidence.

## Run RCA

```bash
python3 -m src.main owner/repo
```

or launch the Streamlit UI:

```bash
streamlit run app.py
```

## Environment Variables

- `GITHUB_ACCESS_TOKEN`
- `TAVILY_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

## Minimum Dataset Guidance

This is a baseline model, not a production-grade predictor. Expect noisy results with very small datasets.

- Below 50 runs: weak signal.
- Around 100-300 runs: usable baseline.
- More than 500 runs: materially better for workflow-level patterns.

Class imbalance is common, so precision, recall, and F1 matter more than raw accuracy.

## Limitations

- Failure category prediction is intentionally optional and not forced if the training data is insufficient.
- Predictions depend on historical run metadata that may be incomplete in some repositories.
- The model does not replace the existing LLM-based RCA path.
- The app does not fabricate confidence where the model does not support it.

## Notes

The UI now shows a failure-risk section before the existing RCA output. If a failure has already happened, the RCA pipeline still runs exactly as before.


