#  CI/CD Root Cause Analyzer

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

```
Developer
    ↓
Commit / PR
    ↓
Pre-CI Prediction (.github/workflows/predict.yml + CLI/UI)
    ↓
GitHub Actions CI (.github/workflows/ci.yml)
    ├── Lint
    ├── Unit Tests
    ├── Prediction Tests
    ├── Security
    ├── Build
    └── Docker
    ↓
Workflow Result
    ↓
Prediction Feedback (prediction_history.csv)
    ↓
If failure → RCA (LangGraph)
    ├── Triage (LLM)
    ├── Research (LLM + Tavily)
    ├── Root Cause + Fixes (LLM)
    └── Likely Culprit Commit (deterministic)
    ↓
Historical Dataset (data/historical_runs.csv)
    ↓
Future Model Training
```

### ML prediction flow (before CI completes)

Historical GitHub Actions runs → feature extraction → Random Forest → failure probability + optional category + feature importance

### RCA flow (after failure)

GitHub Repository → failed run logs → parse/classify error → triage → research → synthesis → debugging brief + culprit commit evidence

### Combined product flow

Commit / PR → predict risk → CI runs → record actual outcome → if failed, run RCA → collect more labeled history → retrain

## Modules

### Prediction layer
- `src/prediction/data_collector.py` — historical run collection
- `src/prediction/feature_extractor.py` — deterministic features
- `src/prediction/category_mapper.py` — expanded failure category taxonomy
- `src/prediction/dataset_validator.py` — dataset quality + leakage checks
- `src/prediction/trainer.py` / `predictor.py` / `evaluator.py`
- `src/prediction/history_store.py` / `feedback.py` / `service.py`
- `src/prediction/cli.py`

### RCA layer (unchanged core)
- `src/graph/workflow.py` — LangGraph supervisor workflow
- `src/agents/` — triage, research, synthesis
- `src/tools/log_parser.py`, `src/tools/github_loader.py`
- `src/tools/commit_analyzer.py` — likely culprit commit scoring

## Data Safety / Leakage Prevention

Target/metadata columns are **never** model input features:

- `actual_failure`, `actual_category`, `status`, `conclusion`, `duration_seconds`

Historical features for a target run are computed only from **prior** completed runs. The target run itself is excluded from its own history window. Pre-CI prediction uses commit metadata + prior run history only — never the target run's final conclusion.

Inspect any dataset before training:

```bash
python3 -m src.prediction.cli inspect --dataset data/historical_runs.csv
```

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

## Automated CI Training Data Generation

Phase 2 converts the deterministic scenarios into **real GitHub Actions workflow runs**. It never writes fabricated rows to `data/historical_runs.csv`: generated-run metadata is kept separately in the gitignored `data/generated_runs.json`, and `--collect` retrieves actual GitHub history.

Before live generation, make sure:

- GitHub CLI is installed and authenticated (`gh auth login`).
- The working directory is clean (`git status`).
- Git author identity is configured (`git config user.name` and `git config user.email`).

Use this staged runbook. Do not skip the two-run validation.

1. **Pre-flight check**

   ```bash
   python3 scripts/generate_training_data.py --check-env
   ```

2. **Dry run (planning mode)**

   ```bash
   python3 scripts/generate_training_data.py --runs 30 --success-runs 15 --failure-runs 15 --dry-run
   ```

3. **Live two-run validation**

   ```bash
   python3 scripts/generate_training_data.py --runs 2 --success-runs 1 --failure-runs 1 --execute
   ```

4. **Full 30-run generation and collection**

   ```bash
   python3 scripts/generate_training_data.py --runs 30 --success-runs 15 --failure-runs 15 --execute --collect
   ```

5. **Model inspection, training, and temporal evaluation**

   ```bash
   python3 -m src.prediction.cli inspect --dataset data/historical_runs.csv
   python3 -m src.prediction.cli train --dataset data/historical_runs.csv --model models/failure_predictor.joblib
   python3 -m src.prediction.cli evaluate --dataset data/historical_runs.csv --model models/failure_predictor.joblib
   ```

6. **Safe cleanup**

   ```bash
   python3 scripts/generate_training_data.py --cleanup
   ```

`--cleanup` considers only branches recorded by this generator whose names start with `ml-data/`; it never targets `main` or developer branches. A resumed scenario uses its existing completed metadata rather than dispatching a duplicate workflow. If an abandoned generated branch already exists, it is reset from `origin/main` and pushed using `--force-with-lease`, which limits the reset to that known generator branch and rejects stale remote changes.

Safety limits:

- Default maximum: **50 runs**; use `--allow-large-run-set` only when intentionally needed.
- `--execute` is required for generation, collection, and training.
- Polling defaults to 10 seconds with a 900-second per-workflow timeout; completed metadata is persisted after each run for safe resume.

## Building a Real Historical CI Dataset

This project learns from **genuine GitHub Actions workflow runs**. Do not fabricate CSV rows or synthetic labels.

### Step 1 — Generate successful CI history

Every normal commit/PR to `main` runs `.github/workflows/ci.yml` and creates real successful runs when CI passes:

```bash
git checkout -b feature/my-change
# make a small legitimate change, commit, push
git push -u origin feature/my-change
# open a PR to main → CI runs on pull_request
# merge to main → CI runs again on push
```

You can also trigger CI manually: **Actions → CI/CD Pipeline → Run workflow**.

Each completed workflow run is later visible to the collector.

### Step 2 — Generate a small number of legitimate failures (dev/testing only)

Use the manual utility workflow — it does **not** run automatically:

**Actions → Controlled CI Failure Generator (Dev/Testing Only) → Run workflow**

Choose a `failure_type`:

| Input | What fails | Typical category |
|-------|------------|------------------|
| `lint` | Syntax/compile check | lint |
| `test` | pytest selection | test |
| `build` | Import/build step | build |
| `dependency` | pip install | dependency |
| `docker` | Docker build | docker |

This creates a **real failed GitHub Actions run**. It does not modify `historical_runs.csv` directly.

Recommended pattern:

```
success → success → success → controlled failure → fix → success → ...
```

### Step 3 — Recollect history

```bash
export GITHUB_ACCESS_TOKEN=ghp_...
python3 -m src.prediction.cli collect \
  Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer \
  --output data/historical_runs.csv
python3 -m src.prediction.cli inspect --dataset data/historical_runs.csv
```

The collector gathers **all completed workflows** (CI, pre-CI prediction, controlled failures, etc.).

### Step 4 — Train only when inspect says sufficient

Requires **≥20 runs** and **both success and failure classes**. Training refuses one-class datasets.

```bash
python3 -m src.prediction.cli train --dataset data/historical_runs.csv
python3 -m src.prediction.cli evaluate --dataset data/historical_runs.csv
```

### Step 5 — Publish model artifact (GitHub Actions)

Run **Train Failure Predictor** workflow (`train-model.yml`) after enough real history exists.  
`predict.yml` downloads the artifact for non-blocking pre-CI prediction.

### Important

- Do not insert fake rows into `data/historical_runs.csv`.
- Real predictive quality depends on real, varied CI history.
- The controlled failure workflow is for development/testing only.
- Validation thresholds are intentional and are not weakened.

## Collect Historical Data

Requires `GITHUB_ACCESS_TOKEN` with access to the target repository.

```bash
python3 -m src.prediction.cli collect owner/repo --output data/historical_runs.csv
python3 -m src.prediction.cli inspect --dataset data/historical_runs.csv
```

This creates one training example per workflow run from real GitHub Actions history.

## Record Prediction Feedback

After a CI run completes, close the loop manually if needed:

```bash
python3 -m src.prediction.cli feedback owner/repo COMMIT_SHA --conclusion success --run-id 123456
```

Supported conclusions: `success`, `failure`, `cancelled`, `skipped`, etc. Non-binary outcomes (e.g. `cancelled`) are recorded without treating them as success.

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

## Model Artifact Strategy

Models are **not** committed to git (`models/*.joblib` is gitignored).

| Workflow | Purpose |
|----------|---------|
| `.github/workflows/train-model.yml` | Manual (`workflow_dispatch`) or weekly schedule — collect real history, train, evaluate, upload artifact |
| `.github/workflows/predict.yml` | Downloads latest `failure-predictor-model` artifact, runs pre-CI prediction (non-blocking), uploads `prediction-history` artifact |
| `.github/workflows/feedback.yml` | After CI or controlled-failure workflows complete, downloads `prediction-history`, records actual outcomes, re-uploads artifact |

Train a model in GitHub Actions:

1. Actions → **Train Failure Predictor** → Run workflow
2. After success, download artifact from that run or let `predict.yml` consume it automatically

Local training still works:

```bash
python3 -m src.prediction.cli collect owner/repo --output data/historical_runs.csv
python3 -m src.prediction.cli train --dataset data/historical_runs.csv
```

## Pre-CI Prediction in GitHub Actions

Optional workflow: `.github/workflows/predict.yml`

Runs on `pull_request` and `push` to `main`. It loads the trained model if present, records the prediction to `data/prediction_history.csv`, uploads that file as the `prediction-history` artifact, and prints failure probability **before** the main CI result is known. If no model exists, it skips gracefully and does not block CI.

`feedback.yml` listens for completed **CI/CD Pipeline** and **Controlled CI Failure Generator** runs only (not pre-CI prediction itself), matches pending predictions by commit SHA, and persists outcomes via the same `prediction-history` artifact.

## Validation Status (honest)

| Component | Status |
|-----------|--------|
| ML pipeline (collect, features, train, predict, feedback) | **IMPLEMENTED** · **TESTED WITH SYNTHETIC DATA** |
| Dataset validation & leakage guards | **IMPLEMENTED** · **TESTED WITH SYNTHETIC DATA** |
| GitHub history collection CLI | **IMPLEMENTED** · **TESTED WITH REAL GITHUB DATA** (currently ~1 run in this repo) |
| Model training on real repo history | **NOT TESTED** (insufficient runs / one outcome class) |
| Pre-CI prediction in GitHub Actions | **IMPLEMENTED** · **NOT TESTED LIVE IN GITHUB ACTIONS** |
| Feedback loop via `prediction-history` artifact | **IMPLEMENTED** · **NOT TESTED LIVE IN GITHUB ACTIONS** |
| LangGraph RCA (Bedrock + Tavily) | **IMPLEMENTED** · **TESTED WITH SYNTHETIC/MOCKED DATA** · **NOT TESTED LIVE** |
| Culprit commit analyzer | **IMPLEMENTED** · **TESTED WITH SYNTHETIC DATA** |
| Streamlit UI | **IMPLEMENTED** · **TESTED LOCALLY** (manual) |
| Controlled failure workflow | **IMPLEMENTED** · **NOT TESTED LIVE IN GITHUB ACTIONS** |

Unit tests do **not** prove production readiness. Live validation requires pushing workflows, generating ≥20 varied CI runs, training a model artifact, and observing predict + feedback on real Actions runs.

## Limitations

- Failure category prediction is optional and requires enough real labeled failures.
- Predictions depend on historical run metadata that may be incomplete in some repositories.
- The model does not replace the existing LLM-based RCA path.
- The app does not fabricate confidence where the model does not support it.
- **Real GitHub credentials are required** to collect history and score live commits.
- Baseline Random Forest quality depends on dataset size; below ~50 runs expect weak signal.
- Culprit commit analysis uses deterministic evidence; it does not replace git bisect or human review.

## Notes

The UI now shows a failure-risk section before the existing RCA output. If a failure has already happened, the RCA pipeline still runs exactly as before.

## CI/CD Pipeline

This repository uses a multi-stage GitHub Actions pipeline at [`.github/workflows/ci.yml`](.github/workflows/ci.yml). The stages are intentionally separated so historical workflow outcomes map to meaningful failure categories for the ML prediction layer.

```
Code Checkout
    ↓
Lint / Static Checks
    ↓
Unit Tests
    ↓
Prediction Module Tests
    ↓
Build / App Validation ──┐
                         ├──→ Docker Build & Validate
Security Check ──────────┘
    ↓ (main branch pushes only)
Release Artifact
```

| Job | Purpose | Typical failure category |
|-----|---------|--------------------------|
| `lint` | `ruff` + `compileall` | Code quality / syntax |
| `unit-tests` | Full `pytest -q` suite with coverage | Test failure |
| `prediction-tests` | ML import, feature extraction, train/load/predict/evaluate via test fixtures | ML / prediction failure |
| `build` | Compile sources and import modules without live API credentials | Application / build failure |
| `security` | `pip-audit` on installed dependencies | Dependency / security failure |
| `docker-build` | Build image and verify Streamlit health endpoint | Container / build failure |
| `release-artifact` | Package a deployment bundle on `main` (simulated release, no cloud deploy) | Release packaging failure |

The pipeline does **not** require AWS, Bedrock, Tavily, or GitHub tokens for ordinary CI runs. Tests use mocked fixtures already present in `tests/test_prediction.py`.

Run CI-like checks locally:

```bash
python -m pip install -r requirements.txt pytest pytest-cov ruff pip-audit
ruff check .
python -m compileall -q .
pytest -q
pip-audit
docker build -t cicd-root-cause-analyzer:test .
docker run --rm -d -p 8501:8501 --name cicd-test cicd-root-cause-analyzer:test
curl -f http://127.0.0.1:8501/_stcore/health
docker stop cicd-test
```

To build historical CI/CD data for the failure predictor, run this workflow on real pull requests and merges to `main`. Do not fabricate workflow outcomes — introduce controlled, real failures by temporarily breaking the relevant stage (for example, a lint violation, a failing assertion, or an outdated dependency) and reverting after the run is recorded.


