# CI/CD Root Cause Analyzer-New

This project has three cooperating layers:

1. CI/CD failure risk prediction before a workflow finishes.
2. Deterministic root cause analysis, with optional provider-backed enrichment.
3. A deployable operations dashboard with credential-free local log triage.

The dashboard fetches bounded, redacted GitHub Actions logs directly. Its
deterministic RCA works without an LLM; an opt-in provider layer can enrich the
brief with AWS Bedrock or a locally hosted Ollama model, with Tavily as optional
web research.

## What Changed

- Added historical workflow-run collection for supervised learning.
- Added deterministic feature extraction for workflow and commit metadata.
- Added a baseline Random Forest failure predictor with saved model artifacts.
- Added an optional failure-category classifier when enough labeled historical failures exist.
- Added prediction history storage so predictions can be compared with actual outcomes later.
- Rebuilt the Streamlit UI as an operational dashboard for readiness, local and
  live RCA, risk prediction, dataset/model management, and feedback metrics.
- Added a deterministic offline analyzer so pasted or uploaded logs can be
  triaged without GitHub, AWS, or Tavily credentials.
- Added a hardened single-replica Docker Compose deployment with persistent
  volumes, health checks, password protection, and a repository allowlist.
- Fixed latest-failed-run selection, common pytest/ANSI/exit-code parsing,
  training/inference feature parity, terminal retry state, and secret redaction.
- Added real workflow discovery, session-only GitHub authentication, and
  provider-neutral Bedrock/Ollama/no-LLM runtime status.

## Architecture

```
Developer
    ↓
Commit / PR
    ↓
GitHub Actions CI (.github/workflows/ci.yml)
    ├── Advisory prediction on the real commit SHA
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
If failure → deterministic local RCA (always available)
    ├── Parse/classify logs
    ├── Root Cause + Fixes
    └── Likely Culprit Commit (evidence-based)
Optional enrichment (explicitly configured)
    ├── Bedrock or Ollama triage/synthesis
    └── Tavily web research (optional)
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
- `src/tools/log_parser.py`, `src/integrations/github_automation.py`
- `src/tools/commit_analyzer.py` — likely culprit commit scoring
- `src/utils/llm.py` — optional Bedrock/Ollama/no-LLM provider boundary

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
- Optional AWS Bedrock or local Ollama model
- Optional Tavily API research
- PyGithub
- Streamlit
- pandas / numpy / scikit-learn / joblib

## Quick start

The offline dashboard and local log analyzer do not require credentials:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://127.0.0.1:8501`. Public GitHub metadata works anonymously; Actions
logs require a fine-grained token entered for the session or supplied at
deployment. For model-backed risk scores, collect at least 20 valid two-class
runs and train or install a trusted model artifact.

For a persistent single-container deployment:

```bash
# An .env file is optional for credential-free local use. Before public
# exposure, set APP_PASSWORD and ALLOWED_REPOSITORIES.
docker compose up -d --build
curl -fsS http://127.0.0.1:8501/_stcore/health
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for secrets, IAM/token scope,
volumes, model installation, hardening, and production acceptance checks.

## Dashboard

The Streamlit dashboard provides:

- runtime/readiness status without revealing secret values;
- credential-free paste/upload log analysis and report download;
- GitHub-backed analysis of the most recent failed run, even when a newer run
  succeeded;
- pre-CI model prediction that is disabled clearly when no model is installed;
- historical-data upload/inspection, collection, and training controls;
- prediction feedback metrics and downloadable history;
- optional shared-password authentication, exact repository allowlisting, and
  a cooldown on external operations.

The health endpoint proves only that Streamlit is alive. Automatic GitHub log
RCA additionally needs a scoped token. AI enrichment needs that token plus an
explicit Bedrock or Ollama provider; Tavily is optional. Prediction needs a
trained model under `models/`.

### Optional AI enrichment providers

No provider is enabled by default, so starting the application cannot incur an
AWS charge. Choose exactly one provider only when enriched triage is wanted:

```dotenv
# AWS SDK credentials/profile/workload identity are resolved on use.
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

or run Ollama separately and configure its local endpoint:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
```

Inside Compose, use `http://host.docker.internal:11434` when Ollama runs on the
host. `LLM_PROVIDER=auto` selects a detectable AWS identity first, then an
explicitly configured Ollama endpoint; otherwise it stays in no-LLM mode.
`TAVILY_API_KEY` adds web findings but is never required for the LLM or offline
paths.

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

The collector includes completed runs with binary conclusions (`success` or
`failure`) and excludes product-automation workflows such as Prediction Feedback
and Train Failure Predictor by default. This prevents the
same commit from gaining contradictory labels merely because support workflows
also ran.

### Step 4 — Train only when inspect says sufficient

Requires **≥20 runs** and **both success and failure classes**. Training refuses one-class datasets.

```bash
python3 -m src.prediction.cli train --dataset data/historical_runs.csv
python3 -m src.prediction.cli evaluate --dataset data/historical_runs.csv
```

### Step 5 — Publish model artifact (GitHub Actions)

Run **Train Failure Predictor** workflow (`train-model.yml`) after enough real history exists.  
The non-blocking prediction job in `ci.yml` downloads the newest trusted model
artifact produced from the default branch.

### Important

- Do not insert fake rows into `data/historical_runs.csv`.
- Real predictive quality depends on real, varied CI history.
- The controlled failure workflow is for development/testing only.
- Validation thresholds are intentional and are not weakened.

## Collect Historical Data

Private repositories and normal API rate limits require `GITHUB_ACCESS_TOKEN`
with access to the target repository. Public repository metadata can be
collected anonymously at GitHub's lower unauthenticated rate limit; authenticated
access is still required to retrieve failed-run log archives and category labels.

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
- `LLM_PROVIDER` (`none` by default; `auto`, `bedrock`, or `ollama`)
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_REQUEST_TIMEOUT_SECONDS`
- `TAVILY_API_KEY` (optional web research)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `BEDROCK_MODEL_ID` (optional model override)
- `AWS_SESSION_TOKEN` (when temporary credentials are used)
- `APP_PASSWORD` (recommended for any shared deployment)
- `ALLOWED_REPOSITORIES` (comma-separated exact `owner/repo` allowlist)
- `DEFAULT_REPOSITORY`
- `DATA_DIR`, `MODELS_DIR`, `OUTPUT_DIR` (optional runtime path overrides)

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
| `.github/workflows/ci.yml` | Scores the exact commit being tested, stores a run-linked `prediction-snapshot`, runs all checks, publishes the dashboard image on `main` |
| `.github/workflows/feedback.yml` | After that exact CI run completes, merges its immutable snapshot into the finalized history and records the actual outcome by run ID |
| `.github/workflows/train-model.yml` | After completed `main` CI/controlled-failure runs, weekly, or manually — collect up to 500 real runs, quality-gate, train, evaluate, and upload auditable artifacts |

Train a model in GitHub Actions:

1. Actions → **Train Failure Predictor** → Run workflow
2. After success, download the artifact or let the next `CI/CD Pipeline` run consume it automatically

Local training still works:

```bash
python3 -m src.prediction.cli collect owner/repo --output data/historical_runs.csv
python3 -m src.prediction.cli train --dataset data/historical_runs.csv
```

## Pre-CI Prediction in GitHub Actions

The experimental `predict-failure-risk` job is part of `.github/workflows/ci.yml`.
For pull requests it explicitly checks out and scores `pull_request.head.sha`, not
GitHub's synthetic merge SHA. For pushes it scores `github.sha`. It restores the
latest model produced on the default branch, calculates risk from that commit's
real changed files plus prior completed-run history, and writes a job summary.
Prediction is advisory: a missing/weak model or prediction error does not gate CI.

The CI run uploads an immutable `prediction-snapshot`. `feedback.yml` runs only
after that same **CI/CD Pipeline** run completes, addresses the prediction by the
shared run ID, merges out-of-order snapshots under a repository-wide non-cancelling
concurrency lock, and uploads the cumulative `prediction-history`. This avoids
duplicate push/PR predictor workflows and commit-SHA matching races.

## Validation Status (honest)

| Component | Status |
|-----------|--------|
| ML pipeline (collect, features, train, predict, feedback) | **IMPLEMENTED** · **AUTOMATED TESTS PASS** |
| Dataset validation, temporal split & leakage guards | **IMPLEMENTED** · **AUTOMATED TESTS PASS** |
| GitHub history collection CLI | **VALIDATED** · collected 37 real public runs (16 success / 21 failure) |
| Model training on real repo history | **VALIDATED, EXPERIMENTAL** · chronological 21/16 split; holdout accuracy 12.5%, F1 0.0 |
| Pre-CI prediction in GitHub Actions | **LIVE VALIDATED** · PR run `33322344708` completed successfully on Python 3.11 |
| Feedback loop via `prediction-history` artifact | **IMPLEMENTED** · **NOT TESTED LIVE IN GITHUB ACTIONS** |
| GHCR dashboard image publishing on `main` | **IMPLEMENTED** · **NOT TESTED LIVE IN GITHUB ACTIONS** |
| LangGraph AI enrichment (Bedrock or Ollama; Tavily optional) | **IMPLEMENTED** · **AUTOMATED PROVIDER/WORKFLOW TESTS PASS** · external model invocation not performed in this audit |
| Culprit commit analyzer | **IMPLEMENTED** · **AUTOMATED TESTS PASS** |
| Offline log RCA | **IMPLEMENTED** · **AUTOMATED TESTS PASS** |
| Streamlit dashboard | **IMPLEMENTED** · **APPTEST + LOCAL HEALTH + BROWSER JOURNEY PASS** |
| Docker/Compose deployment | **IMPLEMENTED** · Compose statically validated; image build + container health passed in PR run `33322344708` |
| Controlled failure workflow | **IMPLEMENTED** · 17 completed runs observed in upstream history |

Local verification currently passes `ruff`, `compileall`, `pip-audit`, and all
142 tests. The complete dependency set also resolves for the supported Python
3.11 runtime. The locally generated real-history CSV and model artifacts are
gitignored runtime state, not source-controlled release assets.
Unit tests do **not** prove production readiness. Live validation requires valid
runtime credentials, ≥20 varied two-class CI runs, a trusted trained model, and
an observed prediction/feedback/RCA cycle against real GitHub Actions.

## Limitations

- Failure category prediction is optional and requires enough real labeled failures.
- Predictions depend on historical run metadata that may be incomplete in some repositories.
- The prediction model and optional LLM enrichment do not replace the
  deterministic RCA path or human review.
- The app does not fabricate confidence where the model does not support it.
- Public workflow metadata can be collected anonymously with low rate limits;
  private repositories, dashboard-backed live GitHub operations, failed-run log
  archives, and category labeling require a scoped GitHub token.
- Baseline Random Forest quality depends on dataset size; below ~50 runs expect weak signal.
- The current 37-run local model is explicitly marked experimental because its
  chronological holdout scored 12.5% accuracy and F1 0.0; it must not gate releases.
- Culprit commit analysis uses deterministic evidence; it does not replace git bisect or human review.
- Runtime CSV storage is protected within one process only; use exactly one
  application replica with persistent `data/`, `models/`, and `output/` mounts.
- Public deployment must add TLS/network controls and set both `APP_PASSWORD`
  and `ALLOWED_REPOSITORIES`; the built-in password is a shared gate, not SSO.

## Notes

The UI now shows a failure-risk section before the existing RCA output. If a failure has already happened, the RCA pipeline still runs exactly as before.

## CI/CD Pipeline

This repository uses a multi-stage GitHub Actions pipeline at [`.github/workflows/ci.yml`](.github/workflows/ci.yml). The stages are intentionally separated so historical workflow outcomes map to meaningful failure categories for the ML prediction layer.

```
Code Checkout
    ├── Advisory Real-Commit Prediction
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
GHCR Image + Release Artifact
```

| Job | Purpose | Typical failure category |
|-----|---------|--------------------------|
| `predict-failure-risk` | Score the exact PR head/push commit and upload a run-linked snapshot; explicitly non-gating | Advisory only |
| `lint` | `ruff` + `compileall` | Code quality / syntax |
| `unit-tests` | Full `pytest -q` suite with coverage | Test failure |
| `prediction-tests` | ML import, feature extraction, train/load/predict/evaluate via test fixtures | ML / prediction failure |
| `build` | Compile sources and import modules without live API credentials | Application / build failure |
| `security` | `pip-audit` on installed dependencies | Dependency / security failure |
| `docker-build` | Build image and verify Streamlit health endpoint | Container / build failure |
| `container-publish` | Publish lowercase `ghcr.io/<owner>/<repo>` tags `latest` and immutable `sha-<12>` on `main` | Container publishing failure |
| `release-artifact` | Package a self-hostable deployment bundle on `main` (no managed-cloud rollout) | Release packaging failure |

The pipeline does **not** require AWS, Bedrock, Tavily, or a separately created
GitHub secret. Prediction/history access and GHCR publishing use GitHub's
short-lived workflow token with job-scoped `actions: read` or `packages: write`
permissions. Tests use mocked fixtures already present in `tests/test_prediction.py`.

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

For the supported persistent deployment and security requirements, use
`docker compose up -d --build` and follow [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

To build historical CI/CD data for the failure predictor, run this workflow on real pull requests and merges to `main`. Do not fabricate workflow outcomes — introduce controlled, real failures by temporarily breaking the relevant stage (for example, a lint violation, a failing assertion, or an outdated dependency) and reverting after the run is recorded.
