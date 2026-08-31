# Deployment Guide

The dashboard is a Streamlit application served on port `8501`. The supported
deployment shape is one application process or container with persistent
storage for its model, prediction history, and generated reports.

The application uses local files with in-process locking and atomic history
writes, but it does not coordinate across multiple processes. Do not run
multiple replicas against the same volumes. Place any externally reachable
deployment behind TLS and enable both access controls below.

## Runtime requirements

- Python 3.11 or newer
- Network access to GitHub for automatic repository/log operations
- Optional access to AWS Bedrock or a separately running Ollama server for AI enrichment
- Optional Tavily network access for web-research enrichment
- A writable filesystem for `data/`, `models/`, and `output/`
- A trained predictor artifact for model-backed failure-risk scores

Create a `.env` file locally and keep it out of source control:

```dotenv
# Optional: automatic Actions log retrieval. A session token can instead be
# entered in the dashboard and is never persisted by the application.
GITHUB_ACCESS_TOKEN=replace_with_a_fine_grained_token

# AI enrichment is opt-in and disabled by default.
LLM_PROVIDER=none

# Bedrock option (prefer a profile/workload identity over static keys).
# LLM_PROVIDER=bedrock
# AWS_REGION=us-east-1
# BEDROCK_MODEL_ID=replace_with_an_enabled_bedrock_model_id
# AWS_PROFILE=replace_with_a_profile

# Local Ollama option.
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OLLAMA_MODEL=llama3.1:8b

# Optional for web research; not required for either provider.
# TAVILY_API_KEY=replace_with_a_tavily_key

# Set both controls before exposing the dashboard publicly.
APP_PASSWORD=replace_with_a_long_random_password
ALLOWED_REPOSITORIES=owner/repository,owner/second-repository
DEFAULT_REPOSITORY=owner/repository

# APP_HOST_PORT=8501
```

Use a workload identity or an instance/task IAM role instead of long-lived AWS
keys when the target platform supports it. The AWS principal must be allowed to
invoke the configured Bedrock model. The GitHub token should be fine-grained
and limited to repository contents and Actions/log read access for only the
repositories the deployment is intended to analyze.

The server and credential-free log triage can run without external API
credentials. Users can paste or upload a `.log` or `.txt` file and receive a
deterministic local debugging brief; that log content is not sent to GitHub,
Bedrock, or Tavily. GitHub collection and live failed-run analysis still require
a repository name and, for Actions logs, a scoped token. AI enrichment requires
an explicit Bedrock or Ollama provider; Tavily remains optional. Model-backed
risk scores require a trained model artifact. A healthy process therefore does
not by itself prove that every external integration is configured.

`APP_PASSWORD` enables the dashboard's shared-password gate.
`ALLOWED_REPOSITORIES` is a comma-separated, case-insensitive repository
allowlist used by GitHub-backed operations. An empty password disables the
gate, and an empty allowlist permits any syntactically valid repository. Set
both variables for every publicly reachable deployment.

### AI provider behavior

`LLM_PROVIDER=none` is the default and guarantees the application does not
construct a Bedrock or Ollama client. `bedrock` uses the normal AWS SDK
credential chain and performs model calls only when a user starts AI-enriched
analysis. `ollama` uses the configured local HTTP endpoint; the Ollama server
and selected model must be installed separately. `auto` chooses a detectable
AWS identity first, then an explicitly configured Ollama endpoint, and otherwise
stays disabled. Tavily is an independent optional enhancement.

For Compose with Ollama running on the host, set:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b
```

## Local virtual environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://127.0.0.1:8501`. Validate the process separately:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
```

## Docker

Build and start a disposable container:

```bash
docker build -t cicd-root-cause-analyzer:local .
docker run --rm --name cicd-root-cause-analyzer \
  -p 8501:8501 \
  cicd-root-cause-analyzer:local
```

Add `--env-file .env` only when integrations or deployment access controls are
configured. Credential-free startup does not need an environment file.

This direct command does not preserve runtime state after the container is
removed. Use Compose for normal operation.

## Docker Compose

`compose.yaml` runs exactly one container, drops Linux capabilities, uses a
read-only root filesystem, and keeps the three writable application paths in
named volumes. It starts successfully without `.env`; Compose automatically
uses that file when present and passes the supported variables explicitly.

Validate the Compose definition, build the image, and start it:

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl -fsS http://127.0.0.1:${APP_HOST_PORT:-8501}/_stcore/health
```

Inspect startup errors without printing the contents of `.env`:

```bash
docker compose logs --tail=100 dashboard
```

`docker compose down` removes the container and network but retains the named
volumes. Do not use `docker compose down --volumes` unless the prediction
history, model artifacts, and generated reports may be permanently deleted.

### Install or update a trained model

The image intentionally does not contain local model or data artifacts. Copy a
validated training artifact into the persistent model volume, then restart the
application so its cached prediction service reloads it:

```bash
docker compose cp models/. dashboard:/home/appuser/app/models/
docker compose restart dashboard
```

Expected filenames are:

- `failure_predictor.joblib`
- `failure_predictor_metadata.json`
- `failure_category_predictor.joblib` and its metadata file, when available

Treat model artifacts as executable inputs: only install artifacts produced by
a trusted training workflow. Back up the `analyzer-data`, `analyzer-models`, and
`analyzer-output` volumes according to the host's normal volume-backup policy.
Treat downloaded CI logs and generated reports as sensitive operational data:
restrict volume access, encrypt host backups, and apply a retention policy.

## CI release bundle

Every successful full pipeline for a push to `main` produces a `release-bundle` GitHub Actions
artifact. Its archive contains the application sources plus the Dockerfile,
Compose definition, Streamlit server config, `.dockerignore`, environment
template, and this deployment guide. The release job validates that manifest
and rejects generated Python bytecode before uploading the archive.

After downloading and extracting the artifact, optionally create `.env` from
`.env.example`, configure the access controls required for the intended
exposure, then run the same `docker compose up -d --build` command. The artifact is a
self-hosting bundle; it is not evidence of a rollout to a managed cloud target.

## CI-published container image

The same successful `main` pipeline publishes the tested dashboard image to
GitHub Container Registry using the workflow's short-lived token. Images use a
lowercase repository path and receive both `latest` and immutable
`sha-<first-12-commit-characters>` tags:

```text
ghcr.io/<owner>/<repository>:latest
ghcr.io/<owner>/<repository>:sha-<12-character-commit>
```

Prefer the immutable SHA tag for a deployment. If the package is private,
authenticate the target host to GHCR with a token that has package-read access,
then run it with the same secrets, single-replica constraint, persistent mounts,
and health probe described in this guide. Publishing an image creates a
deployable artifact; a managed platform still must pull and run it. Models and
runtime CSV/report data remain outside the image and must be restored into the
persistent volumes.

## External hosting contract

Any container platform can run the image if it provides all of the following:

1. One replica listening on container port `8501` (or the platform-provided
   `PORT` outside Compose), including WebSocket support.
2. Persistent writable mounts at `/home/appuser/app/data`,
   `/home/appuser/app/models`, and `/home/appuser/app/output`.
3. Runtime secret injection only for capabilities that are enabled. A Bedrock
   deployment may instead use an AWS workload identity. Never bake `.env` into
   the image.
4. `APP_PASSWORD` and `ALLOWED_REPOSITORIES` configured as runtime secrets or
   environment values, plus an HTTPS reverse proxy or ingress.
5. A health probe against `/_stcore/health`.

The application implements a shared-password gate and an exact repository
allowlist through `APP_PASSWORD` and `ALLOWED_REPOSITORIES`. Set both for any
public exposure. These controls complement, rather than replace, TLS and an
identity-aware reverse proxy, VPN, or ingress policy. The GitHub token must also
be scoped only to approved repositories; do not expose a deployment publicly
with a token that can read unrestricted private repositories.

## Validation checklist

- `docker compose ps` reports the service as healthy.
- The dashboard loads through the final HTTPS endpoint and Streamlit WebSockets
  remain connected.
- The password gate rejects an invalid password, and a repository outside
  `ALLOWED_REPOSITORIES` cannot trigger a GitHub-backed operation.
- Uploaded-log analysis produces a local debugging brief with all external
  credentials unset.
- Invalid or missing credentials produce an actionable UI error without
  revealing secret values in application logs.
- The expected predictor model is visible in the `models` volume and a test
  prediction returns a model version and risk score.
- A controlled failed workflow can be analyzed and its report downloaded.
- Prediction history and reports remain after a container restart.
- Authentication blocks anonymous access and the GitHub token cannot access a
  repository outside the approved set.

The Streamlit health endpoint is a liveness check only. Deterministic RCA is
ready whenever the process is healthy. Each optional capability has its own
readiness contract: GitHub Actions logs need a scoped token, prediction needs a
usable model artifact, Bedrock needs AWS model access, Ollama needs its local
server/model, and Tavily research needs its API key and network connectivity.
