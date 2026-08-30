"""
config.py - Project Configuration

Centralized configuration management for the CI/CD Root Cause Analyzer.
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _configured_path(environment_name: str, default: Path) -> Path:
    """Resolve a configurable runtime path without depending on the CWD."""

    raw_value = os.getenv(environment_name)
    if not raw_value:
        return default

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = default.parent / candidate
    return candidate.resolve()


def _configured_int(environment_name: str, default: int, minimum: int = 0) -> int:
    """Read a bounded integer setting, falling back on malformed input."""

    try:
        return max(int(os.getenv(environment_name, str(default))), minimum)
    except (TypeError, ValueError):
        return max(default, minimum)


class Config:
    """Main configuration class."""
    
    # Project Info
    PROJECT_NAME = "CI/CD Root Cause Analyzer"
    VERSION = "1.0.0"
    
    # Directories
    PROJECT_ROOT = Path(__file__).parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = _configured_path("DATA_DIR", PROJECT_ROOT / "data")
    MODELS_DIR = _configured_path("MODELS_DIR", PROJECT_ROOT / "models")
    OUTPUT_DIR = _configured_path("OUTPUT_DIR", PROJECT_ROOT / "output")
    TESTS_DIR = PROJECT_ROOT / "tests"

    # Prediction Artifacts
    HISTORICAL_DATASET_PATH = DATA_DIR / "historical_runs.csv"
    PREDICTION_HISTORY_PATH = DATA_DIR / "prediction_history.csv"
    GENERATED_RUNS_PATH = DATA_DIR / "generated_runs.json"
    GENERATED_RUN_PLAN_PATH = DATA_DIR / "generated_run_plan.json"
    PREDICTOR_MODEL_PATH = MODELS_DIR / "failure_predictor.joblib"
    PREDICTOR_METADATA_PATH = MODELS_DIR / "failure_predictor_metadata.json"
    CATEGORY_MODEL_PATH = MODELS_DIR / "failure_category_predictor.joblib"
    CATEGORY_METADATA_PATH = MODELS_DIR / "failure_category_predictor_metadata.json"

    # GitHub Actions artifact names (must match workflow upload/download steps)
    MODEL_ARTIFACT_NAME = "failure-predictor-model"
    PREDICTION_HISTORY_ARTIFACT_NAME = "prediction-history"
    
    # AWS Configuration
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_MODEL_ID = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    # AI enrichment is opt-in. The deterministic parser/RCA path never needs
    # an LLM, cloud account, or billable API. ``auto`` selects a detectable AWS
    # identity first, then an explicitly configured Ollama endpoint.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
    OLLAMA_EXPLICITLY_CONFIGURED = bool(os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"))
    OLLAMA_REQUEST_TIMEOUT_SECONDS = _configured_int("OLLAMA_REQUEST_TIMEOUT_SECONDS", 60, 1)
    
    # API Keys
    GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Rate Limiting
    DELAY_BETWEEN_LLM_CALLS = 3
    MAX_FAILURES_PER_STEP = 3
    MIN_DELAY_BETWEEN_CALLS = 2
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    
    # File Processing
    MAX_FILE_SIZE = 100 * 1024  # 100KB
    MAX_CONTENT_LENGTH = 10000
    MAX_LOG_LINES = 50000
    
    # Default Repository for Testing
    DEFAULT_TEST_REPO = os.getenv(
        "DEFAULT_REPOSITORY",
        "Vnj91/AI-Powered-CI-CD-failure-and-root-cause-analyzer",
    )

    # Dashboard access controls. APP_PASSWORD should be set for any public
    # deployment. ALLOWED_REPOSITORIES is a comma-separated allowlist.
    APP_PASSWORD = os.getenv("APP_PASSWORD")
    ALLOWED_REPOSITORIES = tuple(
        repository.strip().lower()
        for repository in os.getenv("ALLOWED_REPOSITORIES", "").split(",")
        if repository.strip()
    )
    ANALYSIS_COOLDOWN_SECONDS = _configured_int("ANALYSIS_COOLDOWN_SECONDS", 5)
    MAX_LOG_UPLOAD_BYTES = _configured_int("MAX_LOG_UPLOAD_BYTES", 2 * 1024 * 1024, 1024)
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate required settings without failing optional capabilities."""
        issues = []
        supported_providers = {"none", "auto", "bedrock", "ollama"}
        if cls.LLM_PROVIDER not in supported_providers:
            issues.append(
                "LLM_PROVIDER must be one of: none, auto, bedrock, ollama"
            )
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": {
                "aws_region": cls.AWS_REGION,
                "model_id": cls.BEDROCK_MODEL_ID,
                "llm_provider": cls.LLM_PROVIDER,
                "has_ollama_endpoint": bool(cls.OLLAMA_BASE_URL),
                "ollama_model": cls.OLLAMA_MODEL,
                "has_github_token": bool(cls.GITHUB_ACCESS_TOKEN),
                "has_tavily_key": bool(cls.TAVILY_API_KEY),
                "has_aws_credentials": cls.has_aws_credentials(),
                "has_app_password": bool(cls.APP_PASSWORD),
                "allowed_repositories": list(cls.ALLOWED_REPOSITORIES),
            }
        }

    @classmethod
    def has_aws_credentials(cls) -> bool:
        """Return whether a locally detectable Bedrock credential source exists."""

        return bool(
            (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
            or os.getenv("AWS_PROFILE")
            or os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
            or os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or (Path.home() / ".aws" / "credentials").exists()
            or (Path.home() / ".aws" / "config").exists()
        )

    @classmethod
    def repository_is_allowed(cls, repository: str) -> bool:
        """Enforce the optional deployment repository allowlist."""

        if not cls.ALLOWED_REPOSITORIES:
            return True
        return repository.strip().lower() in cls.ALLOWED_REPOSITORIES
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUT_DIR, cls.TESTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Create global config instance
config = Config()

# Make validation available to callers without treating optional integrations
# as a startup failure for the credential-free dashboard paths.
validation = config.validate()

config.ensure_directories()
