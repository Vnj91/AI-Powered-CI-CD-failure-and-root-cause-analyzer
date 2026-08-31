"""Operational Streamlit dashboard for CI/CD prediction and RCA.

Run with: ``streamlit run app.py``
"""

from __future__ import annotations

import hmac
import io
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CI Signal · Failure Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _bootstrap_streamlit_secrets() -> None:
    """Expose supported Streamlit secrets to the env-based project config."""

    names = {
        "GITHUB_ACCESS_TOKEN",
        "TAVILY_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "BEDROCK_MODEL_ID",
        "LLM_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS",
        "APP_PASSWORD",
        "ALLOWED_REPOSITORIES",
        "DEFAULT_REPOSITORY",
        "ANALYSIS_COOLDOWN_SECONDS",
        "MAX_LOG_UPLOAD_BYTES",
        "DATA_DIR",
        "MODELS_DIR",
        "OUTPUT_DIR",
    }
    try:
        for name in names:
            value = st.secrets.get(name)
            if value is not None and name not in os.environ:
                os.environ[name] = str(value)
    except Exception:
        # A local deployment normally has no secrets.toml.
        return


_bootstrap_streamlit_secrets()

from config import Config  # noqa: E402
from src.analysis import build_local_debugging_brief  # noqa: E402
from src.graph.state import DebuggingBrief  # noqa: E402
from src.graph.workflow import run_enriched_analysis  # noqa: E402
from src.prediction import (  # noqa: E402
    FailurePredictionService,
    FailurePredictorTrainer,
    HistoricalRunCollector,
)
from src.prediction.dataset_validator import DatasetValidationReport, validate_dataset  # noqa: E402
from src.prediction.feedback import PredictionFeedbackService  # noqa: E402
from src.prediction.history_store import PredictionHistoryStore  # noqa: E402
from src.tools.log_parser import LogParseResult, parse_log_content  # noqa: E402
from src.utils.redaction import redact_sensitive_text  # noqa: E402
from src.utils.llm import get_llm_provider_status  # noqa: E402


_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAMPLE_LOG = """2026-08-29T10:01:00.0000000Z ##[group]Run pytest -q
2026-08-29T10:01:01.0000000Z Traceback (most recent call last):
2026-08-29T10:01:01.0000000Z   File \"/workspace/tests/test_api.py\", line 42, in test_health
2026-08-29T10:01:01.0000000Z     import requests_mock
2026-08-29T10:01:01.0000000Z ModuleNotFoundError: No module named 'requests_mock'
2026-08-29T10:01:01.0000000Z ##[error]Process completed with exit code 1.
"""


def _install_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#142238; --muted:#5f6f82; --line:#dce4ec; --accent:#0f766e; --soft:#edf8f6; }
        .stApp { background:#f7f9fc; }
        .block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:4rem; }
        h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:1rem 1.1rem; box-shadow:0 2px 12px rgba(20,34,56,.04); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--ink); font-size:clamp(1.4rem,1.8vw,2rem); }
        [data-testid="stSidebar"] { border-right:1px solid var(--line); }
        .signal-hero { background:linear-gradient(120deg,#10243e 0%,#153c4c 56%,#0f766e 100%);
          border-radius:20px; color:white; padding:1.8rem 2rem; margin-bottom:1.35rem;
          box-shadow:0 12px 34px rgba(16,36,62,.16); }
        .signal-hero h1 { color:white; margin:0 0 .35rem 0; }
        .signal-hero p { color:#d7e7e8; margin:0; max-width:850px; }
        .signal-kicker { color:#7dd3c7; font-size:.78rem; font-weight:700; letter-spacing:.14em;
          text-transform:uppercase; }
        .signal-note { background:var(--soft); border:1px solid #b9e1dc; border-radius:12px;
          color:#174943; padding:.85rem 1rem; }
        .signal-flow { background:white; border:1px solid var(--line); border-radius:14px;
          padding:1rem 1.2rem; color:var(--muted); line-height:1.75; }
        .connection-card { background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:1rem 1.2rem; margin:.35rem 0 1rem; }
        .connection-card strong { color:var(--ink); }
        .status-dot { display:inline-block; width:.62rem; height:.62rem; border-radius:50%;
          margin-right:.45rem; background:#94a3b8; }
        .status-dot.connected { background:#16a34a; box-shadow:0 0 0 4px #dcfce7; }
        .action-card { background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:.85rem 1rem .3rem; min-height:142px; }
        .action-card h4 { color:var(--ink); margin:.1rem 0 .35rem; }
        .action-card p { color:var(--muted); font-size:.9rem; margin:0 0 .5rem; }
        div[data-baseweb="tab-list"] { gap:.25rem; }
        button[data-baseweb="tab"] { border-radius:9px; padding-left:1rem; padding-right:1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_dashboard_access() -> None:
    expected = Config.APP_PASSWORD
    if not expected or st.session_state.get("dashboard_authenticated"):
        return
    st.markdown(
        """<div class="signal-hero"><div class="signal-kicker">Protected operations console</div>
        <h1>CI Signal</h1><p>Enter the deployment password to access logs, models, and AI analysis.</p></div>""",
        unsafe_allow_html=True,
    )
    with st.form("dashboard_login"):
        supplied = st.text_input("Dashboard password", type="password", key="dashboard_password")
        submitted = st.form_submit_button("Unlock dashboard", type="primary", key="dashboard_unlock")
    if submitted:
        if hmac.compare_digest(supplied, expected):
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        st.error("The password is not valid.")
    st.stop()


def _model_signature() -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 0)
        for path in (Config.PREDICTOR_MODEL_PATH, Config.CATEGORY_MODEL_PATH)
    )


@st.cache_resource(show_spinner=False)
def get_prediction_service(signature: tuple[tuple[str, int], ...]) -> FailurePredictionService:
    """Cache loaded models and invalidate the cache after artifact changes."""

    _ = signature
    return FailurePredictionService(
        model_path=Config.PREDICTOR_MODEL_PATH,
        history_path=Config.PREDICTION_HISTORY_PATH,
        metadata_path=Config.PREDICTOR_METADATA_PATH,
        category_model_path=Config.CATEGORY_MODEL_PATH,
        category_metadata_path=Config.CATEGORY_METADATA_PATH,
    )


def _prediction_service() -> FailurePredictionService:
    return get_prediction_service(_model_signature())


def _repository_error(repository: str) -> str | None:
    candidate = repository.strip()
    if not _REPOSITORY_PATTERN.fullmatch(candidate):
        return "Use the GitHub repository format owner/repository."
    owner, name = candidate.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        return "The repository name is not valid."
    if not Config.repository_is_allowed(candidate):
        return "This repository is not in the deployment allowlist."
    return None


def _cooldown_allows(action: str) -> bool:
    cooldown = Config.ANALYSIS_COOLDOWN_SECONDS
    now = time.monotonic()
    key = f"last_action_{action}"
    last = float(st.session_state.get(key, 0.0))
    if cooldown and now - last < cooldown:
        remaining = max(1, int(cooldown - (now - last) + 0.999))
        st.warning(f"Please wait {remaining}s before running this operation again.")
        return False
    st.session_state[key] = now
    return True


def _safe_message(exc: Exception) -> str:
    return redact_sensitive_text(str(exc).strip() or exc.__class__.__name__, limit=600)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _atomic_save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_model_metrics() -> dict[str, Any]:
    try:
        return json.loads(Config.PREDICTOR_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _model_status(service: FailurePredictionService) -> str:
    if not service.model_available:
        return "Artifact invalid" if Config.PREDICTOR_MODEL_PATH.exists() else "Model needed"
    metrics = _read_model_metrics()
    if int(metrics.get("test_rows", 0) or 0) < 20 or float(metrics.get("f1", 0.0) or 0.0) < 0.5:
        return "Experimental"
    return "Ready"


def _github_runtime_token() -> str | None:
    """Return the session-only token first, then the deployment fallback."""

    session_token = str(st.session_state.get("github_session_token", "") or "").strip()
    return session_token or Config.GITHUB_ACCESS_TOKEN


def _runtime_rows(service: FailurePredictionService) -> list[dict[str, str]]:
    report = validate_dataset(Config.HISTORICAL_DATASET_PATH)
    github_token = _github_runtime_token()
    provider = get_llm_provider_status()
    return [
        {"Capability": "Offline log triage", "Status": "Ready", "Requirement": "None"},
        {"Capability": "Public GitHub metadata", "Status": "Ready", "Requirement": "Repository name"},
        {"Capability": "GitHub Actions logs", "Status": "Ready" if github_token else "Setup needed", "Requirement": "Session or deployment token"},
        {
            "Capability": "AI RCA enrichment",
            "Status": "Ready" if provider.ready and github_token else "Setup needed",
            "Requirement": f"GitHub token + {provider.display_name}",
        },
        {
            "Capability": "Optional web research",
            "Status": "Ready" if Config.TAVILY_API_KEY else "Optional",
            "Requirement": "Tavily key",
        },
        {"Capability": "Failure-risk prediction", "Status": _model_status(service), "Requirement": "Trained, validated model"},
        {"Capability": "Model retraining", "Status": "Ready" if report.sufficient_for_training else "Data needed", "Requirement": "20+ real, two-class runs"},
    ]


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert integration models to display-safe dictionaries."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_")
    }


def _as_rows(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    return [_as_dict(value) for value in values]


def _github_snapshot(repository: str) -> dict[str, Any]:
    snapshot = st.session_state.get("github_snapshot")
    if not isinstance(snapshot, dict):
        return {}
    if str(snapshot.get("repository", "")).lower() != repository.lower():
        return {}
    return snapshot


def _persist_github_snapshot(
    repository: str,
    *,
    status: Any,
    commits: Any,
    runs: Any,
    workflows: Any = None,
    workflow_error: str | None = None,
) -> None:
    st.session_state["github_snapshot"] = {
        "repository": repository,
        "status": _as_dict(status),
        "commits": _as_rows(commits),
        "runs": _as_rows(runs),
        "workflows": _as_rows(workflows),
        "workflow_error": workflow_error,
        "refreshed_at": datetime.now(UTC).isoformat(),
    }


def _connection_is_active(repository: str) -> bool:
    return bool(_github_snapshot(repository).get("status", {}).get("connected"))


def _extract_log_text(context: Any) -> str:
    """Read the combined log field without binding the UI to one model version."""

    payload = _as_dict(context)
    candidates = [payload]
    for key in ("logs", "log_bundle", "workflow_logs"):
        nested = payload.get(key)
        if nested is not None:
            candidates.append(_as_dict(nested))
    for candidate in candidates:
        for key in ("combined_text", "combined_logs", "log_text", "content", "text"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _dataset_timeline(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "actual_failure"}
    if dataset.empty or not required.issubset(dataset.columns):
        return pd.DataFrame()
    timeline = dataset.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
    timeline["actual_failure"] = pd.to_numeric(timeline["actual_failure"], errors="coerce")
    timeline = timeline.dropna(subset=["timestamp", "actual_failure"]).sort_values("timestamp")
    if timeline.empty:
        return timeline
    timeline["rolling_failure_rate"] = timeline["actual_failure"].rolling(10, min_periods=1).mean()
    timeline["outcome"] = timeline["actual_failure"].map({0.0: "Passed", 1.0: "Failed"}).fillna("Unknown")
    return timeline


def _render_failure_trend(dataset: pd.DataFrame) -> bool:
    timeline = _dataset_timeline(dataset)
    if timeline.empty:
        st.info("Sync completed GitHub Actions runs to unlock the failure trend.")
        return False
    chart_data = timeline[["timestamp", "rolling_failure_rate"]].copy()
    # Arrow-backed Altair charts currently serialize pandas 3's microsecond
    # datetime dtype inconsistently in the browser. ISO-8601 keeps every run
    # timestamp portable while the explicit temporal encoding preserves scale.
    chart_data["timestamp"] = chart_data["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    chart = {
        "data": {"values": chart_data.to_dict(orient="records")},
        "mark": {"type": "line", "point": {"filled": True, "size": 36}},
        "encoding": {
            "x": {
                "field": "timestamp",
                "type": "temporal",
                "title": "Workflow start time",
                "axis": {"format": "%b %d"},
            },
            "y": {
                "field": "rolling_failure_rate",
                "type": "quantitative",
                "title": "Failure rate",
                "scale": {"domain": [0, 1]},
                "axis": {"format": "%"},
            },
            "tooltip": [
                {
                    "field": "timestamp",
                    "type": "temporal",
                    "title": "Run started",
                    "format": "%Y-%m-%d %H:%M",
                },
                {
                    "field": "rolling_failure_rate",
                    "type": "quantitative",
                    "title": "Rolling failure rate",
                    "format": ".1%",
                },
            ],
        },
    }
    st.vega_lite_chart(chart, width="stretch", height=285)
    st.caption("Ten-run rolling failure rate, ordered by workflow start time.")
    return True


def _render_failure_categories(dataset: pd.DataFrame) -> bool:
    if dataset.empty or "actual_category" not in dataset.columns:
        st.info("Failure categories appear after failed-run logs are classified.")
        return False
    failures = dataset
    if "actual_failure" in failures.columns:
        failures = failures[pd.to_numeric(failures["actual_failure"], errors="coerce") == 1]
    labels = failures["actual_category"].fillna("Unclassified").astype(str).str.strip()
    labels = labels.replace("", "Unclassified")
    counts = labels.value_counts().rename_axis("Failure category").to_frame("Runs")
    if counts.empty:
        st.info("No failed-run categories are available yet.")
        return False
    chart_data = counts.reset_index()
    chart = {
        "data": {"values": chart_data.to_dict(orient="records")},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "encoding": {
            "x": {
                "field": "Failure category",
                "type": "nominal",
                "title": None,
                "sort": "-y",
                "axis": {"labelLimit": 160},
            },
            "y": {
                "field": "Runs",
                "type": "quantitative",
                "title": "Failed runs",
                "stack": None,
                "axis": {"tickMinStep": 1},
            },
            "tooltip": [
                {"field": "Failure category", "type": "nominal", "title": "Category"},
                {"field": "Runs", "type": "quantitative", "title": "Failed runs", "format": "d"},
            ],
        },
    }
    st.vega_lite_chart(chart, width="stretch", height=285)
    st.caption("Observed failure categories in the active workflow history.")
    return True


def _render_change_risk(dataset: pd.DataFrame) -> bool:
    required = {"lines_added", "lines_deleted", "actual_failure"}
    if dataset.empty or not required.issubset(dataset.columns):
        st.info("Commit change metrics are needed for change-size risk analysis.")
        return False
    changes = dataset.copy()
    changes["change_size"] = (
        pd.to_numeric(changes["lines_added"], errors="coerce").fillna(0)
        + pd.to_numeric(changes["lines_deleted"], errors="coerce").fillna(0)
    )
    changes["actual_failure"] = pd.to_numeric(changes["actual_failure"], errors="coerce")
    changes["Change size"] = pd.cut(
        changes["change_size"],
        bins=[-1, 25, 100, 500, float("inf")],
        labels=["0–25", "26–100", "101–500", "500+"],
    )
    grouped = (
        changes.dropna(subset=["actual_failure", "Change size"])
        .groupby("Change size", observed=False)["actual_failure"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] > 0].rename(columns={"mean": "Failure rate", "count": "Runs"})
    if grouped.empty:
        st.info("No comparable change-size samples are available yet.")
        return False
    grouped["Change size"] = grouped["Change size"].astype(str)
    chart = {
        "data": {"values": grouped.to_dict(orient="records")},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "encoding": {
            "x": {
                "field": "Change size",
                "type": "ordinal",
                "title": "Added + deleted lines",
                "sort": ["0–25", "26–100", "101–500", "500+"],
            },
            "y": {
                "field": "Failure rate",
                "type": "quantitative",
                "title": "Failure rate",
                "stack": None,
                "scale": {"domain": [0, 1]},
                "axis": {"format": "%"},
            },
            "tooltip": [
                {"field": "Change size", "type": "ordinal", "title": "Change size"},
                {
                    "field": "Failure rate",
                    "type": "quantitative",
                    "title": "Failure rate",
                    "format": ".1%",
                },
                {"field": "Runs", "type": "quantitative", "title": "Runs", "format": "d"},
            ],
        },
    }
    st.vega_lite_chart(chart, width="stretch", height=285)
    st.caption("Observed failure rate by added-plus-deleted line count; association is not causality.")
    return True


def _render_prediction_trend(history: pd.DataFrame) -> bool:
    required = {"timestamp", "failure_probability"}
    if history.empty or not required.issubset(history.columns):
        st.info("Automated predictions will appear here after the model scores commits.")
        return False
    trend = history.copy()
    trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce", utc=True)
    trend["failure_probability"] = pd.to_numeric(trend["failure_probability"], errors="coerce")
    trend = trend.dropna(subset=["timestamp", "failure_probability"]).sort_values("timestamp")
    if trend.empty:
        st.info("No probability-bearing prediction records are available yet.")
        return False
    chart_data = trend[["timestamp", "failure_probability"]].copy()
    chart_data["timestamp"] = chart_data["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    chart = {
        "data": {"values": chart_data.to_dict(orient="records")},
        "mark": {"type": "line", "point": {"filled": True, "size": 36}},
        "encoding": {
            "x": {
                "field": "timestamp",
                "type": "temporal",
                "title": "Prediction time",
                "axis": {"format": "%b %d"},
            },
            "y": {
                "field": "failure_probability",
                "type": "quantitative",
                "title": "Predicted failure risk",
                "scale": {"domain": [0, 1]},
                "axis": {"format": "%"},
            },
            "tooltip": [
                {
                    "field": "timestamp",
                    "type": "temporal",
                    "title": "Predicted at",
                    "format": "%Y-%m-%d %H:%M",
                },
                {
                    "field": "failure_probability",
                    "type": "quantitative",
                    "title": "Failure risk",
                    "format": ".1%",
                },
            ],
        },
    }
    st.vega_lite_chart(chart, width="stretch", height=285)
    st.caption("Recorded pre-CI predictions only; retrospective scores are excluded from feedback history.")
    return True


def render_prediction(prediction, *, context: str) -> None:
    st.subheader(context)
    if not prediction.model_available:
        st.warning("No trained predictor is loaded. Add real history and train a model in Data & model.")
        return
    probability = float(prediction.failure_probability or 0.0)
    columns = st.columns(4)
    columns[0].metric("Failure probability", f"{probability:.1%}")
    columns[1].metric("Risk level", prediction.risk_level)
    columns[2].metric("Predicted outcome", "Likely fail" if prediction.predicted_failure else "Likely pass")
    columns[3].metric("Model version", str(prediction.model_version or "unknown")[:19])
    st.progress(probability, text=f"Estimated failure risk · {probability:.1%}")
    if prediction.predicted_category:
        suffix = f" · {prediction.category_confidence:.1%}" if prediction.category_confidence is not None else ""
        st.info(f"Predicted category: {prediction.predicted_category}{suffix}")
    for warning in prediction.warnings:
        st.warning(warning)
    if prediction.top_risk_factors:
        st.markdown("#### Strongest model signals")
        st.caption("Global model importance weighted by this input; association is not proof of causality.")
        rows = [
            {"Feature": f.feature, "Value": f.value, "Importance": f.importance, "Weighted signal": f.contribution}
            for f in prediction.top_risk_factors
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("All feature signals"):
        st.dataframe(
            pd.DataFrame([factor.model_dump() for factor in prediction.feature_importances]),
            width="stretch",
            hide_index=True,
        )


def render_brief(brief: DebuggingBrief, *, key_prefix: str) -> None:
    if brief.workflow_run_id is not None:
        st.caption(f"GitHub Actions run · {brief.workflow_run_id}")
    columns = st.columns(4)
    columns[0].metric("Severity", brief.severity.upper())
    columns[1].metric("Confidence", f"{brief.confidence_score:.0%}")
    columns[2].metric("Category", brief.error_category.replace("_", " ").title())
    columns[3].metric("Suggested fixes", len(brief.fix_suggestions))
    st.markdown("#### Primary error")
    st.code(f"{brief.error_type}: {brief.error_message}")
    st.markdown("#### Root-cause assessment")
    st.write(brief.root_cause_summary)
    with st.expander("Technical explanation"):
        st.write(brief.root_cause_detailed)
        if brief.affected_files:
            st.write("Affected files:", ", ".join(brief.affected_files))
    if brief.likely_culprit_sha:
        st.markdown("#### Most likely culprit commit")
        st.caption("Evidence-ranked candidate, not definitive proof of causality.")
        st.code(brief.likely_culprit_sha[:12])
        if brief.likely_culprit_message:
            st.write(brief.likely_culprit_message)
        for item in brief.likely_culprit_evidence:
            st.markdown(f"- {item}")
    st.markdown("#### Action plan")
    for fix in brief.fix_suggestions:
        with st.expander(f"{fix.priority}. {fix.title} · {fix.confidence:.0%}", expanded=fix.priority == 1):
            st.write(fix.description)
            for index, step in enumerate(fix.implementation_steps, start=1):
                st.markdown(f"{index}. {step}")
            if fix.code_example:
                st.code(fix.code_example)
            st.caption(f"Source: {fix.source.replace('_', ' ')}")
    st.download_button(
        "Download debugging brief",
        brief.to_markdown(),
        file_name="ci-debugging-brief.md",
        mime="text/markdown",
        key=f"{key_prefix}_download",
    )


def render_feedback_summary(*, include_table: bool = True) -> dict[str, Any]:
    store = PredictionHistoryStore(Config.PREDICTION_HISTORY_PATH)
    summary = PredictionFeedbackService(store).feedback_accuracy_summary()
    columns = st.columns(3)
    columns[0].metric("Outcomes recorded", summary["recorded"])
    columns[1].metric("Correct predictions", summary["correct"])
    columns[2].metric("Feedback accuracy", f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "N/A")
    if include_table:
        recent = store.recent_feedback_summary(limit=25)
        if recent.empty:
            st.info("No prediction outcomes have been recorded yet.")
        else:
            visible = [c for c in ("timestamp", "repository", "workflow", "commit_sha", "failure_probability", "predicted_failure", "actual_conclusion", "prediction_correct") if c in recent.columns]
            st.dataframe(recent[visible], width="stretch", hide_index=True)
    return summary


def render_overview(repository: str, service: FailurePredictionService) -> None:
    dataset = _load_csv(Config.HISTORICAL_DATASET_PATH)
    report = validate_dataset(dataset) if not dataset.empty else validate_dataset(Config.HISTORICAL_DATASET_PATH)
    feedback = PredictionFeedbackService(PredictionHistoryStore(Config.PREDICTION_HISTORY_PATH)).feedback_accuracy_summary()
    metrics = st.columns(4)
    metrics[0].metric("Historical runs", report.total_runs)
    metrics[1].metric("Observed failure rate", f"{report.class_balance_failure_rate:.1%}")
    metrics[2].metric("Risk model", _model_status(service) if service.model_available else "Not trained")
    metrics[3].metric("Feedback accuracy", f"{feedback['accuracy']:.1%}" if feedback["accuracy"] is not None else "No feedback")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Operational readiness")
        st.dataframe(pd.DataFrame(_runtime_rows(service)), width="stretch", hide_index=True)
    with right:
        st.subheader("Current scope")
        st.markdown(
            f"""<div class="signal-flow"><strong>Repository</strong><br>{repository or 'Not selected'}<br><br>
            <strong>Product loop</strong><br>Commit → risk score → CI outcome → failed-run RCA → feedback → retraining</div>""",
            unsafe_allow_html=True,
        )
    if dataset.empty:
        st.markdown(
            """<div class="signal-note">Start in <strong>Automation</strong>: connect the repository, sync genuine
            Actions history, then score the latest application change. Manual log upload remains an advanced fallback.</div>""",
            unsafe_allow_html=True,
        )
        return
    st.subheader("Historical signal")
    chart_left, chart_right = st.columns(2)
    if "timestamp" in dataset.columns:
        timeline = dataset.copy()
        timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
        timeline = timeline.dropna(subset=["timestamp"])
        if not timeline.empty:
            timeline["day"] = timeline["timestamp"].dt.tz_convert(None).dt.floor("D")
            timeline["failure"] = timeline["actual_failure"] if "actual_failure" in timeline else 0
            trend = timeline.groupby("day", as_index=False).agg(runs=("day", "size"), failures=("failure", "sum"))
            trend["failure rate"] = trend["failures"] / trend["runs"]
            trend["day"] = trend["day"].dt.strftime("%Y-%m-%d")
            with chart_left:
                st.caption("Runs and failures by day")
                st.dataframe(
                    trend,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "failure rate": st.column_config.ProgressColumn(
                            "Failure rate",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        )
                    },
                )
    if report.failures_by_workflow:
        with chart_right:
            st.caption("Failures by workflow")
            failure_chart = pd.DataFrame(
                report.failures_by_workflow.items(),
                columns=["workflow", "failures"],
            )
            st.dataframe(failure_chart, width="stretch", hide_index=True)


def render_analytics(repository: str) -> None:
    """Render outcome, model-risk, and change-impact charts from real records."""

    dataset = _load_csv(Config.HISTORICAL_DATASET_PATH)
    history = _load_csv(Config.PREDICTION_HISTORY_PATH)
    snapshot = _github_snapshot(repository)
    runs = pd.DataFrame(snapshot.get("runs", []))

    st.subheader("Automated service analytics")
    st.caption("Charts use synced GitHub Actions outcomes and recorded pre-CI predictions—never sample data.")
    top = st.columns(4)
    top[0].metric("Synced runs", len(dataset))
    completed_predictions = int(history["failure_probability"].notna().sum()) if "failure_probability" in history else 0
    top[1].metric("Recorded predictions", completed_predictions)
    connected = bool(snapshot.get("status", {}).get("connected"))
    top[2].metric("Repository feed", "Live" if connected else "Not connected")
    refreshed_at = snapshot.get("refreshed_at")
    top[3].metric("Live runs loaded", len(runs), help=f"Last refresh: {refreshed_at or 'never'}")

    row_one = st.columns(2)
    with row_one[0]:
        st.markdown("#### Failure-rate trend")
        _render_failure_trend(dataset)
    with row_one[1]:
        st.markdown("#### Predicted-risk trend")
        _render_prediction_trend(history)

    row_two = st.columns(2)
    with row_two[0]:
        st.markdown("#### Failure categories")
        _render_failure_categories(dataset)
    with row_two[1]:
        st.markdown("#### Change-size exposure")
        _render_change_risk(dataset)

    st.markdown("#### Recent workflow signal")
    if not runs.empty:
        visible = [
            column
            for column in ("created_at", "name", "workflow_name", "branch", "status", "conclusion", "head_sha", "html_url")
            if column in runs.columns
        ]
        st.dataframe(
            runs[visible].head(20),
            width="stretch",
            hide_index=True,
            column_config={
                "html_url": st.column_config.LinkColumn("Run"),
            },
        )
    elif not dataset.empty:
        visible = [
            column
            for column in ("timestamp", "workflow_name", "branch", "conclusion", "commit_sha", "files_changed", "lines_added", "lines_deleted")
            if column in dataset.columns
        ]
        st.dataframe(dataset.sort_values("timestamp", ascending=False)[visible].head(20), width="stretch", hide_index=True)
    else:
        st.info("Connect the repository and sync Actions history to populate service analytics.")


def render_local_analysis(repository: str) -> None:
    st.markdown("#### Credential-free log triage")
    st.caption("Parsing and guidance run locally; log content is not sent to an external service.")
    if "manual_log_content" not in st.session_state:
        st.session_state["manual_log_content"] = ""
    actions = st.columns([1, 1, 4])
    if actions[0].button("Load sample log", key="load_sample_log"):
        st.session_state["manual_log_content"] = _SAMPLE_LOG
        st.rerun()
    if actions[1].button("Clear log", key="clear_manual_log"):
        st.session_state["manual_log_content"] = ""
        st.session_state.pop("local_analysis", None)
        st.rerun()
    upload = st.file_uploader(
        "Upload a CI log",
        type=["log", "txt"],
        key="ci_log_upload",
        max_upload_size=max(1, Config.MAX_LOG_UPLOAD_BYTES // (1024 * 1024)),
    )
    pasted = st.text_area("Or paste log output", key="manual_log_content", height=260, placeholder="Paste the failing job output here…")
    if st.button("Analyze log locally", type="primary", key="analyze_local"):
        content_bytes = upload.getvalue() if upload is not None else pasted.encode("utf-8")
        if len(content_bytes) > Config.MAX_LOG_UPLOAD_BYTES:
            st.error(f"The log exceeds the {Config.MAX_LOG_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        elif not content_bytes.strip():
            st.error("Upload or paste a non-empty CI log.")
        elif _cooldown_allows("local_analysis"):
            parsed = parse_log_content(content_bytes.decode("utf-8", errors="replace"))
            brief = build_local_debugging_brief(parsed.primary_error, repository or None) if parsed.primary_error else None
            st.session_state["local_analysis"] = {"parsed": parsed, "brief": brief}
    result: dict[str, Any] | None = st.session_state.get("local_analysis")
    if not result:
        return
    parsed: LogParseResult = result["parsed"]
    st.divider()
    if not parsed.primary_error:
        st.warning("No concrete error was found. Include the failed command and surrounding diagnostic lines.")
        return
    st.success(f"Detected {parsed.error_count} error signal(s) across {parsed.total_lines} log lines.")
    render_brief(result["brief"], key_prefix="local")
    with st.expander("Parser evidence"):
        for error in parsed.errors:
            st.code(error.raw_error_block or f"{error.error_type}: {error.error_message}")


def render_live_analysis(repository: str) -> None:
    st.markdown("#### AI-enriched GitHub failed-run analysis")
    provider = get_llm_provider_status()
    st.caption(
        f"Fetches bounded, redacted Actions logs and uses {provider.display_name}. "
        "Tavily web research is optional."
    )
    missing = []
    runtime_token = _github_runtime_token()
    if not runtime_token:
        missing.append("a session or deployment GitHub token with Actions: read")
    if not provider.ready:
        missing.append(provider.detail)
    if missing:
        st.warning("AI enrichment needs " + "; ".join(missing) + " Offline triage remains available.")
    if Config.TAVILY_API_KEY:
        st.info("Tavily research is enabled for this analysis.")
    else:
        st.info("Tavily is not configured; analysis will continue with logs, code context, and the selected LLM.")
    repository_problem = _repository_error(repository)
    if repository_problem:
        st.info(repository_problem)
    if st.button(
        "Analyze latest failed run",
        type="primary",
        key="ai_enriched_rca",
        disabled=bool(missing or repository_problem),
    ):
        if _cooldown_allows("live_rca"):
            with st.spinner("Fetching the latest failed run and building the debugging brief…"):
                try:
                    from src.integrations.github_automation import GitHubAutomationService

                    workflow_choice = st.session_state.get("automation_workflow", "CI/CD Pipeline")
                    workflow_name = None if workflow_choice == "All workflows" else workflow_choice
                    automation = GitHubAutomationService(token=runtime_token, repository=repository)
                    context = automation.latest_failed_run_context(
                        repository=repository,
                        workflow_name=workflow_name,
                        include_system_workflows=workflow_name is None,
                        include_logs=True,
                    )
                    log_text = _extract_log_text(context)
                    if not log_text:
                        context_error = _as_dict(context).get("logs_error")
                        raise ValueError(context_error or "GitHub returned no readable log content.")
                    run_payload = _as_dict(_as_dict(context).get("run"))
                    st.session_state["live_analysis"] = run_enriched_analysis(
                        repository,
                        log_text,
                        workflow_run_id=run_payload.get("id"),
                        github_token=runtime_token,
                    )
                except Exception as exc:
                    st.error(f"Live analysis failed: {_safe_message(exc)}")
    state = st.session_state.get("live_analysis")
    if state is None:
        return
    st.divider()
    if state.debugging_brief:
        st.success("AI analysis completed.")
        render_brief(state.debugging_brief, key_prefix="live")
    else:
        st.error(state.error_message or "No debugging brief was generated.")


def _refresh_repository_snapshot(automation: Any, repository: str) -> Any:
    status = automation.connect_repository(repository)
    status_payload = _as_dict(status)
    if not status_payload.get("connected"):
        raise ValueError(status_payload.get("error") or "GitHub could not connect to this repository.")
    commits = automation.get_recent_commits(repository=repository, limit=10)
    runs = automation.get_workflow_runs(repository=repository, limit=20)
    get_workflows = getattr(automation, "get_workflows", None)
    workflows = []
    workflow_error = None
    if callable(get_workflows):
        try:
            workflows = get_workflows(repository=repository, limit=100)
        except Exception as exc:
            workflow_error = _safe_message(exc)
    _persist_github_snapshot(
        repository,
        status=status,
        commits=commits,
        runs=runs,
        workflows=workflows,
        workflow_error=workflow_error,
    )
    return status


def _render_repository_activity(snapshot: dict[str, Any]) -> None:
    commits = snapshot.get("commits", [])
    runs = snapshot.get("runs", [])
    if not commits and not runs:
        return
    st.markdown("#### Live repository activity")
    left, right = st.columns(2)
    with left:
        st.caption("Recent application commits")
        commit_frame = pd.DataFrame(commits)
        if not commit_frame.empty:
            if "total_changes" in commit_frame.columns:
                commit_frame["changes"] = commit_frame["total_changes"]
            if "sha" in commit_frame.columns:
                commit_frame["sha"] = commit_frame["sha"].astype(str).str[:10]
            visible = [
                column
                for column in ("committed_at", "sha", "message", "author_login", "author_name", "changes", "html_url")
                if column in commit_frame.columns
            ]
            st.dataframe(
                commit_frame[visible].head(10),
                width="stretch",
                hide_index=True,
                column_config={"html_url": st.column_config.LinkColumn("Commit")},
            )
    with right:
        st.caption("Recent GitHub Actions runs")
        run_frame = pd.DataFrame(runs)
        if not run_frame.empty:
            if "head_sha" in run_frame.columns:
                run_frame["head_sha"] = run_frame["head_sha"].astype(str).str[:10]
            visible = [
                column
                for column in ("created_at", "name", "workflow_name", "branch", "status", "conclusion", "head_sha", "html_url")
                if column in run_frame.columns
            ]
            st.dataframe(
                run_frame[visible].head(10),
                width="stretch",
                hide_index=True,
                column_config={"html_url": st.column_config.LinkColumn("Run")},
            )


def _render_automated_prediction(result: Any) -> None:
    prediction = getattr(result, "prediction", None)
    prediction_input = getattr(result, "input", None)
    result_payload = _as_dict(result)
    input_payload = _as_dict(prediction_input or result_payload.get("input"))
    commit = _as_dict(input_payload.get("commit"))
    files = commit.get("files") or []

    st.divider()
    st.markdown("### Latest application change")
    summary = st.columns(5)
    summary[0].metric("Commit", str(commit.get("sha") or "unknown")[:10])
    summary[1].metric("Files", len(files))
    summary[2].metric("Lines added", int(commit.get("additions", 0) or 0))
    summary[3].metric("Lines deleted", int(commit.get("deletions", 0) or 0))
    workflow_status = (
        result_payload.get("workflow_status")
        or input_payload.get("workflow_status")
        or input_payload.get("run_status")
        or "not completed"
    )
    summary[4].metric("Target run", str(workflow_status).replace("_", " ").title())
    if commit.get("message"):
        st.write(commit["message"])
    commit_author = commit.get("author_login") or commit.get("author_name")
    details = [value for value in (commit_author, commit.get("committed_at")) if value]
    if details:
        st.caption(" · ".join(map(str, details)))

    prediction_id = result_payload.get("prediction_id") or getattr(result, "prediction_id", None)
    retrospective = bool(
        result_payload.get("is_retrospective")
        or result_payload.get("retrospective")
        or str(result_payload.get("mode", "")).lower() == "retrospective"
    )
    if retrospective or not prediction_id:
        st.info("Retrospective display-only score. This completed change is not recorded as pre-CI feedback.")
    else:
        st.success("Pre-CI prediction recorded; its eventual workflow outcome can close the feedback loop.")
    if result_payload.get("record_reason"):
        st.caption(str(result_payload["record_reason"]))
    if prediction is not None:
        render_prediction(prediction, context="Automated failure-risk prediction")

    if files:
        with st.expander("Changed files", expanded=False):
            file_frame = pd.DataFrame([_as_dict(item) for item in files])
            visible = [
                column
                for column in ("filename", "status", "additions", "deletions", "changes")
                if column in file_frame.columns
            ]
            st.dataframe(file_frame[visible], width="stretch", hide_index=True)


def _render_automated_failure(result: dict[str, Any], *, key_prefix: str = "automated") -> None:
    st.divider()
    context = _as_dict(result.get("context"))
    run = _as_dict(context.get("run"))
    if run:
        st.caption(
            f"GitHub Actions run {run.get('run_number') or run.get('id') or 'unknown'} · "
            f"{run.get('workflow_name') or run.get('name') or 'workflow'} · "
            f"{run.get('conclusion') or run.get('status') or 'unknown'}"
        )
    brief = result.get("brief")
    parsed = result.get("parsed")
    if brief is not None:
        st.success("Latest failed-run logs were fetched and analyzed automatically.")
        render_brief(brief, key_prefix=key_prefix)
    elif parsed is not None:
        st.warning("The logs were fetched, but no concrete error signal could be isolated.")


def render_automation(repository: str, prediction_service: FailurePredictionService) -> None:
    """Repository-first controls for automatic metadata, prediction, and RCA."""

    from src.integrations.github_automation import GitHubAutomationService

    if st.session_state.pop("clear_github_token_on_next_run", False):
        st.session_state.pop("github_session_token", None)
    problem = _repository_error(repository)
    snapshot = _github_snapshot(repository)
    status = snapshot.get("status", {})
    connected = bool(status.get("connected"))
    authenticated = bool(status.get("authenticated"))

    st.subheader("GitHub automation cockpit")
    st.caption("Connect once, then score real commits and analyze Actions failures without uploading code or logs.")
    with st.container(border=True):
        heading = "Connected" if connected else "Repository not connected"
        st.markdown(f"**{heading}**")
        if connected:
            access = f"Authenticated as {status.get('account_login')}" if authenticated else "Public GitHub metadata access"
            st.write(f"{status.get('repository') or repository} · {access}")
            if snapshot.get("refreshed_at"):
                st.caption(f"Last activity refresh: {snapshot['refreshed_at']}")
        else:
            st.write("Verify repository access and load its latest commits and workflow runs.")
        session_token = st.text_input(
            "GitHub token (session only)",
            type="password",
            key="github_session_token",
            help="Kept only in this Streamlit session; it is never written to disk or displayed back.",
            placeholder="Optional for public metadata",
        )
        if session_token.strip():
            st.caption("Session token supplied. Fine-grained access should include repository Contents: read and Actions: read.")
        elif Config.GITHUB_ACCESS_TOKEN:
            st.caption("Using the deployment GitHub token. It should have repository Contents: read and Actions: read.")
        else:
            st.caption("Anonymous access supports public commits and run metadata. A fine-grained token is required for Actions logs.")
        if problem:
            st.error(problem)
        runtime_token = _github_runtime_token()
        automation = GitHubAutomationService(
            token=runtime_token,
            repository=repository if repository and not problem else None,
        )
        controls = st.columns([1.3, 1, 4])
        connect_label = "Refresh GitHub activity" if connected else "Connect repository"
        if controls[0].button(
            connect_label,
            type="primary",
            key="github_connect_refresh",
            disabled=bool(problem),
        ) and _cooldown_allows("github_connect"):
            with st.spinner("Verifying GitHub access and loading repository activity…"):
                try:
                    _refresh_repository_snapshot(automation, repository)
                    st.session_state["artifact_notice"] = f"GitHub repository {repository} is connected."
                    st.rerun()
                except Exception as exc:
                    st.error(f"GitHub connection failed: {_safe_message(exc)}")
        if controls[1].button("Disconnect", key="github_disconnect", disabled=not connected):
            for key in ("github_snapshot", "automated_prediction", "automated_failure"):
                st.session_state.pop(key, None)
            st.session_state["clear_github_token_on_next_run"] = True
            st.session_state["artifact_notice"] = "The dashboard repository connection was cleared."
            st.rerun()

    snapshot = _github_snapshot(repository)
    status = snapshot.get("status", {})
    connected = bool(status.get("connected"))
    authenticated = bool(status.get("authenticated"))

    discovered_workflow_names = {
        str(workflow.get("name"))
        for workflow in snapshot.get("workflows", [])
        if workflow.get("name") and str(workflow.get("state") or "active").lower() == "active"
    }
    # Run names are a compatibility fallback for repositories or tokens that
    # cannot enumerate workflow definitions.
    run_workflow_names = {
        str(run.get("workflow_name"))
        for run in snapshot.get("runs", [])
        if run.get("workflow_name")
    }
    workflow_names = sorted(discovered_workflow_names or run_workflow_names)
    if snapshot.get("workflow_error"):
        st.caption(
            "Workflow definitions could not be enumerated; the selector is using recent run names. "
            + str(snapshot["workflow_error"])
        )
    default_workflow = "CI/CD Pipeline"
    workflow_options = ["All workflows"] + workflow_names
    if default_workflow not in workflow_options:
        workflow_options.insert(1, default_workflow)
    default_index = workflow_options.index(default_workflow)
    workflow_choice = st.selectbox(
        "Target workflow",
        workflow_options,
        index=default_index,
        key="automation_workflow",
        help="Prediction and failed-run analysis stay scoped to the application pipeline by default.",
    )
    selected_workflow = None if workflow_choice == "All workflows" else workflow_choice

    st.markdown("#### Automatic actions")
    actions = st.columns(3)
    with actions[0]:
        with st.container(border=True):
            st.markdown("**Predict latest change**")
            st.caption("Build leakage-safe features from the newest commit and prior CI outcomes.")
            predict_disabled = bool(problem or not connected or not prediction_service.model_available)
            if st.button(
                "Predict failure percentage",
                type="primary",
                key="github_predict_latest",
                disabled=predict_disabled,
            ) and _cooldown_allows("github_predict_latest"):
                with st.spinner("Reading the latest application changes and scoring CI risk…"):
                    try:
                        st.session_state["automated_prediction"] = automation.predict_latest_change(
                            prediction_service,
                            repository=repository,
                            workflow_name=selected_workflow,
                            record=True,
                        )
                        st.success("Latest change scored.")
                    except Exception as exc:
                        st.error(f"Latest-change prediction failed: {_safe_message(exc)}")
            if not prediction_service.model_available:
                st.caption("Train or download the failure model to enable scoring.")
            elif not connected:
                st.caption("Connect this repository before scoring its latest commit.")
    with actions[1]:
        with st.container(border=True):
            st.markdown("**Analyze latest failure**")
            st.caption("Fetch Actions logs and run deterministic root-cause triage locally.")
            logs_need_auth = connected and not authenticated
            if st.button(
                "Analyze latest failed run",
                key="github_analyze_latest_failure",
                disabled=bool(problem or not connected or logs_need_auth),
            ) and _cooldown_allows("github_latest_failure"):
                with st.spinner("Fetching the latest failed-run logs from GitHub…"):
                    try:
                        context = automation.latest_failed_run_context(
                            repository=repository,
                            workflow_name=selected_workflow,
                            include_system_workflows=selected_workflow is None,
                            include_logs=True,
                        )
                        log_text = _extract_log_text(context)
                        if not log_text:
                            context_error = _as_dict(context).get("logs_error")
                            raise ValueError(
                                context_error
                                or "GitHub returned no readable log content for the latest failed run."
                            )
                        parsed = parse_log_content(log_text)
                        brief = (
                            build_local_debugging_brief(parsed.primary_error, repository)
                            if parsed.primary_error
                            else None
                        )
                        st.session_state["automated_failure"] = {
                            "context": context,
                            "parsed": parsed,
                            "brief": brief,
                        }
                        st.success("Latest failure analyzed.")
                    except Exception as exc:
                        st.error(f"Automatic failed-run analysis failed: {_safe_message(exc)}")
            if logs_need_auth:
                st.caption("Add a repository-scoped token with Actions: read permission to fetch logs.")
            elif not connected:
                st.caption("Connect this repository before fetching failed-run logs.")
    with actions[2]:
        with st.container(border=True):
            st.markdown("**Sync CI history**")
            st.caption("Refresh the real workflow dataset used by analytics and future retraining.")
            if st.button(
                "Sync completed workflow runs",
                key="github_sync_history",
                disabled=bool(problem or not connected),
            ) and _cooldown_allows("github_sync_history"):
                with st.spinner("Collecting completed GitHub Actions runs and commit metadata…"):
                    try:
                        collected = HistoricalRunCollector(token=runtime_token).collect_repository_runs(
                            repository,
                            limit=200,
                        )
                        if collected.empty:
                            raise ValueError("No completed trainable workflow runs were found.")
                        _atomic_save_csv(collected, Config.HISTORICAL_DATASET_PATH)
                        st.session_state["artifact_notice"] = f"Synced {len(collected)} genuine workflow runs."
                        st.rerun()
                    except Exception as exc:
                        st.error(f"History sync failed: {_safe_message(exc)}")
            if not connected:
                st.caption("Connect this repository before synchronizing workflow history.")

    _render_repository_activity(snapshot)
    if result := st.session_state.get("automated_prediction"):
        _render_automated_prediction(result)
    if failure := st.session_state.get("automated_failure"):
        _render_automated_failure(failure)


def render_analyze(repository: str) -> None:
    mode = st.radio(
        "Analysis source",
        ("GitHub automatic RCA", "AI-enriched GitHub RCA", "Advanced: paste or upload log"),
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode == "GitHub automatic RCA":
        st.markdown("#### Latest automatic GitHub analysis")
        st.caption("The Automation tab fetches the failed-run logs directly; nothing needs to be uploaded.")
        failure = st.session_state.get("automated_failure")
        if failure:
            _render_automated_failure(failure, key_prefix="automatic_analysis")
        else:
            st.info("Connect the repository, then choose **Analyze latest failed run** in Automation.")
    elif mode == "AI-enriched GitHub RCA":
        render_live_analysis(repository)
    else:
        render_local_analysis(repository)


def _manual_prediction_record() -> dict[str, Any]:
    first = st.columns(4)
    files_changed = first[0].number_input("Files changed", 0, 10000, 4)
    lines_added = first[1].number_input("Lines added", 0, 1_000_000, 80)
    lines_deleted = first[2].number_input("Lines deleted", 0, 1_000_000, 15)
    commits = first[3].number_input("Commits", 0, 1000, 1)
    flags = st.columns(5)
    dependency = flags[0].checkbox("Dependencies")
    workflow = flags[1].checkbox("CI workflow")
    docker = flags[2].checkbox("Docker")
    tests = flags[3].checkbox("Tests")
    infrastructure = flags[4].checkbox("Infrastructure")
    history = st.columns(4)
    previous = history[0].selectbox("Previous run", ("success", "failure", "unknown"))
    recent = history[1].number_input("Recent failures", 0, 1000, 1)
    same_workflow = history[2].number_input("Prior workflow failures", 0, 1000, 1)
    same_branch = history[3].number_input("Prior branch failures", 0, 1000, 0)
    paths_text = st.text_area("Changed file paths (one per line)", "app.py\ntests/test_app.py\nrequirements.txt", height=105)
    paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
    return {
        "files_changed": int(files_changed), "lines_added": int(lines_added), "lines_deleted": int(lines_deleted),
        "number_of_commits": int(commits), "changed_files_json": json.dumps(paths),
        "dependency_files_changed": dependency, "ci_workflow_files_changed": workflow,
        "docker_files_changed": docker, "test_files_changed": tests,
        "infrastructure_files_changed": infrastructure, "previous_run_status": previous,
        "previous_failure_count": int(recent), "recent_failure_count": int(recent),
        "recent_failure_rate": min(float(recent) / 10.0, 1.0),
        "previous_failures_same_workflow": int(same_workflow),
        "previous_failures_same_branch": int(same_branch),
        "similar_previous_failures": int(same_workflow + same_branch),
    }


def render_predict(repository: str, service: FailurePredictionService) -> None:
    if not service.model_available:
        if Config.PREDICTOR_MODEL_PATH.exists():
            st.error("Prediction is inactive because the model artifact could not be loaded. Install a trusted compatible artifact.")
        else:
            st.warning("Prediction is inactive because no trained model artifact is loaded.")
        st.markdown("Use **Data & model** to validate real workflow history and train the baseline model.")
        return
    source = st.radio("Prediction input", ("Manual feature snapshot", "GitHub commit"), horizontal=True)
    if source == "Manual feature snapshot":
        st.caption("Scenario scores are not written to feedback history.")
        record = _manual_prediction_record()
        if st.button("Score feature snapshot", type="primary", key="score_manual_snapshot") and _cooldown_allows("manual_prediction"):
            st.session_state["prediction_result"] = service.predict(record)
            st.session_state["prediction_context"] = "Scenario risk score"
            st.session_state.pop("prediction_id", None)
    else:
        st.caption("Use a commit whose target CI workflow has not completed. Live scores are recorded for feedback.")
        branch = st.text_input("Branch", "main", key="prediction_branch")
        commit_sha = st.text_input("Commit SHA", key="prediction_sha")
        workflow = st.text_input("Workflow name", "CI/CD Pipeline", key="prediction_workflow")
        confirmed = st.checkbox("I confirm this commit has not completed the target CI workflow.")
        problem = _repository_error(repository)
        if problem:
            st.info(problem)
        github_token = _github_runtime_token()
        if not github_token:
            st.info("Using anonymous GitHub access; public repositories remain available.")
        disabled = bool(problem or not commit_sha.strip() or not confirmed)
        if st.button(
            "Predict pre-CI risk",
            type="primary",
            key="predict_manual_commit",
            disabled=disabled,
        ) and _cooldown_allows("github_prediction"):
            with st.spinner("Collecting leakage-safe commit and prior-run features…"):
                try:
                    features = HistoricalRunCollector(token=github_token).build_prediction_features(
                        repository,
                        commit_sha.strip(),
                        branch or None,
                        workflow or None,
                    )
                    prediction, prediction_id = service.predict_and_record(
                        repository=repository, workflow=workflow or None, run_id=None,
                        commit_sha=commit_sha.strip(), features=features,
                    )
                    st.session_state["prediction_result"] = prediction
                    st.session_state["prediction_context"] = "Pre-CI commit risk"
                    st.session_state["prediction_id"] = prediction_id
                except Exception as exc:
                    st.error(f"Prediction failed: {_safe_message(exc)}")
    prediction = st.session_state.get("prediction_result")
    if prediction is not None:
        st.divider()
        render_prediction(prediction, context=st.session_state.get("prediction_context", "Prediction"))
        if prediction_id := st.session_state.get("prediction_id"):
            st.caption(f"Feedback record: {prediction_id}")


def render_dataset_report(report: DatasetValidationReport) -> None:
    columns = st.columns(4)
    columns[0].metric("Runs", report.total_runs)
    columns[1].metric("Successes", report.success_runs)
    columns[2].metric("Failures", report.failure_runs)
    columns[3].metric("Failure rate", f"{report.class_balance_failure_rate:.1%}")
    if report.sufficient_for_training:
        st.success("Dataset passes the minimum training gate.")
    else:
        st.info("Training requires at least 20 real runs with both success and failure outcomes.")
    for warning in report.warnings:
        st.warning(warning)


def render_data_and_model(repository: str, service: FailurePredictionService) -> None:
    dataset = _load_csv(Config.HISTORICAL_DATASET_PATH)
    report = validate_dataset(dataset) if not dataset.empty else validate_dataset(Config.HISTORICAL_DATASET_PATH)
    columns = st.columns(3)
    columns[0].metric("Dataset", f"{report.total_runs} runs" if report.total_runs else "Missing")
    columns[1].metric("Failure model", _model_status(service) if service.model_available else "Missing")
    columns[2].metric("Category model", "Loaded" if service.predictor.category_available else "Optional / missing")
    with st.expander("Artifact locations"):
        st.code(f"dataset: {Config.HISTORICAL_DATASET_PATH}\nmodel: {Config.PREDICTOR_MODEL_PATH}\nhistory: {Config.PREDICTION_HISTORY_PATH}")
    upload_tab, collect_tab, train_tab = st.tabs(("Upload & inspect", "Collect from GitHub", "Train & evaluate"))
    with upload_tab:
        if report.total_runs:
            render_dataset_report(report)
            with st.expander("Dataset preview"):
                st.dataframe(dataset.tail(50), width="stretch", hide_index=True)
        upload = st.file_uploader(
            "Upload real historical workflow CSV",
            type=["csv"],
            key="dataset_upload",
            max_upload_size=25,
        )
        upload_frame = None
        if upload is not None:
            if len(upload.getvalue()) > 25 * 1024 * 1024:
                st.error("The dataset exceeds the 25 MB limit.")
            else:
                try:
                    upload_frame = pd.read_csv(io.BytesIO(upload.getvalue()))
                    render_dataset_report(validate_dataset(upload_frame))
                except Exception as exc:
                    st.error(f"Could not read the CSV: {_safe_message(exc)}")
        can_save = bool(upload_frame is not None and not upload_frame.empty and {"timestamp", "actual_failure"}.issubset(upload_frame.columns))
        if st.button("Activate uploaded dataset", key="activate_uploaded_dataset", disabled=not can_save):
            _atomic_save_csv(upload_frame, Config.HISTORICAL_DATASET_PATH)
            st.session_state["artifact_notice"] = "The uploaded dataset is now active."
            st.rerun()
    with collect_tab:
        st.write("Collect genuine completed workflow runs and commit metadata from GitHub Actions.")
        limit = st.slider("Maximum runs", 20, 500, 100, 10)
        problem = _repository_error(repository)
        if problem:
            st.info(problem)
        github_token = _github_runtime_token()
        if not github_token:
            st.info("Using anonymous GitHub access for this public repository; failure categories may remain unclassified.")
        if st.button(
            "Collect and activate history",
            key="collect_history_advanced",
            disabled=bool(problem),
        ) and _cooldown_allows("collect_history"):
            with st.spinner("Collecting genuine workflow history…"):
                try:
                    collected = HistoricalRunCollector(token=github_token).collect_repository_runs(
                        repository,
                        limit=int(limit),
                    )
                    if collected.empty:
                        raise ValueError("No completed trainable workflow runs were found.")
                    _atomic_save_csv(collected, Config.HISTORICAL_DATASET_PATH)
                    st.session_state["artifact_notice"] = f"Collected {len(collected)} workflow runs."
                    st.rerun()
                except Exception as exc:
                    st.error(f"Collection failed: {_safe_message(exc)}")
    with train_tab:
        render_dataset_report(report)
        st.caption("Training uses a chronological holdout and excludes final outcome metadata from features.")
        if st.button(
            "Train and activate model",
            type="primary",
            key="train_model",
            disabled=not report.sufficient_for_training,
        ) and _cooldown_allows("train_model"):
            with st.spinner("Training models and evaluating the temporal holdout…"):
                try:
                    artifact = FailurePredictorTrainer().train(
                        Config.HISTORICAL_DATASET_PATH, Config.PREDICTOR_MODEL_PATH,
                        Config.PREDICTOR_METADATA_PATH, Config.CATEGORY_MODEL_PATH,
                        Config.CATEGORY_METADATA_PATH,
                    )
                    st.session_state["training_result"] = artifact["metrics"]
                    st.session_state["artifact_notice"] = "Training completed and the new model is active."
                    get_prediction_service.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {_safe_message(exc)}")
        metrics = st.session_state.get("training_result") or _read_model_metrics()
        if metrics:
            score_columns = st.columns(4)
            for column, name in zip(score_columns, ("accuracy", "precision", "recall", "f1")):
                column.metric(name.title(), f"{float(metrics.get(name, 0)):.1%}")
            with st.expander("Evaluation details"):
                st.json(metrics)


def render_feedback() -> None:
    st.subheader("Prediction feedback loop")
    st.caption("Only real model predictions are stored; scenario scores and unavailable-model attempts are excluded.")
    render_feedback_summary(include_table=True)
    history = _load_csv(Config.PREDICTION_HISTORY_PATH)
    if not history.empty:
        st.download_button("Download prediction history", history.to_csv(index=False), "prediction_history.csv", "text/csv")


_install_theme()
_require_dashboard_access()

with st.sidebar:
    st.markdown("### ◆ CI Signal")
    st.caption("Failure intelligence console")
    repository_name = st.text_input(
        "GitHub repository",
        Config.DEFAULT_TEST_REPO,
        key="repository_name",
        placeholder="owner/repository",
    ).strip()
    st.divider()
    st.markdown("**Runtime**")
    st.write("● Offline deterministic RCA · ready")
    sidebar_snapshot = _github_snapshot(repository_name)
    sidebar_status = sidebar_snapshot.get("status", {})
    if sidebar_status.get("connected"):
        identity = sidebar_status.get("account_login") or "public metadata"
        st.write(f"● GitHub connected · {identity}")
    else:
        st.write("○ GitHub repository · setup needed")
    if st.button(
        "Open Automation setup" if not sidebar_status.get("connected") else "Open Automation",
        key="open_automation_setup",
        width="stretch",
    ):
        st.session_state["dashboard_tab"] = "Automation"
        st.rerun()
    sidebar_provider = get_llm_provider_status()
    if sidebar_provider.ready:
        st.write(f"● AI provider · {sidebar_provider.display_name}")
    else:
        st.write("○ AI enrichment · disabled")
    st.write("● Tavily research · optional/ready" if Config.TAVILY_API_KEY else "○ Tavily research · optional")
    st.divider()
    st.caption(f"Version {Config.VERSION} · single-replica local persistence")

st.markdown(
    """<div class="signal-hero"><div class="signal-kicker">CI/CD failure intelligence</div>
    <h1>Predict earlier. Diagnose automatically.</h1><p>Connect GitHub once to score real application changes,
    analyze failed workflows, and close the loop with verified CI outcomes.</p></div>""",
    unsafe_allow_html=True,
)

if notice := st.session_state.pop("artifact_notice", None):
    st.success(notice)

prediction_service = _prediction_service()
automation_tab, overview_tab, analytics_tab, analyze_tab, predict_tab, model_tab, feedback_tab = st.tabs(
    ("Automation", "Overview", "Analytics", "Failure analysis", "Risk lab", "Data & model", "Feedback"),
    default="Automation",
    key="dashboard_tab",
    on_change="rerun",
)
with automation_tab:
    render_automation(repository_name, prediction_service)
with overview_tab:
    render_overview(repository_name, prediction_service)
with analytics_tab:
    render_analytics(repository_name)
with analyze_tab:
    render_analyze(repository_name)
with predict_tab:
    render_predict(repository_name, prediction_service)
with model_tab:
    render_data_and_model(repository_name, prediction_service)
with feedback_tab:
    render_feedback()
