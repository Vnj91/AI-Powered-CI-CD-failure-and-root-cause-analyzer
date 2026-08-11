"""
app.py - Simple Web Dashboard for CI/CD Analyzer
Run: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.graph.workflow import run_analysis
from src.graph.state import GraphState
from src.prediction import FailurePredictionService, HistoricalRunCollector
from config import Config

st.set_page_config(
    page_title="CI/CD Root Cause Analyzer",
    page_icon="🔧",
    layout="wide"
)

st.title("CI/CD Root Cause Analyzer")
st.markdown("Analyze failed GitHub Actions builds using AI agents")


@st.cache_resource
def get_prediction_service() -> FailurePredictionService:
    return FailurePredictionService(
        model_path=Config.PREDICTOR_MODEL_PATH,
        history_path=Config.PREDICTION_HISTORY_PATH,
        metadata_path=Config.PREDICTOR_METADATA_PATH,
        category_model_path=Config.CATEGORY_MODEL_PATH,
        category_metadata_path=Config.CATEGORY_METADATA_PATH,
    )


@st.cache_resource
def get_history_collector() -> HistoricalRunCollector:
    return HistoricalRunCollector()


def render_prediction(prediction):
    st.subheader("CI/CD Failure Risk Prediction")

    if not prediction.model_available:
        st.info("No trained failure predictor is available yet. Collect historical runs and train the model first.")
        if prediction.warnings:
            for warning in prediction.warnings:
                st.warning(warning)
        return

    probability = prediction.failure_probability or 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Failure Probability", f"{probability:.0%}")
    col2.metric("Risk Level", prediction.risk_level)
    col3.metric("Predicted Outcome", "LIKELY TO FAIL" if prediction.predicted_failure else "LIKELY TO PASS")
    col4.metric("Model Version", prediction.model_version or "unknown")

    if prediction.predicted_category:
        category_suffix = (
            f" ({prediction.category_confidence:.0%} confidence)"
            if prediction.category_confidence is not None
            else ""
        )
        st.write(f"**Predicted Category:** {prediction.predicted_category}{category_suffix}")

    if prediction.top_risk_factors:
        st.markdown("**Top Risk Factors**")
        for factor in prediction.top_risk_factors:
            st.markdown(f"- {factor.feature}: value={factor.value:.2f}, importance={factor.importance:.4f}")

    with st.expander("All Feature Contributions"):
        for factor in prediction.feature_importances[:25]:
            st.markdown(
                f"- {factor.feature}: value={factor.value:.2f}, importance={factor.importance:.4f}, contribution={factor.contribution:.4f}"
            )


def build_prediction_inputs(repo_name: str, branch: str | None, commit_sha: str | None, workflow_name: str | None):
    if not commit_sha:
        from src.tools.github_loader import get_latest_workflow_run

        latest_run = get_latest_workflow_run(repo_name)
        if latest_run is None:
            raise ValueError("No recent workflow run found. Provide a commit SHA to score.")

        commit_sha = latest_run.head_sha
        branch = branch or latest_run.head_branch
        workflow_name = workflow_name or getattr(getattr(latest_run, "workflow", None), "name", None) or getattr(latest_run, "name", None)

    collector = get_history_collector()
    features = collector.build_prediction_features(
        repository=repo_name,
        commit_sha=commit_sha,
        branch=branch,
        workflow_name=workflow_name,
    )
    return features

repo_name = st.text_input(
    "GitHub Repository",
    placeholder="owner/repo",
    value="Yasshu55/Test-repo"
)

with st.expander("Failure Risk Prediction", expanded=True):
    st.caption("Provide a commit SHA and branch for a true pre-run prediction. If omitted, the app will use the latest workflow run as a proxy input.")
    branch_name = st.text_input("Branch", placeholder="main")
    commit_sha = st.text_input("Commit SHA", placeholder="Optional for latest run fallback")
    workflow_name = st.text_input("Workflow Name", placeholder="Optional")

    if st.button("Predict Failure Risk", type="primary"):
        if "/" not in repo_name:
            st.error("Enter repository as owner/repo")
        else:
            with st.spinner("Scoring failure risk..."):
                try:
                    service = get_prediction_service()
                    features = build_prediction_inputs(repo_name, branch_name or None, commit_sha or None, workflow_name or None)
                    prediction, _ = service.predict_and_record(
                        repository=features["repository"],
                        workflow=features.get("workflow_name"),
                        run_id=features.get("run_id"),
                        commit_sha=features.get("commit_sha"),
                        features=features,
                    )
                    render_prediction(prediction)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

st.divider()

if st.button("Analyze Latest Failed Run", type="secondary"):
    if "/" not in repo_name:
        st.error("Enter repository as owner/repo")
    else:
        with st.spinner("Analyzing... (this takes ~30 seconds)"):
            try:
                result = run_analysis(repo_name)

                if result.debugging_brief:
                    brief = result.debugging_brief

                    st.success("Analysis Complete!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Severity", brief.severity.upper())
                    col2.metric("Confidence", f"{brief.confidence_score:.0%}")
                    col3.metric("Fixes Found", len(brief.fix_suggestions))

                    st.subheader("Error")
                    st.code(f"{brief.error_type}: {brief.error_message}")

                    st.subheader("Root Cause")
                    st.write(brief.root_cause_summary)

                    st.subheader("Fix Suggestions")
                    for fix in brief.fix_suggestions:
                        with st.expander(f"Fix #{fix.priority}: {fix.title} ({fix.confidence:.0%})"):
                            st.write(fix.description)
                            if fix.implementation_steps:
                                st.markdown("**Steps:**")
                                for i, step in enumerate(fix.implementation_steps, 1):
                                    st.markdown(f"{i}. {step}")
                            if fix.code_example:
                                st.code(fix.code_example)

                    if brief.relevant_links:
                        st.subheader("Helpful Links")
                        for link in brief.relevant_links:
                            st.markdown(f"- {link}")

                    st.download_button(
                        "Download Report (Markdown)",
                        brief.to_markdown(),
                        file_name="debugging_brief.md",
                        mime="text/markdown"
                    )

                    try:
                        from src.tools.github_loader import get_latest_workflow_run

                        latest_run = get_latest_workflow_run(repo_name)
                        if latest_run:
                            service = get_prediction_service()
                            collector = get_history_collector()
                            latest_run_features = collector.build_prediction_features(
                                repository=repo_name,
                                commit_sha=latest_run.head_sha,
                                branch=getattr(latest_run, "head_branch", None),
                                workflow_name=getattr(getattr(latest_run, "workflow", None), "name", None) or getattr(latest_run, "name", None),
                            )
                            prediction, _ = service.predict_and_record(
                                repository=latest_run_features["repository"],
                                workflow=latest_run_features.get("workflow_name"),
                                run_id=latest_run_features.get("run_id"),
                                commit_sha=latest_run_features.get("commit_sha"),
                                features=latest_run_features,
                                actual_failure=1,
                                actual_category=brief.error_category,
                            )
                            render_prediction(prediction)
                    except Exception:
                        pass
                else:
                    st.warning("No debugging brief generated")
                    if result.error_message:
                        st.error(result.error_message)

            except Exception as e:
                st.error(f"Error: {e}")

with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses AI agents to:
    1. Fetch failed build logs
    2. Parse and extract errors
    3. Analyze root cause
    4. Research solutions
    5. Generate fix suggestions
    6. Predict CI/CD failure risk before the run completes
    
    **Tech Stack:**
    - LangGraph (Supervisor Pattern)
    - Claude 3.5 Sonnet
    - Tavily Search
    - PyGithub
    - Scikit-learn
    """)