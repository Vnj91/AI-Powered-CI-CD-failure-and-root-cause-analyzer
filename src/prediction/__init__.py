"""Prediction package for CI/CD failure risk modeling."""

from .schemas import (
    FailurePrediction,
    FeatureImportance,
    EvaluationMetrics,
    PredictionHistoryRecord,
    WorkflowRunRecord,
)
from .feature_extractor import FailureFeatureExtractor
from .data_collector import HistoricalRunCollector
from .predictor import FailurePredictor
from .trainer import FailurePredictorTrainer
from .history_store import PredictionHistoryStore
from .service import FailurePredictionService
from .feedback import PredictionFeedbackService
from .dataset_validator import validate_dataset, DatasetValidationReport, LEAKAGE_COLUMNS
from .category_mapper import FailureCategory, map_failure_category, conclusion_to_actual_failure

__all__ = [
    "FailurePrediction",
    "FeatureImportance",
    "EvaluationMetrics",
    "PredictionHistoryRecord",
    "WorkflowRunRecord",
    "FailureFeatureExtractor",
    "HistoricalRunCollector",
    "FailurePredictor",
    "FailurePredictorTrainer",
    "PredictionHistoryStore",
    "FailurePredictionService",
    "PredictionFeedbackService",
    "validate_dataset",
    "DatasetValidationReport",
    "LEAKAGE_COLUMNS",
    "FailureCategory",
    "map_failure_category",
    "conclusion_to_actual_failure",
]