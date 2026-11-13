"""
Pydantic Schemas for FastAPI Speech Recognition Endpoints.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str
    models_loaded: int
    commands: List[str]


class CommandsResponse(BaseModel):
    commands: List[str]
    count: int


class MetricSummary(BaseModel):
    dataset: str
    split: str
    num_test_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    avg_inference_time_sec: float
    per_class_metrics: Dict[str, Dict[str, float]]


class ConfusionMatrixResponse(BaseModel):
    commands: List[str]
    matrix: List[List[int]]
    accuracy: float
    macro_f1: float


class ModelAnalysisResponse(BaseModel):
    commands: List[str]
    n_states: int
    covariance_type: str
    feature_dimensions: int
    metrics: Optional[Dict[str, Any]]
    model_details: Dict[str, Any]


class RecognitionResponse(BaseModel):
    predicted_command: str
    confidence_normalized: float
    best_log_likelihood: float
    processing_time_ms: float
    scores: Dict[str, float]
    normalized_probabilities: Dict[str, float]
    viterbi_state_path: List[int]
    status: str = "success"


class MFCCDataResponse(BaseModel):
    time_frames: int
    feature_dim: int
    hop_length: int
    sample_rate: int
    mfcc_matrix: List[List[float]]
    summary_statistics: Dict[str, float]
