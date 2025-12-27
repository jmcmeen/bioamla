"""ML model inference and prediction models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class PredictionResult(ToDictMixin):
    """Single prediction result."""

    filepath: str
    start_time: float
    end_time: float
    predicted_label: str
    confidence: float
    top_k_labels: List[str] = field(default_factory=list)
    top_k_scores: List[float] = field(default_factory=list)


@dataclass
class InferenceSummary(ToDictMixin):
    """Summary of inference results."""

    total_files: int
    total_predictions: int
    unique_labels: int
    label_counts: Dict[str, int]
    output_path: Optional[str] = None


@dataclass
class BatchInferenceResult(ToDictMixin):
    """Result of batch inference."""

    predictions: List[PredictionResult]
    summary: InferenceSummary
    errors: List[str] = field(default_factory=lambda: [])
