# models/ast.py
"""
Data models for Audio Spectrogram Transformer operations.
"""

from dataclasses import dataclass
from typing import List, Optional

from .base import ToDictMixin


@dataclass
class PredictionResult(ToDictMixin):
    """Result of a single prediction."""

    filepath: str
    label: str
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class TrainResult(ToDictMixin):
    """Result of model training."""

    model_path: str
    epochs: int
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None


@dataclass
class EvaluationResult(ToDictMixin):
    """Result of model evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    confusion_matrix: Optional[List[List[int]]] = None
