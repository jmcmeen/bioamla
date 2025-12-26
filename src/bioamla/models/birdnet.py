# models/birdnet.py
"""
Data models for BirdNET operations.
"""

from dataclasses import dataclass
from typing import Optional

from .base import ToDictMixin


@dataclass
class PredictionResult(ToDictMixin):
    """Result of a single prediction."""

    filepath: str
    label: str
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
