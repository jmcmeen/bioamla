"""
API Response Models
==================

This module defines Pydantic models for API responses and request/response
data structures used in the bioamla web API and other interfaces.

These models ensure consistent data validation and serialization
across all API endpoints and client interactions.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """
    Model for individual prediction results from audio classification.

    Attributes:
        label (str): The predicted class label (e.g., species name)
        score (float): Confidence score for this prediction (0-1)
        rank (int): Rank of this prediction among all results
    """

    label: str
    score: float
    rank: int


class AudioClassificationResponse(BaseModel):
    """
    Complete response model for audio classification API endpoints.

    Attributes:
        success (bool): Whether the classification was successful
        predictions (List[PredictionResult]): List of prediction results
        audio_duration (Optional[float]): Duration of processed audio in seconds
        sample_rate (int): Sample rate of the processed audio
        model_used (str): Identifier of the model used for classification
        processing_time (Optional[float]): Time taken for processing in seconds
    """

    success: bool
    predictions: List[PredictionResult]
    audio_duration: Optional[float] = None
    sample_rate: int
    model_used: str
    processing_time: Optional[float] = None


class Base64AudioRequest(BaseModel):
    """
    Request model for submitting audio data as base64 encoded string.

    This model is used for API endpoints that accept audio data
    encoded as base64 strings, typically for web uploads.
    """

    audio_base64: str = Field(..., description="Base64 encoded audio data")
    top_k: Optional[int] = Field(default=5, description="Number of top predictions to return")


class ErrorResponse(BaseModel):
    """
    Model for error responses from API endpoints.

    Attributes:
        success (bool): Always False for error responses
        error (str): Brief error message
        detail (Optional[str]): Detailed error information if available
    """

    success: bool = False
    error: str
    detail: Optional[str] = None
