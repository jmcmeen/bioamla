from pydantic import BaseModel, Field
from typing import Optional, List

# Response models
class PredictionResult(BaseModel):
    label: str
    score: float
    rank: int

class AudioClassificationResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]
    audio_duration: Optional[float] = None
    sample_rate: int
    model_used: str
    processing_time: Optional[float] = None

class Base64AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    top_k: Optional[int] = Field(default=5, description="Number of top predictions to return")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None