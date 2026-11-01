"""Operations for ``batch models``: thin adapters over :mod:`bioamla.ml.batch`."""

from __future__ import annotations

from bioamla.batch import BatchRequest, BatchResult
from bioamla.common.config import config_dataclass
from bioamla.exceptions import InvalidInputError
from bioamla.ml.batch import batch_embed, batch_predict
from bioamla.operations import OperationContext, operation


@config_dataclass
class BatchPredictRequest(BatchRequest):
    """AST classification over a batch, whole files or in segments.

    Attributes:
        model: Model (Hugging Face ID or local path).
        top_k: Rank each prediction's ``top_k`` most likely labels (``top`` and
            ``margin``; 0 leaves them out).
        min_confidence: Leave out predictions below this confidence.
        segment_duration: Classify N-second segments (0: whole files).
        overlap: Overlap between segments (s).
    """

    model: str = ""
    top_k: int = 5
    min_confidence: float = 0.0
    segment_duration: int = 0
    overlap: int = 0

    def __post_init__(self) -> None:
        if not self.model:
            raise InvalidInputError("model is required")
        if self.top_k < 0:
            raise InvalidInputError(f"top_k must be 0 or more, got {self.top_k}")


@operation("batch.models.predict", request=BatchPredictRequest, kind="job")
def batch_models_predict(request: BatchPredictRequest, context: OperationContext) -> BatchResult:
    """Classify every file with an AST model."""
    return batch_predict(
        request.to_config(),
        model_path=request.model,
        top_k=request.top_k,
        min_confidence=request.min_confidence,
        segment_duration=request.segment_duration,
        overlap=request.overlap,
        on_progress=context.progress,
    )


@config_dataclass
class BatchEmbedRequest(BatchRequest):
    """AST embeddings over a batch.

    Attributes:
        model: Model (Hugging Face ID or local path).
    """

    model: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise InvalidInputError("model is required")


@operation("batch.models.embed", request=BatchEmbedRequest, kind="job")
def batch_models_embed(request: BatchEmbedRequest, context: OperationContext) -> BatchResult:
    """Save an AST embedding for every file."""
    return batch_embed(request.to_config(), model_path=request.model, on_progress=context.progress)
