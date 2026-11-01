"""The ``batch index`` operation: a thin adapter over :func:`bioamla.indices.batch_indices`."""

from __future__ import annotations

from dataclasses import field

from bioamla.batch import BatchRequest, BatchResult
from bioamla.common.config import config_dataclass
from bioamla.indices.batch import batch_indices
from bioamla.indices.compute import DEFAULT_INDICES
from bioamla.operations import OperationContext, operation


@config_dataclass
class BatchIndexRequest(BatchRequest):
    """Acoustic indices over a batch.

    Attributes:
        indices: Index names (add ``h_spectral`` and ``h_temporal`` for entropy).
    """

    indices: list[str] = field(default_factory=lambda: list(DEFAULT_INDICES))


@operation("batch.index", request=BatchIndexRequest, kind="job")
def batch_index(request: BatchIndexRequest, context: OperationContext) -> BatchResult:
    """Compute acoustic indices for every file."""
    return batch_indices(request.to_config(), request.indices, on_progress=context.progress)
