"""The ``batch cluster`` operation: a thin adapter over :func:`bioamla.cluster.batch_cluster`."""

from __future__ import annotations

from typing import Literal

from bioamla.batch import BatchRequest, BatchResult
from bioamla.cluster.batch import batch_cluster
from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, operation


@config_dataclass
class BatchClusterRequest(BatchRequest):
    """Cluster a batch of embeddings (a folder, or a CSV from ``batch models embed``).

    Attributes:
        method: ``hdbscan``, ``kmeans``, ``dbscan`` or ``agglomerative``.
        n_clusters: Number of clusters (k-means, agglomerative).
        min_cluster_size: Minimum cluster size (HDBSCAN).
        min_samples: Minimum samples per cluster.
    """

    method: Literal["hdbscan", "kmeans", "dbscan", "agglomerative"] = "hdbscan"
    n_clusters: int | None = None
    min_cluster_size: int = 5
    min_samples: int = 3


@operation("batch.cluster", request=BatchClusterRequest, kind="job")
def batch_cluster_op(request: BatchClusterRequest, context: OperationContext) -> BatchResult:
    """Cluster the embeddings of a batch."""
    return batch_cluster(
        request.to_config(),
        method=request.method,
        n_clusters=request.n_clusters,
        min_cluster_size=request.min_cluster_size,
        min_samples=request.min_samples,
        on_progress=context.progress,
    )
