"""Operations for ``cluster``: thin adapters over :mod:`bioamla.cluster.files`.

See :mod:`bioamla.operations`. Each runs over a whole embeddings file, so each is a
``job``.
"""

from __future__ import annotations

from typing import Literal

from bioamla.cluster.files import (
    ClusterAnalysisReport,
    ClusteredEmbeddings,
    NoveltyReport,
    ReducedEmbeddings,
    analyze_clusters_file,
    cluster_embeddings_file,
    novelty_file,
    reduce_embeddings_file,
)
from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class ReduceRequest:
    """Reduce embeddings to fewer dimensions.

    Attributes:
        embeddings_file: The embeddings (``.npy``, 2-D).
        output: Where to save the reduced embeddings.
        method: ``pca``, ``umap`` or ``tsne``.
        n_components: Output dimensions.
    """

    embeddings_file: PathStr
    output: PathStr
    method: Literal["umap", "tsne", "pca"] = "pca"
    n_components: int = 2


@operation("cluster.reduce", request=ReduceRequest, kind="job")
def reduce_op(request: ReduceRequest, context: OperationContext) -> ReducedEmbeddings:
    """Reduce embeddings to fewer dimensions."""
    return reduce_embeddings_file(
        request.embeddings_file,
        request.output,
        method=request.method,
        n_components=request.n_components,
    )


@config_dataclass
class FitRequest:
    """Cluster embeddings.

    Attributes:
        embeddings_file: The embeddings (``.npy``, 2-D).
        output: Where to save the labels.
        method: ``hdbscan``, ``kmeans``, ``dbscan`` or ``agglomerative``.
        n_clusters: Number of clusters (k-means, agglomerative).
        min_cluster_size: Minimum cluster size (HDBSCAN).
        eps: DBSCAN neighbourhood radius.
        min_samples: Minimum samples per cluster.
    """

    embeddings_file: PathStr
    output: PathStr
    method: Literal["hdbscan", "kmeans", "dbscan", "agglomerative"] = "hdbscan"
    n_clusters: int = 10
    min_cluster_size: int = 5
    eps: float = 0.5
    min_samples: int = 5


@operation("cluster.fit", request=FitRequest, kind="job")
def fit_op(request: FitRequest, context: OperationContext) -> ClusteredEmbeddings:
    """Cluster embeddings."""
    return cluster_embeddings_file(
        request.embeddings_file,
        request.output,
        method=request.method,
        n_clusters=request.n_clusters,
        min_cluster_size=request.min_cluster_size,
        eps=request.eps,
        min_samples=request.min_samples,
    )


@config_dataclass
class AnalyzeRequest:
    """Measure the quality of a clustering.

    Attributes:
        embeddings_file: The embeddings (``.npy``, 2-D).
        labels_file: The labels (``.npy``, 1-D).
        output: A JSON file to save the analysis to.
    """

    embeddings_file: PathStr
    labels_file: PathStr
    output: PathStr | None = None


@operation("cluster.analyze", request=AnalyzeRequest, kind="job")
def analyze_op(request: AnalyzeRequest, context: OperationContext) -> ClusterAnalysisReport:
    """Measure the quality of a clustering."""
    return analyze_clusters_file(
        request.embeddings_file, request.labels_file, output=request.output
    )


@config_dataclass
class NoveltyRequest:
    """Find novel samples among embeddings.

    Attributes:
        embeddings_file: The embeddings (``.npy``, 2-D).
        output: Where to save is-novel flags and scores.
        method: ``distance``, ``isolation_forest`` or ``lof``.
        threshold: Novelty threshold.
        labels_file: Known cluster labels (``.npy``, 1-D).
    """

    embeddings_file: PathStr
    output: PathStr
    method: Literal["distance", "isolation_forest", "lof"] = "distance"
    threshold: float | None = None
    labels_file: PathStr | None = None


@operation("cluster.novelty", request=NoveltyRequest, kind="job")
def novelty_op(request: NoveltyRequest, context: OperationContext) -> NoveltyReport:
    """Find novel samples among embeddings."""
    return novelty_file(
        request.embeddings_file,
        request.output,
        method=request.method,
        threshold=request.threshold,
        labels_file=request.labels_file,
    )
