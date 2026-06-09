"""Clustering, dimensionality reduction, and novel-sound discovery.

Cluster audio embeddings (HDBSCAN, k-means, DBSCAN, agglomerative), reduce
embedding dimensionality (UMAP, t-SNE, PCA), measure cluster quality, and detect
novel/outlier sounds.

Heavy backends (umap-learn, hdbscan, scikit-learn, torch) ship in the base
install but are imported lazily so importing this module stays fast.

Example:
    >>> import numpy as np
    >>> from bioamla.cluster import AudioClusterer, reduce_dimensions
    >>> emb = np.random.rand(100, 32)
    >>> reduced = reduce_dimensions(emb, method="pca", n_components=2)
    >>> labels = AudioClusterer(method="kmeans").fit_predict(reduced)
"""

from bioamla.cluster.batch import (
    cluster_batch_files,
    cluster_embedding_files,
    load_embedding_file,
    load_embeddings_batch,
)
from bioamla.cluster.core import (
    AudioClusterer,
    ClusteringConfig,
    IncrementalReducer,
    NoveltyDetector,
    NoveltyResult,
    ReductionConfig,
    analyze_clusters,
    analyze_clusters_summary,
    cluster_embeddings,
    compute_cluster_similarity,
    detect_novelty,
    discover_novel_sounds,
    export_clusters,
    export_clusters_to_csv,
    extract_embeddings_batch,
    find_optimal_clusters,
    reduce_dimensions,
    sort_by_similarity,
    sort_clusters_by_similarity,
)
from bioamla.cluster.models import (
    ClusterAnalysis,
    ClusteringSummary,
    NoveltyDetectionSummary,
)
from bioamla.exceptions import ClusteringError, DependencyError

__all__ = [
    # Configuration
    "ReductionConfig",
    "ClusteringConfig",
    # Core classes
    "IncrementalReducer",
    "AudioClusterer",
    "NoveltyResult",
    "NoveltyDetector",
    # Reduction / clustering / analysis functions
    "reduce_dimensions",
    "cluster_embeddings",
    "find_optimal_clusters",
    "compute_cluster_similarity",
    "sort_by_similarity",
    "sort_clusters_by_similarity",
    "discover_novel_sounds",
    "detect_novelty",
    "extract_embeddings_batch",
    "analyze_clusters",
    "analyze_clusters_summary",
    "export_clusters",
    "export_clusters_to_csv",
    # Batch
    "cluster_batch_files",
    "cluster_embedding_files",
    "load_embeddings_batch",
    "load_embedding_file",
    # Result dataclasses
    "ClusteringSummary",
    "NoveltyDetectionSummary",
    "ClusterAnalysis",
    # Exceptions
    "ClusteringError",
    "DependencyError",
]
