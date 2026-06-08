"""
Clustering and Discovery
========================

Clustering and discovery capabilities for bioacoustic analysis:

- Embedding-based clustering (HDBSCAN, k-means, DBSCAN, agglomerative)
- UMAP / t-SNE / PCA dimensionality reduction
- Cluster similarity sorting
- Novel sound type discovery

All functions return plain data and raise :class:`~bioamla.exceptions.BioamlaError`
subclasses on failure. Heavy optional dependencies (umap-learn, hdbscan,
scikit-learn, torch) are imported lazily; a missing dependency raises
:class:`~bioamla.exceptions.DependencyError` telling the user which extra to install.

Example:
    >>> from bioamla.cluster import AudioClusterer, reduce_dimensions
    >>> clusterer = AudioClusterer(method="hdbscan")
    >>> labels = clusterer.fit_predict(embeddings)
    >>>
    >>> reduced = reduce_dimensions(embeddings, method="umap", n_components=2)
"""

import csv
import io
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from bioamla.cluster.models import (
    ClusterAnalysis,
    ClusteringSummary,
    NoveltyDetectionSummary,
)
from bioamla.exceptions import ClusteringError, DependencyError, InvalidInputError

logger = logging.getLogger(__name__)


# =============================================================================
# Dimensionality Reduction
# =============================================================================


@dataclass
class ReductionConfig:
    """Configuration for dimensionality reduction."""

    method: str = "umap"  # "umap", "tsne", "pca"
    n_components: int = 2

    # UMAP parameters
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"

    # t-SNE parameters
    perplexity: float = 30.0
    learning_rate: float = 200.0
    max_iter: int = 1000

    # PCA parameters
    whiten: bool = False


def reduce_dimensions(
    embeddings: np.ndarray,
    config: Optional[ReductionConfig] = None,
    method: Optional[str] = None,
    n_components: int = 2,
    random_state: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """
    Reduce dimensionality of embeddings.

    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        config: Reduction configuration
        method: Reduction method (overrides config)
        n_components: Number of output dimensions
        random_state: Random seed
        **kwargs: Additional arguments for the reducer

    Returns:
        Reduced embeddings of shape (n_samples, n_components)

    Raises:
        DependencyError: If a required reduction backend is not installed.
        InvalidInputError: If the reduction method is unknown.
    """
    if config is None:
        config = ReductionConfig(n_components=n_components)

    if method is not None:
        config.method = method

    logger.info(f"Reducing dimensions with {config.method} to {config.n_components}D")

    if config.method == "umap":
        try:
            import umap
        except ImportError as err:
            raise DependencyError(
                "UMAP reduction requires umap-learn — install bioamla[cluster]"
            ) from err

        reducer = umap.UMAP(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(embeddings)

    elif config.method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as err:
            raise DependencyError(
                "t-SNE reduction requires scikit-learn — install bioamla[cluster]"
            ) from err

        reducer = TSNE(
            n_components=config.n_components,
            perplexity=min(config.perplexity, len(embeddings) - 1),
            learning_rate=config.learning_rate,
            max_iter=config.max_iter,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(embeddings)

    elif config.method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as err:
            raise DependencyError(
                "PCA reduction requires scikit-learn — install bioamla[cluster]"
            ) from err

        reducer = PCA(
            n_components=config.n_components,
            whiten=config.whiten,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise InvalidInputError(f"Unknown reduction method: {config.method}")


class IncrementalReducer:
    """
    Incremental dimensionality reducer for streaming data.

    Fits on initial data and can transform new points without refitting.
    """

    def __init__(self, method: str = "umap", n_components: int = 2, **kwargs: Any) -> None:
        """
        Initialize incremental reducer.

        Args:
            method: Reduction method ("umap" or "pca")
            n_components: Number of output dimensions
            **kwargs: Additional arguments for the reducer
        """
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self.reducer = None
        self.fitted = False

    def fit(self, embeddings: np.ndarray) -> "IncrementalReducer":
        """Fit the reducer on embeddings."""
        if self.method == "umap":
            try:
                import umap
            except ImportError as err:
                raise DependencyError(
                    "UMAP reduction requires umap-learn — install bioamla[cluster]"
                ) from err

            self.reducer = umap.UMAP(n_components=self.n_components, **self.kwargs)
        elif self.method == "pca":
            try:
                from sklearn.decomposition import PCA
            except ImportError as err:
                raise DependencyError(
                    "PCA reduction requires scikit-learn — install bioamla[cluster]"
                ) from err

            self.reducer = PCA(n_components=self.n_components, **self.kwargs)
        else:
            raise InvalidInputError(f"Unsupported method for incremental: {self.method}")

        self.reducer.fit(embeddings)
        self.fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings."""
        if not self.fitted:
            raise ClusteringError("Reducer must be fitted before transform")
        return self.reducer.transform(embeddings)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings."""
        self.fit(embeddings)
        return self.transform(embeddings)


# =============================================================================
# Clustering
# =============================================================================


@dataclass
class ClusteringConfig:
    """Configuration for clustering."""

    method: str = "hdbscan"  # "hdbscan", "kmeans", "dbscan", "agglomerative"

    # HDBSCAN parameters
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"  # "eom" or "leaf"

    # K-means parameters
    n_clusters: int = 10
    n_init: int = 10
    max_iter: int = 300

    # DBSCAN parameters
    eps: float = 0.5

    # Agglomerative parameters
    linkage: str = "ward"
    distance_threshold: Optional[float] = None


class AudioClusterer:
    """
    Clustering for audio embeddings.

    Supports multiple clustering algorithms with automatic parameter tuning.
    """

    def __init__(
        self,
        config: Optional[ClusteringConfig] = None,
        method: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize clusterer.

        Args:
            config: Clustering configuration
            method: Clustering method (overrides config)
            **kwargs: Additional arguments for the clusterer
        """
        self.config = config or ClusteringConfig()
        if method is not None:
            self.config.method = method
        self.kwargs = kwargs
        self.clusterer = None
        self.labels_ = None
        self.n_clusters_ = None

    def fit(self, embeddings: np.ndarray) -> "AudioClusterer":
        """Fit the clusterer on embeddings."""
        self._create_clusterer()
        self.clusterer.fit(embeddings)
        self.labels_ = self.clusterer.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        logger.info(f"Found {self.n_clusters_} clusters")
        return self

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(embeddings)
        return self.labels_

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if self.clusterer is None:
            raise ClusteringError("Clusterer must be fitted before predict")

        if hasattr(self.clusterer, "predict"):
            return self.clusterer.predict(embeddings)
        else:
            # For HDBSCAN, use approximate_predict
            if self.config.method == "hdbscan":
                try:
                    import hdbscan

                    labels, strengths = hdbscan.approximate_predict(self.clusterer, embeddings)
                    return labels
                except Exception:
                    pass

            # Fall back to nearest cluster center
            return self._predict_nearest(embeddings)

    def _create_clusterer(self) -> None:
        """Create the clustering algorithm."""
        if self.config.method == "hdbscan":
            try:
                import hdbscan
            except ImportError as err:
                raise DependencyError(
                    "HDBSCAN clustering requires hdbscan — install bioamla[cluster]"
                ) from err

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                cluster_selection_method=self.config.cluster_selection_method,
                prediction_data=True,
                **self.kwargs,
            )

        elif self.config.method == "kmeans":
            try:
                from sklearn.cluster import KMeans
            except ImportError as err:
                raise DependencyError(
                    "K-means clustering requires scikit-learn — install bioamla[cluster]"
                ) from err

            self.clusterer = KMeans(
                n_clusters=self.config.n_clusters,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
                random_state=42,
                **self.kwargs,
            )

        elif self.config.method == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
            except ImportError as err:
                raise DependencyError(
                    "DBSCAN clustering requires scikit-learn — install bioamla[cluster]"
                ) from err

            self.clusterer = DBSCAN(
                eps=self.config.eps, min_samples=self.config.min_samples, **self.kwargs
            )

        elif self.config.method == "agglomerative":
            try:
                from sklearn.cluster import AgglomerativeClustering
            except ImportError as err:
                raise DependencyError(
                    "Agglomerative clustering requires scikit-learn — install bioamla[cluster]"
                ) from err

            self.clusterer = AgglomerativeClustering(
                n_clusters=self.config.n_clusters
                if self.config.distance_threshold is None
                else None,
                linkage=self.config.linkage,
                distance_threshold=self.config.distance_threshold,
                **self.kwargs,
            )

        else:
            raise InvalidInputError(f"Unknown clustering method: {self.config.method}")

    def _predict_nearest(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict by finding nearest cluster center."""
        if not hasattr(self, "_cluster_centers"):
            raise ClusteringError("No cluster centers available for prediction")

        distances = np.linalg.norm(embeddings[:, np.newaxis] - self._cluster_centers, axis=2)
        return np.argmin(distances, axis=1)

    def get_cluster_centers(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cluster centers from embeddings."""
        if self.labels_ is None:
            raise ClusteringError("Clusterer must be fitted first")

        unique_labels = set(self.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        centers = []
        for label in sorted(unique_labels):
            mask = self.labels_ == label
            center = embeddings[mask].mean(axis=0)
            centers.append(center)

        self._cluster_centers = np.array(centers)
        return self._cluster_centers

    def get_cluster_stats(self, embeddings: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each cluster."""
        if self.labels_ is None:
            raise ClusteringError("Clusterer must be fitted first")

        stats = {}
        unique_labels = set(self.labels_)

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_embeddings = embeddings[mask]

            stats[label] = {
                "size": int(mask.sum()),
                "mean": cluster_embeddings.mean(axis=0).tolist(),
                "std": cluster_embeddings.std(axis=0).mean(),
                "is_noise": label == -1,
            }

        return stats


def find_optimal_clusters(
    embeddings: np.ndarray,
    method: str = "silhouette",
    k_range: Tuple[int, int] = (2, 20),
) -> int:
    """
    Find optimal number of clusters.

    Args:
        embeddings: Input embeddings
        method: Method for determining optimal k ("silhouette", "elbow", "gap")
        k_range: Range of k values to try

    Returns:
        Optimal number of clusters

    Raises:
        DependencyError: If scikit-learn is not installed.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError as err:
        raise DependencyError(
            "find_optimal_clusters requires scikit-learn — install bioamla[cluster]"
        ) from err

    scores = []
    k_values = range(k_range[0], k_range[1] + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        if method == "silhouette":
            score = silhouette_score(embeddings, labels)
            scores.append(score)
        elif method == "elbow":
            score = -kmeans.inertia_  # Negative for consistency (higher is better)
            scores.append(score)

    if method == "silhouette":
        optimal_k = k_values[np.argmax(scores)]
    elif method == "elbow":
        # Find elbow point
        diffs = np.diff(scores)
        diffs2 = np.diff(diffs)
        optimal_k = k_values[np.argmax(diffs2) + 1]
    else:
        optimal_k = k_values[np.argmax(scores)]

    logger.info(f"Optimal number of clusters: {optimal_k}")
    return optimal_k


# =============================================================================
# Cluster Similarity and Sorting
# =============================================================================


def compute_cluster_similarity(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> np.ndarray:
    """
    Compute pairwise similarity between clusters.

    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        metric: Distance metric ("cosine", "euclidean")

    Returns:
        Similarity matrix of shape (n_clusters, n_clusters)

    Raises:
        DependencyError: If scikit-learn is required (euclidean) but missing.
        InvalidInputError: If the metric is unknown.
    """
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    centers = []
    for label in unique_labels:
        mask = labels == label
        center = embeddings[mask].mean(axis=0)
        centers.append(center)

    centers = np.array(centers)

    if metric == "cosine":
        # Normalize centers
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers_norm = centers / (norms + 1e-10)
        similarity = centers_norm @ centers_norm.T
    elif metric == "euclidean":
        try:
            from sklearn.metrics import pairwise_distances
        except ImportError as err:
            raise DependencyError(
                "euclidean cluster similarity requires scikit-learn — install bioamla[cluster]"
            ) from err

        distances = pairwise_distances(centers, metric="euclidean")
        similarity = 1 / (1 + distances)
    else:
        raise InvalidInputError(f"Unknown metric: {metric}")

    return similarity


def sort_by_similarity(
    embeddings: np.ndarray,
    reference: Optional[np.ndarray] = None,
    method: str = "nearest_neighbor",
) -> np.ndarray:
    """
    Sort embeddings by similarity.

    Args:
        embeddings: Input embeddings
        reference: Reference embedding (uses first embedding if None)
        method: Sorting method ("nearest_neighbor", "spectral")

    Returns:
        Sorted indices

    Raises:
        DependencyError: If scikit-learn is required (spectral) but missing.
        InvalidInputError: If the method is unknown.
    """
    n_samples = len(embeddings)

    if method == "nearest_neighbor":
        # Start from reference and greedily add nearest neighbors
        if reference is None:
            start_idx = 0
        else:
            distances = np.linalg.norm(embeddings - reference, axis=1)
            start_idx = np.argmin(distances)

        sorted_indices = [start_idx]
        remaining = set(range(n_samples)) - {start_idx}

        while remaining:
            current = embeddings[sorted_indices[-1]]
            distances = {idx: np.linalg.norm(embeddings[idx] - current) for idx in remaining}
            nearest = min(distances, key=distances.get)
            sorted_indices.append(nearest)
            remaining.remove(nearest)

        return np.array(sorted_indices)

    elif method == "spectral":
        # Use spectral ordering for global structure
        try:
            from sklearn.metrics import pairwise_distances
        except ImportError as err:
            raise DependencyError(
                "spectral sorting requires scikit-learn — install bioamla[cluster]"
            ) from err

        distances = pairwise_distances(embeddings)
        similarity = 1 / (1 + distances)

        # Compute Laplacian eigenvector
        degree = np.diag(similarity.sum(axis=1))
        laplacian = degree - similarity

        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Use second smallest eigenvector for ordering
        fiedler = eigenvectors[:, 1]

        return np.argsort(fiedler)

    else:
        raise InvalidInputError(f"Unknown method: {method}")


def sort_clusters_by_similarity(
    embeddings: np.ndarray, labels: np.ndarray, reference_label: Optional[int] = None
) -> List[int]:
    """
    Sort cluster labels by similarity to each other.

    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        reference_label: Starting cluster (uses largest if None)

    Returns:
        Sorted list of cluster labels
    """
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) <= 1:
        return unique_labels

    # Compute cluster centers
    centers = {}
    for label in unique_labels:
        mask = labels == label
        centers[label] = embeddings[mask].mean(axis=0)

    # Find reference cluster
    if reference_label is None:
        # Use largest cluster as reference
        sizes = {label: (labels == label).sum() for label in unique_labels}
        reference_label = max(sizes, key=sizes.get)

    # Sort by similarity to reference
    sorted_labels = [reference_label]
    remaining = set(unique_labels) - {reference_label}

    while remaining:
        current_center = centers[sorted_labels[-1]]
        similarities = {
            label: np.dot(centers[label], current_center)
            / (np.linalg.norm(centers[label]) * np.linalg.norm(current_center) + 1e-10)
            for label in remaining
        }
        most_similar = max(similarities, key=similarities.get)
        sorted_labels.append(most_similar)
        remaining.remove(most_similar)

    return sorted_labels


# =============================================================================
# Novel Sound Discovery
# =============================================================================


@dataclass
class NoveltyResult:
    """Result from novelty detection."""

    sample_idx: int
    novelty_score: float
    nearest_cluster: int
    distance_to_cluster: float
    is_novel: bool


class NoveltyDetector:
    """
    Detect novel sound types in audio embeddings.

    Uses distance-based and density-based methods to identify
    sounds that don't fit existing clusters.
    """

    def __init__(
        self,
        method: str = "distance",
        threshold: Optional[float] = None,
        contamination: float = 0.1,
    ) -> None:
        """
        Initialize novelty detector.

        Args:
            method: Detection method ("distance", "isolation_forest", "lof")
            threshold: Novelty threshold (auto-computed if None)
            contamination: Expected proportion of outliers
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.cluster_centers = None
        self.cluster_radii = None
        self.detector = None

    def fit(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None) -> "NoveltyDetector":
        """
        Fit the novelty detector.

        Args:
            embeddings: Training embeddings (known sounds)
            labels: Optional cluster labels

        Returns:
            self

        Raises:
            DependencyError: If scikit-learn is required but missing.
            InvalidInputError: If the method is unknown.
        """
        if self.method == "distance":
            if labels is None:
                # Create single cluster from all data
                self.cluster_centers = [embeddings.mean(axis=0)]
                distances = np.linalg.norm(embeddings - self.cluster_centers[0], axis=1)
                self.cluster_radii = [np.percentile(distances, 95)]
            else:
                self._fit_distance_based(embeddings, labels)

        elif self.method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError as err:
                raise DependencyError(
                    "isolation_forest novelty detection requires scikit-learn "
                    "— install bioamla[cluster]"
                ) from err

            self.detector = IsolationForest(contamination=self.contamination, random_state=42)
            self.detector.fit(embeddings)

        elif self.method == "lof":
            try:
                from sklearn.neighbors import LocalOutlierFactor
            except ImportError as err:
                raise DependencyError(
                    "lof novelty detection requires scikit-learn — install bioamla[cluster]"
                ) from err

            self.detector = LocalOutlierFactor(
                n_neighbors=20, contamination=self.contamination, novelty=True
            )
            self.detector.fit(embeddings)

        else:
            raise InvalidInputError(f"Unknown method: {self.method}")

        return self

    def _fit_distance_based(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Fit distance-based novelty detection."""
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        self.cluster_centers = []
        self.cluster_radii = []

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_data = embeddings[mask]
            center = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - center, axis=1)
            radius = np.percentile(distances, 95)

            self.cluster_centers.append(center)
            self.cluster_radii.append(radius)

        self.cluster_centers = np.array(self.cluster_centers)
        self.cluster_radii = np.array(self.cluster_radii)

    def predict(self, embeddings: np.ndarray) -> List[NoveltyResult]:
        """
        Detect novel sounds in new embeddings.

        Args:
            embeddings: Embeddings to check for novelty

        Returns:
            List of NoveltyResult for each embedding
        """
        results = []

        if self.method == "distance":
            for i, emb in enumerate(embeddings):
                distances = np.linalg.norm(self.cluster_centers - emb, axis=1)
                nearest_cluster = np.argmin(distances)
                distance = distances[nearest_cluster]
                radius = self.cluster_radii[nearest_cluster]

                # Novelty score: distance normalized by cluster radius
                novelty_score = distance / (radius + 1e-10)
                is_novel = novelty_score > (self.threshold or 1.5)

                results.append(
                    NoveltyResult(
                        sample_idx=i,
                        novelty_score=novelty_score,
                        nearest_cluster=nearest_cluster,
                        distance_to_cluster=distance,
                        is_novel=is_novel,
                    )
                )

        elif self.method in ["isolation_forest", "lof"]:
            scores = -self.detector.score_samples(embeddings)
            predictions = self.detector.predict(embeddings)

            for i, (score, pred) in enumerate(zip(scores, predictions)):
                results.append(
                    NoveltyResult(
                        sample_idx=i,
                        novelty_score=score,
                        nearest_cluster=-1,
                        distance_to_cluster=0.0,
                        is_novel=pred == -1,
                    )
                )

        return results

    def get_novel_samples(
        self, embeddings: np.ndarray, n_samples: Optional[int] = None
    ) -> List[int]:
        """
        Get indices of most novel samples.

        Args:
            embeddings: Embeddings to check
            n_samples: Number of novel samples to return (all if None)

        Returns:
            Indices of novel samples, sorted by novelty score
        """
        results = self.predict(embeddings)
        novel = [r for r in results if r.is_novel]

        if n_samples is not None:
            novel = sorted(novel, key=lambda x: x.novelty_score, reverse=True)
            novel = novel[:n_samples]

        return [r.sample_idx for r in novel]


def discover_novel_sounds(
    embeddings: np.ndarray,
    known_labels: Optional[np.ndarray] = None,
    method: str = "distance",
    threshold: Optional[float] = None,
    return_scores: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Discover novel sound types in embeddings.

    Args:
        embeddings: All embeddings to analyze
        known_labels: Labels for known sounds (None = all unknown)
        method: Detection method
        threshold: Novelty threshold
        return_scores: Whether to return novelty scores

    Returns:
        Binary array indicating novel sounds (and optionally scores)
    """
    detector = NoveltyDetector(method=method, threshold=threshold)

    if known_labels is not None:
        # Split known and unknown
        known_mask = known_labels >= 0
        known_embeddings = embeddings[known_mask]
        known_cluster_labels = known_labels[known_mask]
        detector.fit(known_embeddings, known_cluster_labels)
    else:
        detector.fit(embeddings)

    results = detector.predict(embeddings)
    is_novel = np.array([r.is_novel for r in results])

    if return_scores:
        scores = np.array([r.novelty_score for r in results])
        return is_novel, scores

    return is_novel


# =============================================================================
# Embedding Extraction Utilities
# =============================================================================


def extract_embeddings_batch(
    model, dataloader, device=None, layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Extract embeddings from a model for a batch of data.

    Args:
        model: PyTorch model
        dataloader: DataLoader with audio data
        device: Device to use
        layer_name: Name of layer to extract (uses last hidden if None)

    Returns:
        Embeddings array of shape (n_samples, embedding_dim)

    Raises:
        DependencyError: If torch is not installed.
    """
    try:
        import torch
    except ImportError as err:
        raise DependencyError(
            "extract_embeddings_batch requires torch — install bioamla[ml]"
        ) from err

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    embeddings = []
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output.detach().cpu())

    # Register hook if specific layer requested
    hook = None
    if layer_name is not None:
        for name, module in model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            if hook is not None:
                hook_output.clear()
                _ = model(inputs)
                if hook_output:
                    emb = hook_output[0]
                    if emb.dim() > 2:
                        emb = emb.mean(dim=list(range(2, emb.dim())))
                    embeddings.append(emb.numpy())
            else:
                output = model(inputs)
                if hasattr(output, "last_hidden_state"):
                    emb = output.last_hidden_state.mean(dim=1)
                elif hasattr(output, "pooler_output"):
                    emb = output.pooler_output
                else:
                    emb = output
                embeddings.append(emb.cpu().numpy())

    if hook is not None:
        hook.remove()

    return np.vstack(embeddings)


# =============================================================================
# Cluster Analysis Utilities
# =============================================================================


def analyze_clusters(
    embeddings: np.ndarray, labels: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Comprehensive cluster analysis.

    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        metadata: Optional metadata for each sample

    Returns:
        Analysis results as a dictionary

    Raises:
        DependencyError: If scikit-learn is not installed.
    """
    try:
        from sklearn.metrics import calinski_harabasz_score, silhouette_score
    except ImportError as err:
        raise DependencyError(
            "analyze_clusters requires scikit-learn — install bioamla[cluster]"
        ) from err

    unique_labels = sorted(set(labels))
    n_noise = (labels == -1).sum() if -1 in labels else 0
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Compute metrics
    mask = labels >= 0
    if mask.sum() > n_clusters > 1:
        silhouette = silhouette_score(embeddings[mask], labels[mask])
        calinski = calinski_harabasz_score(embeddings[mask], labels[mask])
    else:
        silhouette = 0.0
        calinski = 0.0

    # Per-cluster statistics
    cluster_stats = {}
    for label in unique_labels:
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_emb = embeddings[cluster_mask]

        cluster_stats[label] = {
            "size": int(cluster_mask.sum()),
            "percentage": float(cluster_mask.sum() / len(labels) * 100),
            "centroid": cluster_emb.mean(axis=0).tolist(),
            "spread": float(cluster_emb.std()),
        }

        if metadata is not None:
            cluster_meta = [metadata[i] for i in np.where(cluster_mask)[0]]
            cluster_stats[label]["metadata_sample"] = cluster_meta[:5]

    return {
        "n_clusters": n_clusters,
        "n_samples": len(labels),
        "n_noise": n_noise,
        "noise_percentage": float(n_noise / len(labels) * 100),
        "silhouette_score": float(silhouette),
        "calinski_harabasz_score": float(calinski),
        "cluster_stats": cluster_stats,
    }


def export_clusters(
    labels: np.ndarray, filepaths: List[str], output_dir: str, copy_files: bool = False
) -> str:
    """
    Export clustering results to directory structure.

    Args:
        labels: Cluster labels
        filepaths: List of file paths
        output_dir: Output directory
        copy_files: Whether to copy files to cluster directories

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create cluster directories and manifest
    manifest = {}
    for label in sorted(set(labels)):
        cluster_name = f"cluster_{label}" if label >= 0 else "noise"
        cluster_dir = output_path / cluster_name

        if copy_files:
            cluster_dir.mkdir(parents=True, exist_ok=True)

        mask = labels == label
        cluster_files = [filepaths[i] for i in np.where(mask)[0]]
        manifest[cluster_name] = cluster_files

        if copy_files:
            for filepath in cluster_files:
                src = Path(filepath)
                dst = cluster_dir / src.name
                if src.exists():
                    shutil.copy2(src, dst)

    # Save manifest
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(f"Exported clusters to {output_dir}")
    return str(output_path)


# =============================================================================
# High-level summary helpers (folded from the former ClusteringService)
# =============================================================================


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    **kwargs: Any,
) -> ClusteringSummary:
    """
    Cluster embeddings and return a summary.

    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        method: Clustering method ("hdbscan", "kmeans", "dbscan", "agglomerative")
        n_clusters: Number of clusters (for k-means/agglomerative)
        min_cluster_size: Minimum cluster size (for HDBSCAN)
        min_samples: Minimum samples per cluster
        **kwargs: Additional arguments forwarded to the clusterer constructor

    Returns:
        A :class:`ClusteringSummary` whose ``labels`` hold the per-sample
        cluster assignments.

    Raises:
        DependencyError: If a required clustering backend is not installed.
        ClusteringError: If clustering fails for any other reason.
    """
    config = ClusteringConfig(
        method=method,
        n_clusters=n_clusters or 10,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    clusterer = AudioClusterer(config=config, **kwargs)
    try:
        labels = clusterer.fit_predict(embeddings)
    except (DependencyError, InvalidInputError):
        raise
    except Exception as e:
        raise ClusteringError(str(e)) from e

    n_noise = int((labels == -1).sum())
    unique_labels = set(labels)
    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

    cluster_sizes = {int(label): int((labels == label).sum()) for label in unique_labels}

    silhouette = 0.0
    if n_clusters_found > 1:
        try:
            from sklearn.metrics import silhouette_score
        except ImportError as err:
            raise DependencyError(
                "silhouette scoring requires scikit-learn — install bioamla[cluster]"
            ) from err

        mask = labels >= 0
        if mask.sum() > n_clusters_found:
            silhouette = silhouette_score(embeddings[mask], labels[mask])

    return ClusteringSummary(
        n_clusters=n_clusters_found,
        n_samples=len(embeddings),
        n_noise=n_noise,
        noise_percentage=n_noise / len(embeddings) * 100,
        silhouette_score=float(silhouette),
        method=method,
        labels=labels.tolist(),
        cluster_sizes=cluster_sizes,
    )


def analyze_clusters_summary(
    embeddings: np.ndarray,
    labels: np.ndarray,
    filepaths: Optional[List[str]] = None,
) -> ClusterAnalysis:
    """
    Perform detailed analysis of clustering results and return a dataclass.

    Args:
        embeddings: Input embeddings
        labels: Cluster labels
        filepaths: Optional list of file paths for per-cluster metadata

    Returns:
        A :class:`ClusterAnalysis` with quality metrics and per-cluster stats.

    Raises:
        DependencyError: If scikit-learn is not installed.
    """
    metadata = [{"filepath": fp} for fp in filepaths] if filepaths else None
    analysis = analyze_clusters(embeddings, labels, metadata)

    return ClusterAnalysis(
        n_clusters=analysis["n_clusters"],
        n_samples=analysis["n_samples"],
        n_noise=analysis["n_noise"],
        silhouette_score=analysis["silhouette_score"],
        calinski_harabasz_score=analysis["calinski_harabasz_score"],
        cluster_stats=analysis["cluster_stats"],
    )


def detect_novelty(
    embeddings: np.ndarray,
    known_embeddings: Optional[np.ndarray] = None,
    known_labels: Optional[np.ndarray] = None,
    method: str = "distance",
    threshold: Optional[float] = None,
    contamination: float = 0.1,
) -> Tuple[NoveltyDetectionSummary, np.ndarray, np.ndarray]:
    """
    Detect novel/outlier samples in embeddings.

    Args:
        embeddings: Embeddings to check for novelty
        known_embeddings: Optional known embeddings to fit the detector on
        known_labels: Optional labels for known embeddings
        method: Detection method ("distance", "isolation_forest", "lof")
        threshold: Novelty threshold (auto if None)
        contamination: Expected proportion of outliers

    Returns:
        Tuple of ``(summary, is_novel, novelty_scores)`` where ``is_novel`` is a
        boolean array and ``novelty_scores`` a float array, both per sample.

    Raises:
        DependencyError: If scikit-learn is required but not installed.
    """
    detector = NoveltyDetector(method=method, threshold=threshold, contamination=contamination)

    if known_embeddings is not None:
        detector.fit(known_embeddings, known_labels)
    else:
        detector.fit(embeddings)

    results = detector.predict(embeddings)

    novel_indices = [r.sample_idx for r in results if r.is_novel]
    n_novel = len(novel_indices)

    summary = NoveltyDetectionSummary(
        n_samples=len(embeddings),
        n_novel=n_novel,
        n_known=len(embeddings) - n_novel,
        novel_percentage=n_novel / len(embeddings) * 100,
        method=method,
        threshold=threshold,
        novel_indices=novel_indices,
    )

    is_novel = np.array([r.is_novel for r in results])
    novelty_scores = np.array([r.novelty_score for r in results])

    return summary, is_novel, novelty_scores


def export_clusters_to_csv(
    labels: np.ndarray,
    filepaths: List[str],
    output_path: str,
    reduced_embeddings: Optional[np.ndarray] = None,
) -> str:
    """
    Export clustering results to a CSV file.

    Args:
        labels: Cluster labels
        filepaths: List of file paths
        output_path: Output CSV path
        reduced_embeddings: Optional 2D coordinates to include as ``x``/``y`` columns

    Returns:
        Path to the written CSV file.
    """
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    include_xy = reduced_embeddings is not None and reduced_embeddings.shape[1] >= 2
    fieldnames = ["filepath", "cluster"]
    if include_xy:
        fieldnames.extend(["x", "y"])

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for i, (fp, label) in enumerate(zip(filepaths, labels)):
        row = {"filepath": fp, "cluster": int(label)}
        if include_xy:
            row["x"] = float(reduced_embeddings[i, 0])
            row["y"] = float(reduced_embeddings[i, 1])
        writer.writerow(row)

    output_path_obj.write_text(buffer.getvalue(), encoding="utf-8")
    return str(output_path_obj)
