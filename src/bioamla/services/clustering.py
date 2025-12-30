# services/clustering.py
"""
Service for clustering and novelty detection operations on audio embeddings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseService, ServiceResult


@dataclass
class ClusteringSummary:
    """Summary of clustering results."""

    n_clusters: int
    n_samples: int
    n_noise: int
    noise_percentage: float
    silhouette_score: float
    method: str
    labels: List[int] = field(default_factory=list)
    cluster_sizes: Dict[int, int] = field(default_factory=dict)


@dataclass
class NoveltyDetectionSummary:
    """Summary of novelty detection results."""

    n_samples: int
    n_novel: int
    n_known: int
    novel_percentage: float
    method: str
    threshold: Optional[float]
    novel_indices: List[int] = field(default_factory=list)


@dataclass
class ClusterAnalysis:
    """Detailed cluster analysis results."""

    n_clusters: int
    n_samples: int
    n_noise: int
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_stats: Dict[int, Dict[str, Any]]


class ClusteringService(BaseService):
    """
    Service for clustering and novelty detection operations.

    Provides high-level methods for:
    - Embedding clustering (HDBSCAN, k-means, DBSCAN)
    - Dimensionality reduction (UMAP, t-SNE, PCA)
    - Novelty/outlier detection
    - Cluster quality metrics
    - Cluster visualization
    - Cluster export
    """

    def __init__(self) -> None:
        """Initialize clustering service."""
        super().__init__()
        self._clusterer = None
        self._reducer = None
        self._novelty_detector = None

    # =========================================================================
    # Clustering
    # =========================================================================

    def cluster(
        self,
        embeddings: np.ndarray,
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        **kwargs,
    ) -> ServiceResult[ClusteringSummary]:
        """
        Cluster embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            method: Clustering method ("hdbscan", "kmeans", "dbscan", "agglomerative")
            n_clusters: Number of clusters (for k-means/agglomerative)
            min_cluster_size: Minimum cluster size (for HDBSCAN)
            min_samples: Minimum samples per cluster
            **kwargs: Additional arguments for the clusterer

        Returns:
            Result with clustering summary
        """
        # Start run tracking
        self._start_run(
            name=f"Clustering: {method}",
            action="cluster",
            parameters={
                "method": method,
                "n_clusters": n_clusters,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_samples": len(embeddings),
                "n_features": embeddings.shape[1] if len(embeddings.shape) > 1 else 1,
            },
        )

        try:
            from bioamla.core.clustering import AudioClusterer, ClusteringConfig

            config = ClusteringConfig(
                method=method,
                n_clusters=n_clusters or 10,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            clusterer = AudioClusterer(config=config, **kwargs)
            labels = clusterer.fit_predict(embeddings)

            # Compute metrics
            n_noise = int((labels == -1).sum())
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            # Cluster sizes
            cluster_sizes = {}
            for label in unique_labels:
                cluster_sizes[int(label)] = int((labels == label).sum())

            # Silhouette score
            silhouette = 0.0
            if n_clusters > 1:
                from sklearn.metrics import silhouette_score

                mask = labels >= 0
                if mask.sum() > n_clusters:
                    silhouette = silhouette_score(embeddings[mask], labels[mask])

            summary = ClusteringSummary(
                n_clusters=n_clusters,
                n_samples=len(embeddings),
                n_noise=n_noise,
                noise_percentage=n_noise / len(embeddings) * 100,
                silhouette_score=float(silhouette),
                method=method,
                labels=labels.tolist(),
                cluster_sizes=cluster_sizes,
            )

            self._clusterer = clusterer

            # Complete run with results
            self._complete_run(
                results={
                    "n_clusters": n_clusters,
                    "n_samples": len(embeddings),
                    "n_noise": n_noise,
                    "noise_percentage": n_noise / len(embeddings) * 100,
                    "silhouette_score": float(silhouette),
                },
            )

            return ServiceResult.ok(
                data=summary,
                message=f"Found {n_clusters} clusters with {method}",
                labels=labels,
                clusterer=clusterer,
            )
        except Exception as e:
            self._fail_run(str(e))
            return ServiceResult.fail(str(e))

    def find_optimal_k(
        self,
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (2, 20),
        method: str = "silhouette",
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Find optimal number of clusters for k-means.

        Args:
            embeddings: Input embeddings
            k_range: Range of k values to try (min, max)
            method: Method for evaluation ("silhouette", "elbow")

        Returns:
            Result with optimal k and scores
        """
        try:
            from bioamla.core.clustering import find_optimal_clusters

            optimal_k = find_optimal_clusters(
                embeddings,
                method=method,
                k_range=k_range,
            )

            return ServiceResult.ok(
                data={
                    "optimal_k": optimal_k,
                    "k_range": k_range,
                    "method": method,
                },
                message=f"Optimal k = {optimal_k} using {method} method",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def analyze_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filepaths: Optional[List[str]] = None,
    ) -> ServiceResult[ClusterAnalysis]:
        """
        Perform detailed analysis of clustering results.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            filepaths: Optional list of file paths for metadata

        Returns:
            Result with detailed cluster analysis
        """
        try:
            from bioamla.core.clustering import analyze_clusters

            metadata = None
            if filepaths:
                metadata = [{"filepath": fp} for fp in filepaths]

            analysis = analyze_clusters(embeddings, labels, metadata)

            result = ClusterAnalysis(
                n_clusters=analysis["n_clusters"],
                n_samples=analysis["n_samples"],
                n_noise=analysis["n_noise"],
                silhouette_score=analysis["silhouette_score"],
                calinski_harabasz_score=analysis["calinski_harabasz_score"],
                cluster_stats=analysis["cluster_stats"],
            )

            return ServiceResult.ok(
                data=result,
                message=f"Analyzed {result.n_clusters} clusters with silhouette={result.silhouette_score:.3f}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Dimensionality Reduction
    # =========================================================================

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            method: Reduction method ("umap", "tsne", "pca")
            n_components: Number of output dimensions
            output_path: Optional path to save reduced embeddings
            **kwargs: Additional arguments for the reducer

        Returns:
            Result with reduced embeddings
        """
        try:
            from bioamla.core.clustering import reduce_dimensions

            reduced = reduce_dimensions(
                embeddings,
                method=method,
                n_components=n_components,
                **kwargs,
            )

            if output_path:
                np.save(output_path, reduced)

            return ServiceResult.ok(
                data={
                    "original_shape": embeddings.shape,
                    "reduced_shape": reduced.shape,
                    "method": method,
                    "n_components": n_components,
                    "output_path": output_path,
                },
                message=f"Reduced from {embeddings.shape[-1]}D to {n_components}D using {method}",
                reduced_embeddings=reduced,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def reduce_for_visualization(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Reduce embeddings to 2D for visualization.

        Args:
            embeddings: Input embeddings
            method: Reduction method ("umap", "tsne", "pca")
            labels: Optional cluster labels for coloring
            **kwargs: Additional arguments for the reducer

        Returns:
            Result with 2D coordinates
        """
        result = self.reduce_dimensions(
            embeddings,
            method=method,
            n_components=2,
            **kwargs,
        )

        if result.success:
            coords = result.metadata["reduced_embeddings"]
            result.data["x"] = coords[:, 0].tolist()
            result.data["y"] = coords[:, 1].tolist()
            if labels is not None:
                result.data["labels"] = (
                    labels.tolist() if hasattr(labels, "tolist") else list(labels)
                )

        return result

    # =========================================================================
    # Novelty Detection
    # =========================================================================

    def detect_novelty(
        self,
        embeddings: np.ndarray,
        known_embeddings: Optional[np.ndarray] = None,
        known_labels: Optional[np.ndarray] = None,
        method: str = "distance",
        threshold: Optional[float] = None,
        contamination: float = 0.1,
    ) -> ServiceResult[NoveltyDetectionSummary]:
        """
        Detect novel/outlier samples in embeddings.

        Args:
            embeddings: Embeddings to check for novelty
            known_embeddings: Optional known embeddings to fit detector on
            known_labels: Optional labels for known embeddings
            method: Detection method ("distance", "isolation_forest", "lof")
            threshold: Novelty threshold (auto if None)
            contamination: Expected proportion of outliers

        Returns:
            Result with novelty detection summary
        """
        try:
            from bioamla.core.clustering import NoveltyDetector

            detector = NoveltyDetector(
                method=method,
                threshold=threshold,
                contamination=contamination,
            )

            # Fit on known embeddings or self
            if known_embeddings is not None:
                detector.fit(known_embeddings, known_labels)
            else:
                detector.fit(embeddings)

            # Predict novelty
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

            # Create boolean mask
            is_novel = np.array([r.is_novel for r in results])
            novelty_scores = np.array([r.novelty_score for r in results])

            self._novelty_detector = detector

            return ServiceResult.ok(
                data=summary,
                message=f"Detected {n_novel} novel samples ({summary.novel_percentage:.1f}%)",
                is_novel=is_novel,
                novelty_scores=novelty_scores,
                detector=detector,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def get_most_novel(
        self,
        embeddings: np.ndarray,
        n_samples: int = 10,
        known_embeddings: Optional[np.ndarray] = None,
        method: str = "distance",
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Get the most novel samples.

        Args:
            embeddings: Embeddings to check
            n_samples: Number of novel samples to return
            known_embeddings: Optional known embeddings to compare against
            method: Detection method

        Returns:
            Result with indices and scores of most novel samples
        """
        try:
            from bioamla.core.clustering import NoveltyDetector

            detector = NoveltyDetector(method=method)

            if known_embeddings is not None:
                detector.fit(known_embeddings)
            else:
                detector.fit(embeddings)

            results = detector.predict(embeddings)

            # Sort by novelty score
            sorted_results = sorted(results, key=lambda r: r.novelty_score, reverse=True)
            top_n = sorted_results[:n_samples]

            indices = [r.sample_idx for r in top_n]
            scores = [r.novelty_score for r in top_n]

            return ServiceResult.ok(
                data={
                    "indices": indices,
                    "scores": scores,
                    "n_samples": len(indices),
                },
                message=f"Found {len(indices)} most novel samples",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Cluster Similarity
    # =========================================================================

    def compute_similarity(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        metric: str = "cosine",
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Compute pairwise similarity between clusters.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            metric: Distance metric ("cosine", "euclidean")

        Returns:
            Result with similarity matrix
        """
        try:
            from bioamla.core.clustering import compute_cluster_similarity

            similarity = compute_cluster_similarity(embeddings, labels, metric=metric)

            return ServiceResult.ok(
                data={
                    "n_clusters": similarity.shape[0],
                    "metric": metric,
                },
                message=f"Computed {similarity.shape[0]}x{similarity.shape[0]} similarity matrix",
                similarity_matrix=similarity,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def sort_clusters_by_similarity(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        reference_cluster: Optional[int] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Sort clusters by similarity to each other.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels
            reference_cluster: Starting cluster (uses largest if None)

        Returns:
            Result with sorted cluster order
        """
        try:
            from bioamla.core.clustering import sort_clusters_by_similarity

            sorted_labels = sort_clusters_by_similarity(
                embeddings, labels, reference_label=reference_cluster
            )

            return ServiceResult.ok(
                data={
                    "sorted_labels": sorted_labels,
                    "reference_cluster": reference_cluster or "largest",
                },
                message=f"Sorted {len(sorted_labels)} clusters by similarity",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    # =========================================================================
    # Export
    # =========================================================================

    def export_clusters(
        self,
        labels: np.ndarray,
        filepaths: List[str],
        output_dir: str,
        copy_files: bool = False,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Export clustering results to directory structure.

        Args:
            labels: Cluster labels
            filepaths: List of file paths
            output_dir: Output directory
            copy_files: Whether to copy files to cluster directories

        Returns:
            Result with export info
        """
        try:
            from bioamla.core.clustering import export_clusters

            output_path = export_clusters(labels, filepaths, output_dir, copy_files=copy_files)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            return ServiceResult.ok(
                data={
                    "output_dir": output_path,
                    "n_clusters": n_clusters,
                    "n_files": len(filepaths),
                    "files_copied": copy_files,
                },
                message=f"Exported {n_clusters} clusters to {output_path}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))

    def export_to_csv(
        self,
        labels: np.ndarray,
        filepaths: List[str],
        output_path: str,
        embeddings: Optional[np.ndarray] = None,
        reduced_embeddings: Optional[np.ndarray] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Export clustering results to CSV.

        Args:
            labels: Cluster labels
            filepaths: List of file paths
            output_path: Output CSV path
            embeddings: Optional embeddings to include
            reduced_embeddings: Optional 2D coordinates to include

        Returns:
            Result with export info
        """
        try:
            from bioamla.services.file import FileService

            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = ["filepath", "cluster"]
            if reduced_embeddings is not None and reduced_embeddings.shape[1] >= 2:
                fieldnames.extend(["x", "y"])

            rows = []
            for i, (fp, label) in enumerate(zip(filepaths, labels)):
                row = {
                    "filepath": fp,
                    "cluster": int(label),
                }
                if reduced_embeddings is not None and reduced_embeddings.shape[1] >= 2:
                    row["x"] = float(reduced_embeddings[i, 0])
                    row["y"] = float(reduced_embeddings[i, 1])
                rows.append(row)

            file_svc = FileService()
            file_svc.write_csv_dicts(str(output_path_obj), rows, fieldnames=fieldnames)

            return ServiceResult.ok(
                data={
                    "output_path": str(output_path_obj),
                    "n_rows": len(labels),
                },
                message=f"Exported {len(labels)} rows to {output_path_obj}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))
