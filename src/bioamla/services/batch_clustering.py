"""Batch clustering service."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase
from bioamla.services.clustering import ClusteringService


class BatchClusteringService(BatchServiceBase):
    """Service for batch clustering operations.

    This service delegates to ClusteringService for actual clustering operations,
    following the dependency injection pattern.

    Note: Clustering is different from other batch operations - it loads all
    embeddings first, then clusters the combined dataset once.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        clustering_service: ClusteringService,
    ) -> None:
        """Initialize batch clustering service.

        Args:
            file_repository: File repository for file discovery
            clustering_service: Single-operation clustering service to delegate to
        """
        super().__init__(file_repository)
        self.clustering_service = clustering_service
        self._embeddings_cache: List[np.ndarray] = []
        self._file_mapping: List[str] = []

    def process_file(self, file_path: Path) -> Any:
        """Load a single embedding file and cache it.

        This is Phase 1 of clustering - loading all embeddings.

        Args:
            file_path: Path to the embedding file

        Returns:
            Embedding shape

        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Load embedding based on file extension
            if file_path.suffix == ".npy":
                embedding = np.load(str(file_path))
            elif file_path.suffix in {".pkl", ".pickle"}:
                import pickle

                with open(file_path, "rb") as f:
                    embedding = pickle.load(f)
            elif file_path.suffix == ".json":
                with open(file_path) as f:
                    data = json.load(f)
                    embedding = np.array(data)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Ensure 1D or 2D
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                # Flatten to 2D if needed
                embedding = embedding.reshape(embedding.shape[0], -1)

            # Cache embedding and file reference
            self._embeddings_cache.append(embedding)
            self._file_mapping.append(str(file_path))

            return embedding.shape

        except Exception as e:
            raise RuntimeError(f"Failed to load embedding: {e}")

    def cluster_batch(
        self,
        config: BatchConfig,
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        **kwargs,
    ) -> BatchResult:
        """Cluster embeddings from batch files.

        This is a two-phase operation:
        1. Load all embeddings via process_file()
        2. Run clustering once on combined data
        3. Save cluster assignments

        Args:
            config: Batch processing configuration
            method: Clustering method ("hdbscan", "kmeans", "dbscan", "agglomerative")
            n_clusters: Number of clusters (for k-means/agglomerative)
            min_cluster_size: Minimum cluster size (for HDBSCAN)
            min_samples: Minimum samples per cluster
            **kwargs: Additional arguments for the clusterer

        Returns:
            BatchResult with processing summary
        """
        # Reset caches
        self._embeddings_cache = []
        self._file_mapping = []

        start_time = datetime.now()
        BatchResult(start_time=start_time.isoformat())

        # Phase 1: Load all embeddings
        def data_filter(path: Path) -> bool:
            data_exts = {".npy", ".pkl", ".pickle", ".json"}
            return path.suffix.lower() in data_exts

        # Use parent class to load files
        load_result = self.process_batch_auto(config, file_filter=data_filter)

        if load_result.total_files == 0:
            return load_result

        # Phase 2: Combine embeddings and run clustering
        try:
            # Concatenate all embeddings
            combined_embeddings = np.vstack(self._embeddings_cache)

            # Run clustering
            cluster_result = self.clustering_service.cluster(
                combined_embeddings,
                method=method,
                n_clusters=n_clusters,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                **kwargs,
            )

            if not cluster_result.success:
                load_result.errors.append(f"Clustering failed: {cluster_result.error}")
                load_result.failed = load_result.total_files
                load_result.successful = 0
                return load_result

            # Phase 3: Save cluster assignments
            output_path = Path(config.output_dir) / "cluster_assignments.json"
            self.file_repository.mkdir(str(output_path.parent), parents=True)

            # Map cluster labels back to files
            assignments = []
            label_idx = 0
            for file_path, embedding in zip(self._file_mapping, self._embeddings_cache):
                n_samples = embedding.shape[0]
                file_labels = cluster_result.data.labels[label_idx : label_idx + n_samples]
                assignments.append(
                    {
                        "filepath": file_path,
                        "cluster_labels": file_labels,
                        "n_samples": n_samples,
                    }
                )
                label_idx += n_samples

            # Save results
            output_data = {
                "method": method,
                "n_clusters": cluster_result.data.n_clusters,
                "n_samples": cluster_result.data.n_samples,
                "n_noise": cluster_result.data.n_noise,
                "noise_percentage": cluster_result.data.noise_percentage,
                "silhouette_score": cluster_result.data.silhouette_score,
                "cluster_sizes": cluster_result.data.cluster_sizes,
                "file_assignments": assignments,
            }

            content = json.dumps(output_data, indent=2)
            self.file_repository.write_text(output_path, content)

            # Update result metadata
            load_result.metadata = {
                "n_clusters": cluster_result.data.n_clusters,
                "n_samples": cluster_result.data.n_samples,
                "n_noise": cluster_result.data.n_noise,
                "output_file": str(output_path),
            }

        except Exception as e:
            load_result.errors.append(f"Clustering processing failed: {e}")
            load_result.failed = load_result.total_files
            load_result.successful = 0

        # Finalize result
        end_time = datetime.now()
        load_result.end_time = end_time.isoformat()
        load_result.duration_seconds = (end_time - start_time).total_seconds()

        return load_result
