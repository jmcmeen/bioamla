"""
Batch clustering
================

Two-phase batch clustering of embedding files:

1. Load every embedding file under an input directory (Phase 1).
2. Concatenate and cluster the combined dataset once (Phase 2), then map the
   resulting labels back to their source files and write an assignments JSON.

This folds the former ``services/batch_clustering.py`` into plain functions that
use :func:`bioamla.batch.run_batch` / :func:`bioamla.batch.discover_files` and
direct ``pathlib`` I/O.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bioamla.batch import BatchResult, run_batch
from bioamla.batch import discover_files as _discover_files
from bioamla.cluster.core import cluster_embeddings
from bioamla.exceptions import InvalidInputError

EMBEDDING_EXTENSIONS = {".npy", ".pkl", ".pickle", ".json"}


def load_embedding_file(file_path: str | Path) -> np.ndarray:
    """
    Load a single embedding file as a 2D array.

    Supports ``.npy``, ``.pkl``/``.pickle`` and ``.json``. 1D arrays are
    reshaped to ``(1, n_features)`` and arrays of rank > 2 are flattened to
    ``(n_samples, n_features)``.

    Args:
        file_path: Path to the embedding file.

    Returns:
        Embedding array of shape (n_samples, n_features).

    Raises:
        InvalidInputError: If the file extension is unsupported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        embedding = np.load(str(path))
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            embedding = pickle.load(f)
        embedding = np.asarray(embedding)
    elif suffix == ".json":
        embedding = np.array(json.loads(path.read_text(encoding="utf-8")))
    else:
        raise InvalidInputError(f"Unsupported embedding file format: {path.suffix}")

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    elif embedding.ndim > 2:
        embedding = embedding.reshape(embedding.shape[0], -1)

    return embedding


def load_embeddings_batch(
    input_dir: str | Path,
    *,
    recursive: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Phase 1: discover and load every embedding file under ``input_dir``.

    Args:
        input_dir: Directory to search for embedding files.
        recursive: Whether to recurse into subdirectories.

    Returns:
        Tuple ``(embeddings, filepaths)`` of per-file arrays and their string
        paths, in matching order.
    """

    def _is_embedding(path: Path) -> bool:
        return path.suffix.lower() in EMBEDDING_EXTENSIONS

    files = _discover_files(input_dir, recursive=recursive, file_filter=_is_embedding)

    embeddings: list[np.ndarray] = []
    filepaths: list[str] = []
    for path in files:
        embeddings.append(load_embedding_file(path))
        filepaths.append(str(path))

    return embeddings, filepaths


def cluster_batch_files(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    method: str = "hdbscan",
    n_clusters: int | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    recursive: bool = True,
    continue_on_error: bool = True,
    **kwargs: Any,
) -> BatchResult:
    """
    Cluster all embedding files under a directory (two-phase).

    Discovers embedding files under ``input_dir`` and delegates to
    :func:`cluster_embedding_files`.

    Args:
        input_dir: Directory containing embedding files.
        output_dir: Directory where ``cluster_assignments.json`` is written.
        method: Clustering method ("hdbscan", "kmeans", "dbscan", "agglomerative").
        n_clusters: Number of clusters (for k-means/agglomerative).
        min_cluster_size: Minimum cluster size (for HDBSCAN).
        min_samples: Minimum samples per cluster.
        recursive: Whether to recurse into subdirectories.
        continue_on_error: Keep going when an individual file fails to load.
        **kwargs: Additional arguments forwarded to the clusterer.

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run; on success its
        ``metadata`` holds clustering counts and the output file path.
    """

    def _is_embedding(path: Path) -> bool:
        return path.suffix.lower() in EMBEDDING_EXTENSIONS

    files = _discover_files(input_dir, recursive=recursive, file_filter=_is_embedding)
    return cluster_embedding_files(
        files,
        output_dir,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        continue_on_error=continue_on_error,
        **kwargs,
    )


def cluster_embedding_files(
    files: list[str | Path],
    output_dir: str | Path,
    *,
    method: str = "hdbscan",
    n_clusters: int | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    continue_on_error: bool = True,
    **kwargs: Any,
) -> BatchResult:
    """
    Cluster an explicit list of embedding files (two-phase).

    Phase 1 loads each embedding file (tracked via :func:`bioamla.batch.run_batch`
    so per-file failures are recorded). Phase 2 concatenates the successfully
    loaded embeddings, clusters them once, maps labels back to files and writes
    ``<output_dir>/cluster_assignments.json``.

    Args:
        files: Explicit list of embedding file paths.
        output_dir: Directory where ``cluster_assignments.json`` is written.
        method: Clustering method ("hdbscan", "kmeans", "dbscan", "agglomerative").
        n_clusters: Number of clusters (for k-means/agglomerative).
        min_cluster_size: Minimum cluster size (for HDBSCAN).
        min_samples: Minimum samples per cluster.
        continue_on_error: Keep going when an individual file fails to load.
        **kwargs: Additional arguments forwarded to the clusterer.

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run.
    """
    files = [Path(f) for f in files]

    # Phase 1: load each file, recording successes/failures.
    loaded: list[tuple[str, np.ndarray]] = []

    def _load(path: Path) -> str | None:
        loaded.append((str(path), load_embedding_file(path)))
        return None

    result = run_batch(files, _load, continue_on_error=continue_on_error)

    if not loaded:
        return result

    filepaths = [fp for fp, _ in loaded]
    embeddings = [emb for _, emb in loaded]

    # Phase 2: cluster the combined dataset once.
    try:
        combined = np.vstack(embeddings)
        summary = cluster_embeddings(
            combined,
            method=method,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            **kwargs,
        )

        labels = summary.labels

        # Phase 3: map labels back to their source files and persist.
        assignments = []
        label_idx = 0
        for file_path, embedding in zip(filepaths, embeddings):
            n_samples = embedding.shape[0]
            file_labels = labels[label_idx : label_idx + n_samples]
            assignments.append(
                {
                    "filepath": file_path,
                    "cluster_labels": [int(label) for label in file_labels],
                    "n_samples": int(n_samples),
                }
            )
            label_idx += n_samples

        output_path = Path(output_dir) / "cluster_assignments.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "method": method,
            "n_clusters": summary.n_clusters,
            "n_samples": summary.n_samples,
            "n_noise": summary.n_noise,
            "noise_percentage": summary.noise_percentage,
            "silhouette_score": summary.silhouette_score,
            "cluster_sizes": summary.cluster_sizes,
            "file_assignments": assignments,
        }
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

        result.metadata = {
            "n_clusters": summary.n_clusters,
            "n_samples": summary.n_samples,
            "n_noise": summary.n_noise,
            "output_file": str(output_path),
        }
    except Exception as e:
        result.errors.append(f"Clustering processing failed: {e}")
        result.failed = result.total_files
        result.successful = 0

    return result
