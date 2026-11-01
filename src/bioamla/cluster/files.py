"""Clustering on ``.npy`` files: load embeddings, run one step, save the result.

These are the file-level calls behind the ``bioamla cluster`` commands. Embeddings
are a 2-D array (samples x features) in a ``.npy`` file; labels are 1-D.

Example:
    >>> from bioamla.cluster import cluster_embeddings_file, reduce_embeddings_file
    >>> reduce_embeddings_file("emb.npy", "emb2d.npy", method="umap")
    >>> cluster_embeddings_file("emb2d.npy", "labels.npy").n_clusters
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bioamla.exceptions import ClusteringError, InvalidInputError


def load_array_file(path: str | Path, ndim: int | None = 2) -> np.ndarray:
    """Load a ``.npy`` array, checking its dimensions.

    Raises:
        InvalidInputError: If it can't be read or doesn't have ``ndim`` dimensions.
    """
    try:
        array = np.load(path)
    except (OSError, ValueError, EOFError) as e:
        raise InvalidInputError(f"Cannot read {path} as a .npy array: {e}") from e
    if ndim is not None and array.ndim != ndim:
        raise InvalidInputError(f"{path} must be a {ndim}-D array, got shape {array.shape}")
    return array


def _npy_path(path: str | Path) -> str:
    """The file ``np.save`` writes for ``path``: ``.npy`` is added when it's missing."""
    path = str(path)
    return path if path.endswith(".npy") else path + ".npy"


def save_array_file(path: str | Path, array: np.ndarray) -> str:
    """Save a ``.npy`` array, creating its folder.

    Returns:
        The file written (``path``, with ``.npy`` added if it was missing).

    Raises:
        ClusteringError: If it can't be written.
    """
    out = _npy_path(path)
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.save(out, array)
    except OSError as e:
        raise ClusteringError(f"Failed to write {path}: {e}") from e
    return out


@dataclass
class ReducedEmbeddings:
    """Embeddings reduced to fewer dimensions and saved.

    Attributes:
        file: The input embeddings.
        output: The reduced embeddings written.
        n_samples: Samples.
        n_features: Input dimensions.
        shape: The reduced array's shape.
    """

    file: str
    output: str
    n_samples: int
    n_features: int
    shape: list[int]

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output", "n_samples", "n_features", "shape"}``."""
        return {
            "file": self.file,
            "output": self.output,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "shape": self.shape,
        }


def reduce_embeddings_file(
    embeddings_file: str | Path,
    output: str | Path,
    *,
    method: str = "pca",
    n_components: int = 2,
) -> ReducedEmbeddings:
    """Reduce embeddings to ``n_components`` dimensions (``pca``, ``umap``, ``tsne``).

    Raises:
        InvalidInputError: If the embeddings can't be read or aren't 2-D.
        ClusteringError: If reduction or saving fails.
    """
    from bioamla.cluster.core import reduce_dimensions

    embeddings = load_array_file(embeddings_file)
    reduced = reduce_dimensions(embeddings, method=method, n_components=n_components)
    written = save_array_file(output, reduced)
    return ReducedEmbeddings(
        file=str(embeddings_file),
        output=written,
        n_samples=len(embeddings),
        n_features=int(embeddings.shape[1]),
        shape=list(reduced.shape),
    )


@dataclass
class ClusteredEmbeddings:
    """Cluster labels for embeddings, saved.

    Attributes:
        file: The input embeddings.
        output: The labels written (one per sample; -1 is noise).
        n_samples: Samples.
        n_clusters: Clusters found.
        n_noise: Samples labelled noise.
        cluster_sizes: Label -> sample count.
    """

    file: str
    output: str
    n_samples: int
    n_clusters: int
    n_noise: int
    cluster_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output", "n_samples", "n_clusters", "n_noise", "cluster_sizes"}``."""
        return {
            "file": self.file,
            "output": self.output,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "cluster_sizes": self.cluster_sizes,
        }


def cluster_embeddings_file(
    embeddings_file: str | Path,
    output: str | Path,
    *,
    method: str = "hdbscan",
    n_clusters: int = 10,
    min_cluster_size: int = 5,
    eps: float = 0.5,
    min_samples: int = 5,
) -> ClusteredEmbeddings:
    """Cluster embeddings and save one label per sample.

    Args:
        embeddings_file: The embeddings (``.npy``, 2-D).
        output: Where to save the labels (``.npy``).
        method: ``hdbscan``, ``kmeans``, ``dbscan`` or ``agglomerative``.
        n_clusters: Number of clusters (k-means, agglomerative).
        min_cluster_size: Minimum cluster size (HDBSCAN).
        eps: DBSCAN neighbourhood radius.
        min_samples: Minimum samples per cluster.

    Raises:
        InvalidInputError: If the embeddings can't be read or aren't 2-D.
        ClusteringError: If clustering or saving fails.
    """
    from bioamla.cluster.core import AudioClusterer, ClusteringConfig

    embeddings = load_array_file(embeddings_file)
    config = ClusteringConfig(
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        eps=eps,
    )
    clusterer = AudioClusterer(config=config)
    labels = np.asarray(clusterer.fit_predict(embeddings))
    written = save_array_file(output, labels)
    unique, counts = np.unique(labels, return_counts=True)
    return ClusteredEmbeddings(
        file=str(embeddings_file),
        output=written,
        n_samples=len(labels),
        n_clusters=int(clusterer.n_clusters_),
        n_noise=int((labels == -1).sum()),
        cluster_sizes={str(int(u)): int(c) for u, c in zip(unique, counts, strict=True)},
    )


@dataclass
class ClusterAnalysisReport:
    """Quality of a clustering.

    Attributes:
        file: The embeddings.
        labels_file: The labels.
        analysis: Clusters, samples, noise, silhouette and Calinski-Harabasz scores,
            per-cluster statistics.
        noise_percentage: Share of samples labelled noise (%).
        output: The JSON file the analysis was saved to, if any.
    """

    file: str
    labels_file: str
    analysis: Any
    noise_percentage: float
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "labels_file", **analysis, "noise_percentage", "output"}``."""
        from bioamla.common.jsonable import to_jsonable

        return {
            "file": self.file,
            "labels_file": self.labels_file,
            **to_jsonable(self.analysis),
            "noise_percentage": self.noise_percentage,
            "output": self.output,
        }


def analyze_clusters_file(
    embeddings_file: str | Path,
    labels_file: str | Path,
    *,
    output: str | Path | None = None,
) -> ClusterAnalysisReport:
    """Measure how well labels cluster embeddings, optionally saving the analysis as JSON.

    Raises:
        InvalidInputError: If a file can't be read or has the wrong shape.
        ClusteringError: If the analysis can't be saved.
    """
    from bioamla.cluster.core import analyze_clusters_summary
    from bioamla.common.jsonable import to_jsonable

    embeddings = load_array_file(embeddings_file)
    labels = load_array_file(labels_file, ndim=1)
    analysis = analyze_clusters_summary(embeddings, labels)
    noise_pct = analysis.n_noise / analysis.n_samples * 100 if analysis.n_samples > 0 else 0
    if output:
        data = {**to_jsonable(analysis), "noise_percentage": noise_pct}
        try:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            raise ClusteringError(f"Failed to write {output}: {e}") from e
    return ClusterAnalysisReport(
        file=str(embeddings_file),
        labels_file=str(labels_file),
        analysis=analysis,
        noise_percentage=noise_pct,
        output=str(output) if output else None,
    )


@dataclass
class NoveltyReport:
    """Novel samples among embeddings.

    Attributes:
        file: The embeddings.
        output: The ``(n_samples, 2)`` array written: is-novel (0/1), score.
        n_samples: Samples.
        n_novel: Novel samples.
        novel_percentage: Their share (%).
        novel_indices: Their indices.
    """

    file: str
    output: str
    n_samples: int
    n_novel: int
    novel_percentage: float
    novel_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        """``{"file", "output", "n_samples", "n_novel", "novel_percentage", "novel_indices"}``."""
        return {
            "file": self.file,
            "output": self.output,
            "n_samples": self.n_samples,
            "n_novel": self.n_novel,
            "novel_percentage": self.novel_percentage,
            "novel_indices": self.novel_indices,
        }


def novelty_file(
    embeddings_file: str | Path,
    output: str | Path,
    *,
    method: str = "distance",
    threshold: float | None = None,
    labels_file: str | Path | None = None,
) -> NoveltyReport:
    """Find novel samples among embeddings and save is-novel flags with scores.

    Args:
        embeddings_file: The embeddings (``.npy``, 2-D).
        output: Where to save the ``(n_samples, 2)`` result (``.npy``).
        method: ``distance``, ``isolation_forest`` or ``lof``.
        threshold: Novelty threshold (default: the method's).
        labels_file: Known cluster labels (``.npy``, 1-D).

    Raises:
        InvalidInputError: If a file can't be read or has the wrong shape.
        ClusteringError: If detection or saving fails.
    """
    from bioamla.cluster.core import detect_novelty

    embeddings = load_array_file(embeddings_file)
    known = load_array_file(labels_file, ndim=1) if labels_file else None
    summary, is_novel, scores = detect_novelty(
        embeddings, known_labels=known, method=method, threshold=threshold
    )
    written = save_array_file(output, np.column_stack([is_novel.astype(int), scores]))
    return NoveltyReport(
        file=str(embeddings_file),
        output=written,
        n_samples=len(is_novel),
        n_novel=int(summary.n_novel),
        novel_percentage=float(summary.novel_percentage),
        novel_indices=[int(i) for i in summary.novel_indices],
    )


__all__ = [
    "ClusterAnalysisReport",
    "ClusteredEmbeddings",
    "NoveltyReport",
    "ReducedEmbeddings",
    "analyze_clusters_file",
    "cluster_embeddings_file",
    "load_array_file",
    "novelty_file",
    "reduce_embeddings_file",
    "save_array_file",
]
