"""Clustering and dimensionality reduction models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bioamla.models.base import ToDictMixin


@dataclass
class ClusteringSummary(ToDictMixin):
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
class NoveltyDetectionSummary(ToDictMixin):
    """Summary of novelty detection results."""

    n_samples: int
    n_novel: int
    n_known: int
    novel_percentage: float
    method: str
    threshold: Optional[float] = None
    novel_indices: List[int] = field(default_factory=list)


@dataclass
class ClusterAnalysis(ToDictMixin):
    """Detailed cluster analysis results."""

    n_clusters: int
    n_samples: int
    n_noise: int
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
