"""Result dataclasses for the cluster domain.

These summarize clustering, novelty-detection, and cluster-analysis runs. They
were deduplicated from the former ``services/clustering.py`` and
``models/clustering.py`` (which defined identical dataclasses) — this is the
single canonical home.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClusteringSummary:
    """Summary of clustering results."""

    n_clusters: int
    n_samples: int
    n_noise: int
    noise_percentage: float
    silhouette_score: float
    method: str
    labels: list[int] = field(default_factory=list)
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


@dataclass
class NoveltyDetectionSummary:
    """Summary of novelty detection results."""

    n_samples: int
    n_novel: int
    n_known: int
    novel_percentage: float
    method: str
    threshold: float | None = None
    novel_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


@dataclass
class ClusterAnalysis:
    """Detailed cluster analysis results."""

    n_clusters: int
    n_samples: int
    n_noise: int
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_stats: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)
