"""
Active Learning Annotation Loop
===============================

This module provides active learning functionality for efficiently annotating
bioacoustic datasets. It implements various sampling strategies to select the
most informative samples for human annotation.

Key Features:
- Uncertainty sampling (least confidence, margin, entropy)
- Diversity sampling (clustering-based, feature-space)
- Query-by-committee
- Hybrid strategies combining uncertainty and diversity
- Annotation queue management
- Oracle interface for labeling
- Progress tracking and statistics

Example:
    >>> from bioamla.active_learning import ActiveLearner, UncertaintySampler
    >>> sampler = UncertaintySampler(strategy="entropy")
    >>> learner = ActiveLearner(model=model, sampler=sampler)
    >>>
    >>> # Get samples to annotate
    >>> queries = learner.query(unlabeled_pool, n_samples=10)
    >>>
    >>> # Simulate annotation
    >>> for sample in queries:
    ...     label = oracle.annotate(sample)
    ...     learner.teach(sample, label)
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ActiveLearningSample:
    """
    Represents a sample in the active learning pool.

    Attributes:
        id: Unique identifier for the sample
        filepath: Path to the audio file
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
        features: Optional precomputed features (e.g., embeddings)
        metadata: Additional metadata (e.g., recording info)
        label: Ground truth label (None if unlabeled)
        predicted_label: Model's predicted label
        confidence: Model's confidence for the predicted label
        probabilities: Full probability distribution over classes
        uncertainty_score: Computed uncertainty score for sampling
    """

    id: str
    filepath: str
    start_time: float = 0.0
    end_time: float = 0.0
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[np.ndarray] = None
    uncertainty_score: Optional[float] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, ActiveLearningSample):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "id": self.id,
            "filepath": self.filepath,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "uncertainty_score": self.uncertainty_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActiveLearningSample":
        """Create a Sample from a dictionary."""
        return cls(
            id=data["id"],
            filepath=data["filepath"],
            start_time=data.get("start_time", 0.0),
            end_time=data.get("end_time", 0.0),
            label=data.get("label"),
            predicted_label=data.get("predicted_label"),
            confidence=data.get("confidence"),
            uncertainty_score=data.get("uncertainty_score"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AnnotationRecord:
    """
    Record of a single annotation event.

    Attributes:
        sample_id: ID of the annotated sample
        label: Assigned label
        annotator: Identifier of the annotator
        timestamp: When the annotation was made
        duration_seconds: How long annotation took
        confidence: Annotator's confidence in the label
        notes: Optional notes from annotator
    """

    sample_id: str
    label: str
    annotator: str = "unknown"
    timestamp: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    confidence: Optional[float] = None
    notes: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "annotator": self.annotator,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class ActiveLearningState:
    """
    Tracks the state of an active learning session.

    Attributes:
        iteration: Current iteration number
        total_labeled: Total number of labeled samples
        total_unlabeled: Total number of unlabeled samples
        labels_per_class: Count of labels per class
        query_history: History of queried sample IDs
        performance_history: History of model performance metrics
    """

    iteration: int = 0
    total_labeled: int = 0
    total_unlabeled: int = 0
    labels_per_class: Dict[str, int] = field(default_factory=dict)
    query_history: List[List[str]] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "iteration": self.iteration,
            "total_labeled": self.total_labeled,
            "total_unlabeled": self.total_unlabeled,
            "labels_per_class": self.labels_per_class,
            "query_history": self.query_history,
            "performance_history": self.performance_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActiveLearningState":
        """Create state from dictionary."""
        return cls(
            iteration=data.get("iteration", 0),
            total_labeled=data.get("total_labeled", 0),
            total_unlabeled=data.get("total_unlabeled", 0),
            labels_per_class=data.get("labels_per_class", {}),
            query_history=data.get("query_history", []),
            performance_history=data.get("performance_history", []),
        )


# =============================================================================
# Sampling Strategies
# =============================================================================


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """
        Compute informativeness scores for samples.

        Args:
            samples: List of samples to score

        Returns:
            Array of scores (higher = more informative)
        """
        pass

    def select(
        self, samples: List[ActiveLearningSample], n_samples: int, exclude_ids: Optional[set] = None
    ) -> List[ActiveLearningSample]:
        """
        Select the most informative samples.

        Args:
            samples: Pool of samples to select from
            n_samples: Number of samples to select
            exclude_ids: Sample IDs to exclude from selection

        Returns:
            List of selected samples
        """
        if exclude_ids:
            samples = [s for s in samples if s.id not in exclude_ids]

        if len(samples) == 0:
            return []

        if len(samples) <= n_samples:
            return samples

        scores = self.score(samples)
        indices = np.argsort(scores)[::-1][:n_samples]

        return [samples[i] for i in indices]


class UncertaintySampler(SamplingStrategy):
    """
    Uncertainty-based sampling strategy.

    Selects samples where the model is most uncertain about the prediction.

    Strategies:
        - "least_confidence": 1 - max(probabilities)
        - "margin": 1 - (p_1 - p_2), difference between top two probs
        - "entropy": -sum(p * log(p)), Shannon entropy
    """

    def __init__(self, strategy: str = "entropy"):
        """
        Initialize uncertainty sampler.

        Args:
            strategy: One of "least_confidence", "margin", or "entropy"
        """
        if strategy not in ["least_confidence", "margin", "entropy"]:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """Compute uncertainty scores for samples."""
        scores = np.zeros(len(samples))

        for i, sample in enumerate(samples):
            if sample.probabilities is None:
                # Use confidence as fallback
                if sample.confidence is not None:
                    scores[i] = 1 - sample.confidence
                else:
                    scores[i] = 0.5  # Default uncertainty
            else:
                probs = sample.probabilities

                if self.strategy == "least_confidence":
                    scores[i] = 1 - np.max(probs)

                elif self.strategy == "margin":
                    sorted_probs = np.sort(probs)[::-1]
                    if len(sorted_probs) >= 2:
                        scores[i] = 1 - (sorted_probs[0] - sorted_probs[1])
                    else:
                        scores[i] = 1 - sorted_probs[0]

                elif self.strategy == "entropy":
                    # Add small epsilon to avoid log(0)
                    probs_safe = np.clip(probs, 1e-10, 1.0)
                    scores[i] = -np.sum(probs_safe * np.log(probs_safe))

        return scores


class DiversitySampler(SamplingStrategy):
    """
    Diversity-based sampling strategy.

    Selects samples that are diverse in feature space to maximize coverage.
    Uses k-medoids or greedy farthest-first traversal.
    """

    def __init__(self, method: str = "greedy"):
        """
        Initialize diversity sampler.

        Args:
            method: One of "greedy" (farthest-first) or "kmeans"
        """
        if method not in ["greedy", "kmeans"]:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """
        Compute diversity scores.

        Note: For diversity sampling, scores represent distance to
        nearest selected sample. Initial scores are based on distance
        to feature centroid.
        """
        # Get features
        features = self._get_features(samples)

        if features is None:
            # No features available, use random scores
            return np.random.rand(len(samples))

        # Compute distance to centroid as initial diversity score
        centroid = np.mean(features, axis=0)
        distances = np.linalg.norm(features - centroid, axis=1)

        return distances

    def select(
        self, samples: List[ActiveLearningSample], n_samples: int, exclude_ids: Optional[set] = None
    ) -> List[ActiveLearningSample]:
        """Select diverse samples using greedy farthest-first traversal."""
        if exclude_ids:
            samples = [s for s in samples if s.id not in exclude_ids]

        if len(samples) == 0:
            return []

        if len(samples) <= n_samples:
            return samples

        features = self._get_features(samples)

        if features is None or self.method != "greedy":
            # Fall back to score-based selection
            return super().select(samples, n_samples, exclude_ids=None)

        # Greedy farthest-first traversal
        selected_indices = []

        # Start with the sample closest to centroid
        centroid = np.mean(features, axis=0)
        first_idx = np.argmin(np.linalg.norm(features - centroid, axis=1))
        selected_indices.append(first_idx)

        # Iteratively add farthest sample
        while len(selected_indices) < n_samples:
            selected_features = features[selected_indices]

            # Compute minimum distance to selected set for each sample
            min_distances = np.full(len(samples), np.inf)
            for i in range(len(samples)):
                if i not in selected_indices:
                    dists = np.linalg.norm(selected_features - features[i], axis=1)
                    min_distances[i] = np.min(dists)
                else:
                    min_distances[i] = -np.inf

            # Select sample with maximum minimum distance
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(farthest_idx)

        return [samples[i] for i in selected_indices]

    def _get_features(self, samples: List[ActiveLearningSample]) -> Optional[np.ndarray]:
        """Extract features from samples."""
        features = []
        for sample in samples:
            if sample.features is not None:
                features.append(sample.features)
            elif sample.probabilities is not None:
                features.append(sample.probabilities)
            else:
                return None

        return np.array(features)


class HybridSampler(SamplingStrategy):
    """
    Hybrid sampling combining uncertainty and diversity.

    Uses a two-stage approach:
    1. Filter top-k uncertain samples
    2. Select diverse samples from the filtered set
    """

    def __init__(
        self,
        uncertainty_strategy: str = "entropy",
        diversity_method: str = "greedy",
        uncertainty_ratio: float = 0.5,
    ):
        """
        Initialize hybrid sampler.

        Args:
            uncertainty_strategy: Strategy for uncertainty sampling
            diversity_method: Method for diversity sampling
            uncertainty_ratio: Ratio of samples to pre-filter by uncertainty
        """
        self.uncertainty_sampler = UncertaintySampler(strategy=uncertainty_strategy)
        self.diversity_sampler = DiversitySampler(method=diversity_method)
        self.uncertainty_ratio = uncertainty_ratio

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """Compute hybrid scores (uncertainty weighted by diversity)."""
        uncertainty_scores = self.uncertainty_sampler.score(samples)
        diversity_scores = self.diversity_sampler.score(samples)

        # Normalize scores
        if uncertainty_scores.max() > 0:
            uncertainty_scores = uncertainty_scores / uncertainty_scores.max()
        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()

        # Combine scores
        return (
            self.uncertainty_ratio * uncertainty_scores
            + (1 - self.uncertainty_ratio) * diversity_scores
        )

    def select(
        self, samples: List[ActiveLearningSample], n_samples: int, exclude_ids: Optional[set] = None
    ) -> List[ActiveLearningSample]:
        """Select samples using two-stage approach."""
        if exclude_ids:
            samples = [s for s in samples if s.id not in exclude_ids]

        if len(samples) <= n_samples:
            return samples

        # Stage 1: Pre-filter by uncertainty
        pre_filter_k = min(len(samples), n_samples * 3)
        uncertain_samples = self.uncertainty_sampler.select(samples, pre_filter_k, exclude_ids=None)

        # Stage 2: Select diverse samples from filtered set
        return self.diversity_sampler.select(uncertain_samples, n_samples, exclude_ids=None)


class RandomSampler(SamplingStrategy):
    """Random sampling strategy (baseline)."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize random sampler with optional seed."""
        self.rng = np.random.default_rng(seed)

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """Return random scores."""
        return self.rng.random(len(samples))


class QueryByCommittee(SamplingStrategy):
    """
    Query by Committee sampling strategy.

    Uses disagreement among multiple models (committee) to identify
    informative samples.
    """

    def __init__(self, disagreement_measure: str = "vote_entropy"):
        """
        Initialize QBC sampler.

        Args:
            disagreement_measure: One of "vote_entropy" or "kl_divergence"
        """
        self.disagreement_measure = disagreement_measure
        self.committee_predictions: List[Dict[str, np.ndarray]] = []

    def add_committee_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        """
        Add predictions from a committee member.

        Args:
            predictions: Dict mapping sample_id to probability array
        """
        self.committee_predictions.append(predictions)

    def clear_committee(self) -> None:
        """Clear all committee predictions."""
        self.committee_predictions = []

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """Compute disagreement scores."""
        scores = np.zeros(len(samples))

        if not self.committee_predictions:
            logger.warning("No committee predictions available, using random scores")
            return np.random.rand(len(samples))

        for i, sample in enumerate(samples):
            member_probs = []
            for member_preds in self.committee_predictions:
                if sample.id in member_preds:
                    member_probs.append(member_preds[sample.id])

            if len(member_probs) < 2:
                scores[i] = 0
                continue

            member_probs = np.array(member_probs)

            if self.disagreement_measure == "vote_entropy":
                # Vote entropy: entropy of hard vote distribution
                votes = np.argmax(member_probs, axis=1)
                vote_counts = np.bincount(votes, minlength=member_probs.shape[1])
                vote_dist = vote_counts / len(votes)
                vote_dist = np.clip(vote_dist, 1e-10, 1.0)
                scores[i] = -np.sum(vote_dist * np.log(vote_dist))

            elif self.disagreement_measure == "kl_divergence":
                # Average KL divergence from consensus
                consensus = np.mean(member_probs, axis=0)
                consensus = np.clip(consensus, 1e-10, 1.0)
                kl_divs = []
                for probs in member_probs:
                    probs = np.clip(probs, 1e-10, 1.0)
                    kl = np.sum(probs * np.log(probs / consensus))
                    kl_divs.append(kl)
                scores[i] = np.mean(kl_divs)

        return scores


class BalancedSampler(SamplingStrategy):
    """
    Class-balanced sampling strategy.

    Prioritizes samples from underrepresented classes to maintain
    balanced training data.
    """

    def __init__(self, class_counts: Optional[Dict[str, int]] = None):
        """
        Initialize balanced sampler.

        Args:
            class_counts: Current count of samples per class
        """
        self.class_counts = class_counts or {}

    def update_counts(self, class_counts: Dict[str, int]) -> None:
        """Update class counts."""
        self.class_counts = class_counts.copy()

    def score(self, samples: List[ActiveLearningSample]) -> np.ndarray:
        """Score samples inversely by class frequency."""
        scores = np.zeros(len(samples))

        if not self.class_counts:
            # No class info, use uniform scores
            return np.ones(len(samples))

        total = sum(self.class_counts.values()) + 1

        for i, sample in enumerate(samples):
            pred_label = sample.predicted_label
            if pred_label is not None and pred_label in self.class_counts:
                # Inverse frequency weighting
                class_freq = self.class_counts[pred_label] / total
                scores[i] = 1 / (class_freq + 1e-10)
            else:
                # Unknown class gets high score
                scores[i] = total

        return scores


# =============================================================================
# Active Learner
# =============================================================================


class ActiveLearner:
    """
    Main active learning orchestrator.

    Manages the active learning loop including:
    - Querying informative samples from unlabeled pool
    - Tracking annotation progress
    - Managing labeled/unlabeled pools
    - Saving/loading state
    """

    def __init__(
        self,
        sampler: Optional[SamplingStrategy] = None,
        prediction_fn: Optional[
            Callable[[List[ActiveLearningSample]], List[ActiveLearningSample]]
        ] = None,
        state: Optional[ActiveLearningState] = None,
    ):
        """
        Initialize active learner.

        Args:
            sampler: Sampling strategy to use
            prediction_fn: Function to get model predictions for samples
            state: Optional existing state to resume from
        """
        self.sampler = sampler or UncertaintySampler(strategy="entropy")
        self.prediction_fn = prediction_fn
        self.state = state or ActiveLearningState()

        self.labeled_pool: Dict[str, ActiveLearningSample] = {}
        self.unlabeled_pool: Dict[str, ActiveLearningSample] = {}
        self.annotation_history: List[AnnotationRecord] = []

    def add_unlabeled(self, samples: List[ActiveLearningSample]) -> None:
        """
        Add samples to the unlabeled pool.

        Args:
            samples: List of samples to add
        """
        for sample in samples:
            if sample.id not in self.labeled_pool:
                self.unlabeled_pool[sample.id] = sample

        self.state.total_unlabeled = len(self.unlabeled_pool)
        logger.info(f"Added {len(samples)} samples to unlabeled pool")

    def add_labeled(self, samples: List[ActiveLearningSample]) -> None:
        """
        Add pre-labeled samples (e.g., initial seed set).

        Args:
            samples: List of labeled samples
        """
        for sample in samples:
            if sample.label is not None:
                self.labeled_pool[sample.id] = sample
                self.unlabeled_pool.pop(sample.id, None)

                # Update class counts
                if sample.label not in self.state.labels_per_class:
                    self.state.labels_per_class[sample.label] = 0
                self.state.labels_per_class[sample.label] += 1

        self.state.total_labeled = len(self.labeled_pool)
        self.state.total_unlabeled = len(self.unlabeled_pool)
        logger.info(f"Added {len(samples)} samples to labeled pool")

    def query(
        self, n_samples: int = 10, update_predictions: bool = True
    ) -> List[ActiveLearningSample]:
        """
        Query the most informative samples for annotation.

        Args:
            n_samples: Number of samples to query
            update_predictions: Whether to update model predictions first

        Returns:
            List of selected samples for annotation
        """
        if len(self.unlabeled_pool) == 0:
            logger.warning("Unlabeled pool is empty")
            return []

        samples = list(self.unlabeled_pool.values())

        # Update predictions if requested
        if update_predictions and self.prediction_fn is not None:
            samples = self.prediction_fn(samples)
            # Update pool with predictions
            for sample in samples:
                self.unlabeled_pool[sample.id] = sample

        # Select samples using strategy
        selected = self.sampler.select(
            samples, n_samples, exclude_ids=set(self.labeled_pool.keys())
        )

        # Record query
        self.state.query_history.append([s.id for s in selected])
        self.state.iteration += 1

        logger.info(f"Queried {len(selected)} samples (iteration {self.state.iteration})")
        return selected

    def teach(
        self,
        sample: ActiveLearningSample,
        label: str,
        annotator: str = "unknown",
        duration_seconds: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """
        Record an annotation and move sample to labeled pool.

        Args:
            sample: The annotated sample
            label: The assigned label
            annotator: Identifier of the annotator
            duration_seconds: How long annotation took
            notes: Optional notes
        """
        # Create annotation record
        record = AnnotationRecord(
            sample_id=sample.id,
            label=label,
            annotator=annotator,
            duration_seconds=duration_seconds,
            notes=notes,
        )
        self.annotation_history.append(record)

        # Update sample
        sample.label = label

        # Move to labeled pool
        self.labeled_pool[sample.id] = sample
        self.unlabeled_pool.pop(sample.id, None)

        # Update state
        self.state.total_labeled = len(self.labeled_pool)
        self.state.total_unlabeled = len(self.unlabeled_pool)

        if label not in self.state.labels_per_class:
            self.state.labels_per_class[label] = 0
        self.state.labels_per_class[label] += 1

        logger.debug(f"Recorded annotation for {sample.id}: {label}")

    def teach_batch(
        self, annotations: List[Tuple[ActiveLearningSample, str]], annotator: str = "unknown"
    ) -> None:
        """
        Record multiple annotations at once.

        Args:
            annotations: List of (sample, label) tuples
            annotator: Identifier of the annotator
        """
        for sample, label in annotations:
            self.teach(sample, label, annotator=annotator)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current active learning statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "iteration": self.state.iteration,
            "total_labeled": self.state.total_labeled,
            "total_unlabeled": self.state.total_unlabeled,
            "labels_per_class": self.state.labels_per_class.copy(),
            "total_annotations": len(self.annotation_history),
            "queries_made": len(self.state.query_history),
        }

    def get_labeled_samples(self) -> List[ActiveLearningSample]:
        """Get all labeled samples."""
        return list(self.labeled_pool.values())

    def get_unlabeled_samples(self) -> List[ActiveLearningSample]:
        """Get all unlabeled samples."""
        return list(self.unlabeled_pool.values())

    def record_performance(self, metrics: Dict[str, float]) -> None:
        """
        Record model performance metrics.

        Args:
            metrics: Dictionary of performance metrics (e.g., accuracy, f1)
        """
        self.state.performance_history.append(
            {"iteration": self.state.iteration, "timestamp": datetime.now().isoformat(), **metrics}
        )

    def save_state(self, filepath: str) -> str:
        """
        Save active learning state to file.

        Args:
            filepath: Path to save state

        Returns:
            Path to saved file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "state": self.state.to_dict(),
            "labeled_pool": {k: v.to_dict() for k, v in self.labeled_pool.items()},
            "unlabeled_pool": {k: v.to_dict() for k, v in self.unlabeled_pool.items()},
            "annotation_history": [r.to_dict() for r in self.annotation_history],
        }

        with TextFile(path, mode="w", encoding="utf-8") as f:
            json.dump(state_dict, f.handle, indent=2)

        logger.info(f"Saved active learning state to {filepath}")
        return str(path)

    @classmethod
    def load_state(
        cls,
        filepath: str,
        sampler: Optional[SamplingStrategy] = None,
        prediction_fn: Optional[Callable] = None,
    ) -> "ActiveLearner":
        """
        Load active learning state from file.

        Args:
            filepath: Path to state file
            sampler: Sampling strategy to use
            prediction_fn: Function to get model predictions

        Returns:
            ActiveLearner instance with loaded state
        """
        with TextFile(filepath, mode="r", encoding="utf-8") as f:
            state_dict = json.load(f.handle)

        learner = cls(
            sampler=sampler,
            prediction_fn=prediction_fn,
            state=ActiveLearningState.from_dict(state_dict["state"]),
        )

        # Restore pools
        learner.labeled_pool = {
            k: ActiveLearningSample.from_dict(v)
            for k, v in state_dict.get("labeled_pool", {}).items()
        }
        learner.unlabeled_pool = {
            k: ActiveLearningSample.from_dict(v)
            for k, v in state_dict.get("unlabeled_pool", {}).items()
        }

        # Restore annotation history
        for record_dict in state_dict.get("annotation_history", []):
            record = AnnotationRecord(
                sample_id=record_dict["sample_id"],
                label=record_dict["label"],
                annotator=record_dict.get("annotator", "unknown"),
                timestamp=datetime.fromisoformat(record_dict["timestamp"])
                if record_dict.get("timestamp")
                else None,
                duration_seconds=record_dict.get("duration_seconds"),
                notes=record_dict.get("notes", ""),
            )
            learner.annotation_history.append(record)

        logger.info(f"Loaded active learning state from {filepath}")
        return learner


# =============================================================================
# Annotation Queue
# =============================================================================


@dataclass
class AnnotationQueue:
    """
    Manages a queue of samples awaiting annotation.

    Features:
    - FIFO ordering with priority support
    - Progress tracking
    - Export/import functionality
    """

    samples: List[ActiveLearningSample] = field(default_factory=list)
    current_index: int = 0
    completed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, samples: List[ActiveLearningSample], priority: bool = False) -> None:
        """
        Add samples to the queue.

        Args:
            samples: Samples to add
            priority: If True, add to front of queue
        """
        if priority:
            self.samples = samples + self.samples
        else:
            self.samples.extend(samples)

    def current(self) -> Optional[ActiveLearningSample]:
        """Get the current sample to annotate."""
        if self.current_index < len(self.samples):
            return self.samples[self.current_index]
        return None

    def next(self) -> Optional[ActiveLearningSample]:
        """Move to and return the next sample."""
        self.current_index += 1
        return self.current()

    def previous(self) -> Optional[ActiveLearningSample]:
        """Move to and return the previous sample."""
        if self.current_index > 0:
            self.current_index -= 1
        return self.current()

    def mark_completed(self, sample_id: str) -> None:
        """Mark a sample as completed."""
        if sample_id not in self.completed:
            self.completed.append(sample_id)

    def mark_skipped(self, sample_id: str) -> None:
        """Mark a sample as skipped."""
        if sample_id not in self.skipped:
            self.skipped.append(sample_id)

    def get_progress(self) -> Dict[str, Any]:
        """Get queue progress information."""
        total = len(self.samples)
        return {
            "total": total,
            "completed": len(self.completed),
            "skipped": len(self.skipped),
            "remaining": total - len(self.completed) - len(self.skipped),
            "current_position": self.current_index + 1,
            "progress_percent": (len(self.completed) / total * 100) if total > 0 else 0,
        }

    def export_to_csv(self, filepath: str) -> str:
        """Export queue to CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with TextFile(path, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "filepath",
                "start_time",
                "end_time",
                "predicted_label",
                "confidence",
                "status",
            ]
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()

            for sample in self.samples:
                status = "pending"
                if sample.id in self.completed:
                    status = "completed"
                elif sample.id in self.skipped:
                    status = "skipped"

                writer.writerow(
                    {
                        "id": sample.id,
                        "filepath": sample.filepath,
                        "start_time": sample.start_time,
                        "end_time": sample.end_time,
                        "predicted_label": sample.predicted_label or "",
                        "confidence": sample.confidence or "",
                        "status": status,
                    }
                )

        return str(path)


# =============================================================================
# Oracle Interface
# =============================================================================


class Oracle(ABC):
    """
    Abstract base class for annotation oracles.

    An oracle provides labels for samples. This can be:
    - Human annotator (interactive)
    - Simulated oracle (for experiments using ground truth)
    - External service
    """

    @abstractmethod
    def annotate(self, sample: ActiveLearningSample) -> str:
        """
        Get annotation for a sample.

        Args:
            sample: Sample to annotate

        Returns:
            Label string
        """
        pass

    def annotate_batch(self, samples: List[ActiveLearningSample]) -> List[str]:
        """
        Annotate multiple samples.

        Args:
            samples: List of samples to annotate

        Returns:
            List of labels
        """
        return [self.annotate(sample) for sample in samples]


class SimulatedOracle(Oracle):
    """
    Simulated oracle using ground truth labels.

    Useful for active learning experiments and benchmarking.
    """

    def __init__(
        self,
        ground_truth: Dict[str, str],
        noise_rate: float = 0.0,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize simulated oracle.

        Args:
            ground_truth: Dict mapping sample_id to true label
            noise_rate: Probability of returning wrong label (0-1)
            labels: List of possible labels (for noise generation)
        """
        self.ground_truth = ground_truth
        self.noise_rate = noise_rate
        self.labels = labels or list(set(ground_truth.values()))
        self.rng = np.random.default_rng()

    def annotate(self, sample: ActiveLearningSample) -> str:
        """Get (possibly noisy) label for sample."""
        if sample.id not in self.ground_truth:
            raise ValueError(f"No ground truth for sample {sample.id}")

        true_label = self.ground_truth[sample.id]

        # Optionally add noise
        if self.noise_rate > 0 and self.rng.random() < self.noise_rate:
            other_labels = [label for label in self.labels if label != true_label]
            if other_labels:
                return self.rng.choice(other_labels)

        return true_label


class CallbackOracle(Oracle):
    """Oracle that uses a callback function for annotation."""

    def __init__(self, callback: Callable[[ActiveLearningSample], str]):
        """
        Initialize callback oracle.

        Args:
            callback: Function that takes a Sample and returns a label
        """
        self.callback = callback

    def annotate(self, sample: ActiveLearningSample) -> str:
        """Get label using callback."""
        return self.callback(sample)


# =============================================================================
# Utility Functions
# =============================================================================


def create_samples_from_predictions(
    predictions_csv: str,
    filepath_col: str = "filepath",
    start_col: str = "start_time",
    end_col: str = "end_time",
    label_col: str = "predicted_label",
    confidence_col: str = "confidence",
) -> List[ActiveLearningSample]:
    """
    Create Sample objects from a predictions CSV file.

    Args:
        predictions_csv: Path to predictions CSV
        filepath_col: Column name for file path
        start_col: Column name for start time
        end_col: Column name for end time
        label_col: Column name for predicted label
        confidence_col: Column name for confidence

    Returns:
        List of Sample objects
    """
    samples = []

    with TextFile(predictions_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f.handle)

        for _i, row in enumerate(reader):
            filepath = row.get(filepath_col, "")
            start_time = float(row.get(start_col, 0))
            end_time = float(row.get(end_col, 0))

            # Create unique ID
            sample_id = f"{Path(filepath).stem}_{start_time:.2f}_{end_time:.2f}"

            sample = ActiveLearningSample(
                id=sample_id,
                filepath=filepath,
                start_time=start_time,
                end_time=end_time,
                predicted_label=row.get(label_col),
                confidence=float(row.get(confidence_col, 0)) if row.get(confidence_col) else None,
            )
            samples.append(sample)

    logger.info(f"Created {len(samples)} samples from {predictions_csv}")
    return samples


def export_annotations(learner: ActiveLearner, filepath: str, format: str = "csv") -> str:
    """
    Export annotations from active learner.

    Args:
        learner: ActiveLearner instance
        filepath: Output file path
        format: Output format ("csv" or "raven")

    Returns:
        Path to exported file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    labeled_samples = learner.get_labeled_samples()

    if format == "csv":
        with TextFile(path, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "filepath",
                "start_time",
                "end_time",
                "label",
                "predicted_label",
                "confidence",
            ]
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()

            for sample in labeled_samples:
                writer.writerow(
                    {
                        "id": sample.id,
                        "filepath": sample.filepath,
                        "start_time": sample.start_time,
                        "end_time": sample.end_time,
                        "label": sample.label,
                        "predicted_label": sample.predicted_label or "",
                        "confidence": sample.confidence or "",
                    }
                )

    elif format == "raven":
        # Export as Raven selection table
        from bioamla.core.annotations import Annotation, save_raven_selection_table

        annotations = []
        for sample in labeled_samples:
            ann = Annotation(
                start_time=sample.start_time,
                end_time=sample.end_time,
                label=sample.label or "",
                confidence=sample.confidence,
                custom_fields={"sample_id": sample.id, "filepath": sample.filepath},
            )
            annotations.append(ann)

        save_raven_selection_table(annotations, str(path))

    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Exported {len(labeled_samples)} annotations to {filepath}")
    return str(path)


def compute_sample_uncertainty(probabilities: np.ndarray, strategy: str = "entropy") -> float:
    """
    Compute uncertainty score for a single probability distribution.

    Args:
        probabilities: Probability distribution over classes
        strategy: One of "least_confidence", "margin", "entropy"

    Returns:
        Uncertainty score
    """
    probs = np.asarray(probabilities)
    probs = np.clip(probs, 1e-10, 1.0)

    if strategy == "least_confidence":
        return 1 - np.max(probs)
    elif strategy == "margin":
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2:
            return 1 - (sorted_probs[0] - sorted_probs[1])
        return 1 - sorted_probs[0]
    elif strategy == "entropy":
        return -np.sum(probs * np.log(probs))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def summarize_annotation_session(learner: ActiveLearner) -> Dict[str, Any]:
    """
    Generate summary statistics for an annotation session.

    Args:
        learner: ActiveLearner instance

    Returns:
        Dictionary with session summary
    """
    stats = learner.get_statistics()

    # Compute annotation rate
    total_time = 0.0
    for record in learner.annotation_history:
        if record.duration_seconds:
            total_time += record.duration_seconds

    annotations_per_hour = 0
    if total_time > 0:
        annotations_per_hour = len(learner.annotation_history) / (total_time / 3600)

    # Class balance
    class_counts = stats["labels_per_class"]
    if class_counts:
        counts = list(class_counts.values())
        class_balance = min(counts) / max(counts) if max(counts) > 0 else 0
    else:
        class_balance = 0

    return {
        **stats,
        "total_annotation_time_seconds": total_time,
        "annotations_per_hour": annotations_per_hour,
        "class_balance_ratio": class_balance,
        "performance_history": learner.state.performance_history,
    }
