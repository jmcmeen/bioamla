"""
Unit tests for bioamla.active_learning module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bioamla.core.active_learning import (
    ActiveLearner,
    ActiveLearningState,
    AnnotationQueue,
    AnnotationRecord,
    BalancedSampler,
    CallbackOracle,
    DiversitySampler,
    HybridSampler,
    QueryByCommittee,
    RandomSampler,
    ActiveLearningSample,
    SimulatedOracle,
    UncertaintySampler,
    compute_sample_uncertainty,
    create_samples_from_predictions,
    export_annotations,
    summarize_annotation_session,
)


# =============================================================================
# Test Data Structures
# =============================================================================

class TestSample:
    """Tests for Sample dataclass."""

    def test_basic_sample(self):
        """Test creating a basic sample."""
        sample = ActiveLearningSample(
            id="test_001",
            filepath="/path/to/audio.wav",
            start_time=0.0,
            end_time=5.0,
        )
        assert sample.id == "test_001"
        assert sample.filepath == "/path/to/audio.wav"
        assert sample.start_time == 0.0
        assert sample.end_time == 5.0
        assert sample.label is None

    def test_sample_with_predictions(self):
        """Test sample with model predictions."""
        probs = np.array([0.1, 0.3, 0.6])
        sample = ActiveLearningSample(
            id="test_002",
            filepath="/path/to/audio.wav",
            predicted_label="class_c",
            confidence=0.6,
            probabilities=probs,
        )
        assert sample.predicted_label == "class_c"
        assert sample.confidence == 0.6
        assert np.array_equal(sample.probabilities, probs)

    def test_sample_hash_and_eq(self):
        """Test sample hashing and equality."""
        sample1 = ActiveLearningSample(id="test_001", filepath="/path/1.wav")
        sample2 = ActiveLearningSample(id="test_001", filepath="/path/2.wav")
        sample3 = ActiveLearningSample(id="test_002", filepath="/path/1.wav")

        assert sample1 == sample2  # Same ID
        assert sample1 != sample3  # Different ID
        assert hash(sample1) == hash(sample2)

    def test_sample_to_dict(self):
        """Test converting sample to dictionary."""
        sample = ActiveLearningSample(
            id="test_001",
            filepath="/path/to/audio.wav",
            start_time=1.0,
            end_time=2.0,
            label="bird",
            metadata={"source": "xeno-canto"},
        )
        d = sample.to_dict()

        assert d["id"] == "test_001"
        assert d["filepath"] == "/path/to/audio.wav"
        assert d["start_time"] == 1.0
        assert d["end_time"] == 2.0
        assert d["label"] == "bird"
        assert d["metadata"]["source"] == "xeno-canto"

    def test_sample_from_dict(self):
        """Test creating sample from dictionary."""
        data = {
            "id": "test_001",
            "filepath": "/path/to/audio.wav",
            "start_time": 1.5,
            "end_time": 3.0,
            "label": "frog",
            "confidence": 0.9,
        }
        sample = ActiveLearningSample.from_dict(data)

        assert sample.id == "test_001"
        assert sample.filepath == "/path/to/audio.wav"
        assert sample.start_time == 1.5
        assert sample.end_time == 3.0
        assert sample.label == "frog"
        assert sample.confidence == 0.9


class TestAnnotationRecord:
    """Tests for AnnotationRecord dataclass."""

    def test_basic_record(self):
        """Test creating a basic annotation record."""
        record = AnnotationRecord(
            sample_id="test_001",
            label="bird",
            annotator="user1",
        )
        assert record.sample_id == "test_001"
        assert record.label == "bird"
        assert record.annotator == "user1"
        assert record.timestamp is not None

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = AnnotationRecord(
            sample_id="test_001",
            label="frog",
            annotator="user1",
            duration_seconds=5.5,
            notes="Clear call",
        )
        d = record.to_dict()

        assert d["sample_id"] == "test_001"
        assert d["label"] == "frog"
        assert d["annotator"] == "user1"
        assert d["duration_seconds"] == 5.5
        assert d["notes"] == "Clear call"


class TestActiveLearningState:
    """Tests for ActiveLearningState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = ActiveLearningState()
        assert state.iteration == 0
        assert state.total_labeled == 0
        assert state.total_unlabeled == 0
        assert state.labels_per_class == {}

    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = ActiveLearningState(
            iteration=5,
            total_labeled=100,
            total_unlabeled=500,
            labels_per_class={"bird": 50, "frog": 50},
        )
        d = state.to_dict()

        assert d["iteration"] == 5
        assert d["total_labeled"] == 100
        assert d["total_unlabeled"] == 500
        assert d["labels_per_class"]["bird"] == 50

    def test_state_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            "iteration": 3,
            "total_labeled": 30,
            "total_unlabeled": 270,
            "labels_per_class": {"a": 10, "b": 20},
            "query_history": [["s1", "s2"], ["s3"]],
        }
        state = ActiveLearningState.from_dict(data)

        assert state.iteration == 3
        assert state.total_labeled == 30
        assert len(state.query_history) == 2


# =============================================================================
# Test Sampling Strategies
# =============================================================================

class TestUncertaintySampler:
    """Tests for UncertaintySampler."""

    def _create_samples_with_probs(self) -> list:
        """Create test samples with probability distributions."""
        samples = [
            ActiveLearningSample(id="high_conf", filepath="a.wav",
                  probabilities=np.array([0.9, 0.05, 0.05])),  # High confidence
            ActiveLearningSample(id="low_conf", filepath="b.wav",
                  probabilities=np.array([0.4, 0.35, 0.25])),  # Low confidence
            ActiveLearningSample(id="uniform", filepath="c.wav",
                  probabilities=np.array([0.33, 0.34, 0.33])),  # Most uncertain
        ]
        return samples

    def test_least_confidence(self):
        """Test least confidence sampling."""
        sampler = UncertaintySampler(strategy="least_confidence")
        samples = self._create_samples_with_probs()

        scores = sampler.score(samples)

        # Uniform should have highest uncertainty (1 - 0.34 = 0.66)
        # High confidence should have lowest (1 - 0.9 = 0.1)
        assert scores[0] < scores[1]  # high_conf < low_conf
        assert scores[0] < scores[2]  # high_conf < uniform

    def test_margin_sampling(self):
        """Test margin sampling."""
        sampler = UncertaintySampler(strategy="margin")
        samples = self._create_samples_with_probs()

        scores = sampler.score(samples)

        # High confidence has large margin, low uncertainty
        assert scores[0] < scores[1]
        assert scores[0] < scores[2]

    def test_entropy_sampling(self):
        """Test entropy sampling."""
        sampler = UncertaintySampler(strategy="entropy")
        samples = self._create_samples_with_probs()

        scores = sampler.score(samples)

        # Uniform distribution has highest entropy
        assert scores[2] > scores[0]

    def test_select_samples(self):
        """Test selecting most uncertain samples."""
        sampler = UncertaintySampler(strategy="entropy")
        samples = self._create_samples_with_probs()

        selected = sampler.select(samples, n_samples=2)

        assert len(selected) == 2
        # Should not include the high confidence sample
        assert "high_conf" not in [s.id for s in selected]

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            UncertaintySampler(strategy="invalid")

    def test_confidence_fallback(self):
        """Test using confidence when probabilities unavailable."""
        sampler = UncertaintySampler(strategy="entropy")
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav", confidence=0.9),
            ActiveLearningSample(id="s2", filepath="b.wav", confidence=0.5),
            ActiveLearningSample(id="s3", filepath="c.wav", confidence=0.3),
        ]

        scores = sampler.score(samples)

        # Lower confidence = higher uncertainty
        assert scores[2] > scores[1] > scores[0]


class TestDiversitySampler:
    """Tests for DiversitySampler."""

    def _create_samples_with_features(self) -> list:
        """Create test samples with features."""
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav",
                  features=np.array([0.0, 0.0])),  # Origin
            ActiveLearningSample(id="s2", filepath="b.wav",
                  features=np.array([1.0, 0.0])),  # Far from origin
            ActiveLearningSample(id="s3", filepath="c.wav",
                  features=np.array([0.0, 1.0])),  # Far from origin
            ActiveLearningSample(id="s4", filepath="d.wav",
                  features=np.array([5.0, 5.0])),  # Farthest from origin
        ]
        return samples

    def test_greedy_selection(self):
        """Test greedy farthest-first selection."""
        sampler = DiversitySampler(method="greedy")
        samples = self._create_samples_with_features()

        selected = sampler.select(samples, n_samples=2)

        assert len(selected) == 2
        # Should select diverse points

    def test_score_based_on_centroid_distance(self):
        """Test that scores are based on distance to centroid."""
        sampler = DiversitySampler(method="greedy")
        samples = self._create_samples_with_features()

        scores = sampler.score(samples)

        # Sample furthest from centroid should have highest score
        assert scores[3] > scores[0]  # (5,5) is furthest

    def test_no_features_random_fallback(self):
        """Test random fallback when no features available."""
        sampler = DiversitySampler(method="greedy")
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
        ]

        scores = sampler.score(samples)
        assert len(scores) == 2

    def test_uses_probabilities_as_features(self):
        """Test using probabilities as features when no explicit features."""
        sampler = DiversitySampler(method="greedy")
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav",
                  probabilities=np.array([0.9, 0.1])),
            ActiveLearningSample(id="s2", filepath="b.wav",
                  probabilities=np.array([0.1, 0.9])),
        ]

        selected = sampler.select(samples, n_samples=2)
        assert len(selected) == 2


class TestHybridSampler:
    """Tests for HybridSampler."""

    def test_hybrid_combines_strategies(self):
        """Test that hybrid sampler combines uncertainty and diversity."""
        sampler = HybridSampler(
            uncertainty_strategy="entropy",
            diversity_method="greedy",
            uncertainty_ratio=0.5,
        )

        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav",
                  probabilities=np.array([0.9, 0.05, 0.05]),
                  features=np.array([0.0, 0.0])),
            ActiveLearningSample(id="s2", filepath="b.wav",
                  probabilities=np.array([0.33, 0.34, 0.33]),
                  features=np.array([5.0, 5.0])),
        ]

        scores = sampler.score(samples)
        assert len(scores) == 2

    def test_two_stage_selection(self):
        """Test two-stage selection process."""
        sampler = HybridSampler(uncertainty_ratio=0.5)

        samples = [
            ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav",
                  probabilities=np.random.dirichlet([1, 1, 1]),
                  features=np.random.randn(5))
            for i in range(20)
        ]

        selected = sampler.select(samples, n_samples=5)
        assert len(selected) == 5


class TestRandomSampler:
    """Tests for RandomSampler."""

    def test_random_selection(self):
        """Test random sample selection."""
        sampler = RandomSampler(seed=42)
        samples = [ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav") for i in range(10)]

        selected = sampler.select(samples, n_samples=3)
        assert len(selected) == 3

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results."""
        samples = [ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav") for i in range(10)]

        sampler1 = RandomSampler(seed=42)
        sampler2 = RandomSampler(seed=42)

        scores1 = sampler1.score(samples)
        scores2 = sampler2.score(samples)

        np.testing.assert_array_equal(scores1, scores2)


class TestQueryByCommittee:
    """Tests for QueryByCommittee."""

    def test_vote_entropy(self):
        """Test vote entropy disagreement measure."""
        sampler = QueryByCommittee(disagreement_measure="vote_entropy")

        # Add predictions from 3 committee members
        # Sample where all agree
        # Sample where they disagree
        sampler.add_committee_predictions({
            "agree": np.array([0.9, 0.05, 0.05]),
            "disagree": np.array([0.5, 0.3, 0.2]),
        })
        sampler.add_committee_predictions({
            "agree": np.array([0.85, 0.1, 0.05]),
            "disagree": np.array([0.2, 0.5, 0.3]),
        })
        sampler.add_committee_predictions({
            "agree": np.array([0.88, 0.07, 0.05]),
            "disagree": np.array([0.3, 0.2, 0.5]),
        })

        samples = [
            ActiveLearningSample(id="agree", filepath="a.wav"),
            ActiveLearningSample(id="disagree", filepath="b.wav"),
        ]

        scores = sampler.score(samples)

        # Disagreement sample should have higher score
        assert scores[1] > scores[0]

    def test_clear_committee(self):
        """Test clearing committee predictions."""
        sampler = QueryByCommittee()
        sampler.add_committee_predictions({"s1": np.array([0.5, 0.5])})

        assert len(sampler.committee_predictions) == 1

        sampler.clear_committee()
        assert len(sampler.committee_predictions) == 0


class TestBalancedSampler:
    """Tests for BalancedSampler."""

    def test_prioritizes_underrepresented_classes(self):
        """Test that underrepresented classes get higher scores."""
        sampler = BalancedSampler(class_counts={"bird": 100, "frog": 10})

        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav", predicted_label="bird"),
            ActiveLearningSample(id="s2", filepath="b.wav", predicted_label="frog"),
        ]

        scores = sampler.score(samples)

        # Frog (underrepresented) should have higher score
        assert scores[1] > scores[0]

    def test_update_counts(self):
        """Test updating class counts."""
        sampler = BalancedSampler()
        sampler.update_counts({"a": 5, "b": 10})

        assert sampler.class_counts["a"] == 5
        assert sampler.class_counts["b"] == 10


# =============================================================================
# Test Active Learner
# =============================================================================

class TestActiveLearner:
    """Tests for ActiveLearner class."""

    def test_add_unlabeled(self):
        """Test adding samples to unlabeled pool."""
        learner = ActiveLearner()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
        ]

        learner.add_unlabeled(samples)

        assert learner.state.total_unlabeled == 2
        assert "s1" in learner.unlabeled_pool
        assert "s2" in learner.unlabeled_pool

    def test_add_labeled(self):
        """Test adding pre-labeled samples."""
        learner = ActiveLearner()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav", label="bird"),
            ActiveLearningSample(id="s2", filepath="b.wav", label="frog"),
        ]

        learner.add_labeled(samples)

        assert learner.state.total_labeled == 2
        assert learner.state.labels_per_class["bird"] == 1
        assert learner.state.labels_per_class["frog"] == 1

    def test_query_samples(self):
        """Test querying samples for annotation."""
        learner = ActiveLearner(sampler=RandomSampler(seed=42))

        samples = [
            ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav",
                  confidence=np.random.rand())
            for i in range(20)
        ]
        learner.add_unlabeled(samples)

        queried = learner.query(n_samples=5, update_predictions=False)

        assert len(queried) == 5
        assert learner.state.iteration == 1
        assert len(learner.state.query_history) == 1

    def test_teach(self):
        """Test teaching (recording annotation)."""
        learner = ActiveLearner()
        sample = ActiveLearningSample(id="s1", filepath="a.wav")
        learner.add_unlabeled([sample])

        learner.teach(sample, label="bird", annotator="user1")

        assert "s1" in learner.labeled_pool
        assert "s1" not in learner.unlabeled_pool
        assert learner.state.total_labeled == 1
        assert learner.state.labels_per_class["bird"] == 1
        assert len(learner.annotation_history) == 1

    def test_teach_batch(self):
        """Test teaching multiple samples."""
        learner = ActiveLearner()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
        ]
        learner.add_unlabeled(samples)

        learner.teach_batch([
            (samples[0], "bird"),
            (samples[1], "frog"),
        ])

        assert learner.state.total_labeled == 2

    def test_get_statistics(self):
        """Test getting statistics."""
        learner = ActiveLearner()
        learner.add_labeled([ActiveLearningSample(id="s1", filepath="a.wav", label="bird")])
        learner.add_unlabeled([ActiveLearningSample(id="s2", filepath="b.wav")])

        stats = learner.get_statistics()

        assert stats["total_labeled"] == 1
        assert stats["total_unlabeled"] == 1
        assert "bird" in stats["labels_per_class"]

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        learner = ActiveLearner()
        learner.add_labeled([ActiveLearningSample(id="s1", filepath="a.wav", label="bird")])
        learner.add_unlabeled([ActiveLearningSample(id="s2", filepath="b.wav")])
        learner.state.iteration = 5

        state_file = tmp_path / "al_state.json"
        learner.save_state(str(state_file))

        # Load into new learner
        loaded = ActiveLearner.load_state(str(state_file))

        assert loaded.state.iteration == 5
        assert loaded.state.total_labeled == 1
        assert "s1" in loaded.labeled_pool
        assert "s2" in loaded.unlabeled_pool

    def test_record_performance(self):
        """Test recording performance metrics."""
        learner = ActiveLearner()
        learner.record_performance({"accuracy": 0.85, "f1": 0.82})

        assert len(learner.state.performance_history) == 1
        assert learner.state.performance_history[0]["accuracy"] == 0.85


# =============================================================================
# Test Annotation Queue
# =============================================================================

class TestAnnotationQueue:
    """Tests for AnnotationQueue class."""

    def test_add_samples(self):
        """Test adding samples to queue."""
        queue = AnnotationQueue()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
        ]

        queue.add(samples)

        assert len(queue) == 2

    def test_add_priority(self):
        """Test adding samples with priority."""
        queue = AnnotationQueue()
        queue.add([ActiveLearningSample(id="s1", filepath="a.wav")])
        queue.add([ActiveLearningSample(id="s2", filepath="b.wav")], priority=True)

        current = queue.current()
        assert current.id == "s2"  # Priority sample should be first

    def test_navigation(self):
        """Test queue navigation."""
        queue = AnnotationQueue()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
            ActiveLearningSample(id="s3", filepath="c.wav"),
        ]
        queue.add(samples)

        assert queue.current().id == "s1"
        assert queue.next().id == "s2"
        assert queue.next().id == "s3"
        assert queue.previous().id == "s2"

    def test_mark_completed(self):
        """Test marking samples as completed."""
        queue = AnnotationQueue()
        queue.add([ActiveLearningSample(id="s1", filepath="a.wav")])

        queue.mark_completed("s1")

        assert "s1" in queue.completed

    def test_get_progress(self):
        """Test getting queue progress."""
        queue = AnnotationQueue()
        samples = [ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav") for i in range(10)]
        queue.add(samples)

        queue.mark_completed("s0")
        queue.mark_completed("s1")
        queue.mark_skipped("s2")

        progress = queue.get_progress()

        assert progress["total"] == 10
        assert progress["completed"] == 2
        assert progress["skipped"] == 1
        assert progress["remaining"] == 7

    def test_export_to_csv(self, tmp_path):
        """Test exporting queue to CSV."""
        queue = AnnotationQueue()
        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav", predicted_label="bird", confidence=0.9),
            ActiveLearningSample(id="s2", filepath="b.wav", predicted_label="frog", confidence=0.7),
        ]
        queue.add(samples)
        queue.mark_completed("s1")

        csv_path = tmp_path / "queue.csv"
        queue.export_to_csv(str(csv_path))

        assert csv_path.exists()

        # Verify content
        with open(csv_path, "r") as f:
            content = f.read()
            assert "s1" in content
            assert "completed" in content


# =============================================================================
# Test Oracles
# =============================================================================

class TestSimulatedOracle:
    """Tests for SimulatedOracle class."""

    def test_annotate_returns_ground_truth(self):
        """Test that oracle returns ground truth labels."""
        ground_truth = {"s1": "bird", "s2": "frog"}
        oracle = SimulatedOracle(ground_truth=ground_truth)

        sample1 = ActiveLearningSample(id="s1", filepath="a.wav")
        sample2 = ActiveLearningSample(id="s2", filepath="b.wav")

        assert oracle.annotate(sample1) == "bird"
        assert oracle.annotate(sample2) == "frog"

    def test_missing_ground_truth_raises(self):
        """Test that missing ground truth raises error."""
        oracle = SimulatedOracle(ground_truth={"s1": "bird"})
        sample = ActiveLearningSample(id="unknown", filepath="x.wav")

        with pytest.raises(ValueError):
            oracle.annotate(sample)

    def test_noisy_oracle(self):
        """Test oracle with noise."""
        ground_truth = {"s1": "bird"}
        oracle = SimulatedOracle(
            ground_truth=ground_truth,
            noise_rate=1.0,  # Always return wrong label
            labels=["bird", "frog"]
        )

        sample = ActiveLearningSample(id="s1", filepath="a.wav")
        label = oracle.annotate(sample)

        # With 100% noise rate, should not return true label
        assert label == "frog"

    def test_annotate_batch(self):
        """Test batch annotation."""
        ground_truth = {"s1": "bird", "s2": "frog", "s3": "bird"}
        oracle = SimulatedOracle(ground_truth=ground_truth)

        samples = [
            ActiveLearningSample(id="s1", filepath="a.wav"),
            ActiveLearningSample(id="s2", filepath="b.wav"),
            ActiveLearningSample(id="s3", filepath="c.wav"),
        ]

        labels = oracle.annotate_batch(samples)

        assert labels == ["bird", "frog", "bird"]


class TestCallbackOracle:
    """Tests for CallbackOracle class."""

    def test_callback_oracle(self):
        """Test oracle with callback function."""
        def my_annotator(sample: ActiveLearningSample) -> str:
            return f"label_for_{sample.id}"

        oracle = CallbackOracle(callback=my_annotator)
        sample = ActiveLearningSample(id="test123", filepath="a.wav")

        label = oracle.annotate(sample)
        assert label == "label_for_test123"


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_sample_uncertainty_entropy(self):
        """Test computing entropy uncertainty."""
        probs = np.array([0.5, 0.5])
        uncertainty = compute_sample_uncertainty(probs, strategy="entropy")

        # Maximum entropy for uniform distribution over 2 classes is ln(2)
        expected = -2 * (0.5 * np.log(0.5))
        np.testing.assert_almost_equal(uncertainty, expected)

    def test_compute_sample_uncertainty_least_confidence(self):
        """Test computing least confidence uncertainty."""
        probs = np.array([0.7, 0.3])
        uncertainty = compute_sample_uncertainty(probs, strategy="least_confidence")

        np.testing.assert_almost_equal(uncertainty, 0.3)  # 1 - 0.7

    def test_compute_sample_uncertainty_margin(self):
        """Test computing margin uncertainty."""
        probs = np.array([0.7, 0.2, 0.1])
        uncertainty = compute_sample_uncertainty(probs, strategy="margin")

        # Margin = 1 - (0.7 - 0.2) = 0.5
        assert uncertainty == 0.5

    def test_create_samples_from_predictions(self, tmp_path):
        """Test creating samples from predictions CSV."""
        csv_path = tmp_path / "predictions.csv"
        with open(csv_path, "w") as f:
            f.write("filepath,start_time,end_time,predicted_label,confidence\n")
            f.write("audio1.wav,0.0,5.0,bird,0.9\n")
            f.write("audio2.wav,5.0,10.0,frog,0.8\n")

        samples = create_samples_from_predictions(str(csv_path))

        assert len(samples) == 2
        assert samples[0].predicted_label == "bird"
        assert samples[0].confidence == 0.9
        assert samples[1].predicted_label == "frog"

    def test_export_annotations_csv(self, tmp_path):
        """Test exporting annotations to CSV."""
        learner = ActiveLearner()
        learner.add_labeled([
            ActiveLearningSample(id="s1", filepath="a.wav", start_time=0, end_time=5, label="bird"),
            ActiveLearningSample(id="s2", filepath="b.wav", start_time=0, end_time=5, label="frog"),
        ])

        csv_path = tmp_path / "annotations.csv"
        export_annotations(learner, str(csv_path), format="csv")

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            content = f.read()
            assert "bird" in content
            assert "frog" in content

    def test_summarize_annotation_session(self):
        """Test summarizing annotation session."""
        learner = ActiveLearner()
        learner.add_labeled([
            ActiveLearningSample(id="s1", filepath="a.wav", label="bird"),
            ActiveLearningSample(id="s2", filepath="b.wav", label="bird"),
            ActiveLearningSample(id="s3", filepath="c.wav", label="frog"),
        ])
        learner.add_unlabeled([
            ActiveLearningSample(id="s4", filepath="d.wav"),
        ])

        # Add some annotation records with timing
        learner.annotation_history = [
            AnnotationRecord(sample_id="s1", label="bird", duration_seconds=10.0),
            AnnotationRecord(sample_id="s2", label="bird", duration_seconds=8.0),
            AnnotationRecord(sample_id="s3", label="frog", duration_seconds=12.0),
        ]

        summary = summarize_annotation_session(learner)

        assert summary["total_labeled"] == 3
        assert summary["total_unlabeled"] == 1
        assert summary["total_annotation_time_seconds"] == 30.0
        assert summary["class_balance_ratio"] == 0.5  # 1 frog / 2 birds


# =============================================================================
# Integration Tests
# =============================================================================

class TestActiveLearningLoop:
    """Integration tests for active learning loop."""

    def test_full_active_learning_loop(self):
        """Test complete active learning loop."""
        # Setup
        ground_truth = {f"s{i}": "bird" if i % 2 == 0 else "frog" for i in range(100)}
        oracle = SimulatedOracle(ground_truth=ground_truth)
        sampler = UncertaintySampler(strategy="entropy")
        learner = ActiveLearner(sampler=sampler)

        # Create unlabeled pool with synthetic predictions
        samples = []
        for i in range(100):
            probs = np.random.dirichlet([1, 1])  # Random probs for bird/frog
            samples.append(ActiveLearningSample(
                id=f"s{i}",
                filepath=f"{i}.wav",
                probabilities=probs,
                confidence=np.max(probs),
                predicted_label="bird" if np.argmax(probs) == 0 else "frog",
            ))

        learner.add_unlabeled(samples)

        # Run 3 iterations
        for iteration in range(3):
            # Query samples
            queried = learner.query(n_samples=10, update_predictions=False)
            assert len(queried) == 10

            # Annotate using oracle
            for sample in queried:
                label = oracle.annotate(sample)
                learner.teach(sample, label, annotator="oracle")

        # Verify results
        assert learner.state.iteration == 3
        assert learner.state.total_labeled == 30
        assert learner.state.total_unlabeled == 70
        assert len(learner.annotation_history) == 30

    def test_active_learning_with_hybrid_sampler(self):
        """Test active learning with hybrid sampling."""
        ground_truth = {f"s{i}": "bird" if i < 50 else "frog" for i in range(100)}
        oracle = SimulatedOracle(ground_truth=ground_truth)
        sampler = HybridSampler(uncertainty_ratio=0.5)
        learner = ActiveLearner(sampler=sampler)

        # Create samples with features and probabilities
        samples = []
        for i in range(100):
            probs = np.random.dirichlet([1, 1])
            features = np.random.randn(10)
            samples.append(ActiveLearningSample(
                id=f"s{i}",
                filepath=f"{i}.wav",
                probabilities=probs,
                features=features,
            ))

        learner.add_unlabeled(samples)

        # Query and annotate
        queried = learner.query(n_samples=10, update_predictions=False)
        for sample in queried:
            label = oracle.annotate(sample)
            learner.teach(sample, label)

        assert learner.state.total_labeled == 10

    def test_save_load_resume_learning(self, tmp_path):
        """Test saving, loading, and resuming active learning."""
        # Initial session
        ground_truth = {f"s{i}": f"class_{i % 3}" for i in range(50)}
        oracle = SimulatedOracle(ground_truth=ground_truth)

        learner1 = ActiveLearner(sampler=RandomSampler(seed=42))
        samples = [ActiveLearningSample(id=f"s{i}", filepath=f"{i}.wav") for i in range(50)]
        learner1.add_unlabeled(samples)

        # First batch
        queried = learner1.query(n_samples=5, update_predictions=False)
        for sample in queried:
            learner1.teach(sample, oracle.annotate(sample))

        # Save state
        state_file = tmp_path / "state.json"
        learner1.save_state(str(state_file))

        # Load and resume
        learner2 = ActiveLearner.load_state(
            str(state_file),
            sampler=RandomSampler(seed=42)
        )

        assert learner2.state.total_labeled == 5
        assert learner2.state.total_unlabeled == 45
        assert learner2.state.iteration == 1

        # Continue learning
        queried = learner2.query(n_samples=5, update_predictions=False)
        for sample in queried:
            learner2.teach(sample, oracle.annotate(sample))

        assert learner2.state.total_labeled == 10
        assert learner2.state.iteration == 2
