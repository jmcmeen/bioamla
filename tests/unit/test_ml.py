"""
Unit tests for bioamla.ml module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestCNNClassifier:
    """Tests for CNNClassifier."""

    def test_initialization_default_params(self):
        """Test CNN classifier with default parameters."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(n_classes=10)
        assert model.n_classes == 10
        assert model.n_mels == 128
        assert model.n_time == 256
        assert len(model.conv_blocks) == 4

    def test_initialization_custom_params(self):
        """Test CNN classifier with custom parameters."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(
            n_classes=5,
            n_mels=64,
            n_time=128,
            channels=[16, 32, 64],
            dropout=0.5,
        )
        assert model.n_classes == 5
        assert model.n_mels == 64
        assert len(model.conv_blocks) == 3

    def test_forward_3d_input(self):
        """Test forward pass with 3D input."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(n_classes=10)
        x = torch.randn(4, 128, 256)  # batch, n_mels, n_time
        output = model(x)
        assert output.shape == (4, 10)

    def test_forward_4d_input(self):
        """Test forward pass with 4D input."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(n_classes=10)
        x = torch.randn(4, 1, 128, 256)  # batch, channel, n_mels, n_time
        output = model(x)
        assert output.shape == (4, 10)

    def test_predict(self):
        """Test predict method."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(n_classes=5)
        x = torch.randn(2, 128, 256)
        predictions = model.predict(x)
        assert predictions.shape == (2,)
        assert all(0 <= p < 5 for p in predictions)

    def test_predict_proba(self):
        """Test predict_proba method."""
        from bioamla.ml import CNNClassifier

        model = CNNClassifier(n_classes=5)
        x = torch.randn(2, 128, 256)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


class TestCRNNClassifier:
    """Tests for CRNNClassifier."""

    def test_initialization_gru(self):
        """Test CRNN with GRU."""
        from bioamla.ml import CRNNClassifier

        model = CRNNClassifier(n_classes=10, rnn_type="gru")
        assert model.n_classes == 10
        assert isinstance(model.rnn, nn.GRU)

    def test_initialization_lstm(self):
        """Test CRNN with LSTM."""
        from bioamla.ml import CRNNClassifier

        model = CRNNClassifier(n_classes=10, rnn_type="lstm")
        assert isinstance(model.rnn, nn.LSTM)

    def test_forward(self):
        """Test forward pass."""
        from bioamla.ml import CRNNClassifier

        model = CRNNClassifier(n_classes=10)
        x = torch.randn(4, 128, 256)
        output = model(x)
        assert output.shape == (4, 10)

    def test_bidirectional(self):
        """Test bidirectional RNN."""
        from bioamla.ml import CRNNClassifier

        model_bi = CRNNClassifier(n_classes=10, bidirectional=True, rnn_hidden=64)
        model_uni = CRNNClassifier(n_classes=10, bidirectional=False, rnn_hidden=64)

        # Bidirectional should have more parameters
        bi_params = sum(p.numel() for p in model_bi.parameters())
        uni_params = sum(p.numel() for p in model_uni.parameters())
        assert bi_params > uni_params


class TestAttentionClassifier:
    """Tests for AttentionClassifier."""

    def test_initialization(self):
        """Test attention classifier initialization."""
        from bioamla.ml import AttentionClassifier

        model = AttentionClassifier(n_classes=10)
        assert model.n_classes == 10
        assert model.attention is not None

    def test_forward(self):
        """Test forward pass."""
        from bioamla.ml import AttentionClassifier

        model = AttentionClassifier(n_classes=10)
        x = torch.randn(4, 128, 256)
        output = model(x)
        assert output.shape == (4, 10)

    def test_attention_heads(self):
        """Test different number of attention heads."""
        from bioamla.ml import AttentionClassifier

        model = AttentionClassifier(n_classes=10, attention_heads=8, hidden_dim=256)
        assert model.attention.num_heads == 8


class TestLabelHierarchy:
    """Tests for LabelHierarchy."""

    def test_add_node_root(self):
        """Test adding root node."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        assert "Order1" in hierarchy.nodes
        assert "Order1" in hierarchy.root_nodes
        assert hierarchy.nodes["Order1"].level == 0

    def test_add_node_with_parent(self):
        """Test adding node with parent."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")
        assert hierarchy.nodes["Family1"].parent == "Order1"
        assert hierarchy.nodes["Family1"].level == 1
        assert "Family1" in hierarchy.nodes["Order1"].children

    def test_get_ancestors(self):
        """Test getting ancestors."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")
        hierarchy.add_node("Genus1", parent="Family1")
        hierarchy.add_node("Species1", parent="Genus1")

        ancestors = hierarchy.get_ancestors("Species1")
        assert ancestors == ["Genus1", "Family1", "Order1"]

    def test_get_descendants(self):
        """Test getting descendants."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")
        hierarchy.add_node("Genus1", parent="Family1")

        descendants = hierarchy.get_descendants("Order1")
        assert "Family1" in descendants
        assert "Genus1" in descendants

    def test_get_path(self):
        """Test getting path from root to node."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")
        hierarchy.add_node("Species1", parent="Family1")

        path = hierarchy.get_path("Species1")
        assert path == ["Order1", "Family1", "Species1"]

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")

        data = hierarchy.to_dict()
        restored = LabelHierarchy.from_dict(data)

        assert "Order1" in restored.nodes
        assert "Family1" in restored.nodes
        assert restored.nodes["Family1"].parent == "Order1"

    def test_save_and_load(self, tmp_path):
        """Test saving and loading hierarchy."""
        from bioamla.ml import LabelHierarchy

        hierarchy = LabelHierarchy()
        hierarchy.add_node("Order1")
        hierarchy.add_node("Family1", parent="Order1")

        filepath = tmp_path / "hierarchy.json"
        hierarchy.save(str(filepath))

        loaded = LabelHierarchy.load(str(filepath))
        assert "Order1" in loaded.nodes
        assert "Family1" in loaded.nodes

    def test_from_taxonomy_csv(self, tmp_path):
        """Test loading from taxonomy CSV."""
        from bioamla.ml import LabelHierarchy

        csv_path = tmp_path / "taxonomy.csv"
        csv_path.write_text(
            "order,family,genus,species\n"
            "Passeriformes,Paridae,Parus,major\n"
            "Passeriformes,Paridae,Cyanistes,caeruleus\n"
        )

        hierarchy = LabelHierarchy.from_taxonomy_csv(str(csv_path))
        assert "Passeriformes" in hierarchy.nodes
        assert "Paridae" in hierarchy.nodes
        assert hierarchy.nodes["Paridae"].parent == "Passeriformes"


class TestHierarchicalClassifier:
    """Tests for HierarchicalClassifier."""

    @pytest.fixture
    def hierarchy(self):
        """Create test hierarchy."""
        from bioamla.ml import LabelHierarchy

        h = LabelHierarchy()
        h.add_node("Order1", level=0)
        h.add_node("Order2", level=0)
        h.add_node("Family1", parent="Order1", level=1)
        h.add_node("Family2", parent="Order2", level=1)
        h.add_node("Species1", parent="Family1", level=2)
        h.add_node("Species2", parent="Family2", level=2)
        return h

    def test_initialization(self, hierarchy):
        """Test hierarchical classifier initialization."""
        from bioamla.ml import CNNClassifier, HierarchicalClassifier

        backbone = CNNClassifier(n_classes=10)
        model = HierarchicalClassifier(
            backbone=backbone,
            hierarchy=hierarchy,
            feature_dim=256,
        )

        assert len(model.level_classifiers) == 3  # 3 levels
        assert 0 in model.level_labels
        assert 1 in model.level_labels
        assert 2 in model.level_labels

    def test_forward(self, hierarchy):
        """Test forward pass."""
        from bioamla.ml import CNNClassifier, HierarchicalClassifier

        backbone = CNNClassifier(n_classes=10)
        model = HierarchicalClassifier(
            backbone=backbone,
            hierarchy=hierarchy,
            feature_dim=256,
        )

        x = torch.randn(2, 128, 256)
        outputs = model(x)

        assert 0 in outputs
        assert 1 in outputs
        assert 2 in outputs


class TestMultiLabelClassifier:
    """Tests for MultiLabelClassifier."""

    def test_initialization(self):
        """Test multi-label classifier initialization."""
        from bioamla.ml import CNNClassifier, MultiLabelClassifier

        backbone = CNNClassifier(n_classes=10)
        model = MultiLabelClassifier(
            backbone=backbone,
            n_labels=20,
            feature_dim=256,
        )

        assert model.n_labels == 20
        assert model.threshold == 0.5

    def test_forward(self):
        """Test forward pass."""
        from bioamla.ml import CNNClassifier, MultiLabelClassifier

        backbone = CNNClassifier(n_classes=10)
        model = MultiLabelClassifier(
            backbone=backbone,
            n_labels=20,
            feature_dim=256,
        )

        x = torch.randn(2, 128, 256)
        output = model(x)
        assert output.shape == (2, 20)

    def test_predict(self):
        """Test binary predictions."""
        from bioamla.ml import CNNClassifier, MultiLabelClassifier

        backbone = CNNClassifier(n_classes=10)
        model = MultiLabelClassifier(
            backbone=backbone,
            n_labels=5,
            feature_dim=256,
        )

        x = torch.randn(2, 128, 256)
        predictions = model.predict(x)
        assert predictions.shape == (2, 5)
        assert all(p in [0.0, 1.0] for p in predictions.flatten())

    def test_predict_proba(self):
        """Test probability predictions."""
        from bioamla.ml import CNNClassifier, MultiLabelClassifier

        backbone = CNNClassifier(n_classes=10)
        model = MultiLabelClassifier(
            backbone=backbone,
            n_labels=5,
            feature_dim=256,
        )

        x = torch.randn(2, 128, 256)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 5)
        assert all(0 <= p <= 1 for p in probs.flatten())


class TestEnsembleStrategies:
    """Tests for ensemble combination strategies."""

    def test_averaging_strategy(self):
        """Test averaging strategy."""
        from bioamla.ml import AveragingStrategy

        strategy = AveragingStrategy()
        predictions = [
            torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
            torch.tensor([[0.6, 0.4], [0.5, 0.5]]),
        ]

        combined = strategy.combine(predictions)
        assert combined.shape == (2, 2)
        assert torch.allclose(combined[0], torch.tensor([0.7, 0.3]), atol=1e-5)

    def test_averaging_strategy_with_weights(self):
        """Test weighted averaging."""
        from bioamla.ml import AveragingStrategy

        strategy = AveragingStrategy()
        predictions = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.0, 1.0]]),
        ]

        combined = strategy.combine(predictions, weights=[0.75, 0.25])
        assert torch.allclose(combined, torch.tensor([[0.75, 0.25]]))

    def test_voting_strategy_soft(self):
        """Test soft voting strategy."""
        from bioamla.ml import VotingStrategy

        strategy = VotingStrategy(soft=True)
        predictions = [
            torch.tensor([[0.8, 0.2]]),
            torch.tensor([[0.6, 0.4]]),
        ]

        combined = strategy.combine(predictions)
        assert combined.shape == (1, 2)

    def test_voting_strategy_hard(self):
        """Test hard voting strategy."""
        from bioamla.ml import VotingStrategy

        strategy = VotingStrategy(soft=False)
        predictions = [
            torch.tensor([[0.9, 0.1]]),  # Predicts class 0
            torch.tensor([[0.8, 0.2]]),  # Predicts class 0
            torch.tensor([[0.3, 0.7]]),  # Predicts class 1
        ]

        combined = strategy.combine(predictions)
        # Class 0 should have 2/3 votes
        assert combined[0, 0] > combined[0, 1]

    def test_max_strategy(self):
        """Test max probability strategy."""
        from bioamla.ml import MaxStrategy

        strategy = MaxStrategy()
        predictions = [
            torch.tensor([[0.5, 0.3]]),
            torch.tensor([[0.4, 0.9]]),
        ]

        combined = strategy.combine(predictions)
        assert combined[0, 0] == 0.5
        assert combined[0, 1] == 0.9

    def test_stacking_strategy_untrained(self):
        """Test stacking strategy falls back to averaging when untrained."""
        from bioamla.ml import StackingStrategy

        strategy = StackingStrategy(n_classes=3, n_models=2)
        predictions = [
            torch.tensor([[0.5, 0.3, 0.2]]),
            torch.tensor([[0.4, 0.4, 0.2]]),
        ]

        combined = strategy.combine(predictions)
        assert combined.shape == (1, 3)


class TestEnsemble:
    """Tests for Ensemble class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        from bioamla.ml import CNNClassifier, Ensemble

        models = [CNNClassifier(n_classes=5) for _ in range(3)]
        ensemble = Ensemble(models)

        assert len(ensemble.models) == 3
        assert ensemble.weights is None

    def test_predict(self):
        """Test ensemble predictions."""
        from bioamla.ml import CNNClassifier, Ensemble

        models = [CNNClassifier(n_classes=5) for _ in range(3)]
        ensemble = Ensemble(models)

        x = torch.randn(2, 128, 256)
        probs = ensemble.predict(x)

        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_predict_classes(self):
        """Test class predictions."""
        from bioamla.ml import CNNClassifier, Ensemble

        models = [CNNClassifier(n_classes=5) for _ in range(3)]
        ensemble = Ensemble(models)

        x = torch.randn(2, 128, 256)
        classes = ensemble.predict_classes(x)

        assert classes.shape == (2,)
        assert all(0 <= c < 5 for c in classes)

    def test_get_uncertainty(self):
        """Test uncertainty estimation."""
        from bioamla.ml import CNNClassifier, Ensemble

        models = [CNNClassifier(n_classes=5) for _ in range(3)]
        ensemble = Ensemble(models)

        x = torch.randn(2, 128, 256)
        uncertainty = ensemble.get_uncertainty(x)

        assert uncertainty.shape == (2,)
        assert all(u >= 0 for u in uncertainty)

    def test_save(self, tmp_path):
        """Test saving ensemble."""
        from bioamla.ml import CNNClassifier, Ensemble

        models = [CNNClassifier(n_classes=5) for _ in range(2)]
        ensemble = Ensemble(models, weights=[0.6, 0.4])

        save_path = ensemble.save(str(tmp_path / "ensemble"))

        assert Path(save_path).exists()
        assert (Path(save_path) / "model_0.pt").exists()
        assert (Path(save_path) / "model_1.pt").exists()
        assert (Path(save_path) / "config.json").exists()


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from bioamla.ml import TrainerConfig

        config = TrainerConfig()
        assert config.epochs == 50
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.patience == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        from bioamla.ml import TrainerConfig

        config = TrainerConfig(
            epochs=100,
            learning_rate=1e-4,
            batch_size=64,
        )
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.batch_size == 64


class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_no_stop_improving(self):
        """Test no early stop when improving."""
        from bioamla.ml import EarlyStopping

        early_stop = EarlyStopping(patience=3)
        assert not early_stop(0.5)
        assert not early_stop(0.6)
        assert not early_stop(0.7)

    def test_stop_no_improvement(self):
        """Test early stop after patience exceeded."""
        from bioamla.ml import EarlyStopping

        early_stop = EarlyStopping(patience=3)
        early_stop(0.7)
        early_stop(0.65)  # Worse
        early_stop(0.66)  # Still worse
        result = early_stop(0.67)  # 3rd worse

        assert result is True

    def test_reset_on_improvement(self):
        """Test counter reset on improvement."""
        from bioamla.ml import EarlyStopping

        early_stop = EarlyStopping(patience=3)
        early_stop(0.5)
        early_stop(0.4)  # Worse
        early_stop(0.4)  # Worse
        early_stop(0.6)  # Better - should reset
        assert not early_stop(0.55)  # Only 1 worse now


class TestTrainClassifier:
    """Tests for train_classifier function."""

    def test_training_runs(self, tmp_path):
        """Test that training runs without errors."""
        from bioamla.ml import CNNClassifier, TrainerConfig, train_classifier

        model = CNNClassifier(n_classes=3)

        # Create simple dataset
        X = torch.randn(16, 128, 256)
        y = torch.randint(0, 3, (16,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=4)

        config = TrainerConfig(
            epochs=2,
            output_dir=str(tmp_path / "output"),
        )

        result = train_classifier(
            model=model,
            train_loader=train_loader,
            config=config,
            device=torch.device("cpu"),
        )

        assert "history" in result
        assert len(result["history"]["train_loss"]) == 2

    def test_training_with_validation(self, tmp_path):
        """Test training with validation set."""
        from bioamla.ml import CNNClassifier, TrainerConfig, train_classifier

        model = CNNClassifier(n_classes=3)

        X = torch.randn(16, 128, 256)
        y = torch.randint(0, 3, (16,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        config = TrainerConfig(
            epochs=2,
            output_dir=str(tmp_path / "output"),
        )

        result = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )

        assert len(result["history"]["val_loss"]) == 2


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path
