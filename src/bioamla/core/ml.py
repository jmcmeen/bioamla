"""
Advanced Machine Learning Module
================================

This module provides advanced ML capabilities for bioacoustic classification:
- Custom classifier training from scratch (CNN, CRNN architectures)
- Multi-label hierarchical classification
- Model ensemble predictions with various strategies

Example:
    >>> from bioamla.ml import CNNClassifier, train_classifier
    >>> model = CNNClassifier(n_classes=10, n_mels=128)
    >>> train_classifier(model, train_loader, val_loader, epochs=50)
    >>>
    >>> from bioamla.ml import Ensemble
    >>> ensemble = Ensemble(models=[model1, model2, model3])
    >>> predictions = ensemble.predict(audio_batch)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bioamla.core.files import TextFile

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Classifier Architectures
# =============================================================================

class AudioClassifierBase(nn.Module, ABC):
    """Base class for audio classifiers."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class CNNClassifier(AudioClassifierBase):
    """
    Convolutional Neural Network for audio classification.

    Takes mel-spectrogram input and outputs class logits.

    Architecture:
        - 4 convolutional blocks with batch norm and max pooling
        - Global average pooling
        - Fully connected classifier head
    """

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 128,
        n_time: int = 256,
        channels: List[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize CNN classifier.

        Args:
            n_classes: Number of output classes
            n_mels: Number of mel frequency bins
            n_time: Number of time frames
            channels: List of channel sizes for conv layers
            dropout: Dropout probability
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_mels = n_mels
        self.n_time = n_time

        if channels is None:
            channels = [32, 64, 128, 256]

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 1

        for out_channels in channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout),
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, n_time) or (batch, n_mels, n_time)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        return self.classifier(x)


class CRNNClassifier(AudioClassifierBase):
    """
    Convolutional Recurrent Neural Network for audio classification.

    Combines CNN feature extraction with LSTM/GRU for temporal modeling.
    """

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 128,
        conv_channels: List[int] = None,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        bidirectional: bool = True,
        rnn_type: str = "gru",
        dropout: float = 0.3,
    ):
        """
        Initialize CRNN classifier.

        Args:
            n_classes: Number of output classes
            n_mels: Number of mel frequency bins
            conv_channels: List of channel sizes for conv layers
            rnn_hidden: Hidden size for RNN
            rnn_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            rnn_type: Type of RNN ("gru" or "lstm")
            dropout: Dropout probability
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_mels = n_mels

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        # Convolutional feature extractor (only pool in frequency dim)
        self.conv_blocks = nn.ModuleList()
        in_channels = 1

        for out_channels in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # Only pool frequency
                nn.Dropout2d(dropout),
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate feature size after conv layers
        freq_size = n_mels // (2 ** len(conv_channels))
        rnn_input_size = conv_channels[-1] * freq_size

        # RNN layer
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        # Classifier
        rnn_output_size = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, n_time) or (batch, n_mels, n_time)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # Conv feature extraction
        for block in self.conv_blocks:
            x = block(x)

        # Reshape for RNN: (batch, channels, freq, time) -> (batch, time, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, features)

        # RNN
        rnn_out, _ = self.rnn(x)

        # Use last hidden state
        x = rnn_out[:, -1, :]

        # Classifier
        return self.classifier(x)


class AttentionClassifier(AudioClassifierBase):
    """
    CNN with attention mechanism for audio classification.

    Uses self-attention to weight temporal features.
    """

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 128,
        conv_channels: List[int] = None,
        attention_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize attention classifier.

        Args:
            n_classes: Number of output classes
            n_mels: Number of mel frequency bins
            conv_channels: List of channel sizes for conv layers
            attention_heads: Number of attention heads
            hidden_dim: Hidden dimension for attention
            dropout: Dropout probability
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_mels = n_mels

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        # Convolutional feature extractor
        self.conv_blocks = nn.ModuleList()
        in_channels = 1

        for out_channels in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Dropout2d(dropout),
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Calculate feature size
        freq_size = n_mels // (2 ** len(conv_channels))
        feature_size = conv_channels[-1] * freq_size

        # Project to hidden dimension
        self.feature_proj = nn.Linear(feature_size, hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        # Conv feature extraction
        for block in self.conv_blocks:
            x = block(x)

        # Reshape: (batch, channels, freq, time) -> (batch, time, features)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)

        # Project features
        x = self.feature_proj(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Global average pooling over time
        x = attn_out.mean(dim=1)

        # Classifier
        return self.classifier(x)


# =============================================================================
# Multi-Label Hierarchical Classification
# =============================================================================

@dataclass
class HierarchyNode:
    """Node in the label hierarchy."""
    name: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    level: int = 0


class LabelHierarchy:
    """
    Manages hierarchical label structure.

    Supports multi-level taxonomy (e.g., Order -> Family -> Genus -> Species).
    """

    def __init__(self):
        """Initialize empty hierarchy."""
        self.nodes: Dict[str, HierarchyNode] = {}
        self.root_nodes: List[str] = []
        self.levels: Dict[int, List[str]] = {}

    def add_node(
        self,
        name: str,
        parent: Optional[str] = None,
        level: Optional[int] = None
    ) -> None:
        """
        Add a node to the hierarchy.

        Args:
            name: Node name (label)
            parent: Parent node name
            level: Hierarchy level (auto-computed if None)
        """
        if level is None:
            if parent is None:
                level = 0
            elif parent in self.nodes:
                level = self.nodes[parent].level + 1
            else:
                level = 0

        node = HierarchyNode(name=name, parent=parent, level=level)
        self.nodes[name] = node

        if parent is None:
            self.root_nodes.append(name)
        elif parent in self.nodes:
            self.nodes[parent].children.append(name)

        if level not in self.levels:
            self.levels[level] = []
        self.levels[level].append(name)

    def get_ancestors(self, name: str) -> List[str]:
        """Get all ancestors of a node."""
        ancestors = []
        current = name
        while current in self.nodes and self.nodes[current].parent is not None:
            current = self.nodes[current].parent
            ancestors.append(current)
        return ancestors

    def get_descendants(self, name: str) -> List[str]:
        """Get all descendants of a node."""
        descendants = []
        if name not in self.nodes:
            return descendants

        stack = [name]
        while stack:
            current = stack.pop()
            for child in self.nodes[current].children:
                descendants.append(child)
                stack.append(child)
        return descendants

    def get_path(self, name: str) -> List[str]:
        """Get full path from root to node."""
        if name not in self.nodes:
            return []
        path = [name]
        current = name
        while self.nodes[current].parent is not None:
            current = self.nodes[current].parent
            path.insert(0, current)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert hierarchy to dictionary."""
        return {
            "nodes": {
                name: {
                    "name": node.name,
                    "parent": node.parent,
                    "children": node.children,
                    "level": node.level,
                }
                for name, node in self.nodes.items()
            },
            "root_nodes": self.root_nodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelHierarchy":
        """Create hierarchy from dictionary."""
        hierarchy = cls()
        hierarchy.root_nodes = data.get("root_nodes", [])

        for name, node_data in data.get("nodes", {}).items():
            node = HierarchyNode(
                name=node_data["name"],
                parent=node_data.get("parent"),
                children=node_data.get("children", []),
                level=node_data.get("level", 0),
            )
            hierarchy.nodes[name] = node

            level = node.level
            if level not in hierarchy.levels:
                hierarchy.levels[level] = []
            hierarchy.levels[level].append(name)

        return hierarchy

    @classmethod
    def from_taxonomy_csv(
        cls,
        filepath: str,
        columns: List[str] = None
    ) -> "LabelHierarchy":
        """
        Create hierarchy from taxonomy CSV.

        Args:
            filepath: Path to CSV file
            columns: Column names for hierarchy levels (in order)

        Returns:
            LabelHierarchy instance
        """
        import csv

        if columns is None:
            columns = ["order", "family", "genus", "species"]

        hierarchy = cls()

        with TextFile(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f.handle)

            for row in reader:
                parent = None
                for level, col in enumerate(columns):
                    if col not in row or not row[col]:
                        continue

                    name = row[col].strip()
                    if name not in hierarchy.nodes:
                        hierarchy.add_node(name, parent=parent, level=level)
                    parent = name

        return hierarchy

    def save(self, filepath: str) -> str:
        """Save hierarchy to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with TextFile(path, mode="w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f.handle, indent=2)

        return str(path)

    @classmethod
    def load(cls, filepath: str) -> "LabelHierarchy":
        """Load hierarchy from JSON file."""
        with TextFile(filepath, mode="r", encoding="utf-8") as f:
            data = json.load(f.handle)
        return cls.from_dict(data)


class HierarchicalClassifier(nn.Module):
    """
    Multi-label hierarchical classifier.

    Outputs predictions at multiple levels of a taxonomy hierarchy.
    Enforces hierarchical consistency in predictions.
    """

    def __init__(
        self,
        backbone: AudioClassifierBase,
        hierarchy: LabelHierarchy,
        feature_dim: int = 256,
        enforce_consistency: bool = True,
    ):
        """
        Initialize hierarchical classifier.

        Args:
            backbone: Feature extraction backbone (without final classifier)
            hierarchy: Label hierarchy
            feature_dim: Dimension of backbone features
            enforce_consistency: Whether to enforce parent-child consistency
        """
        super().__init__()

        self.backbone = backbone
        self.hierarchy = hierarchy
        self.enforce_consistency = enforce_consistency
        self.feature_dim = feature_dim

        # Create classifier head for each level
        self.level_classifiers = nn.ModuleDict()
        self.level_labels: Dict[int, List[str]] = {}

        for level in sorted(hierarchy.levels.keys()):
            labels = hierarchy.levels[level]
            self.level_labels[level] = labels
            n_classes = len(labels)

            self.level_classifiers[str(level)] = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(feature_dim // 2, n_classes),
            )

        # Create label to index mappings
        self.label2idx: Dict[int, Dict[str, int]] = {}
        self.idx2label: Dict[int, Dict[int, str]] = {}

        for level, labels in self.level_labels.items():
            self.label2idx[level] = {label: idx for idx, label in enumerate(labels)}
            self.idx2label[level] = dict(enumerate(labels))

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor
            return_features: Whether to return intermediate features

        Returns:
            Dictionary mapping level to logits
        """
        # Get backbone features (need to access before final classifier)
        # This assumes backbone has a feature extraction part we can access
        features = self._extract_features(x)

        outputs = {}
        for level_str, classifier in self.level_classifiers.items():
            level = int(level_str)
            outputs[level] = classifier(features)

        if return_features:
            outputs["features"] = features

        return outputs

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        # For CNNClassifier, we need to run through conv blocks and pool
        if isinstance(self.backbone, CNNClassifier):
            if x.dim() == 3:
                x = x.unsqueeze(1)

            for block in self.backbone.conv_blocks:
                x = block(x)

            x = self.backbone.global_pool(x)
            x = x.view(x.size(0), -1)

            # Project to feature_dim if needed
            if x.size(1) != self.feature_dim:
                if not hasattr(self, "feature_proj"):
                    self.feature_proj = nn.Linear(x.size(1), self.feature_dim).to(x.device)
                x = self.feature_proj(x)

            return x
        else:
            # Generic fallback - use backbone directly and project
            logits = self.backbone(x)
            if not hasattr(self, "feature_proj"):
                self.feature_proj = nn.Linear(logits.size(1), self.feature_dim).to(logits.device)
            return self.feature_proj(logits)

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[int, List[str]]:
        """
        Get multi-label predictions for all levels.

        Args:
            x: Input tensor
            threshold: Probability threshold for positive prediction

        Returns:
            Dictionary mapping level to list of predicted labels
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            predictions = {}
            for level, logits in outputs.items():
                if not isinstance(level, int):
                    continue

                probs = F.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=-1).item()
                predictions[level] = self.idx2label[level][pred_idx]

            # Enforce consistency if enabled
            if self.enforce_consistency:
                predictions = self._enforce_consistency(predictions)

            return predictions

    def _enforce_consistency(
        self,
        predictions: Dict[int, str]
    ) -> Dict[int, str]:
        """Ensure predictions are consistent with hierarchy."""
        levels = sorted(predictions.keys(), reverse=True)

        for i, level in enumerate(levels[:-1]):
            child_label = predictions[level]
            parent_level = levels[i + 1]

            # Get expected parent from hierarchy
            if child_label in self.hierarchy.nodes:
                expected_parent = self.hierarchy.nodes[child_label].parent
                if expected_parent and expected_parent in self.level_labels.get(parent_level, []):
                    predictions[parent_level] = expected_parent

        return predictions


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier for predicting multiple non-exclusive labels.

    Uses binary cross-entropy loss for each label independently.
    """

    def __init__(
        self,
        backbone: AudioClassifierBase,
        n_labels: int,
        feature_dim: int = 256,
        threshold: float = 0.5,
    ):
        """
        Initialize multi-label classifier.

        Args:
            backbone: Feature extraction backbone
            n_labels: Number of labels
            feature_dim: Feature dimension
            threshold: Prediction threshold
        """
        super().__init__()

        self.backbone = backbone
        self.n_labels = n_labels
        self.threshold = threshold

        # Multi-label head (no softmax - use sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self._extract_features(x)
        return self.classifier(features)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        if isinstance(self.backbone, CNNClassifier):
            if x.dim() == 3:
                x = x.unsqueeze(1)

            for block in self.backbone.conv_blocks:
                x = block(x)

            x = self.backbone.global_pool(x)
            return x.view(x.size(0), -1)
        else:
            return self.backbone(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > self.threshold).float()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# =============================================================================
# Model Ensemble
# =============================================================================

class EnsembleStrategy(ABC):
    """Base class for ensemble combination strategies."""

    @abstractmethod
    def combine(
        self,
        predictions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Combine predictions from multiple models."""
        pass


class AveragingStrategy(EnsembleStrategy):
    """Average probability predictions."""

    def combine(
        self,
        predictions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Average predictions with optional weights."""
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)

        combined = torch.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            combined += weight * pred

        return combined


class VotingStrategy(EnsembleStrategy):
    """Majority voting on class predictions."""

    def __init__(self, soft: bool = True):
        """
        Initialize voting strategy.

        Args:
            soft: If True, use soft voting (average probabilities).
                  If False, use hard voting (majority class).
        """
        self.soft = soft

    def combine(
        self,
        predictions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Combine predictions via voting."""
        if self.soft:
            # Soft voting: average probabilities
            return AveragingStrategy().combine(predictions, weights)
        else:
            # Hard voting: majority class
            batch_size, n_classes = predictions[0].shape
            votes = torch.zeros(batch_size, n_classes, device=predictions[0].device)

            for pred in predictions:
                class_pred = torch.argmax(pred, dim=-1)
                for i in range(batch_size):
                    votes[i, class_pred[i]] += 1

            return votes / len(predictions)


class MaxStrategy(EnsembleStrategy):
    """Take maximum probability for each class."""

    def combine(
        self,
        predictions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Take element-wise maximum."""
        stacked = torch.stack(predictions, dim=0)
        return torch.max(stacked, dim=0).values


class StackingStrategy(EnsembleStrategy):
    """
    Stacking ensemble with learned meta-classifier.

    Trains a meta-model on base model predictions.
    """

    def __init__(self, n_classes: int, n_models: int):
        """
        Initialize stacking strategy.

        Args:
            n_classes: Number of output classes
            n_models: Number of base models
        """
        self.n_classes = n_classes
        self.n_models = n_models

        # Simple linear meta-classifier
        self.meta_classifier = nn.Linear(n_classes * n_models, n_classes)
        self.trained = False

    def fit(
        self,
        predictions: List[torch.Tensor],
        labels: torch.Tensor,
        epochs: int = 10,
        lr: float = 0.01
    ) -> None:
        """
        Train meta-classifier.

        Args:
            predictions: List of model predictions on validation set
            labels: True labels
            epochs: Training epochs
            lr: Learning rate
        """
        # Concatenate predictions
        stacked = torch.cat(predictions, dim=-1)

        optimizer = torch.optim.Adam(self.meta_classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.meta_classifier.train()
        for _epoch in range(epochs):
            optimizer.zero_grad()
            output = self.meta_classifier(stacked)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        self.trained = True

    def combine(
        self,
        predictions: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Combine using meta-classifier."""
        if not self.trained:
            # Fall back to averaging if not trained
            return AveragingStrategy().combine(predictions, weights)

        stacked = torch.cat(predictions, dim=-1)
        self.meta_classifier.eval()
        with torch.no_grad():
            output = self.meta_classifier(stacked)
        return F.softmax(output, dim=-1)


class Ensemble:
    """
    Model ensemble for combining multiple classifiers.

    Supports various combination strategies including averaging,
    voting, and stacking.
    """

    def __init__(
        self,
        models: List[nn.Module],
        strategy: Optional[EnsembleStrategy] = None,
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ensemble.

        Args:
            models: List of classifier models
            strategy: Combination strategy (default: averaging)
            weights: Model weights for weighted averaging
            device: Device to use for inference
        """
        self.models = models
        self.strategy = strategy or AveragingStrategy()
        self.weights = weights
        self.device = device or torch.device("cpu")

        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble predictions.

        Args:
            x: Input tensor

        Returns:
            Combined probability predictions
        """
        x = x.to(self.device)
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        return self.strategy.combine(predictions, self.weights)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict(x)
        return torch.argmax(probs, dim=-1)

    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction uncertainty (disagreement between models).

        Args:
            x: Input tensor

        Returns:
            Uncertainty scores
        """
        x = x.to(self.device)
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

        # Compute variance across models
        stacked = torch.stack(predictions, dim=0)
        variance = torch.var(stacked, dim=0)
        return variance.mean(dim=-1)  # Average uncertainty across classes

    def save(self, directory: str) -> str:
        """
        Save ensemble models.

        Args:
            directory: Output directory

        Returns:
            Path to saved directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model in enumerate(self.models):
            model_path = path / f"model_{i}.pt"
            torch.save(model.state_dict(), model_path)

        # Save config
        config = {
            "n_models": len(self.models),
            "weights": self.weights,
            "strategy": type(self.strategy).__name__,
        }
        with TextFile(path / "config.json", mode="w") as f:
            json.dump(config, f.handle, indent=2)

        return str(path)


# =============================================================================
# Training Utilities
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for model training."""

    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 1e-4
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    grad_clip: Optional[float] = 1.0
    mixed_precision: bool = False
    save_best: bool = True
    output_dir: str = "./training_output"


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainerConfig] = None,
    device: Optional[torch.device] = None,
    callbacks: Optional[List[Callable]] = None,
) -> Dict[str, Any]:
    """
    Train a classifier model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use
        callbacks: List of callback functions

    Returns:
        Training history and metrics
    """
    if config is None:
        config = TrainerConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    callbacks = callbacks or []

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for _batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if config.mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()

                if config.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Save best model
            if config.save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), output_dir / "best_model.pt")

            # Learning rate scheduling
            if scheduler is not None:
                if config.scheduler == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            # Early stopping
            if early_stopping(val_acc):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        else:
            if scheduler is not None and config.scheduler != "plateau":
                scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

        # Run callbacks
        for callback in callbacks:
            callback(epoch, model, history)

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "output_dir": str(output_dir),
    }


def train_multilabel_classifier(
    model: MultiLabelClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainerConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Train a multi-label classifier.

    Uses binary cross-entropy loss.
    """
    if config is None:
        config = TrainerConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.float())
                    val_loss += loss.item()

                    preds = (torch.sigmoid(outputs) > model.threshold).float()
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Compute F1 score
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Micro F1
            tp = (all_preds * all_targets).sum()
            fp = (all_preds * (1 - all_targets)).sum()
            fn = ((1 - all_preds) * all_targets).sum()

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            history["val_f1"].append(f1.item())

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), output_dir / "best_model.pt")

            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}"
            )

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    return {"history": history, "best_f1": best_f1, "output_dir": str(output_dir)}
