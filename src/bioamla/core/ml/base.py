"""
Base Model Classes
==================

This module provides abstract base classes and common utilities for all
ML models in bioamla. It defines a unified interface for model loading,
inference, embedding extraction, and batch processing.

Supported model backends:
- AST (Audio Spectrogram Transformer)
- BirdNET (bird species classification)
- OpenSoundscape (ResNet18/50 CNNs)
- Custom CNN (transfer learning)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ModelBackend(Enum):
    """Supported model backends."""

    AST = "ast"
    BIRDNET = "birdnet"
    OPENSOUNDSCAPE = "opensoundscape"
    CUSTOM_CNN = "custom_cnn"


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    label: str
    confidence: float
    logits: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    start_time: float = 0.0
    end_time: float = 0.0
    filepath: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "label": self.label,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        if self.filepath:
            d["filepath"] = self.filepath
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class BatchPredictionResult:
    """Results from batch prediction."""

    predictions: List[PredictionResult]
    total_files: int
    files_processed: int
    files_failed: int
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "processing_time": self.processing_time,
        }


@dataclass
class ModelConfig:
    """Base configuration for all models."""

    sample_rate: int = 16000
    clip_duration: float = 3.0
    overlap: float = 0.0
    min_confidence: float = 0.0
    top_k: int = 1
    batch_size: int = 8
    num_workers: int = 4
    use_fp16: bool = False
    device: Optional[str] = None

    def get_device(self) -> torch.device:
        """Get the torch device."""
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseAudioModel(ABC):
    """
    Abstract base class for audio classification models.

    This class defines the interface that all model backends must implement.
    It provides common functionality for audio preprocessing, batch processing,
    and result filtering.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.device = self.config.get_device()
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}

    @property
    @abstractmethod
    def backend(self) -> ModelBackend:
        """Return the model backend type."""
        pass

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.id2label)

    @property
    def classes(self) -> List[str]:
        """Return list of class labels."""
        return list(self.id2label.values())

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> "BaseAudioModel":
        """
        Load model from path.

        Args:
            model_path: Path to model file or HuggingFace identifier.
            **kwargs: Additional model-specific arguments.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> List[PredictionResult]:
        """
        Run prediction on audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is an array/tensor.

        Returns:
            List of prediction results.
        """
        pass

    def predict_file(self, filepath: str) -> List[PredictionResult]:
        """
        Run prediction on an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            List of prediction results.
        """
        return self.predict(filepath)

    @abstractmethod
    def extract_embeddings(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        layer: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract embeddings from audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is an array/tensor.
            layer: Optional layer name for extraction.

        Returns:
            Embedding vectors as numpy array.
        """
        pass

    def predict_batch(
        self,
        audio_files: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchPredictionResult:
        """
        Run batch prediction on multiple files.

        Args:
            audio_files: List of audio file paths.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Batch prediction results.
        """
        import time

        start_time = time.time()

        predictions = []
        files_processed = 0
        files_failed = 0

        for i, filepath in enumerate(audio_files):
            try:
                results = self.predict_file(filepath)
                predictions.extend(results)
                files_processed += 1
            except Exception:
                files_failed += 1

            if progress_callback:
                progress_callback(i + 1, len(audio_files))

        processing_time = time.time() - start_time

        return BatchPredictionResult(
            predictions=predictions,
            total_files=len(audio_files),
            files_processed=files_processed,
            files_failed=files_failed,
            processing_time=processing_time,
        )

    def filter_predictions(
        self,
        predictions: List[PredictionResult],
        min_confidence: Optional[float] = None,
        labels: Optional[List[str]] = None,
        exclude_labels: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """
        Filter predictions by confidence and labels.

        Args:
            predictions: List of predictions to filter.
            min_confidence: Minimum confidence threshold.
            labels: Only include these labels.
            exclude_labels: Exclude these labels.

        Returns:
            Filtered list of predictions.
        """
        min_conf = min_confidence if min_confidence is not None else self.config.min_confidence

        filtered = []
        for pred in predictions:
            if pred.confidence < min_conf:
                continue
            if labels and pred.label not in labels:
                continue
            if exclude_labels and pred.label in exclude_labels:
                continue
            filtered.append(pred)

        return filtered

    def save(self, path: str, format: str = "pt") -> str:
        """
        Save model to file.

        Args:
            path: Output path.
            format: Save format ("pt" for PyTorch, "onnx" for ONNX).

        Returns:
            Path to saved model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "pt":
            return self._save_pytorch(str(path))
        elif format == "onnx":
            return self._save_onnx(str(path))
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_pytorch(self, path: str) -> str:
        """Save model in PyTorch format."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        state = {
            "model_state_dict": self.model.state_dict(),
            "id2label": self.id2label,
            "label2id": self.label2id,
            "config": {
                "sample_rate": self.config.sample_rate,
                "clip_duration": self.config.clip_duration,
                "backend": self.backend.value,
            },
        }

        torch.save(state, path)
        return path

    def _save_onnx(self, path: str) -> str:
        """Save model in ONNX format."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Create dummy input for tracing
        dummy_input = self._get_dummy_input()

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=["audio_input"],
            output_names=["logits"],
            dynamic_axes={
                "audio_input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
        )
        return path

    def _get_dummy_input(self) -> torch.Tensor:
        """Get dummy input for model export."""
        # Default: 1 second of audio at model sample rate
        return torch.randn(1, self.config.sample_rate).to(self.device)

    def to(self, device: Union[str, torch.device]) -> "BaseAudioModel":
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    def eval(self) -> "BaseAudioModel":
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(backend={self.backend.value}, classes={self.num_classes})"
        )


class AudioDataset(Dataset):
    """Dataset for batch audio processing."""

    def __init__(
        self,
        filepaths: List[str],
        sample_rate: int = 16000,
        clip_duration: float = 3.0,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            filepaths: List of audio file paths.
            sample_rate: Target sample rate.
            clip_duration: Clip duration in seconds.
            transform: Optional transform to apply.
        """
        self.filepaths = filepaths
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor

        filepath = self.filepaths[idx]
        waveform, sr = load_waveform_tensor(filepath)

        if sr != self.sample_rate:
            waveform = resample_waveform_tensor(waveform, sr, self.sample_rate)

        # Truncate or pad to clip_duration
        target_length = int(self.clip_duration * self.sample_rate)
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0), filepath


def create_dataloader(
    filepaths: List[str],
    config: ModelConfig,
    transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Create a DataLoader for batch processing.

    Args:
        filepaths: List of audio file paths.
        config: Model configuration.
        transform: Optional transform to apply.

    Returns:
        DataLoader instance.
    """
    dataset = AudioDataset(
        filepaths=filepaths,
        sample_rate=config.sample_rate,
        clip_duration=config.clip_duration,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
        shuffle=False,
    )


# Model registry for dynamic loading
_MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type:
    """Get a registered model class by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())
