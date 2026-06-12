"""
Base Model Classes
==================

Abstract base classes and common utilities for ML models in bioamla: a unified
interface for model loading, inference, embedding extraction, and batch
processing.

PyTorch / torchaudio ship in the base install but are imported lazily so this
module imports fast.

numpy is imported at module level.

Supported model backends:
- AST (Audio Spectrogram Transformer)
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from bioamla.exceptions import InvalidInputError, ModelError

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


def _require_torch():
    """Import and return the torch module."""
    import torch

    return torch


class ModelBackend(Enum):
    """Supported model backends."""

    AST = "ast"


@dataclass
class PredictionResult:
    """Result from a single prediction."""

    label: str
    confidence: float
    logits: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    filepath: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

    predictions: list[PredictionResult]
    total_files: int
    files_processed: int
    files_failed: int
    processing_time: float

    def to_dict(self) -> dict[str, Any]:
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
    device: str | None = None

    def get_device(self) -> "torch.device":
        """Get the torch device."""
        torch = _require_torch()
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

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize the model.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.device = self.config.get_device()
        self.model = None
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

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
    def classes(self) -> list[str]:
        """Return list of class labels."""
        return list(self.id2label.values())

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> "BaseAudioModel":
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
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None = None,
    ) -> list[PredictionResult]:
        """
        Run prediction on audio.

        Args:
            audio: Audio file path, numpy array, or torch tensor.
            sample_rate: Sample rate if audio is an array/tensor.

        Returns:
            List of prediction results.
        """
        pass

    def predict_file(self, filepath: str) -> list[PredictionResult]:
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
        audio: Union[str, np.ndarray, "torch.Tensor"],
        sample_rate: int | None = None,
        layer: str | None = None,
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
        audio_files: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
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
        predictions: list[PredictionResult],
        min_confidence: float | None = None,
        labels: list[str] | None = None,
        exclude_labels: list[str] | None = None,
    ) -> list[PredictionResult]:
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
            raise InvalidInputError(f"Unsupported format: {format}")

    def _save_pytorch(self, path: str) -> str:
        """Save model in PyTorch format."""
        torch = _require_torch()
        if self.model is None:
            raise ModelError("No model loaded")

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
        torch = _require_torch()
        if self.model is None:
            raise ModelError("No model loaded")

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

    def _get_dummy_input(self) -> "torch.Tensor":
        """Get dummy input for model export."""
        torch = _require_torch()
        # Default: 1 second of audio at model sample rate
        return torch.randn(1, self.config.sample_rate).to(self.device)

    def to(self, device: Union[str, "torch.device"]) -> "BaseAudioModel":
        """Move model to device."""
        torch = _require_torch()
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


# Cache for the lazily-built torch-backed AudioDataset class.
_AUDIO_DATASET_CLASS: type | None = None


def _audio_dataset_class() -> type:
    """Build (and cache) the torch-backed AudioDataset class.

    Defined lazily because it subclasses ``torch.utils.data.Dataset``, which is
    only importable when the ``[ml]`` extra is installed.
    """
    global _AUDIO_DATASET_CLASS
    if _AUDIO_DATASET_CLASS is not None:
        return _AUDIO_DATASET_CLASS

    _require_torch()
    from torch.utils.data import Dataset

    class _AudioDataset(Dataset):
        """Dataset for batch audio processing."""

        def __init__(
            self,
            filepaths: list[str],
            sample_rate: int = 16000,
            clip_duration: float = 3.0,
            transform: Callable | None = None,
        ):
            self.filepaths = filepaths
            self.sample_rate = sample_rate
            self.clip_duration = clip_duration
            self.transform = transform

        def __len__(self) -> int:
            return len(self.filepaths)

        def __getitem__(self, idx: int) -> tuple["torch.Tensor", str]:
            import torch

            from bioamla.audio.torchaudio import (
                load_waveform_tensor,
                resample_waveform_tensor,
            )

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

    _AUDIO_DATASET_CLASS = _AudioDataset
    return _AUDIO_DATASET_CLASS


def AudioDataset(
    filepaths: list[str],
    sample_rate: int = 16000,
    clip_duration: float = 3.0,
    transform: Callable | None = None,
):
    """Create an AudioDataset (torch ``Dataset``) for batch audio processing.

    This is a factory wrapper around a lazily-built ``torch.utils.data.Dataset``
    subclass so the module imports without torch. The returned object IS a real
    ``Dataset`` instance.

    Args:
        filepaths: List of audio file paths.
        sample_rate: Target sample rate.
        clip_duration: Clip duration in seconds.
        transform: Optional transform to apply.

    """
    cls = _audio_dataset_class()
    return cls(
        filepaths=filepaths,
        sample_rate=sample_rate,
        clip_duration=clip_duration,
        transform=transform,
    )


def create_dataloader(
    filepaths: list[str],
    config: ModelConfig,
    transform: Callable | None = None,
) -> "DataLoader":
    """
    Create a DataLoader for batch processing.

    Args:
        filepaths: List of audio file paths.
        config: Model configuration.
        transform: Optional transform to apply.

    Returns:
        DataLoader instance.

    """
    _require_torch()
    from torch.utils.data import DataLoader

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
_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type:
    """Get a registered model class by name."""
    if name not in _MODEL_REGISTRY:
        raise InvalidInputError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


__all__ = [
    "ModelBackend",
    "ModelConfig",
    "PredictionResult",
    "BatchPredictionResult",
    "BaseAudioModel",
    "AudioDataset",
    "create_dataloader",
    "register_model",
    "get_model_class",
    "list_models",
]
