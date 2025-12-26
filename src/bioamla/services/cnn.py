# services/cnn.py
"""
Service for CNN-based spectrogram classifier operations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class PredictionResult(ToDictMixin):
    """Result of a single prediction."""

    filepath: str
    label: str
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class TrainResult(ToDictMixin):
    """Result of model training."""

    model_path: str
    epochs: int
    architecture: str
    n_classes: int
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None


@dataclass
class EvaluationResult(ToDictMixin):
    """Result of model evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int


class CNNService(BaseService):
    """
    Service for CNN-based spectrogram classifier operations.

    Provides ServiceResult-wrapped methods for CNN model operations.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize CNN service.

        Args:
            file_repository: Repository for file operations (required)
        """
        super().__init__(file_repository=file_repository)
        self._model = None
        self._model_path = None

    def predict(
        self,
        filepath: str,
        model_path: str,
        sample_rate: int = 16000,
        clip_duration: float = 3.0,
        overlap: float = 0.0,
        min_confidence: float = 0.0,
        top_k: int = 1,
    ) -> ServiceResult[List[PredictionResult]]:
        """
        Run prediction on a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Path to trained CNN model
            sample_rate: Target sample rate
            clip_duration: Clip duration in seconds
            overlap: Overlap between clips in seconds
            min_confidence: Minimum confidence threshold
            top_k: Number of top predictions per segment

        Returns:
            ServiceResult containing list of PredictionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.ml import ModelConfig, load_model

            config = ModelConfig(
                sample_rate=sample_rate,
                clip_duration=clip_duration,
                overlap=overlap,
                min_confidence=min_confidence,
                top_k=top_k,
            )

            model = load_model("cnn", model_path, config)
            results = model.predict(filepath)

            predictions = [
                PredictionResult(
                    filepath=r.filepath,
                    label=r.label,
                    confidence=r.confidence,
                    start_time=r.start_time,
                    end_time=r.end_time,
                )
                for r in results
            ]

            return ServiceResult.ok(
                data=predictions,
                message=f"Generated {len(predictions)} predictions",
            )
        except Exception as e:
            return ServiceResult.fail(f"Prediction failed: {e}")


    def train(
        self,
        data_dir: str,
        output_dir: str,
        n_classes: int,
        architecture: str = "cnn",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> ServiceResult[TrainResult]:
        """
        Train a CNN-based spectrogram classifier.

        Args:
            data_dir: Directory containing training data
            output_dir: Output directory for model
            n_classes: Number of classes
            architecture: Model architecture (cnn, crnn, attention)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            ServiceResult containing TrainResult
        """
        error = self._validate_input_path(data_dir)
        if error:
            return ServiceResult.fail(error)

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Note: Full training implementation is complex
            # This is a placeholder for the service interface

            return ServiceResult.fail(
                "Full CNN training not yet implemented in service layer. "
                "Use CLI command 'bioamla models train cnn' directly."
            )
        except Exception as e:
            return ServiceResult.fail(f"Training failed: {e}")

    def extract_embeddings(
        self,
        filepath: str,
        model_path: str,
        layer: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Extract embeddings from audio using a CNN model.

        Args:
            filepath: Path to audio file
            model_path: Path to trained CNN model
            layer: Layer to extract embeddings from
            sample_rate: Target sample rate

        Returns:
            ServiceResult containing embeddings info
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:

            from bioamla.core.ml import ModelConfig, load_model

            config = ModelConfig(sample_rate=sample_rate)
            model = load_model("cnn", model_path, config)

            embeddings = model.extract_embeddings(filepath, layer=layer)

            return ServiceResult.ok(
                data={
                    "filepath": filepath,
                    "shape": embeddings.shape,
                    "dtype": str(embeddings.dtype),
                },
                message=f"Extracted embeddings with shape {embeddings.shape}",
                embeddings=embeddings,
            )
        except Exception as e:
            return ServiceResult.fail(f"Embedding extraction failed: {e}")

    def get_model_info(
        self,
        model_path: str,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Get information about a CNN model.

        Args:
            model_path: Path to model

        Returns:
            ServiceResult containing model info dict
        """
        try:
            from bioamla.core.ml import load_model

            model = load_model("cnn", model_path)

            info = {
                "path": model_path,
                "backend": model.backend.value if hasattr(model.backend, "value") else str(model.backend),
                "num_classes": model.num_classes,
                "classes": model.classes[:10] if model.classes else [],
                "has_more_classes": len(model.classes) > 10 if model.classes else False,
            }

            return ServiceResult.ok(
                data=info,
                message=f"Model has {model.num_classes} classes",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get model info: {e}")
