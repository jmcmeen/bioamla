"""Service for CNN-based audio classifier operations using OpenSoundscape adapter.

This service provides high-level operations for training, prediction, and embedding
extraction using CNN models via the CNNAdapter from the adapters layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


@dataclass
class CNNPrediction:
    """A single CNN prediction result."""

    filepath: str
    start_time: float
    end_time: float
    label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filepath": self.filepath,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass
class CNNTrainResult:
    """Result from CNN training."""

    model_path: str
    epochs: int
    num_classes: int
    architecture: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_path": self.model_path,
            "epochs": self.epochs,
            "num_classes": self.num_classes,
            "architecture": self.architecture,
        }


class CNNService(BaseService):
    """Service for CNN-based audio classifier operations.

    Uses CNNAdapter from the adapters layer to wrap OpenSoundscape CNN
    functionality with a service-layer interface.

    Example:
        >>> from bioamla.services import ServiceFactory
        >>> factory = ServiceFactory()
        >>> cnn_service = factory.cnn
        >>>
        >>> # Get model info
        >>> result = cnn_service.get_model_info("model.pt")
        >>> if result.success:
        ...     print(result.data)
        >>>
        >>> # Run prediction
        >>> result = cnn_service.predict("audio.wav", "model.pt")
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize CNN service.

        Args:
            file_repository: Repository for file operations.
        """
        super().__init__(file_repository=file_repository)
        self._adapter = None
        self._model_path: Optional[str] = None

    def _get_adapter(self, model_path: str) -> Any:
        """Get or load the CNN adapter.

        Args:
            model_path: Path to model file.

        Returns:
            CNNAdapter instance.
        """
        from bioamla.adapters.opensoundscape import CNNAdapter

        if self._adapter is None or self._model_path != model_path:
            self._adapter = CNNAdapter.load(model_path)
            self._model_path = model_path
        return self._adapter

    def create_model(
        self,
        classes: List[str],
        architecture: str = "resnet18",
        sample_duration: float = 3.0,
        sample_rate: int = 16000,
    ) -> ServiceResult[Dict[str, Any]]:
        """Create a new CNN model.

        Args:
            classes: List of class names.
            architecture: Model architecture (resnet18, resnet50, efficientnet_b0, etc.).
            sample_duration: Duration of audio clips in seconds.
            sample_rate: Target sample rate.

        Returns:
            ServiceResult containing model configuration.
        """
        try:
            from bioamla.adapters.opensoundscape import CNNAdapter

            adapter = CNNAdapter.create(
                classes=classes,
                architecture=architecture,
                sample_duration=sample_duration,
                sample_rate=sample_rate,
            )

            self._adapter = adapter
            self._model_path = None  # Not saved yet

            return ServiceResult.ok(
                data=adapter.config(),
                message=f"Created {architecture} model with {len(classes)} classes",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to create model: {e}")

    def train(
        self,
        train_csv: str,
        output_dir: str,
        validation_csv: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        freeze_backbone: bool = False,
        num_workers: int = 0,
    ) -> ServiceResult[CNNTrainResult]:
        """Train the CNN model.

        Args:
            train_csv: Path to training CSV with columns: file, class1, class2, ...
                       Values should be 0 or 1 for each class.
            output_dir: Directory to save model checkpoints.
            validation_csv: Path to validation CSV (same format).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate (uses default if None).
            freeze_backbone: If True, freeze backbone weights for transfer learning.
            num_workers: Number of data loader workers.

        Returns:
            ServiceResult containing training result.
        """
        if self._adapter is None:
            return ServiceResult.fail(
                "No model loaded. Use create_model() or load_model() first."
            )

        error = self._validate_input_path(train_csv)
        if error:
            return ServiceResult.fail(error)

        if validation_csv:
            error = self._validate_input_path(validation_csv)
            if error:
                return ServiceResult.fail(error)

        try:
            # Load training data
            train_df = pd.read_csv(train_csv, index_col=0)

            # Load validation data if provided
            val_df = None
            if validation_csv:
                val_df = pd.read_csv(validation_csv, index_col=0)

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Train the model
            self._adapter.train(
                train_df=train_df,
                validation_df=val_df,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                save_path=output_dir,
                learning_rate=learning_rate,
                freeze_feature_extractor=freeze_backbone,
            )

            # Save the final model
            model_path = str(output_path / "model.pt")
            self._adapter.save(model_path)
            self._model_path = model_path

            result = CNNTrainResult(
                model_path=model_path,
                epochs=epochs,
                num_classes=self._adapter.num_classes,
                architecture=self._adapter.architecture,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Training complete. Model saved to {model_path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Training failed: {e}")

    def predict(
        self,
        filepath: str,
        model_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        min_confidence: float = 0.0,
        top_k: int = 1,
    ) -> ServiceResult[List[CNNPrediction]]:
        """Run prediction on an audio file.

        Args:
            filepath: Path to audio file.
            model_path: Path to trained CNN model.
            batch_size: Batch size for inference.
            num_workers: Number of data loader workers.
            min_confidence: Minimum confidence threshold.
            top_k: Number of top predictions per segment.

        Returns:
            ServiceResult containing list of predictions.
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_input_path(model_path)
        if error:
            return ServiceResult.fail(error)

        try:
            adapter = self._get_adapter(model_path)

            # Run prediction
            predictions_df = adapter.predict(
                samples=filepath,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Convert DataFrame to list of predictions
            predictions: List[CNNPrediction] = []

            for idx, row in predictions_df.iterrows():
                # Parse index for time info (format: "file_start_end")
                if isinstance(idx, tuple):
                    file_path, start, end = idx
                else:
                    file_path = filepath
                    start, end = 0.0, adapter.sample_duration

                # Get top-k predictions
                sorted_scores = row.sort_values(ascending=False)
                for label, score in sorted_scores.head(top_k).items():
                    if score >= min_confidence:
                        predictions.append(
                            CNNPrediction(
                                filepath=str(file_path),
                                start_time=float(start),
                                end_time=float(end),
                                label=str(label),
                                confidence=float(score),
                            )
                        )

            return ServiceResult.ok(
                data=predictions,
                message=f"Generated {len(predictions)} predictions",
            )
        except Exception as e:
            return ServiceResult.fail(f"Prediction failed: {e}")

    def predict_batch(
        self,
        filepaths: List[str],
        model_path: str,
        batch_size: int = 8,
        num_workers: int = 0,
        min_confidence: float = 0.0,
    ) -> ServiceResult[pd.DataFrame]:
        """Run prediction on multiple audio files.

        Args:
            filepaths: List of audio file paths.
            model_path: Path to trained CNN model.
            batch_size: Batch size for inference.
            num_workers: Number of data loader workers.
            min_confidence: Minimum confidence threshold.

        Returns:
            ServiceResult containing predictions DataFrame.
        """
        for fp in filepaths:
            error = self._validate_input_path(fp)
            if error:
                return ServiceResult.fail(error)

        error = self._validate_input_path(model_path)
        if error:
            return ServiceResult.fail(error)

        try:
            adapter = self._get_adapter(model_path)

            predictions_df = adapter.predict(
                samples=filepaths,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Filter by confidence if needed
            if min_confidence > 0:
                predictions_df = predictions_df.where(
                    predictions_df >= min_confidence
                )

            return ServiceResult.ok(
                data=predictions_df,
                message=f"Generated predictions for {len(filepaths)} files",
            )
        except Exception as e:
            return ServiceResult.fail(f"Batch prediction failed: {e}")

    def extract_embeddings(
        self,
        filepath: str,
        model_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        target_layer: Optional[str] = None,
    ) -> ServiceResult[np.ndarray]:
        """Extract embeddings from audio using a CNN model.

        Args:
            filepath: Path to audio file.
            model_path: Path to trained CNN model.
            batch_size: Batch size for inference.
            num_workers: Number of data loader workers.
            target_layer: Layer to extract embeddings from.

        Returns:
            ServiceResult containing embeddings array.
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_input_path(model_path)
        if error:
            return ServiceResult.fail(error)

        try:
            adapter = self._get_adapter(model_path)

            embeddings = adapter.extract_embeddings(
                samples=filepath,
                batch_size=batch_size,
                num_workers=num_workers,
                target_layer=target_layer,
            )

            return ServiceResult.ok(
                data=embeddings,
                message=f"Extracted embeddings with shape {embeddings.shape}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Embedding extraction failed: {e}")

    def get_model_info(self, model_path: str) -> ServiceResult[Dict[str, Any]]:
        """Get information about a CNN model.

        Args:
            model_path: Path to model file.

        Returns:
            ServiceResult containing model info dict.
        """
        error = self._validate_input_path(model_path)
        if error:
            return ServiceResult.fail(error)

        try:
            adapter = self._get_adapter(model_path)

            info = {
                "path": model_path,
                "architecture": adapter.architecture,
                "num_classes": adapter.num_classes,
                "classes": adapter.classes[:10],
                "has_more_classes": adapter.num_classes > 10,
                "sample_duration": adapter.sample_duration,
            }

            return ServiceResult.ok(
                data=info,
                message=f"Model has {adapter.num_classes} classes",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get model info: {e}")

    def list_architectures(self) -> ServiceResult[List[str]]:
        """List available CNN architectures.

        Returns:
            ServiceResult containing list of architecture names.
        """
        architectures = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "densenet121",
            "densenet161",
            "densenet169",
            "densenet201",
            "inception_v3",
        ]

        return ServiceResult.ok(
            data=architectures,
            message=f"{len(architectures)} architectures available",
        )
