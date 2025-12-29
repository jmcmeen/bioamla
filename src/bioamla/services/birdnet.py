# services/birdnet.py
"""
Service for BirdNET model operations.
"""

from typing import Any, Dict, List, Optional

from bioamla.models.birdnet import PredictionResult
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class BirdNETService(BaseService):
    """
    Service for BirdNET model operations.

    Provides ServiceResult-wrapped methods for BirdNET model operations.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize BirdNET service.

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
        sample_rate: int = 48000,
        clip_duration: float = 3.0,
        overlap: float = 0.0,
        min_confidence: float = 0.0,
        top_k: int = 1,
    ) -> ServiceResult[List[PredictionResult]]:
        """
        Run prediction on a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Path to BirdNET model
            sample_rate: Target sample rate (BirdNET uses 48kHz)
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

            model = load_model("birdnet", model_path, config)
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


    def extract_embeddings(
        self,
        filepath: str,
        model_path: str,
        layer: Optional[str] = None,
        sample_rate: int = 48000,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Extract embeddings from audio using a BirdNET model.

        Args:
            filepath: Path to audio file
            model_path: Path to BirdNET model
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
            model = load_model("birdnet", model_path, config)

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
        Get information about a BirdNET model.

        Args:
            model_path: Path to model

        Returns:
            ServiceResult containing model info dict
        """
        try:
            from bioamla.core.ml import load_model

            model = load_model("birdnet", model_path)

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
