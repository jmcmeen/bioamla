# services/inference.py
"""
Service for ML model inference operations.
"""

from typing import Any, Dict, List, Optional

from bioamla.models.inference import PredictionResult
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class InferenceService(BaseService):
    """
    Service for ML model inference operations.

    Provides high-level methods for:
    - Single file prediction
    - CSV/JSON output generation
    - Model information and listing
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        model_path: Optional[str] = None,
    ) -> None:
        """
        Initialize inference service.

        Args:
            file_repository: Repository for file operations (required)
            model_path: Path to model (HuggingFace ID or local path)
        """
        super().__init__(file_repository=file_repository)
        self._model_path = model_path
        self._model = None

    def _get_model(self, model_path: Optional[str] = None) -> "ASTInference":
        """Lazy load the model."""
        path = model_path or self._model_path
        if path is None:
            raise ValueError("No model path specified")

        # Only reload if path changed
        if self._model is None or path != self._model_path:
            from bioamla.core.inference import ASTInference

            self._model = ASTInference(model_path=path)
            self._model_path = path

        return self._model

    # =========================================================================
    # Single File Prediction
    # =========================================================================

    def predict(
        self,
        filepath: str,
        model_path: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> ServiceResult[List[PredictionResult]]:
        """
        Run prediction on a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Model path (uses controller default if not specified)
            top_k: Number of top predictions to return
            min_confidence: Minimum confidence threshold

        Returns:
            Result with list of predictions
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            model = self._get_model(model_path)
            raw_predictions = model.predict(filepath, top_k=top_k)

            results = []
            for pred in raw_predictions:
                if pred.get("confidence", 0) >= min_confidence:
                    results.append(
                        PredictionResult(
                            filepath=filepath,
                            start_time=pred.get("start_time", 0.0),
                            end_time=pred.get("end_time", 0.0),
                            predicted_label=pred.get("label", ""),
                            confidence=pred.get("confidence", 0.0),
                            top_k_labels=pred.get("top_k_labels", []),
                            top_k_scores=pred.get("top_k_scores", []),
                        )
                    )

            return ServiceResult.ok(
                data=results,
                message=f"Generated {len(results)} predictions for {filepath}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))


    # =========================================================================
    # Embedding Extraction
    # =========================================================================

    def extract_embeddings(
        self,
        filepath: str,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Extract embeddings from an audio file.

        Args:
            filepath: Path to audio file
            model_path: Model path
            output_path: Optional path to save embeddings (.npy)

        Returns:
            Result with embedding info
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            model = self._get_model(model_path)
            embeddings = model.get_embeddings(filepath)

            if output_path:
                import numpy as np

                np.save(output_path, embeddings)

            return ServiceResult.ok(
                data={
                    "filepath": filepath,
                    "shape": embeddings.shape,
                    "output_path": output_path,
                },
                message=f"Extracted embeddings with shape {embeddings.shape}",
            )
        except Exception as e:
            return ServiceResult.fail(str(e))


    # =========================================================================
    # Model Information
    # =========================================================================

    def get_model_info(
        self,
        model_path: Optional[str] = None,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Get information about a model.

        Args:
            model_path: Model path

        Returns:
            Result with model information
        """
        try:
            model = self._get_model(model_path)

            info = {
                "model_path": self._model_path,
                "model_type": type(model).__name__,
                "num_labels": getattr(model, "num_labels", None),
                "labels": getattr(model, "labels", None),
            }

            return ServiceResult.ok(data=info)
        except Exception as e:
            return ServiceResult.fail(str(e))

    def list_available_models(self) -> ServiceResult[List[Dict[str, str]]]:
        """
        List commonly available models.

        Returns:
            Result with list of model info dicts
        """
        models = [
            {
                "name": "MIT/ast-finetuned-audioset-10-10-0.4593",
                "description": "AudioSet pre-trained AST model (527 classes)",
                "type": "AudioSet",
            },
            {
                "name": "bioamla/scp-frogs",
                "description": "Frog species classifier",
                "type": "Bioamla",
            },
            {
                "name": "bioamla/ast-esc50",
                "description": "ESC-50 environmental sounds",
                "type": "Bioamla",
            },
        ]

        return ServiceResult.ok(
            data=models,
            message=f"Found {len(models)} available models",
        )
