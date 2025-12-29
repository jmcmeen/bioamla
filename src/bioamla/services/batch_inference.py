"""Batch model inference service (AST-only)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from bioamla.models.batch import BatchConfig, BatchResult
from bioamla.repository.protocol import FileRepositoryProtocol
from bioamla.services.batch_base import BatchServiceBase
from bioamla.services.inference import InferenceService


class BatchInferenceService(BatchServiceBase):
    """Service for batch model inference (AST-only for now).

    This service delegates to InferenceService for actual inference operations,
    following the dependency injection pattern.

    Once AST is perfected, this will be extended to support other model types.
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        inference_service: InferenceService,
    ) -> None:
        """Initialize batch inference service.

        Args:
            file_repository: File repository for file discovery
            inference_service: Single-file inference service to delegate to
        """
        super().__init__(file_repository)
        self.inference_service = inference_service
        self._current_operation: Optional[str] = None
        self._current_model_path: Optional[str] = None
        self._current_params: Dict[str, Any] = {}
        self._all_predictions: list = []

    def process_file(self, file_path: Path) -> Any:
        """Process a single audio file by delegating to InferenceService.

        Dispatches to the appropriate method based on _current_operation.

        Args:
            file_path: Path to the audio file to process

        Returns:
            Inference result

        Raises:
            ValueError: If operation is not set or unknown
            RuntimeError: If the underlying service operation fails
        """
        if self._current_operation is None:
            raise ValueError("Operation not set. Call a batch method first.")

        # Dispatch to appropriate inference operation
        if self._current_operation == "predict":
            result = self.inference_service.predict(
                str(file_path),
                model_path=self._current_model_path,
                top_k=self._current_params.get("top_k", 5),
                min_confidence=self._current_params.get("min_confidence", 0.0),
            )

            if not result.success:
                raise RuntimeError(result.error)

            # Collect predictions for aggregated output
            for pred in result.data:
                pred_dict = {
                    "filepath": pred.filepath,
                    "predicted_label": pred.predicted_label,
                    "confidence": pred.confidence,
                    "start_time": pred.start_time,
                    "end_time": pred.end_time,
                    "top_k_labels": pred.top_k_labels,
                    "top_k_scores": pred.top_k_scores,
                }
                self._all_predictions.append(pred_dict)

            return result.data

        elif self._current_operation == "embed":
            # Calculate output path for embeddings
            output_dir = Path(self._current_params.get("output_dir", "."))
            output_path = output_dir / f"{file_path.stem}_embeddings.npy"

            # Ensure output directory exists
            self.file_repository.mkdir(str(output_path.parent), parents=True)

            result = self.inference_service.extract_embeddings(
                str(file_path),
                model_path=self._current_model_path,
                output_path=str(output_path),
            )

            if not result.success:
                raise RuntimeError(result.error)

            return result.data

        else:
            raise ValueError(f"Unknown operation: {self._current_operation}")

    def _write_aggregated_predictions(self, output_dir: str) -> None:
        """Write all prediction results to a JSON file.

        Args:
            output_dir: Directory to write results to
        """
        if not self._all_predictions:
            return

        output_path = Path(output_dir) / "predictions_results.json"
        self.file_repository.mkdir(str(output_path.parent), parents=True)

        content = json.dumps(self._all_predictions, indent=2)
        self.file_repository.write_text(output_path, content)

    def predict_batch(
        self,
        config: BatchConfig,
        model_path: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> BatchResult:
        """Run predictions on audio files batch-wise (AST-only).

        Args:
            config: Batch processing configuration
            model_path: Path to the model (HuggingFace ID or local path)
            top_k: Number of top predictions to return per file
            min_confidence: Minimum confidence threshold

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "predict"
        self._current_model_path = model_path
        self._current_params = {
            "top_k": top_k,
            "min_confidence": min_confidence,
        }
        self._all_predictions = []

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        result = self.process_batch_auto(config, file_filter=audio_filter)
        self._write_aggregated_predictions(config.output_dir)
        return result

    def embed_batch(
        self,
        config: BatchConfig,
        model_path: str,
    ) -> BatchResult:
        """Extract embeddings from audio files batch-wise (AST-only).

        Args:
            config: Batch processing configuration
            model_path: Path to the model (HuggingFace ID or local path)

        Returns:
            BatchResult with processing summary
        """
        self._current_operation = "embed"
        self._current_model_path = model_path
        self._current_params = {
            "output_dir": config.output_dir,
        }

        def audio_filter(path: Path) -> bool:
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            return path.suffix.lower() in audio_exts

        return self.process_batch_auto(config, file_filter=audio_filter)
