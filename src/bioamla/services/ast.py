# services/ast.py
"""
Service for Audio Spectrogram Transformer (AST) model operations.
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
    confusion_matrix: Optional[List[List[int]]] = None


class ASTService(BaseService):
    """
    Service for AST (Audio Spectrogram Transformer) model operations.

    Provides ServiceResult-wrapped methods for AST model operations.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize AST service.

        Args:
            file_repository: Repository for file operations (required)
        """
        super().__init__(file_repository=file_repository)
        self._model = None
        self._model_path = None

    def predict(
        self,
        filepath: str,
        model_path: str = "bioamla/scp-frogs",
        resample_freq: int = 16000,
    ) -> ServiceResult[List[PredictionResult]]:
        """
        Run prediction on a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Path to model or HuggingFace identifier
            resample_freq: Target sample rate

        Returns:
            ServiceResult containing list of PredictionResult
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.services.inference import InferenceService

            service = InferenceService(
                file_repository=self.file_repository,
                model_path=model_path,
            )
            result = service.predict(filepath=filepath)

            if not result.success:
                return ServiceResult.fail(result.error)

            predictions = [
                PredictionResult(
                    filepath=filepath,
                    label=pred.predicted_label,
                    confidence=pred.confidence,
                )
                for pred in result.data
            ]

            return ServiceResult.ok(
                data=predictions,
                message=f"Generated {len(predictions)} predictions",
            )
        except Exception as e:
            return ServiceResult.fail(f"Prediction failed: {e}")


    def train(
        self,
        train_dataset: str,
        output_dir: str,
        base_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        split: str = "train",
        category_label_column: str = "category",
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        batch_size: int = 8,
        fp16: bool = False,
        bf16: bool = False,
        finetune_mode: str = "full",
        push_to_hub: bool = False,
    ) -> ServiceResult[TrainResult]:
        """
        Fine-tune an AST model on a custom dataset.

        Args:
            train_dataset: Training dataset from HuggingFace Hub
            output_dir: Output directory for training outputs
            base_model: Base model to fine-tune
            split: Dataset split to use
            category_label_column: Column name for category labels
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Training batch size
            fp16: Use FP16 mixed precision
            bf16: Use BF16 mixed precision
            finetune_mode: Training mode (full or feature-extraction)
            push_to_hub: Push model to HuggingFace Hub

        Returns:
            ServiceResult containing TrainResult
        """
        try:
            from pathlib import Path as PathLib

            output_path = PathLib(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            best_model_path = output_path / "best_model"

            # Note: Full training implementation is complex and should be
            # delegated to the core module or handled separately
            # This is a placeholder for the service interface

            return ServiceResult.fail(
                "Full training not yet implemented in service layer. "
                "Use CLI command 'bioamla models train ast' directly."
            )
        except Exception as e:
            return ServiceResult.fail(f"Training failed: {e}")

    def evaluate(
        self,
        audio_dir: str,
        model_path: str,
        ground_truth_csv: str,
        file_column: str = "file_name",
        label_column: str = "label",
        resample_freq: int = 16000,
        batch_size: int = 8,
        fp16: bool = False,
    ) -> ServiceResult[EvaluationResult]:
        """
        Evaluate an AST model on a directory of audio files.

        Args:
            audio_dir: Directory containing audio files
            model_path: Path to model
            ground_truth_csv: Path to CSV with ground truth labels
            file_column: Column name for file names
            label_column: Column name for labels
            resample_freq: Target sample rate
            batch_size: Batch size for inference
            fp16: Use half-precision inference

        Returns:
            ServiceResult containing EvaluationResult
        """
        error = self._validate_input_path(audio_dir)
        if error:
            return ServiceResult.fail(error)

        error = self._validate_input_path(ground_truth_csv)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.evaluate import evaluate_directory

            result = evaluate_directory(
                audio_dir=audio_dir,
                model_path=model_path,
                ground_truth_csv=ground_truth_csv,
                gt_file_column=file_column,
                gt_label_column=label_column,
                resample_freq=resample_freq,
                batch_size=batch_size,
                use_fp16=fp16,
                verbose=False,
            )

            eval_result = EvaluationResult(
                accuracy=result.accuracy,
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                total_samples=result.total_samples,
            )

            return ServiceResult.ok(
                data=eval_result,
                message=f"Accuracy: {result.accuracy:.4f}, F1: {result.f1_score:.4f}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Evaluation failed: {e}")

    def list_models(self) -> ServiceResult[List[str]]:
        """
        List available AST models.

        Returns:
            ServiceResult containing list of model identifiers
        """
        try:
            models = [
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                "bioamla/scp-frogs",
            ]

            return ServiceResult.ok(
                data=models,
                message=f"Found {len(models)} AST models",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to list models: {e}")

    def extract_embeddings(
        self,
        filepath: str,
        model_path: str,
        layer: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Extract embeddings from audio using an AST model.

        Args:
            filepath: Path to audio file
            model_path: Path to model or HuggingFace identifier
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
            model = load_model("ast", model_path, config)

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
        Get information about an AST model.

        Args:
            model_path: Path to model or HuggingFace identifier

        Returns:
            ServiceResult containing model info dict
        """
        try:
            from bioamla.core.ml import load_model

            model = load_model("ast", model_path)

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
