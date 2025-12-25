# services/birdnet.py
"""
Service for BirdNET model operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class PredictionResult(ToDictMixin):
    """Result of a single prediction."""

    filepath: str
    label: str
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class BirdNETService(BaseService):
    """
    Service for BirdNET model operations.

    Provides ServiceResult-wrapped methods for BirdNET model operations.
    """

    def __init__(self) -> None:
        super().__init__()
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

    def predict_batch(
        self,
        directory: str,
        model_path: str,
        output_csv: Optional[str] = None,
        sample_rate: int = 48000,
        clip_duration: float = 3.0,
        overlap: float = 0.0,
        min_confidence: float = 0.0,
        batch_size: int = 8,
        fp16: bool = False,
        recursive: bool = True,
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Run batch prediction on a directory of audio files.

        Args:
            directory: Directory containing audio files
            model_path: Path to BirdNET model
            output_csv: Output CSV file path
            sample_rate: Target sample rate
            clip_duration: Clip duration in seconds
            overlap: Overlap between clips in seconds
            min_confidence: Minimum confidence threshold
            batch_size: Batch size for processing
            fp16: Use half-precision inference
            recursive: Search subdirectories

        Returns:
            ServiceResult containing batch prediction summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ServiceResult.fail(error)

        try:
            import csv
            import time

            from bioamla.core.ml import ModelConfig, load_model
            from bioamla.core.utils import get_audio_files

            config = ModelConfig(
                sample_rate=sample_rate,
                clip_duration=clip_duration,
                overlap=overlap,
                min_confidence=min_confidence,
                batch_size=batch_size,
                use_fp16=fp16,
            )

            model = load_model("birdnet", model_path, config, use_fp16=fp16)

            audio_files = get_audio_files(directory)
            if not audio_files:
                return ServiceResult.fail(f"No audio files found in {directory}")

            start_time = time.time()
            all_results = []

            for filepath in audio_files:
                try:
                    results = model.predict(filepath)
                    all_results.extend(results)
                except Exception:
                    continue

            elapsed = time.time() - start_time

            if output_csv:
                output_path = Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["filepath", "start_time", "end_time", "label", "confidence"])
                    for r in all_results:
                        writer.writerow([
                            r.filepath,
                            f"{r.start_time:.3f}",
                            f"{r.end_time:.3f}",
                            r.label,
                            f"{r.confidence:.4f}",
                        ])

            result = {
                "total_files": len(audio_files),
                "total_predictions": len(all_results),
                "output_path": output_csv,
                "elapsed_seconds": elapsed,
            }

            return ServiceResult.ok(
                data=result,
                message=f"Processed {len(audio_files)} files in {elapsed:.2f}s",
            )
        except Exception as e:
            return ServiceResult.fail(f"Batch prediction failed: {e}")

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
