# controllers/inference.py
"""
Inference Controller
====================

Controller for ML model inference operations.

Orchestrates between CLI/API views and core ML inference functions.
Handles model loading, batch processing, and output formatting.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseController, ControllerResult


@dataclass
class PredictionResult:
    """Single prediction result."""

    filepath: str
    start_time: float
    end_time: float
    predicted_label: str
    confidence: float
    top_k_labels: List[str] = field(default_factory=list)
    top_k_scores: List[float] = field(default_factory=list)


@dataclass
class InferenceSummary:
    """Summary of inference results."""

    total_files: int
    total_predictions: int
    unique_labels: int
    label_counts: Dict[str, int]
    output_path: Optional[str] = None


@dataclass
class BatchInferenceResult:
    """Result of batch inference."""

    predictions: List[PredictionResult]
    summary: InferenceSummary
    errors: List[str] = field(default_factory=list)


class InferenceController(BaseController):
    """
    Controller for ML model inference operations.

    Provides high-level methods for:
    - Single file prediction
    - Batch prediction with progress
    - CSV/JSON output generation
    - Model information and listing
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize inference controller.

        Args:
            model_path: Path to model (HuggingFace ID or local path)
        """
        super().__init__()
        self._model_path = model_path
        self._model = None

    def _get_model(self, model_path: Optional[str] = None):
        """Lazy load the model."""
        path = model_path or self._model_path
        if path is None:
            raise ValueError("No model path specified")

        # Only reload if path changed
        if self._model is None or path != self._model_path:
            from bioamla.core.ml.inference import ASTInference

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
    ) -> ControllerResult[List[PredictionResult]]:
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
            return ControllerResult.fail(error)

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

            return ControllerResult.ok(
                data=results,
                message=f"Generated {len(results)} predictions for {filepath}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Batch Prediction
    # =========================================================================

    def predict_batch(
        self,
        directory: str,
        model_path: Optional[str] = None,
        output_csv: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.0,
        recursive: bool = True,
        clip_duration: Optional[float] = None,
        overlap: float = 0.0,
    ) -> ControllerResult[BatchInferenceResult]:
        """
        Run prediction on multiple audio files.

        Args:
            directory: Directory containing audio files
            model_path: Model path (uses controller default if not specified)
            output_csv: Optional CSV output path
            top_k: Number of top predictions per clip
            min_confidence: Minimum confidence threshold
            recursive: Search subdirectories
            clip_duration: Duration of clips in seconds (for long files)
            overlap: Overlap between clips (0-1)

        Returns:
            Result with batch inference summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        # Start run tracking
        run_id = self._start_run(
            name=f"Batch prediction: {directory}",
            action="predict",
            input_path=directory,
            output_path=output_csv or "",
            parameters={
                "model_path": model_path or self._model_path,
                "top_k": top_k,
                "min_confidence": min_confidence,
                "recursive": recursive,
                "clip_duration": clip_duration,
                "overlap": overlap,
            },
        )

        try:
            from bioamla.core.files import TextFile

            model = self._get_model(model_path)
            files = self._get_audio_files(directory, recursive=recursive)

            if not files:
                self._fail_run("No audio files found")
                return ControllerResult.fail(f"No audio files found in {directory}")

            all_predictions = []
            errors = []
            label_counts: Dict[str, int] = {}

            def process_file(filepath: Path) -> List[PredictionResult]:
                raw_preds = model.predict(
                    str(filepath),
                    top_k=top_k,
                    clip_duration=clip_duration,
                    overlap=overlap,
                )

                results = []
                for pred in raw_preds:
                    conf = pred.get("confidence", 0)
                    if conf >= min_confidence:
                        label = pred.get("label", "")
                        label_counts[label] = label_counts.get(label, 0) + 1
                        results.append(
                            PredictionResult(
                                filepath=str(filepath),
                                start_time=pred.get("start_time", 0.0),
                                end_time=pred.get("end_time", 0.0),
                                predicted_label=label,
                                confidence=conf,
                                top_k_labels=pred.get("top_k_labels", []),
                                top_k_scores=pred.get("top_k_scores", []),
                            )
                        )
                return results

            for filepath, results, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")
                elif results:
                    all_predictions.extend(results)

            # Write CSV if requested
            if output_csv and all_predictions:
                with TextFile(output_csv, mode="w", newline="") as f:
                    writer = csv.writer(f.handle)
                    writer.writerow(
                        [
                            "filepath",
                            "start_time",
                            "end_time",
                            "predicted_label",
                            "confidence",
                        ]
                    )
                    for pred in all_predictions:
                        writer.writerow(
                            [
                                pred.filepath,
                                f"{pred.start_time:.3f}",
                                f"{pred.end_time:.3f}",
                                pred.predicted_label,
                                f"{pred.confidence:.4f}",
                            ]
                        )

            summary = InferenceSummary(
                total_files=len(files),
                total_predictions=len(all_predictions),
                unique_labels=len(label_counts),
                label_counts=label_counts,
                output_path=output_csv,
            )

            # Complete run with results
            self._complete_run(
                results={
                    "total_files": len(files),
                    "total_predictions": len(all_predictions),
                    "unique_labels": len(label_counts),
                    "label_counts": label_counts,
                    "errors_count": len(errors),
                },
                output_files=[output_csv] if output_csv else None,
            )

            return ControllerResult.ok(
                data=BatchInferenceResult(
                    predictions=all_predictions,
                    summary=summary,
                    errors=errors,
                ),
                message=f"Generated {len(all_predictions)} predictions from {len(files)} files",
            )
        except Exception as e:
            self._fail_run(str(e))
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Embedding Extraction
    # =========================================================================

    def extract_embeddings(
        self,
        filepath: str,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> ControllerResult[Dict[str, Any]]:
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
            return ControllerResult.fail(error)

        try:
            model = self._get_model(model_path)
            embeddings = model.get_embeddings(filepath)

            if output_path:
                import numpy as np

                np.save(output_path, embeddings)

            return ControllerResult.ok(
                data={
                    "filepath": filepath,
                    "shape": embeddings.shape,
                    "output_path": output_path,
                },
                message=f"Extracted embeddings with shape {embeddings.shape}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def extract_embeddings_batch(
        self,
        directory: str,
        output_path: str,
        model_path: Optional[str] = None,
        recursive: bool = True,
        format: str = "npy",
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Extract embeddings from multiple audio files.

        Args:
            directory: Directory containing audio files
            output_path: Path to save embeddings
            model_path: Model path
            recursive: Search subdirectories
            format: Output format (npy, parquet, csv)

        Returns:
            Result with embedding extraction summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        # Start run tracking
        run_id = self._start_run(
            name=f"Batch embedding extraction: {directory}",
            action="embed",
            input_path=directory,
            output_path=output_path,
            parameters={
                "model_path": model_path or self._model_path,
                "recursive": recursive,
                "format": format,
            },
        )

        try:
            import numpy as np

            model = self._get_model(model_path)
            files = self._get_audio_files(directory, recursive=recursive)

            if not files:
                self._fail_run("No audio files found")
                return ControllerResult.fail(f"No audio files found in {directory}")

            all_embeddings = []
            file_mapping = []
            errors = []

            def process_file(filepath: Path):
                emb = model.get_embeddings(str(filepath))
                return emb

            for filepath, embeddings, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")
                elif embeddings is not None:
                    all_embeddings.append(embeddings)
                    file_mapping.append(str(filepath))

            if not all_embeddings:
                self._fail_run("No embeddings extracted")
                return ControllerResult.fail("No embeddings extracted")

            # Stack all embeddings
            stacked = np.vstack(all_embeddings)

            # Save based on format
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            output_files = [str(output_file)]
            if format == "npy":
                np.save(str(output_file), stacked)
                # Also save file mapping
                mapping_file = output_file.with_suffix(".files.txt")
                with open(mapping_file, "w") as f:
                    f.write("\n".join(file_mapping))
                output_files.append(str(mapping_file))
            elif format == "parquet":
                try:
                    import pandas as pd

                    df = pd.DataFrame(stacked)
                    df["filepath"] = file_mapping
                    df.to_parquet(str(output_file))
                except ImportError:
                    self._fail_run("pandas required for parquet format")
                    return ControllerResult.fail("pandas required for parquet format")

            # Complete run with results
            self._complete_run(
                results={
                    "total_files": len(files),
                    "extracted": len(all_embeddings),
                    "shape": list(stacked.shape),
                    "errors_count": len(errors),
                },
                output_files=output_files,
            )

            return ControllerResult.ok(
                data={
                    "total_files": len(files),
                    "extracted": len(all_embeddings),
                    "shape": stacked.shape,
                    "output_path": str(output_file),
                    "errors": errors,
                },
                message=f"Extracted embeddings from {len(all_embeddings)} files",
            )
        except Exception as e:
            self._fail_run(str(e))
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Model Information
    # =========================================================================

    def get_model_info(
        self,
        model_path: Optional[str] = None,
    ) -> ControllerResult[Dict[str, Any]]:
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

            return ControllerResult.ok(data=info)
        except Exception as e:
            return ControllerResult.fail(str(e))

    def list_available_models(self) -> ControllerResult[List[Dict[str, str]]]:
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

        return ControllerResult.ok(
            data=models,
            message=f"Found {len(models)} available models",
        )
