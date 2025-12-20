# controllers/embedding.py
"""
Embedding Controller
====================

Controller for audio embedding extraction operations.

Orchestrates between CLI/API views and core embedding extraction functions.
Handles model loading, batch processing, dimensionality reduction, and
output formatting.

Example:
    from bioamla.controllers.embedding import EmbeddingController

    controller = EmbeddingController(model_path="MIT/ast-finetuned-audioset")

    # Single file
    result = controller.extract("audio.wav")
    print(result.data.embeddings.shape)

    # Batch extraction
    result = controller.extract_batch(
        directory="./audio",
        output_path="embeddings.npy",
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseController, ControllerResult


@dataclass
class EmbeddingInfo:
    """Information about extracted embeddings."""

    filepath: str
    shape: Tuple[int, ...]
    embedding_dim: int
    num_segments: int
    normalized: bool
    model: str
    layer: str


@dataclass
class BatchEmbeddingSummary:
    """Summary of batch embedding extraction."""

    total_files: int
    files_processed: int
    files_failed: int
    embedding_dim: int
    total_embeddings: int
    output_path: Optional[str]
    errors: List[str] = field(default_factory=list)


class EmbeddingController(BaseController):
    """
    Controller for audio embedding extraction operations.

    Provides high-level methods for:
    - Single file embedding extraction
    - Batch embedding extraction with progress
    - Dimensionality reduction (PCA, UMAP)
    - Multiple output formats (npy, parquet, csv)
    - Embedding visualization coordinates
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "ast",
    ):
        """
        Initialize embedding controller.

        Args:
            model_path: Path to model (HuggingFace ID or local path)
            model_type: Model type ("ast", "birdnet")
        """
        super().__init__()
        self._model_path = model_path
        self._model_type = model_type
        self._extractor = None

    def _get_extractor(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ):
        """Lazy load the embedding extractor."""
        from bioamla.core.ml.embeddings import EmbeddingConfig, EmbeddingExtractor

        path = model_path or self._model_path
        mtype = model_type or self._model_type

        if path is None:
            # Use default AST model
            path = "MIT/ast-finetuned-audioset-10-10-0.4593"

        # Create new extractor if path changed or first use
        if self._extractor is None or path != self._model_path:
            config = EmbeddingConfig(
                model_path=path,
                model_type=mtype,
                **kwargs,
            )
            self._extractor = EmbeddingExtractor(config=config)
            self._model_path = path

        return self._extractor

    # =========================================================================
    # Single File Extraction
    # =========================================================================

    def extract(
        self,
        filepath: str,
        model_path: Optional[str] = None,
        layer: str = "last_hidden_state",
        normalize: bool = True,
        output_path: Optional[str] = None,
    ) -> ControllerResult[EmbeddingInfo]:
        """
        Extract embeddings from a single audio file.

        Args:
            filepath: Path to audio file
            model_path: Model path (uses controller default if not specified)
            layer: Layer to extract embeddings from
            normalize: Whether to L2-normalize embeddings
            output_path: Optional path to save embeddings (.npy)

        Returns:
            Result with embedding info
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            extractor = self._get_extractor(model_path, normalize=normalize)
            result = extractor.extract(filepath, layer=layer)

            # Save if requested
            if output_path:
                from bioamla.core.ml.embeddings import save_embeddings

                save_embeddings(
                    result.embeddings,
                    [filepath],
                    output_path,
                    format="npy",
                )

            info = EmbeddingInfo(
                filepath=filepath,
                shape=result.embeddings.shape,
                embedding_dim=result.embedding_dim,
                num_segments=result.num_segments,
                normalized=normalize,
                model=self._model_path or "default",
                layer=layer,
            )

            return ControllerResult.ok(
                data=info,
                message=f"Extracted embeddings with shape {result.embeddings.shape}",
                embeddings=result.embeddings,
                segments=result.segments,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Batch Extraction
    # =========================================================================

    def extract_batch(
        self,
        directory: str,
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        output_format: str = "npy",
        layer: str = "last_hidden_state",
        aggregate: str = "mean",
        normalize: bool = True,
        recursive: bool = True,
    ) -> ControllerResult[BatchEmbeddingSummary]:
        """
        Extract embeddings from multiple audio files.

        Args:
            directory: Directory containing audio files
            model_path: Model path (uses controller default if not specified)
            output_path: Path to save embeddings
            output_format: Output format ("npy", "parquet", "csv", "npz")
            layer: Layer to extract embeddings from
            aggregate: How to aggregate segments ("mean", "first", "all")
            normalize: Whether to L2-normalize embeddings
            recursive: Search subdirectories

        Returns:
            Result with batch embedding summary
        """
        error = self._validate_input_path(directory)
        if error:
            return ControllerResult.fail(error)

        try:
            extractor = self._get_extractor(model_path, normalize=normalize)
            files = self._get_audio_files(directory, recursive=recursive)

            if not files:
                return ControllerResult.fail(f"No audio files found in {directory}")

            # Extract with progress
            all_embeddings = []
            filepaths = []
            errors = []

            def process_file(filepath: Path):
                result = extractor.extract(str(filepath), layer=layer)
                if aggregate == "mean":
                    return result.mean_embedding()
                elif aggregate == "first":
                    return result.embeddings[0]
                else:
                    return result.embeddings

            for filepath, embeddings, error in self._process_batch(files, process_file):
                if error:
                    errors.append(f"{filepath.name}: {error}")
                elif embeddings is not None:
                    all_embeddings.append(embeddings)
                    filepaths.append(str(filepath))

            if not all_embeddings:
                return ControllerResult.fail("No embeddings extracted")

            # Stack embeddings
            stacked = np.vstack(all_embeddings)

            # Save if output path specified
            saved_path = None
            if output_path:
                from bioamla.core.ml.embeddings import save_embeddings

                saved_path = save_embeddings(
                    stacked,
                    filepaths,
                    output_path,
                    format=output_format,
                )

            summary = BatchEmbeddingSummary(
                total_files=len(files),
                files_processed=len(filepaths),
                files_failed=len(errors),
                embedding_dim=stacked.shape[-1],
                total_embeddings=stacked.shape[0],
                output_path=saved_path,
                errors=errors,
            )

            return ControllerResult.ok(
                data=summary,
                message=f"Extracted {stacked.shape[0]} embeddings from {len(filepaths)} files",
                embeddings=stacked,
                filepaths=filepaths,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Dimensionality Reduction
    # =========================================================================

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Reduce dimensionality of embeddings.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            method: Reduction method ("pca", "umap", "tsne")
            n_components: Number of output dimensions
            output_path: Optional path to save reduced embeddings
            **kwargs: Additional arguments for the reducer

        Returns:
            Result with reduced embeddings info
        """
        try:
            from bioamla.core.analysis.clustering import reduce_dimensions

            reduced = reduce_dimensions(
                embeddings,
                method=method,
                n_components=n_components,
                **kwargs,
            )

            if output_path:
                np.save(output_path, reduced)

            return ControllerResult.ok(
                data={
                    "original_shape": embeddings.shape,
                    "reduced_shape": reduced.shape,
                    "method": method,
                    "n_components": n_components,
                    "output_path": output_path,
                },
                message=f"Reduced from {embeddings.shape} to {reduced.shape}",
                reduced_embeddings=reduced,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def reduce_for_visualization(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        **kwargs,
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Reduce embeddings to 2D for visualization.

        Args:
            embeddings: Input embeddings
            method: Reduction method ("umap", "tsne", "pca")
            **kwargs: Additional arguments for the reducer

        Returns:
            Result with 2D coordinates
        """
        result = self.reduce_dimensions(
            embeddings,
            method=method,
            n_components=2,
            **kwargs,
        )

        if result.success:
            coords = result.metadata["reduced_embeddings"]
            result.data["x"] = coords[:, 0].tolist()
            result.data["y"] = coords[:, 1].tolist()

        return result

    # =========================================================================
    # Load and Save
    # =========================================================================

    def load_embeddings(
        self,
        filepath: str,
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Load embeddings from file.

        Args:
            filepath: Path to embeddings file

        Returns:
            Result with loaded embeddings
        """
        error = self._validate_input_path(filepath)
        if error:
            return ControllerResult.fail(error)

        try:
            from bioamla.core.ml.embeddings import load_embeddings

            embeddings, filepaths = load_embeddings(filepath)

            return ControllerResult.ok(
                data={
                    "filepath": filepath,
                    "shape": embeddings.shape,
                    "num_files": len(filepaths),
                },
                message=f"Loaded embeddings with shape {embeddings.shape}",
                embeddings=embeddings,
                filepaths=filepaths,
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepaths: List[str],
        output_path: str,
        format: str = "npy",
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Save embeddings to file.

        Args:
            embeddings: Embedding array
            filepaths: List of source file paths
            output_path: Output file path
            format: Output format ("npy", "parquet", "csv", "npz")

        Returns:
            Result with save info
        """
        try:
            from bioamla.core.ml.embeddings import save_embeddings

            saved_path = save_embeddings(
                embeddings,
                filepaths,
                output_path,
                format=format,
            )

            return ControllerResult.ok(
                data={
                    "output_path": saved_path,
                    "format": format,
                    "shape": embeddings.shape,
                    "num_files": len(filepaths),
                },
                message=f"Saved embeddings to {saved_path}",
            )
        except Exception as e:
            return ControllerResult.fail(str(e))

    # =========================================================================
    # Model Information
    # =========================================================================

    def get_model_info(
        self,
        model_path: Optional[str] = None,
    ) -> ControllerResult[Dict[str, Any]]:
        """
        Get information about the embedding model.

        Args:
            model_path: Model path

        Returns:
            Result with model information
        """
        try:
            extractor = self._get_extractor(model_path)
            model = extractor._get_model()

            info = {
                "model_path": self._model_path,
                "model_type": self._model_type,
                "num_classes": getattr(model, "num_classes", None),
                "sample_rate": extractor.config.sample_rate,
                "clip_duration": extractor.config.clip_duration,
            }

            return ControllerResult.ok(data=info)
        except Exception as e:
            return ControllerResult.fail(str(e))
