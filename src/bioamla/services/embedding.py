# services/embedding.py
"""
Service for audio embedding extraction operations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from bioamla.models.embedding import EmbeddingInfo
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class EmbeddingService(BaseService):
    """
    Service for audio embedding extraction operations.

    Provides high-level methods for:
    - Single file embedding extraction
    - Dimensionality reduction (PCA, UMAP)
    - Multiple output formats (npy, parquet, csv)
    - Embedding visualization coordinates
    """

    def __init__(
        self,
        file_repository: FileRepositoryProtocol,
        model_path: Optional[str] = None,
        model_type: str = "ast",
    ) -> None:
        """
        Initialize embedding service.

        Args:
            file_repository: Repository for file operations (required)
            model_path: Path to model (HuggingFace ID or local path)
            model_type: Model type ("ast", "birdnet")
        """
        super().__init__(file_repository=file_repository)
        self._model_path = model_path
        self._model_type = model_type
        self._extractor = None

    def _get_extractor(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs: Any,
    ) -> "EmbeddingExtractor":
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
    ) -> ServiceResult[EmbeddingInfo]:
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
            return ServiceResult.fail(error)

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

            return ServiceResult.ok(
                data=info,
                message=f"Extracted embeddings with shape {result.embeddings.shape}",
                embeddings=result.embeddings,
                segments=result.segments,
            )
        except Exception as e:
            return ServiceResult.fail(str(e))


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
    ) -> ServiceResult[Dict[str, Any]]:
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
            from bioamla.core.audio.clustering import reduce_dimensions

            reduced = reduce_dimensions(
                embeddings,
                method=method,
                n_components=n_components,
                **kwargs,
            )

            if output_path:
                np.save(output_path, reduced)

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def reduce_for_visualization(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        **kwargs,
    ) -> ServiceResult[Dict[str, Any]]:
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
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Load embeddings from file.

        Args:
            filepath: Path to embeddings file

        Returns:
            Result with loaded embeddings
        """
        error = self._validate_input_path(filepath)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.ml.embeddings import load_embeddings

            embeddings, filepaths = load_embeddings(filepath)

            return ServiceResult.ok(
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
            return ServiceResult.fail(str(e))

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepaths: List[str],
        output_path: str,
        format: str = "npy",
    ) -> ServiceResult[Dict[str, Any]]:
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

            return ServiceResult.ok(
                data={
                    "output_path": saved_path,
                    "format": format,
                    "shape": embeddings.shape,
                    "num_files": len(filepaths),
                },
                message=f"Saved embeddings to {saved_path}",
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

            return ServiceResult.ok(data=info)
        except Exception as e:
            return ServiceResult.fail(str(e))
