# core/ml/embeddings.py
"""
Embedding Extraction Module
===========================

Provides unified embedding extraction from audio using various model backends.

This module supports:
- AST (Audio Spectrogram Transformer) embeddings
- BirdNET embeddings
- Custom model adapters
- Batch processing with GPU memory management
- Multiple output formats (npy, parquet, csv)

Example:
    from bioamla.core.ml.embeddings import (
        EmbeddingExtractor,
        extract_embeddings,
        extract_embeddings_batch,
    )

    # Single file extraction
    extractor = EmbeddingExtractor(model_path="MIT/ast-finetuned-audioset")
    embeddings = extractor.extract("audio.wav")

    # Batch extraction
    result = extract_embeddings_batch(
        audio_files=["a.wav", "b.wav"],
        model_path="MIT/ast-finetuned-audioset",
        output_path="embeddings.npy",
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from bioamla.core.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingExtractor",
    "extract_embeddings",
    "extract_embeddings_batch",
    "save_embeddings",
    "load_embeddings",
]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""

    # Model settings
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model_type: str = "ast"  # "ast", "birdnet", "custom"
    layer: str = "last_hidden_state"  # Layer to extract embeddings from

    # Audio settings
    sample_rate: int = 16000
    clip_duration: float = 10.0
    overlap: float = 0.0

    # Processing settings
    batch_size: int = 8
    use_fp16: bool = False
    device: Optional[str] = None

    # Reduction settings
    reduce_method: Optional[str] = None  # None, "pca", "umap"
    n_components: int = 128
    normalize: bool = True


@dataclass
class EmbeddingResult:
    """Result from single file embedding extraction."""

    filepath: str
    embeddings: np.ndarray
    sample_rate: int
    segments: List[Tuple[float, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[-1] if self.embeddings.ndim > 1 else len(self.embeddings)

    @property
    def num_segments(self) -> int:
        """Number of segments (embeddings)."""
        return self.embeddings.shape[0] if self.embeddings.ndim > 1 else 1

    def mean_embedding(self) -> np.ndarray:
        """Get mean embedding across all segments."""
        if self.embeddings.ndim == 1:
            return self.embeddings
        return self.embeddings.mean(axis=0)


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding extraction."""

    embeddings: np.ndarray
    filepaths: List[str]
    total_files: int
    files_processed: int
    files_failed: int
    output_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[-1]

    @property
    def success_rate(self) -> float:
        """Percentage of files successfully processed."""
        return self.files_processed / self.total_files * 100 if self.total_files > 0 else 0


class EmbeddingExtractor:
    """
    Unified embedding extractor supporting multiple model backends.

    This class provides a consistent interface for extracting embeddings
    from audio files using AST, BirdNET, or custom models.

    Example:
        extractor = EmbeddingExtractor(model_path="MIT/ast-finetuned-audioset")

        # Single file
        result = extractor.extract("audio.wav")
        print(f"Embedding shape: {result.embeddings.shape}")

        # Multiple files
        for result in extractor.extract_iter(["a.wav", "b.wav"]):
            print(f"{result.filepath}: {result.embeddings.shape}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to model or HuggingFace model ID
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        if model_path:
            self.config.model_path = model_path

        self._model = None
        self._reducer = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return self._model

        model_type = self.config.model_type.lower()

        if model_type == "ast":
            from bioamla.core.ml.ast_model import ASTModel
            from bioamla.core.ml.base import ModelConfig

            model_config = ModelConfig(
                sample_rate=self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                overlap=self.config.overlap,
                use_fp16=self.config.use_fp16,
                device=self.config.device,
            )
            self._model = ASTModel(config=model_config)
            self._model.load(
                self.config.model_path,
                use_fp16=self.config.use_fp16,
            )

        elif model_type == "birdnet":
            try:
                from bioamla.core.ml.base import ModelConfig
                from bioamla.core.ml.birdnet import BirdNETModel

                model_config = ModelConfig(
                    sample_rate=self.config.sample_rate,
                    clip_duration=self.config.clip_duration,
                    device=self.config.device,
                )
                self._model = BirdNETModel(config=model_config)
                self._model.load(self.config.model_path)
            except ImportError as err:
                raise ImportError(
                    "BirdNET support requires additional dependencies. "
                    "Install with: pip install birdnetlib"
                ) from err

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Loaded {model_type} model from {self.config.model_path}")
        return self._model

    def _get_reducer(self):
        """Get or create dimensionality reducer."""
        if self._reducer is not None or self.config.reduce_method is None:
            return self._reducer

        from bioamla.core.analysis.clustering import IncrementalReducer

        self._reducer = IncrementalReducer(
            method=self.config.reduce_method,
            n_components=self.config.n_components,
        )
        return self._reducer

    def extract(
        self,
        audio: Union[str, np.ndarray],
        sample_rate: Optional[int] = None,
        layer: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from a single audio file or array.

        Args:
            audio: Audio file path or numpy array
            sample_rate: Sample rate (required if audio is array)
            layer: Layer to extract from (uses config default if None)

        Returns:
            EmbeddingResult with extracted embeddings
        """
        model = self._get_model()
        layer = layer or self.config.layer

        filepath = audio if isinstance(audio, str) else None

        # Extract embeddings
        embeddings = model.extract_embeddings(
            audio,
            sample_rate=sample_rate,
            layer=layer,
        )

        # Ensure 2D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize if requested
        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        # Reduce dimensions if configured
        reducer = self._get_reducer()
        if reducer is not None:
            if not reducer.fitted:
                reducer.fit(embeddings)
            embeddings = reducer.transform(embeddings)

        # Calculate segments
        segments = []
        if filepath:
            try:
                import soundfile as sf

                info = sf.info(filepath)
                duration = info.duration
                clip_dur = self.config.clip_duration
                overlap = self.config.overlap

                n_segments = embeddings.shape[0]
                step = clip_dur - overlap
                for i in range(n_segments):
                    start = i * step
                    end = min(start + clip_dur, duration)
                    segments.append((start, end))
            except Exception:
                pass

        return EmbeddingResult(
            filepath=filepath or "<array>",
            embeddings=embeddings,
            sample_rate=sample_rate or self.config.sample_rate,
            segments=segments,
            metadata={
                "model": self.config.model_path,
                "layer": layer,
                "normalized": self.config.normalize,
            },
        )

    def extract_iter(
        self,
        audio_files: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[EmbeddingResult]:
        """
        Extract embeddings from multiple files as an iterator.

        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback(current, total)

        Yields:
            EmbeddingResult for each file
        """
        total = len(audio_files)

        for i, filepath in enumerate(audio_files):
            try:
                result = self.extract(filepath)
                yield result
            except Exception as e:
                logger.warning(f"Failed to extract embeddings from {filepath}: {e}")
                # Yield empty result for failed files
                yield EmbeddingResult(
                    filepath=filepath,
                    embeddings=np.array([]),
                    sample_rate=self.config.sample_rate,
                    metadata={"error": str(e)},
                )

            if progress_callback:
                progress_callback(i + 1, total)

    def extract_batch(
        self,
        audio_files: List[str],
        aggregate: str = "mean",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEmbeddingResult:
        """
        Extract embeddings from multiple files.

        Args:
            audio_files: List of audio file paths
            aggregate: How to aggregate segment embeddings ("mean", "first", "all")
            progress_callback: Optional callback(current, total)

        Returns:
            BatchEmbeddingResult with all embeddings
        """
        embeddings_list = []
        filepaths = []
        errors = []
        files_processed = 0
        files_failed = 0

        for result in self.extract_iter(audio_files, progress_callback):
            if result.embeddings.size == 0:
                files_failed += 1
                errors.append(f"{result.filepath}: {result.metadata.get('error', 'unknown error')}")
                continue

            files_processed += 1
            filepaths.append(result.filepath)

            # Aggregate embeddings for this file
            if aggregate == "mean":
                emb = result.mean_embedding()
            elif aggregate == "first":
                emb = result.embeddings[0]
            else:  # "all"
                emb = result.embeddings

            embeddings_list.append(emb)

        # Stack all embeddings
        if embeddings_list:
            if aggregate == "all":
                # Variable length - use list
                stacked = np.vstack(embeddings_list)
            else:
                stacked = np.vstack(embeddings_list)
        else:
            stacked = np.array([])

        return BatchEmbeddingResult(
            embeddings=stacked,
            filepaths=filepaths,
            total_files=len(audio_files),
            files_processed=files_processed,
            files_failed=files_failed,
            errors=errors,
            metadata={
                "model": self.config.model_path,
                "aggregate": aggregate,
                "normalized": self.config.normalize,
            },
        )

    def fit_reducer(self, embeddings: np.ndarray) -> None:
        """
        Fit the dimensionality reducer on embeddings.

        Args:
            embeddings: Training embeddings
        """
        reducer = self._get_reducer()
        if reducer is not None:
            reducer.fit(embeddings)
            logger.info(f"Fitted {self.config.reduce_method} reducer")


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_embeddings(
    audio: Union[str, np.ndarray],
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    model_type: str = "ast",
    layer: str = "last_hidden_state",
    sample_rate: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract embeddings from audio.

    This is a convenience function for one-off embedding extraction.
    For batch processing, use EmbeddingExtractor directly.

    Args:
        audio: Audio file path or numpy array
        model_path: Path to model or HuggingFace model ID
        model_type: Model type ("ast", "birdnet")
        layer: Layer to extract from
        sample_rate: Sample rate (required if audio is array)
        normalize: Whether to L2-normalize embeddings

    Returns:
        Embedding array of shape (n_segments, embedding_dim)
    """
    config = EmbeddingConfig(
        model_path=model_path,
        model_type=model_type,
        layer=layer,
        normalize=normalize,
    )
    extractor = EmbeddingExtractor(config=config)
    result = extractor.extract(audio, sample_rate=sample_rate, layer=layer)
    return result.embeddings


def extract_embeddings_batch(
    audio_files: List[str],
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    model_type: str = "ast",
    output_path: Optional[str] = None,
    output_format: str = "npy",
    aggregate: str = "mean",
    normalize: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchEmbeddingResult:
    """
    Extract embeddings from multiple audio files.

    Args:
        audio_files: List of audio file paths
        model_path: Path to model or HuggingFace model ID
        model_type: Model type ("ast", "birdnet")
        output_path: Optional path to save embeddings
        output_format: Output format ("npy", "parquet", "csv")
        aggregate: How to aggregate segments ("mean", "first", "all")
        normalize: Whether to L2-normalize embeddings
        progress_callback: Optional callback(current, total)

    Returns:
        BatchEmbeddingResult with extracted embeddings
    """
    config = EmbeddingConfig(
        model_path=model_path,
        model_type=model_type,
        normalize=normalize,
    )
    extractor = EmbeddingExtractor(config=config)
    result = extractor.extract_batch(
        audio_files,
        aggregate=aggregate,
        progress_callback=progress_callback,
    )

    # Save if output path specified
    if output_path and result.embeddings.size > 0:
        save_embeddings(
            result.embeddings,
            result.filepaths,
            output_path,
            format=output_format,
        )
        result.output_path = output_path

    return result


def save_embeddings(
    embeddings: np.ndarray,
    filepaths: List[str],
    output_path: str,
    format: str = "npy",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save embeddings to file.

    Args:
        embeddings: Embedding array
        filepaths: List of source file paths
        output_path: Output file path
        format: Output format ("npy", "parquet", "csv", "npz")
        metadata: Optional metadata to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        np.save(str(output_path), embeddings)
        # Save file mapping
        mapping_path = output_path.with_suffix(".files.txt")
        with open(mapping_path, "w") as f:
            f.write("\n".join(filepaths))
        logger.info(f"Saved embeddings to {output_path} and file mapping to {mapping_path}")

    elif format == "npz":
        np.savez(
            str(output_path),
            embeddings=embeddings,
            filepaths=np.array(filepaths, dtype=object),
            **(metadata or {}),
        )
        logger.info(f"Saved embeddings to {output_path}")

    elif format == "parquet":
        try:
            import pandas as pd

            df = pd.DataFrame(embeddings)
            df.columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
            df["filepath"] = filepaths
            df.to_parquet(str(output_path))
            logger.info(f"Saved embeddings to {output_path}")
        except ImportError as err:
            raise ImportError("pandas and pyarrow required for parquet format") from err

    elif format == "csv":
        try:
            import pandas as pd

            df = pd.DataFrame(embeddings)
            df.columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
            df["filepath"] = filepaths
            df.to_csv(str(output_path), index=False)
            logger.info(f"Saved embeddings to {output_path}")
        except ImportError as err:
            raise ImportError("pandas required for CSV format") from err

    else:
        raise ValueError(f"Unknown format: {format}")

    return str(output_path)


def load_embeddings(
    filepath: str,
    format: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from file.

    Args:
        filepath: Path to embeddings file
        format: File format (auto-detected if None)

    Returns:
        Tuple of (embeddings array, file paths list)
    """
    filepath = Path(filepath)

    if format is None:
        format = filepath.suffix.lstrip(".")

    if format == "npy":
        embeddings = np.load(str(filepath))
        mapping_path = filepath.with_suffix(".files.txt")
        if mapping_path.exists():
            with open(mapping_path) as f:
                filepaths = [line.strip() for line in f]
        else:
            filepaths = []
        return embeddings, filepaths

    elif format == "npz":
        data = np.load(str(filepath), allow_pickle=True)
        embeddings = data["embeddings"]
        filepaths = list(data.get("filepaths", []))
        return embeddings, filepaths

    elif format == "parquet":
        import pandas as pd

        df = pd.read_parquet(str(filepath))
        filepaths = df["filepath"].tolist() if "filepath" in df.columns else []
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        embeddings = df[dim_cols].values
        return embeddings, filepaths

    elif format == "csv":
        import pandas as pd

        df = pd.read_csv(str(filepath))
        filepaths = df["filepath"].tolist() if "filepath" in df.columns else []
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        embeddings = df[dim_cols].values
        return embeddings, filepaths

    else:
        raise ValueError(f"Unknown format: {format}")
