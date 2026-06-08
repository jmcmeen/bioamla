"""
Embedding Extraction
====================

AST-based audio embedding extraction: :class:`EmbeddingExtractor` plus
convenience functions (:func:`extract_embeddings`,
:func:`extract_embeddings_batch`, :func:`save_embeddings`,
:func:`load_embeddings`).

PyTorch / transformers / torchaudio are optional extras (``bioamla[ml]``);
parquet / CSV I/O needs ``pandas`` (and ``pyarrow`` for parquet); dimensionality
reduction needs the clustering extra. Heavy deps are imported lazily, so this
module imports on a slim install and only *using* the relevant feature raises
:class:`~bioamla.exceptions.DependencyError`. numpy is a core dependency.

Example:
    from bioamla.ml import EmbeddingExtractor, save_embeddings

    extractor = EmbeddingExtractor(model_path="MIT/ast-finetuned-audioset-10-10-0.4593")
    result = extractor.extract("audio.wav")
    save_embeddings(result.embeddings, ["audio.wav"], "embeddings.npy")
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bioamla.exceptions import DependencyError, ModelError

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""

    # Model settings
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    model_type: str = "ast"  # "ast"
    layer: str = "last_hidden_state"  # Layer to extract embeddings from

    # Audio settings
    sample_rate: int = 16000
    clip_duration: float = 10.0
    overlap: float = 0.0

    # Processing settings
    batch_size: int = 8
    use_fp16: bool = False
    device: str | None = None

    # Reduction settings
    reduce_method: str | None = None  # None, "pca", "umap"
    n_components: int = 128
    normalize: bool = True


@dataclass
class EmbeddingResult:
    """Result from single-file embedding extraction."""

    filepath: str
    embeddings: np.ndarray
    sample_rate: int
    segments: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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
    filepaths: list[str]
    total_files: int
    files_processed: int
    files_failed: int
    output_path: str | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[-1]

    @property
    def success_rate(self) -> float:
        """Percentage of files successfully processed."""
        return self.files_processed / self.total_files * 100 if self.total_files > 0 else 0


# =============================================================================
# Embedding Extractor
# =============================================================================


class EmbeddingExtractor:
    """
    Unified embedding extractor (AST backend).

    Example:
        extractor = EmbeddingExtractor(model_path="MIT/ast-finetuned-audioset-10-10-0.4593")
        result = extractor.extract("audio.wav")
        print(result.embeddings.shape)
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: EmbeddingConfig | None = None,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_path: Path to model or HuggingFace model ID.
            config: Embedding configuration (uses defaults if None).
        """
        self.config = config or EmbeddingConfig()
        if model_path:
            self.config.model_path = model_path

        self._model = None
        self._reducer = None

    def _get_model(self) -> Any:
        """Lazy-load the backing model.

        Raises:
            DependencyError: If the ML extra is not installed.
            InvalidInputError: If the configured model type is unknown.
            ModelError: If the model fails to load.
        """
        if self._model is not None:
            return self._model

        model_type = self.config.model_type.lower()

        if model_type == "ast":
            from bioamla.ml.ast_model import ASTModel
            from bioamla.ml.base import ModelConfig

            model_config = ModelConfig(
                sample_rate=self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                overlap=self.config.overlap,
                use_fp16=self.config.use_fp16,
                device=self.config.device,
            )
            self._model = ASTModel(config=model_config)
            self._model.load(self.config.model_path, use_fp16=self.config.use_fp16)
        else:
            from bioamla.exceptions import InvalidInputError

            raise InvalidInputError(f"Unknown model type: {model_type}")

        logger.info(f"Loaded {model_type} model from {self.config.model_path}")
        return self._model

    def _get_reducer(self):
        """Get or create the dimensionality reducer.

        Raises:
            DependencyError: If the clustering extra is not installed.
        """
        if self._reducer is not None or self.config.reduce_method is None:
            return self._reducer

        try:
            from bioamla.cluster import IncrementalReducer
        except ImportError as e:
            raise DependencyError(
                "Dimensionality reduction requires the clustering extra — install bioamla[cluster]"
            ) from e

        self._reducer = IncrementalReducer(
            method=self.config.reduce_method,
            n_components=self.config.n_components,
        )
        return self._reducer

    def extract(
        self,
        audio: str | np.ndarray,
        sample_rate: int | None = None,
        layer: str | None = None,
    ) -> EmbeddingResult:
        """
        Extract embeddings from a single audio file or array.

        Args:
            audio: Audio file path or numpy array.
            sample_rate: Sample rate (required if ``audio`` is an array).
            layer: Layer to extract from (uses config default if None).

        Returns:
            An :class:`EmbeddingResult`.

        Raises:
            DependencyError: If the ML / clustering extra is missing.
            ModelError: If extraction fails.
        """
        model = self._get_model()
        layer = layer or self.config.layer

        filepath = audio if isinstance(audio, str) else None

        embeddings = model.extract_embeddings(audio, sample_rate=sample_rate, layer=layer)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        reducer = self._get_reducer()
        if reducer is not None:
            if not reducer.fitted:
                reducer.fit(embeddings)
            embeddings = reducer.transform(embeddings)

        segments = []
        if filepath:
            try:
                from bioamla.audio import get_audio_info

                duration = get_audio_info(filepath).duration
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
        audio_files: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[EmbeddingResult]:
        """
        Extract embeddings from multiple files, yielding per-file results.

        Files that fail are yielded with an empty embeddings array and an
        ``error`` entry in their metadata (the iterator does not raise).

        Args:
            audio_files: Audio file paths.
            progress_callback: Optional ``(current, total)`` callback.

        Yields:
            An :class:`EmbeddingResult` for each file.
        """
        total = len(audio_files)

        for i, filepath in enumerate(audio_files):
            try:
                yield self.extract(filepath)
            except Exception as e:
                logger.warning(f"Failed to extract embeddings from {filepath}: {e}")
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
        audio_files: list[str],
        aggregate: str = "mean",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchEmbeddingResult:
        """
        Extract embeddings from multiple files into a single stacked array.

        Args:
            audio_files: Audio file paths.
            aggregate: Segment aggregation: ``"mean"``, ``"first"``, or ``"all"``.
            progress_callback: Optional ``(current, total)`` callback.

        Returns:
            A :class:`BatchEmbeddingResult`.
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

            if aggregate == "mean":
                emb = result.mean_embedding()
            elif aggregate == "first":
                emb = result.embeddings[0]
            else:  # "all"
                emb = result.embeddings

            embeddings_list.append(emb)

        if embeddings_list:
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
        """Fit the configured dimensionality reducer on ``embeddings``."""
        reducer = self._get_reducer()
        if reducer is not None:
            reducer.fit(embeddings)
            logger.info(f"Fitted {self.config.reduce_method} reducer")


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_embeddings(
    audio: str | np.ndarray,
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    model_type: str = "ast",
    layer: str = "last_hidden_state",
    sample_rate: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract embeddings from a single audio file or array.

    For batch processing, use :class:`EmbeddingExtractor` directly.

    Args:
        audio: Audio file path or numpy array.
        model_path: Path to model or HuggingFace model ID.
        model_type: Model type (``"ast"``).
        layer: Layer to extract from.
        sample_rate: Sample rate (required if ``audio`` is an array).
        normalize: L2-normalize embeddings.

    Returns:
        Embedding array of shape ``(n_segments, embedding_dim)``.

    Raises:
        DependencyError: If the ML extra is missing.
        ModelError: If extraction fails.
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
    audio_files: list[str],
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    model_type: str = "ast",
    output_path: str | None = None,
    output_format: str = "npy",
    aggregate: str = "mean",
    normalize: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchEmbeddingResult:
    """
    Extract embeddings from multiple audio files, optionally saving them.

    Args:
        audio_files: Audio file paths.
        model_path: Path to model or HuggingFace model ID.
        model_type: Model type (``"ast"``).
        output_path: Optional path to save embeddings.
        output_format: Output format (``"npy"``, ``"parquet"``, ``"csv"``).
        aggregate: Segment aggregation (``"mean"``, ``"first"``, ``"all"``).
        normalize: L2-normalize embeddings.
        progress_callback: Optional ``(current, total)`` callback.

    Returns:
        A :class:`BatchEmbeddingResult`.
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

    if output_path and result.embeddings.size > 0:
        save_embeddings(result.embeddings, result.filepaths, output_path, format=output_format)
        result.output_path = output_path

    return result


# =============================================================================
# Save and Load Functions
# =============================================================================


def save_embeddings(
    embeddings: np.ndarray,
    filepaths: list[str],
    output_path: str,
    format: str = "npy",
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save embeddings to disk.

    Args:
        embeddings: Embedding array.
        filepaths: Source file paths.
        output_path: Output file path.
        format: ``"npy"``, ``"npz"``, ``"parquet"``, or ``"csv"``.
        metadata: Optional metadata (saved with the ``npz`` format).

    Returns:
        The path written to.

    Raises:
        DependencyError: If pandas/pyarrow are required but missing.
        InvalidInputError: If the format is unknown.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        np.save(str(output_path), embeddings)
        mapping_path = output_path.with_suffix(".files.txt")
        mapping_path.write_text("\n".join(filepaths), encoding="utf-8")
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
        except ImportError as err:
            raise DependencyError(
                "Parquet output requires pandas and pyarrow — install bioamla[ml]"
            ) from err
        df = pd.DataFrame(embeddings)
        df.columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
        df["filepath"] = filepaths
        df.to_parquet(str(output_path))
        logger.info(f"Saved embeddings to {output_path}")

    elif format == "csv":
        try:
            import pandas as pd
        except ImportError as err:
            raise DependencyError("CSV output requires pandas — install bioamla[ml]") from err
        df = pd.DataFrame(embeddings)
        df.columns = [f"dim_{i}" for i in range(embeddings.shape[1])]
        df["filepath"] = filepaths
        df.to_csv(str(output_path), index=False)
        logger.info(f"Saved embeddings to {output_path}")

    else:
        from bioamla.exceptions import InvalidInputError

        raise InvalidInputError(f"Unknown format: {format}")

    return str(output_path)


def load_embeddings(
    filepath: str,
    format: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load embeddings from disk.

    Args:
        filepath: Path to the embeddings file.
        format: File format (auto-detected from the suffix if None).

    Returns:
        ``(embeddings, filepaths)``.

    Raises:
        DependencyError: If pandas is required but missing.
        InvalidInputError: If the format is unknown.
    """
    filepath = Path(filepath)

    if format is None:
        format = filepath.suffix.lstrip(".")

    if format == "npy":
        embeddings = np.load(str(filepath))
        mapping_path = filepath.with_suffix(".files.txt")
        if mapping_path.exists():
            filepaths = [
                line.strip() for line in mapping_path.read_text(encoding="utf-8").splitlines()
            ]
        else:
            filepaths = []
        return embeddings, filepaths

    elif format == "npz":
        data = np.load(str(filepath), allow_pickle=True)
        embeddings = data["embeddings"]
        filepaths = list(data.get("filepaths", []))
        return embeddings, filepaths

    elif format in ("parquet", "csv"):
        try:
            import pandas as pd
        except ImportError as err:
            raise DependencyError(
                f"Loading {format} embeddings requires pandas — install bioamla[ml]"
            ) from err
        reader = pd.read_parquet if format == "parquet" else pd.read_csv
        df = reader(str(filepath))
        filepaths = df["filepath"].tolist() if "filepath" in df.columns else []
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        embeddings = df[dim_cols].values
        return embeddings, filepaths

    else:
        from bioamla.exceptions import InvalidInputError

        raise InvalidInputError(f"Unknown format: {format}")


def get_ast_model_info(model_path: str) -> dict[str, Any]:
    """
    Return lightweight information about an AST model from its config.

    Loads only the model config (not weights), so this is cheap.

    Args:
        model_path: Path to model or HuggingFace identifier.

    Returns:
        A dict with ``path``, ``model_type``, ``num_classes``, ``classes``,
        ``has_more_classes``, and ``hidden_size``.

    Raises:
        DependencyError: If transformers is not installed.
        ModelError: If the config cannot be loaded.
    """
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise DependencyError("Model info requires transformers — install bioamla[ml]") from e

    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception as e:
        raise ModelError(f"Failed to load model config from {model_path}: {e}") from e

    id2label = getattr(config, "id2label", {})
    classes = list(id2label.values()) if id2label else []

    return {
        "path": model_path,
        "model_type": getattr(config, "model_type", "unknown"),
        "num_classes": len(classes) if classes else getattr(config, "num_labels", 0),
        "classes": classes[:10] if classes else [],
        "has_more_classes": len(classes) > 10 if classes else False,
        "hidden_size": getattr(config, "hidden_size", None),
    }
