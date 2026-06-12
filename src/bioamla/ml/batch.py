"""
ML Batch Processing
===================

Batch wrappers for the ml domain built on :func:`bioamla.batch.run_batch`.
These let cut-over wire the batch CLI to the ml domain later without depending
on the old service layer.

Each wrapper discovers audio files under a directory, runs AST predict / embed
per file, and returns a :class:`bioamla.batch.BatchResult`.

Heavy deps (torch / transformers) are loaded lazily by the underlying ml
functions for fast startup.
"""

from collections.abc import Callable
from pathlib import Path

from bioamla.batch import BatchResult, discover_files, run_batch
from bioamla.exceptions import NotFoundError

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def _audio_filter(path: Path) -> bool:
    """Predicate selecting supported audio files."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def batch_predict_files(
    input_dir: str,
    model_path: str = "bioamla/scp-frogs",
    *,
    top_k: int = 5,
    min_confidence: float = 0.0,
    resample_freq: int = 16000,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Run AST prediction over every audio file in a directory.

    The model is loaded once and reused across files (sequential mode). Each
    file's structured prediction (top-k labels/scores) is collected into
    ``result.metadata["predictions"]`` so callers can write a structured
    ``predictions.json``; ``result.output_files`` holds human-readable summaries.

    Args:
        input_dir: Directory containing input audio files.
        model_path: Path to model or HuggingFace identifier.
        top_k: Number of top predictions to keep per file.
        min_confidence: Drop predictions below this probability.
        resample_freq: Target sample rate.
        recursive: Search subdirectories.
        max_workers: Worker count (kept at 1 in practice — the model is shared).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run.

    Raises:
        NotFoundError: If the input directory does not exist.
        ModelError: On model-load or inference failure.
    """
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    from bioamla.ml.inference import ASTInference

    files = discover_files(in_dir, recursive=recursive, file_filter=_audio_filter)
    inference = ASTInference(model_path=model_path, sample_rate=resample_freq)

    predictions: list = []

    def _process_one(audio_path: Path) -> str:
        pred = inference.predict_topk(str(audio_path), top_k=top_k, min_confidence=min_confidence)
        predictions.append(
            {
                "filepath": str(audio_path),
                "predicted_label": pred.predicted_label,
                "confidence": pred.confidence,
                "start_time": pred.start_time,
                "end_time": pred.end_time,
                "top_k_labels": pred.top_k_labels,
                "top_k_scores": pred.top_k_scores,
            }
        )
        return f"{audio_path}: {pred.predicted_label} ({pred.confidence:.4f})"

    result = run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )
    result.metadata = {"predictions": predictions}
    return result


def batch_predict_segments(
    input_dir: str,
    model_path: str = "bioamla/scp-frogs",
    *,
    segment_duration: int,
    overlap: int = 0,
    min_confidence: float = 0.0,
    resample_freq: int = 16000,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Run segmented AST prediction over every audio file in a directory.

    Each file is split into fixed-length (optionally overlapping) segments and
    classified per segment. The model is loaded once and reused across files
    (sequential mode). Per-segment rows (``filepath, start_time, end_time,
    predicted_label, confidence``) are collected into
    ``result.metadata["segments"]`` so callers can write a flat
    ``predictions.csv``; ``result.output_files`` holds per-file summaries.

    Args:
        input_dir: Directory containing input audio files.
        model_path: Path to model or HuggingFace identifier.
        segment_duration: Duration of each segment in seconds.
        overlap: Overlap between consecutive segments in seconds.
        min_confidence: Drop segments whose prediction is below this probability.
        resample_freq: Target sample rate.
        recursive: Search subdirectories.
        max_workers: Worker count (kept at 1 in practice — the model is shared).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run.

    Raises:
        NotFoundError: If the input directory does not exist.
        ModelError: On model-load or inference failure.
    """
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    from bioamla.ml.inference import ASTInference

    files = discover_files(in_dir, recursive=recursive, file_filter=_audio_filter)
    inference = ASTInference(model_path=model_path, sample_rate=resample_freq)

    segments: list = []

    def _process_one(audio_path: Path) -> str:
        results = inference.predict_segments(
            str(audio_path), clip_length=segment_duration, overlap=overlap
        )
        kept = 0
        for r in results:
            if r.confidence < min_confidence:
                continue
            segments.append(
                {
                    "filepath": str(audio_path),
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                }
            )
            kept += 1
        return f"{audio_path}: {kept} segment(s)"

    result = run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )
    result.metadata = {"segments": segments}
    return result


def batch_embed_files(
    input_dir: str,
    output_dir: str,
    model_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    *,
    layer: str = "last_hidden_state",
    normalize: bool = True,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """
    Extract AST embeddings for every audio file in a directory, saving one
    ``.npy`` per input file under ``output_dir``.

    The extractor (and its model) is loaded once and reused across files.

    Args:
        input_dir: Directory containing input audio files.
        output_dir: Directory to write ``<stem>_embeddings.npy`` files to.
        model_path: Path to model or HuggingFace identifier.
        layer: Layer to extract embeddings from.
        normalize: L2-normalize embeddings.
        recursive: Search subdirectories.
        max_workers: Worker count (kept at 1 — the model is shared).
        continue_on_error: Collect per-file errors and keep going if True.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        A :class:`bioamla.batch.BatchResult` summarizing the run.

    Raises:
        NotFoundError: If the input directory does not exist.
        ModelError: On model-load or inference failure.
    """
    import numpy as np

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    from bioamla.ml.embedding import EmbeddingConfig, EmbeddingExtractor

    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(in_dir, recursive=recursive, file_filter=_audio_filter)

    config = EmbeddingConfig(model_path=model_path, layer=layer, normalize=normalize)
    extractor = EmbeddingExtractor(config=config)

    def _process_one(audio_path: Path) -> str:
        result = extractor.extract(str(audio_path), layer=layer)
        out_path = out_dir / f"{audio_path.stem}_embeddings.npy"
        np.save(str(out_path), result.embeddings)
        return str(out_path)

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


__all__ = [
    "batch_predict_files",
    "batch_predict_segments",
    "batch_embed_files",
]
