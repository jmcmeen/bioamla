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

from pathlib import Path

from bioamla.batch import (
    BatchConfig,
    BatchResult,
    config_csv_context,
    discover_files,
    output_stems,
    require_input_dir,
    require_output_dir,
    run_batch,
    run_csv_merge,
    run_csv_predict_segments,
    run_csv_process,
    write_rows_csv,
)
from bioamla.exceptions import NotFoundError
from bioamla.protocols import OnProgress

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
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Run AST prediction over every audio file in a directory.

    The model is loaded once and reused across files (sequential mode). Each
    file's structured prediction (with its ranked labels, ``top`` and ``margin``, as
    :func:`bioamla.ml.classify_file` reports them) is collected into
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
                **_ranked(pred),
            }
        )
        return f"{audio_path}: {pred.predicted_label} ({pred.confidence:.4f})"

    result = run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="predict",
    )
    result.metadata = {"predictions": predictions}
    return result


def batch_predict_segments(
    input_dir: str,
    model_path: str = "bioamla/scp-frogs",
    *,
    segment_duration: int,
    overlap: int = 0,
    top_k: int = 0,
    min_confidence: float = 0.0,
    resample_freq: int = 16000,
    recursive: bool = True,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: OnProgress | None = None,
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
        top_k: Also rank each segment's ``top_k`` most likely labels (``margin`` and
            ``top`` columns, as ``models ast predict --top-k`` writes them).
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
            str(audio_path), clip_length=segment_duration, overlap=overlap, top_k=top_k
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
                    **(r.ranked_cells() if top_k else {}),
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
        operation="predict-segments",
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
    on_progress: OnProgress | None = None,
) -> BatchResult:
    """
    Extract AST embeddings for every audio file in a directory, saving one
    ``.npy`` per input file under ``output_dir``, mirroring the input tree
    (``<stem>_embeddings.npy``; files that would share a name get distinct ones,
    see :func:`bioamla.batch.output_stems`).

    The extractor (and its model) is loaded once and reused across files.

    Args:
        input_dir: Directory containing input audio files.
        output_dir: Directory to write the ``<stem>_embeddings.npy`` files to.
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

    # Mirror the input tree; files that share a name get distinct outputs.
    stems = output_stems([str(f) for f in files], in_dir)

    def _process_one(audio_path: Path) -> str:
        result = extractor.extract(str(audio_path), layer=layer)
        stem = stems[str(audio_path)]
        out_path = out_dir / stem.parent / f"{stem.name}_embeddings.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), result.embeddings)
        return str(out_path)

    return run_batch(
        files,
        _process_one,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="embed",
    )


__all__ = [
    "batch_predict_files",
    "batch_predict_segments",
    "batch_embed_files",
    "batch_predict",
    "batch_embed",
]


_SEGMENT_COLUMNS = ["filepath", "start_time", "end_time", "predicted_label", "confidence"]


def _ranked(pred: object) -> dict[str, object]:
    """A prediction's ranked labels as JSON reports them (``top``, ``margin``), as
    :func:`bioamla.ml.classify_file` does; nothing when it wasn't ranked."""
    if pred.top_k_labels is None:  # type: ignore[attr-defined]
        return {}
    return {"top": pred.top(), "margin": pred.margin}  # type: ignore[attr-defined]


def _write_json(path: Path, data: object) -> str:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def batch_predict(
    config: BatchConfig,
    *,
    model_path: str,
    top_k: int = 5,
    min_confidence: float = 0.0,
    segment_duration: int = 0,
    overlap: int = 0,
    on_progress: OnProgress | None = None,
    operation: str = "batch models predict",
) -> BatchResult:
    """Classify every file of a batch with an AST model, whole or in segments.

    Whole files: predictions go to ``predictions.json``, in CSV order in CSV mode.
    With ``segment_duration`` (seconds) each file is classified per segment: directory
    mode writes a flat ``predictions.csv`` (one row per segment, those below
    ``min_confidence`` left out); CSV mode replaces each row with one row per segment.
    Files go to ``config.output_dir``, or, without one, beside the input (the input
    folder, or the metadata CSV's folder). The file written, if any, is
    ``metadata["output_file"]``.

    Args:
        config: The batch input and output.
        model_path: AST model (Hugging Face ID or local path).
        top_k: Rank each prediction's ``top_k`` most likely labels: ``top`` (``[{label,
            confidence}]``) and ``margin`` (top-1 minus top-2) in ``predictions.json``;
            ``margin`` and ``top`` (``label:confidence;...``) columns per segment. 0
            leaves them out.
        min_confidence: Leave out predictions below this confidence.
        segment_duration: Classify N-second segments (0: whole files).
        overlap: Overlap between segments (s).
        on_progress: ``(done, total)`` callback.
        operation: Name used in the batch log records.
    """
    if config.output_dir:
        out_dir = Path(config.output_dir)
    elif config.input_file:
        out_dir = Path(config.input_file).parent
    else:
        out_dir = Path(require_input_dir(config))
    if segment_duration > 0:
        if config.input_file:
            from bioamla.ml.inference import ASTInference

            context = config_csv_context(config)  # read (and check) the CSV first
            inference = ASTInference(model_path=model_path)

            def _predict_segments(path: Path) -> list[dict]:
                results = inference.predict_segments(
                    str(path), clip_length=segment_duration, overlap=overlap, top_k=top_k
                )
                return [
                    {
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                        "prediction": r.predicted_label,
                        "confidence": r.confidence,
                        **(r.ranked_cells() if top_k else {}),
                    }
                    for r in results
                    if r.confidence >= min_confidence
                ]

            return run_csv_predict_segments(
                context,
                _predict_segments,
                max_workers=config.max_workers,
                continue_on_error=config.continue_on_error,
                on_progress=on_progress,
                operation=operation,
            )
        result = batch_predict_segments(
            require_input_dir(config),
            model_path=model_path,
            segment_duration=segment_duration,
            overlap=overlap,
            top_k=top_k,
            min_confidence=min_confidence,
            recursive=config.recursive,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
        )
        segments = result.metadata.get("segments", [])
        if segments:
            out = out_dir / "predictions.csv"
            columns = _SEGMENT_COLUMNS + (["margin", "top"] if top_k else [])
            write_rows_csv(out, segments, columns)
            result.metadata["output_file"] = str(out)
        return result

    if config.input_file:
        from bioamla.ml.inference import ASTInference

        context = config_csv_context(config)
        inference = ASTInference(model_path=model_path)
        by_path: dict[str, dict] = {}

        def _process(path: Path) -> None:
            pred = inference.predict_topk(str(path), top_k=top_k, min_confidence=min_confidence)
            by_path[str(path)] = {
                "filepath": str(path),
                "predicted_label": pred.predicted_label,
                "confidence": pred.confidence,
                **_ranked(pred),
            }

        result = run_csv_process(
            context,
            _process,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
        # In CSV order, whatever order the workers finished in.
        predictions = [by_path[p] for p in (str(r.file_path) for r in context.rows) if p in by_path]
    else:
        result = batch_predict_files(
            require_input_dir(config),
            model_path=model_path,
            top_k=top_k,
            min_confidence=min_confidence,
            recursive=config.recursive,
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
        )
        predictions = result.metadata.get("predictions", [])
    if predictions:
        result.metadata["output_file"] = _write_json(out_dir / "predictions.json", predictions)
    return result


def batch_embed(
    config: BatchConfig,
    *,
    model_path: str,
    on_progress: OnProgress | None = None,
    operation: str = "batch models embed",
) -> BatchResult:
    """Save an AST embedding (``<name>_embeddings.npy``) for every file of a batch.

    Embeddings mirror the input (folder or the CSV's folder) under ``config.output_dir``
    (required); files that share a name get distinct embeddings. CSV mode adds each
    row's ``embedding_file`` (blank for a file that failed).

    Raises:
        InvalidInputError: If ``config`` has no ``output_dir``.
    """
    out_dir = Path(require_output_dir(config))
    if config.input_file:
        import numpy as np

        from bioamla.ml.embedding import EmbeddingConfig, EmbeddingExtractor

        context = config_csv_context(config)  # read (and check) the CSV before the model
        extractor = EmbeddingExtractor(config=EmbeddingConfig(model_path=model_path))
        stems = output_stems([str(row.file_path) for row in context.rows], context.csv_dir)

        def _analyze(path: Path) -> dict:
            stem = stems[str(path)]
            target = out_dir / stem.parent / f"{stem.name}_embeddings.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(target), extractor.extract(str(path)).embeddings)
            # A Path: written relative to wherever the metadata CSV is saved.
            return {"embedding_file": target.resolve()}

        return run_csv_merge(
            context,
            _analyze,
            clear=("embedding_file",),
            max_workers=config.max_workers,
            continue_on_error=config.continue_on_error,
            on_progress=on_progress,
            operation=operation,
        )
    return batch_embed_files(
        require_input_dir(config),
        str(out_dir),
        model_path=model_path,
        recursive=config.recursive,
        max_workers=config.max_workers,
        continue_on_error=config.continue_on_error,
        on_progress=on_progress,
    )
