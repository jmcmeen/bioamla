"""Extract annotated regions into a training-ready labeled dataset of clips.

Bridges annotations to ML training: given audio + its annotations (a single
audio/annotation pair, or a directory of sibling pairs), cut each annotated
region into a clip and write a dataset whose layout is directly consumable by
``ast train`` — label-named subdirectories (AudioFolder) and/or a flat directory,
plus a ``metadata.csv`` carrying per-clip provenance (source recording, time
bounds, frequency band, confidence, split).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bioamla.audio import get_audio_files
from bioamla.datasets._io import detect_annotation_format, load_annotations
from bioamla.datasets._metadata import write_metadata_csv
from bioamla.datasets.annotation_utils import create_label_map, filter_labels
from bioamla.datasets.annotations import Annotation
from bioamla.datasets.clip_extraction import extract_audio_clips
from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)

# Annotation file extensions we recognize when pairing with audio in a directory.
_ANNOTATION_SUFFIXES = (".json", ".txt", ".csv")

# metadata.csv columns written for each extracted clip.
CLIP_METADATA_FIELDS = [
    "file_name",
    "label",
    "target",
    "split",
    "source_file",
    "start_time",
    "end_time",
    "low_freq",
    "high_freq",
    "confidence",
    "channel",
    "sample_rate",
    "duration",
]


def _find_sibling_annotation(audio_file: Path) -> Path | None:
    """Find an annotation file sharing the audio file's stem (``.json``/``.txt``/``.csv``)."""
    for suffix in _ANNOTATION_SUFFIXES:
        candidate = audio_file.with_suffix(suffix)
        if candidate.exists() and candidate != audio_file:
            return candidate
    return None


def _resolve_pairs(source: str, annotations: str | None) -> list[tuple[Path, Path]]:
    """Resolve ``source`` into a list of ``(audio_file, annotation_file)`` pairs."""
    source_path = Path(source)
    if not source_path.exists():
        raise NotFoundError(f"Source not found: {source}")

    if source_path.is_dir():
        pairs: list[tuple[Path, Path]] = []
        for audio_file in sorted(get_audio_files(str(source_path))):
            ann_file = _find_sibling_annotation(Path(audio_file))
            if ann_file is not None:
                pairs.append((Path(audio_file), ann_file))
            else:
                logger.warning(f"No annotation file found for {audio_file}, skipping")
        if not pairs:
            raise AnnotationError(f"No audio/annotation pairs found under {source}")
        return pairs

    # Single audio file: annotation must be given or a sibling must exist.
    ann_file = Path(annotations) if annotations else _find_sibling_annotation(source_path)
    if ann_file is None or not ann_file.exists():
        raise AnnotationError(
            f"No annotation file for {source} (pass --annotations or place a sibling file)"
        )
    return [(source_path, ann_file)]


def _load_pair_annotations(ann_file: Path) -> list[Annotation]:
    fmt = detect_annotation_format(ann_file)
    anns, _ = load_annotations(ann_file, fmt)
    return anns


def extract_labeled_dataset(
    source: str,
    output_dir: str,
    annotations: str | None = None,
    layout: str = "both",
    padding_ms: float = 0.0,
    bandpass: bool = False,
    format: str = "wav",
    target_sample_rate: int | None = None,
    include_labels: set[str] | None = None,
    exclude_labels: set[str] | None = None,
    min_duration: float | None = None,
    metadata_filename: str = "metadata.csv",
    verbose: bool = True,
) -> dict[str, Any]:
    """Extract annotated regions into a labeled clip dataset.

    Args:
        source: An audio file, or a directory of audio files each paired with a
            sibling annotation file (same stem, ``.json``/``.txt``/``.csv``).
        output_dir: Destination dataset directory (created if missing).
        annotations: Explicit annotation file when ``source`` is one audio file.
        layout: ``"both"`` (label subdirs + metadata.csv), ``"audiofolder"``
            (label subdirs only), or ``"flat"`` (one dir + metadata.csv).
        padding_ms: Padding added before/after each clip.
        bandpass: Bandpass-filter clips to each annotation's freq band when set.
        format: Output audio extension.
        target_sample_rate: Resample clips to this rate (e.g. 16000 for AST).
        include_labels: If set, keep only these labels.
        exclude_labels: If set, drop these labels.
        min_duration: Drop annotations shorter than this (seconds).
        metadata_filename: Name of the metadata CSV written to ``output_dir``.
        verbose: Log progress.

    Returns:
        Dict with ``clips_written``, ``files_processed``, ``labels`` (sorted),
        ``label_map``, ``output_dir``, ``metadata_file`` (or None), and
        ``failed``.

    Raises:
        NotFoundError: If the source path doesn't exist.
        AnnotationError: If no usable audio/annotation pairs are found.
    """
    if layout not in ("both", "audiofolder", "flat"):
        raise AnnotationError(f"Invalid layout: {layout!r} (expected both|audiofolder|flat)")

    pairs = _resolve_pairs(source, annotations)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subdir_by_label = layout != "flat"
    write_csv = layout != "audiofolder"

    # First pass: load + filter all annotations so the label map spans the whole
    # dataset (consistent target indices across source files).
    per_file: list[tuple[Path, list[Annotation]]] = []
    all_labels: list[str] = []
    for audio_file, ann_file in pairs:
        anns = _load_pair_annotations(ann_file)
        anns = filter_labels(anns, include_labels=include_labels, exclude_labels=exclude_labels)
        if min_duration is not None:
            anns = [a for a in anns if a.duration >= min_duration]
        if anns:
            per_file.append((audio_file, anns))
            all_labels.extend(a.label for a in anns if a.label)

    label_map = create_label_map(all_labels)

    clip_rows: list[dict[str, Any]] = []
    clips_written = 0
    failed: list[str] = []

    for audio_file, anns in per_file:
        result = extract_audio_clips(
            anns,
            str(audio_file),
            str(output_path),
            padding_ms=padding_ms,
            format=format,
            subdir_by_label=subdir_by_label,
            bandpass=bandpass,
            target_sample_rate=target_sample_rate,
        )
        clips_written += len(result["clips"])
        failed.extend(result["failed_clips"])
        for rec in result["clips"]:
            rec["target"] = label_map.get(rec["label"], "")
            rec["split"] = ""
            clip_rows.append(rec)
        if verbose:
            logger.info(f"{audio_file.name}: {len(result['clips'])} clips")

    metadata_file = None
    if write_csv and clip_rows:
        metadata_path = output_path / metadata_filename
        write_metadata_csv(metadata_path, clip_rows, set(CLIP_METADATA_FIELDS), merge_existing=False)
        metadata_file = str(metadata_path)

    return {
        "clips_written": clips_written,
        "files_processed": len(per_file),
        "labels": sorted(label_map, key=lambda label_value: label_map[label_value]),
        "label_map": label_map,
        "output_dir": str(output_path),
        "metadata_file": metadata_file,
        "failed": failed,
    }
