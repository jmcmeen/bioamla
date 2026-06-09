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
from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv
from bioamla.datasets.annotation_utils import create_label_map, filter_labels
from bioamla.datasets.annotations import Annotation
from bioamla.datasets.clip_extraction import extract_audio_clips
from bioamla.exceptions import AnnotationError, NotFoundError

logger = logging.getLogger(__name__)

# Annotation file extensions we recognize when pairing with audio in a directory.
_ANNOTATION_SUFFIXES = (".json", ".txt", ".csv")

# Provenance/license columns copied from a source catalog metadata.csv onto each
# clip so a finished dataset stays traceable to its original recordings.
PROVENANCE_FIELDS = [
    "source",
    "license",
    "attribution",
    "scientific_name",
    "common_name",
    "attr_id",
    "attr_lic",
    "attr_url",
    "attr_note",
]

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


def _resolve_source_metadata(source: str, source_metadata: str | None) -> Path | None:
    """Locate the catalog ``metadata.csv`` to join license/attribution from.

    Precedence: an explicit ``source_metadata`` path (must exist), else a
    ``metadata.csv`` sibling of the source — in the source directory itself, or
    in the parent directory of a single audio file. Returns ``None`` when nothing
    is found, making the join a silent no-op.
    """
    if source_metadata:
        path = Path(source_metadata)
        if not path.exists():
            raise NotFoundError(f"Source metadata not found: {source_metadata}")
        return path

    source_path = Path(source)
    candidate = (
        source_path / "metadata.csv"
        if source_path.is_dir()
        else source_path.parent / "metadata.csv"
    )
    return candidate if candidate.exists() else None


def _build_provenance_index(
    metadata_path: Path,
) -> tuple[dict[str, dict[str, str]], set[str]]:
    """Index a catalog ``metadata.csv`` by recording basename for the clip join.

    Returns ``(index, present_keys)`` where ``index`` maps a recording's file
    basename to the subset of :data:`PROVENANCE_FIELDS` it populates, and
    ``present_keys`` is the union of provenance columns seen across all rows
    (used to extend the written clip-metadata header). Basenames that collide
    with *differing* provenance are dropped from the index so an ambiguous clip
    gets blanks rather than a wrong attribution.
    """
    rows, _ = read_metadata_csv(metadata_path)
    index: dict[str, dict[str, str]] = {}
    ambiguous: set[str] = set()
    present_keys: set[str] = set()

    for row in rows:
        file_name = row.get("file_name", "")
        if not file_name:
            continue
        basename = Path(file_name).name
        provenance = {k: row[k] for k in PROVENANCE_FIELDS if row.get(k)}
        if not provenance:
            continue
        present_keys.update(provenance.keys())

        if basename in ambiguous:
            continue
        if basename in index and index[basename] != provenance:
            del index[basename]
            ambiguous.add(basename)
            continue
        index[basename] = provenance

    return index, present_keys


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
    source_metadata: str | None = None,
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
        source_metadata: Catalog ``metadata.csv`` to join license/attribution
            onto clips (by source-recording basename). Auto-detected as a
            ``metadata.csv`` sibling of ``source`` when omitted.
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
        ``label_map``, ``output_dir``, ``metadata_file`` (or None), ``failed``,
        ``skipped``, and ``provenance`` (join summary: joined/matched/unmatched/
        source_metadata/columns).

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

    # First pass: load + filter annotations per file.
    per_file: list[tuple[Path, list[Annotation]]] = []
    for audio_file, ann_file in pairs:
        anns = _load_pair_annotations(ann_file)
        anns = filter_labels(anns, include_labels=include_labels, exclude_labels=exclude_labels)
        if min_duration is not None:
            anns = [a for a in anns if a.duration >= min_duration]
        if anns:
            per_file.append((audio_file, anns))

    clip_rows: list[dict[str, Any]] = []
    clips_written = 0
    failed: list[str] = []
    skipped: list[str] = []

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
        skipped.extend(result.get("skipped_clips", []))
        for rec in result["clips"]:
            rec["split"] = ""
            clip_rows.append(rec)
        if verbose:
            logger.info(f"{audio_file.name}: {len(result['clips'])} clips")

    # Build the label map from clips actually written, so a label whose only
    # annotations were out of range never becomes an empty class.
    label_map = create_label_map([r["label"] for r in clip_rows if r["label"]])
    for rec in clip_rows:
        rec["target"] = label_map.get(rec["label"], "")

    # Join license/attribution from the source catalog metadata so the dataset
    # stays traceable to its original recordings.
    fieldnames = set(CLIP_METADATA_FIELDS)
    provenance = {
        "joined": False,
        "matched": 0,
        "unmatched": 0,
        "source_metadata": None,
        "columns": [],
    }
    meta_path = _resolve_source_metadata(source, source_metadata)
    if meta_path is not None and clip_rows:
        index, present_keys = _build_provenance_index(meta_path)
        fieldnames |= present_keys
        matched = 0
        for rec in clip_rows:
            prov = index.get(rec["source_file"])
            if prov:
                rec.update(prov)
                matched += 1
        provenance = {
            "joined": True,
            "matched": matched,
            "unmatched": len(clip_rows) - matched,
            "source_metadata": str(meta_path),
            "columns": sorted(present_keys),
        }
        if verbose:
            logger.info(
                f"Provenance: joined {matched}/{len(clip_rows)} clips from {meta_path.name}"
            )

    metadata_file = None
    if write_csv and clip_rows:
        metadata_path = output_path / metadata_filename
        write_metadata_csv(metadata_path, clip_rows, fieldnames, merge_existing=False)
        metadata_file = str(metadata_path)

    return {
        "clips_written": clips_written,
        "files_processed": len(per_file),
        "labels": sorted(label_map, key=lambda label_value: label_map[label_value]),
        "label_map": label_map,
        "output_dir": str(output_path),
        "metadata_file": metadata_file,
        "failed": failed,
        "skipped": skipped,
        "provenance": provenance,
    }
