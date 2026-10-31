"""Shared annotation-format dispatch (raven / csv / bioamla / arbimon).

A single place to resolve an annotation file's format from its extension and to
load/save through the right reader/writer. Used by both the ``annotation`` CLI
group and the dataset-extraction routines so format handling stays consistent.
"""

import logging
from pathlib import Path
from typing import Any

from bioamla.common.constants import (  # noqa: F401  (ANNOTATION_FORMATS re-exported)
    ANNOTATION_FORMATS,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from bioamla.datasets.annotations import (
    Annotation,
    is_arbimon_csv,
    load_arbimon_annotations,
    load_bioamla_annotations,
    load_csv_annotations,
    load_raven_selection_table,
    save_arbimon_annotations,
    save_bioamla_annotations,
    save_csv_annotations,
    save_raven_selection_table,
)

logger = logging.getLogger(__name__)


def recordings_named(annotations: list[Annotation]) -> set[str]:
    """The distinct ``recording`` names the annotations carry (Arbimon rows do)."""
    return {str(r) for a in annotations if (r := a.custom_fields.get("recording"))}


def annotations_for_recording(annotations: list[Annotation], audio: str | Path) -> list[Annotation]:
    """The annotations whose ``recording`` is ``audio``.

    Matched by file name, or by name without the audio extension, ignoring case
    (``REC_01.WAV`` matches ``rec_01.wav`` and ``rec_01``). Annotations that name
    no recording are all returned unchanged.
    """
    if not recordings_named(annotations):
        return annotations
    name = Path(audio).name.casefold()
    stem = Path(audio).stem.casefold()

    def matches(rec: str) -> bool:
        rec = rec.casefold()
        if rec in (name, stem):
            return True
        rec_path = Path(rec)
        return rec_path.suffix in SUPPORTED_AUDIO_EXTENSIONS and rec_path.stem == stem

    return [a for a in annotations if matches(str(a.custom_fields.get("recording", "")))]


def detect_annotation_format(
    path: Path, explicit: str | None = None, *, read_header: bool = False
) -> str:
    """Resolve an annotation format from an explicit flag or the file extension.

    ``.txt`` -> raven selection table, ``.json`` -> bioamla format, anything
    else -> flat CSV. With ``read_header`` (for an input file), a CSV whose header
    has Arbimon's ROI columns (``species``, ``x1``, ``x2``) -> arbimon. Output
    paths are never read, so what a file already holds can't change the format a
    command writes; pass ``explicit="arbimon"`` for Arbimon output.
    """
    if explicit:
        return explicit
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "raven"
    if suffix == ".json":
        return "bioamla"
    if read_header and path.is_file() and is_arbimon_csv(path):
        return "arbimon"
    return "csv"


def load_annotations(
    path: Path, fmt: str, label_column: str | None = None
) -> tuple[list[Annotation], dict[str, Any]]:
    """Load annotations in the given format, returning ``(annotations, metadata)``.

    Only the bioamla format carries file-level metadata; the others return an
    empty metadata dict so callers can treat every format uniformly.
    """
    if fmt == "raven":
        return load_raven_selection_table(str(path), label_column=label_column), {}
    if fmt == "bioamla":
        return load_bioamla_annotations(str(path))
    if fmt == "arbimon":
        return load_arbimon_annotations(str(path)), {}
    return load_csv_annotations(str(path)), {}


def save_annotations(
    annotations: list[Annotation],
    path: Path,
    fmt: str,
    metadata: dict | None = None,
    recording: str | None = None,
) -> None:
    """Save annotations in the given format, preserving metadata for bioamla.

    ``recording`` names the recording for arbimon rows that don't carry one; it
    defaults to the ``audio_file`` in ``metadata`` (as the bioamla format holds).
    """
    if fmt == "arbimon":
        if recording is None and (metadata or {}).get("audio_file"):
            recording = Path(str(metadata["audio_file"])).name
        if recording is None and not all(a.custom_fields.get("recording") for a in annotations):
            logger.warning(
                f"{path}: some rows have no recording name; pass recording= (--recording)"
            )
        save_arbimon_annotations(annotations, str(path), recording=recording)
    elif fmt == "raven":
        save_raven_selection_table(annotations, str(path))
    elif fmt == "bioamla":
        save_bioamla_annotations(annotations, str(path), metadata=metadata)
    else:
        save_csv_annotations(annotations, str(path))
