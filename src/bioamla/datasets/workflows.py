"""Dataset workflows: the multi-step calls behind ``bioamla dataset`` commands.

Each wraps the dataset building blocks (:func:`extract_labeled_dataset`,
:func:`partition_dataset`, the manifest, license, augmentation and remix functions)
into one call with the choices a command makes: default names and paths, checks,
and chaining.

Example:
    >>> from bioamla.datasets import build_dataset
    >>> build_dataset("recordings", "dataset", sample_rate=16000)["manifest_path"]
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bioamla.exceptions import (
    BioamlaError,
    DatasetError,
    InvalidInputError,
    NotFoundError,
    OptionConflictError,
)

logger = logging.getLogger(__name__)


def write_dataset_manifest(
    dataset_dir: str | Path,
    *,
    name: str = "",
    kind: str = "labeled",
    output: str | Path | None = None,
    metadata_filename: str = "metadata.csv",
    sample_rate: int | None = None,
) -> dict[str, Any]:
    """Build a ``dataset.json`` manifest (labels, counts, splits) from ``metadata.csv``.

    Args:
        dataset_dir: The dataset folder.
        name: Dataset name (default: the folder's name).
        kind: ``labeled`` or ``partitioned``.
        output: Where to write it (default: ``<dataset_dir>/dataset.json``).
        metadata_filename: The metadata CSV's name.
        sample_rate: Sample rate to record.

    Returns:
        ``{"output", "manifest"}``.
    """
    from bioamla.datasets.manifest import build_manifest_from_metadata, save_dataset_manifest

    dataset_path = Path(dataset_dir)
    output_path = str(output) if output else str(dataset_path / "dataset.json")
    manifest = build_manifest_from_metadata(
        str(dataset_dir),
        name=name or dataset_path.name,
        kind=kind,
        created=datetime.now(timezone.utc).isoformat(),
        metadata_filename=metadata_filename,
        sample_rate=sample_rate,
    )
    save_dataset_manifest(manifest, output_path)
    return {"output": output_path, "manifest": manifest}


def build_dataset(
    source: str | Path,
    output_dir: str | Path,
    *,
    annotations: str | Path | None = None,
    padding_ms: float = 0.0,
    bandpass: bool = False,
    sample_rate: int | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    min_duration: float | None = None,
    splits: tuple[float, float, float] | None = (0.70, 0.15, 0.15),
    seed: int = 0,
    name: str = "",
    source_metadata: str | Path | None = None,
    attributions: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a training-ready dataset: extract clips, partition, write a manifest.

    Chains :func:`extract_labeled_dataset` (both layouts), :func:`partition_dataset`
    (into ``train/val/test/<label>/``, unless ``splits`` is ``None``) and the
    manifest, then writes ``ATTRIBUTIONS.md`` from the clips' joined license and
    attribution, when there is any (skipped quietly otherwise).

    Args:
        source: An annotated recording, or a folder of them.
        output_dir: The dataset folder to create.
        annotations: The annotation file when ``source`` is one recording.
        padding_ms: Padding around each clip (ms).
        bandpass: Band-pass each clip to its annotation's band.
        sample_rate: Resample clips to this rate.
        include: Keep only these labels.
        exclude: Drop these labels.
        min_duration: Drop clips shorter than this (s).
        splits: Train, validation and test fractions; ``None`` to not partition.
        seed: Shuffle seed.
        name: Dataset name for the manifest (default: the folder's name).
        source_metadata: Catalog ``metadata.csv`` to join license and attribution from.
        attributions: Write ``ATTRIBUTIONS.md``.
        verbose: Log progress.

    Returns:
        ``{"output_dir", "extract", "partition", "manifest", "manifest_path",
        "attributions"}``.
    """
    from bioamla.datasets.labeled_dataset import extract_labeled_dataset
    from bioamla.datasets.licenses import generate_license_for_dataset
    from bioamla.datasets.partition import partition_dataset

    output_path = Path(output_dir)
    extract = extract_labeled_dataset(
        source=str(source),
        output_dir=str(output_dir),
        annotations=str(annotations) if annotations else None,
        layout="both",
        padding_ms=padding_ms,
        bandpass=bandpass,
        target_sample_rate=sample_rate,
        include_labels=set(include) if include else None,
        exclude_labels=set(exclude) if exclude else None,
        min_duration=min_duration,
        source_metadata=str(source_metadata) if source_metadata else None,
        verbose=verbose,
    )
    partition = None
    if splits is not None:
        partition = partition_dataset(
            str(output_dir), splits=splits, seed=seed, mode="subdirs", verbose=verbose
        )
    written = write_dataset_manifest(
        output_dir,
        name=name,
        kind="labeled" if splits is None else "partitioned",
        sample_rate=sample_rate,
    )
    attributions_path = None
    if attributions:
        try:
            attributions_path = generate_license_for_dataset(output_path, format="md")[
                "output_path"
            ]
        except BioamlaError as e:  # usually: no provenance to attribute
            logger.info("No ATTRIBUTIONS.md written for %s: %s", output_dir, e)
            attributions_path = None
    return {
        "output_dir": str(output_dir),
        "extract": extract,
        "partition": partition,
        "manifest": written["manifest"],
        "manifest_path": written["output"],
        "attributions": attributions_path,
    }


def write_dataset_license(
    path: str | Path,
    *,
    template: str | Path | None = None,
    output: str | None = None,
    format: str = "text",
    metadata_filename: str = "metadata.csv",
    batch: bool = False,
) -> dict[str, Any]:
    """Write a LICENSE (``text``) or ATTRIBUTIONS.md (``md``) from a dataset's metadata.

    With ``batch``, every subfolder of ``path`` holding a metadata CSV gets one; the
    result lists each dataset's outcome, and ``datasets_failed`` counts failures.

    Args:
        path: The dataset folder, or (with ``batch``) the folder of datasets.
        template: A file to put at the top of each license file.
        output: The file name (default ``LICENSE`` or ``ATTRIBUTIONS.md``).
        format: ``text`` or ``md``.
        metadata_filename: The metadata CSV's name.
        batch: Process every dataset under ``path``.

    Raises:
        NotFoundError: If the template, folder or metadata CSV is missing.
        DatasetError: With ``batch``, if no dataset was found.
    """
    from bioamla.datasets.licenses import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    folder = Path(path)
    template_path = Path(template) if template else None
    if template_path and not template_path.exists():
        raise NotFoundError(f"Template file '{template}' not found.")
    if not folder.is_dir():
        raise NotFoundError(f"Path '{path}' is not a directory.")
    if batch:
        stats = generate_licenses_for_directory(
            audio_dir=folder,
            template_path=template_path,
            output_filename=output,
            metadata_filename=metadata_filename,
            format=format,
        )
        if stats["datasets_found"] == 0:
            raise DatasetError("No datasets found (no directories with metadata.csv)")
        return stats
    csv_path = folder / metadata_filename
    if not csv_path.exists():
        raise NotFoundError(f"Metadata file '{csv_path}' not found.")
    return generate_license_for_dataset(
        dataset_path=folder,
        template_path=template_path,
        output_filename=output,
        metadata_filename=metadata_filename,
        format=format,
    )


def augment_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    add_noise: tuple[float, float] | None = None,
    time_stretch: tuple[float, float] | None = None,
    pitch_shift: tuple[float, float] | None = None,
    gain: tuple[float, float] | None = None,
    multiply: int = 1,
    sample_rate: int = 16000,
    recursive: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Write augmented copies of every audio file in a folder.

    Each transform given as a ``(min, max)`` range is applied with a value drawn from
    it, per copy.

    Args:
        input_dir: The folder of audio files.
        output_dir: Where the copies go.
        add_noise: Noise SNR range (dB).
        time_stretch: Time-stretch rate range.
        pitch_shift: Pitch-shift range (semitones).
        gain: Gain range (dB).
        multiply: Copies per file.
        sample_rate: Output sample rate (Hz).
        recursive: Search subfolders.
        verbose: Log progress.

    Raises:
        InvalidInputError: If no transform is given.
    """
    from bioamla.datasets.augmentation import WaveformAugmentationConfig, batch_augment

    if not any(r is not None for r in (add_noise, time_stretch, pitch_shift, gain)):
        raise InvalidInputError(
            "At least one augmentation must be given (add_noise, time_stretch, pitch_shift, gain)"
        )
    noise = add_noise or (3.0, 30.0)
    stretch = time_stretch or (0.8, 1.2)
    pitch = pitch_shift or (-2.0, 2.0)
    level = gain or (-12.0, 12.0)
    config = WaveformAugmentationConfig(
        sample_rate=sample_rate,
        multiply=multiply,
        add_noise=add_noise is not None,
        noise_min_snr=noise[0],
        noise_max_snr=noise[1],
        time_stretch=time_stretch is not None,
        time_stretch_min=stretch[0],
        time_stretch_max=stretch[1],
        pitch_shift=pitch_shift is not None,
        pitch_shift_min=pitch[0],
        pitch_shift_max=pitch[1],
        gain=gain is not None,
        gain_min_db=level[0],
        gain_max_db=level[1],
    )
    return batch_augment(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        config=config,
        recursive=recursive,
        verbose=verbose,
    )


def remix_from(
    output_dir: str | Path,
    *,
    recipe: str | Path | None = None,
    source: str | Path | None = None,
    keep: list[str] | None = None,
    drop: list[str] | None = None,
    name: str | None = None,
    label_column: str | None = None,
    template: str | Path | None = None,
    metadata_filename: str = "metadata.csv",
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a derived dataset from a TOML recipe and/or selections (see
    :func:`remix_dataset`).

    The arguments override the recipe's matching fields; ``keep`` or ``drop``
    replaces the recipe's own selection.

    Args:
        output_dir: The new dataset's folder.
        recipe: A TOML recipe (``[remix]`` plus ``[[add]]`` / ``[[replace]]``).
        source: Source dataset folder.
        keep: Keep only these labels.
        drop: Drop these labels.
        name: Dataset name.
        label_column: The column holding labels.
        template: A file to put at the top of the generated LICENSE.
        metadata_filename: The metadata CSV's name.
        verbose: Log progress.

    Raises:
        OptionConflictError: If both ``keep`` and ``drop`` are given, or neither
            ``recipe`` nor ``source``.
    """
    from bioamla.datasets.remix import RemixRecipe, remix_dataset

    if keep and drop:
        raise OptionConflictError("Give keep (--keep) or drop (--drop), not both")
    if not recipe and not source:
        raise OptionConflictError("Give a recipe (--recipe) and/or a source (--source)")
    overrides: dict[str, Any] = {}
    if source:
        overrides["source"] = str(source)
    if keep:
        overrides.update(keep=list(keep), drop=None)
    if drop:
        overrides.update(drop=list(drop), keep=None)
    if name:
        overrides["name"] = name
    if label_column:
        overrides["label_column"] = label_column
    spec = (
        replace(RemixRecipe.from_toml(str(recipe)), **overrides)
        if recipe
        else RemixRecipe(**overrides)
    )
    return remix_dataset(
        spec,
        str(output_dir),
        metadata_filename=metadata_filename,
        license_template=str(template) if template else None,
        verbose=verbose,
    )


__all__ = [
    "augment_dataset",
    "build_dataset",
    "remix_from",
    "write_dataset_license",
    "write_dataset_manifest",
]
