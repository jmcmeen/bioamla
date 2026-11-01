"""Operations for ``dataset``: thin adapters over :mod:`bioamla.datasets`.

See :mod:`bioamla.operations`. Reading stats is ``quick``; anything that reads or
writes a dataset's files is a ``job``. Results are the library's dicts (manifest and
build results hold a :class:`~bioamla.datasets.DatasetManifest`). The library
functions are imported where they're called, from their own modules.

File *names* in requests (``metadata_filename``, a license's ``output``) must be plain
names, so joining them to a checked folder can't leave it; paths named inside a remix
recipe go through the path check like the request's own.
"""

from __future__ import annotations

from typing import Any, Literal

from bioamla.common.config import config_dataclass
from bioamla.exceptions import InvalidInputError
from bioamla.operations import OperationContext, PathStr, operation


def _plain_name(value: str | None, field: str) -> None:
    """Refuse a file name that would leave the folder it's joined to."""
    if value is None:
        return
    if value in ("", ".", "..") or any(c in value for c in "/\\") or value.startswith("~"):
        raise InvalidInputError(f"{field} must be a plain file name, not {value!r}")


@config_dataclass
class MergeRequest:
    """Merge datasets into one.

    Attributes:
        output_dir: The merged dataset's folder.
        dataset_paths: The datasets to merge.
        metadata_filename: Each dataset's metadata CSV name.
        overwrite: Overwrite existing files instead of skipping them.
        organize: Organize files by category (else keep each dataset's layout).
        target_format: Convert all audio to this format.
        verbose: Log progress.
    """

    output_dir: PathStr
    dataset_paths: list[PathStr]
    metadata_filename: str = "metadata.csv"
    overwrite: bool = False
    organize: bool = True
    target_format: str | None = None
    verbose: bool = True


@operation("dataset.merge", request=MergeRequest, kind="job")
def merge_op(request: MergeRequest, context: OperationContext) -> dict[str, Any]:
    """Merge datasets into one."""
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.merge import merge_datasets

    return merge_datasets(
        dataset_paths=list(request.dataset_paths),
        output_dir=request.output_dir,
        metadata_filename=request.metadata_filename,
        skip_existing=not request.overwrite,
        organize_by_category=request.organize,
        target_format=request.target_format,
        verbose=request.verbose,
    )


@config_dataclass
class ExtractClipsRequest:
    """Extract annotated regions into a labelled clip dataset.

    Attributes:
        source: An annotated recording, or a folder of them.
        output_dir: The dataset folder to write.
        annotations: The annotation file when ``source`` is one recording.
        layout: ``both`` (label folders and metadata.csv), ``audiofolder`` or ``flat``.
        padding_ms: Padding around each clip (ms).
        bandpass: Band-pass each clip to its annotation's band.
        audio_format: Clip format.
        sample_rate: Resample clips to this rate.
        include: Keep only these labels.
        exclude: Drop these labels.
        min_duration: Drop clips shorter than this (s).
        source_metadata: Catalog metadata.csv to join license and attribution from.
        verbose: Log progress.
    """

    source: PathStr
    output_dir: PathStr
    annotations: PathStr | None = None
    layout: Literal["both", "audiofolder", "flat"] = "both"
    padding_ms: float = 0.0
    bandpass: bool = False
    audio_format: str = "wav"
    sample_rate: int | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    min_duration: float | None = None
    source_metadata: PathStr | None = None
    verbose: bool = True


@operation("dataset.extract-clips", request=ExtractClipsRequest, kind="job")
def extract_clips_op(request: ExtractClipsRequest, context: OperationContext) -> dict[str, Any]:
    """Extract annotated regions into a labelled clip dataset."""
    from bioamla.datasets.labeled_dataset import extract_labeled_dataset

    return extract_labeled_dataset(
        source=request.source,
        output_dir=request.output_dir,
        annotations=request.annotations,
        layout=request.layout,
        padding_ms=request.padding_ms,
        bandpass=request.bandpass,
        format=request.audio_format,
        target_sample_rate=request.sample_rate,
        include_labels=set(request.include) if request.include else None,
        exclude_labels=set(request.exclude) if request.exclude else None,
        min_duration=request.min_duration,
        source_metadata=request.source_metadata,
        verbose=request.verbose,
    )


@config_dataclass
class StatsRequest:
    """Summary statistics of a dataset's metadata CSV.

    Attributes:
        dataset_dir: The dataset folder, or its metadata CSV.
        metadata_filename: The metadata CSV's name.
    """

    dataset_dir: PathStr
    metadata_filename: str = "metadata.csv"


@operation("dataset.stats", request=StatsRequest)
def stats_op(request: StatsRequest, context: OperationContext) -> dict[str, Any]:
    """Summarize a dataset's metadata."""
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.stats import get_dataset_stats

    return get_dataset_stats(request.dataset_dir, metadata_filename=request.metadata_filename)


@config_dataclass
class ManifestRequest:
    """A ``dataset.json`` manifest from a dataset's metadata CSV.

    Attributes:
        dataset_dir: The dataset folder.
        name: Dataset name (default: the folder's name).
        kind: ``labeled`` or ``partitioned``.
        output: Where to write it (default: ``<dataset_dir>/dataset.json``).
        metadata_filename: The metadata CSV's name.
        sample_rate: Sample rate to record.
    """

    dataset_dir: PathStr
    name: str = ""
    kind: Literal["labeled", "partitioned"] = "labeled"
    output: PathStr | None = None
    metadata_filename: str = "metadata.csv"
    sample_rate: int | None = None


@operation("dataset.manifest", request=ManifestRequest)
def manifest_op(request: ManifestRequest, context: OperationContext) -> dict[str, Any]:
    """Write a dataset's manifest."""
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.workflows import write_dataset_manifest

    return write_dataset_manifest(
        request.dataset_dir,
        name=request.name,
        kind=request.kind,
        output=request.output,
        metadata_filename=request.metadata_filename,
        sample_rate=request.sample_rate,
    )


@config_dataclass
class PartitionRequest:
    """Split a dataset into train, validation and test.

    Attributes:
        dataset_dir: The dataset folder, or its metadata CSV.
        train: Train fraction.
        val: Validation fraction.
        test: Test fraction.
        seed: Shuffle seed.
        stratify: Balance labels across splits.
        mode: ``subdirs`` (move files into split folders) or ``column``.
        group_by: Keep rows sharing this column's value in one split (``""``: none).
        background_label: Partition this label as its own stratum.
        metadata_filename: The metadata CSV's name.
        verbose: Log progress.
    """

    dataset_dir: PathStr
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    stratify: bool = True
    mode: Literal["subdirs", "column"] = "subdirs"
    group_by: str = "source_file"
    background_label: str | None = None
    metadata_filename: str = "metadata.csv"
    verbose: bool = True


@operation("dataset.partition", request=PartitionRequest, kind="job")
def partition_op(request: PartitionRequest, context: OperationContext) -> dict[str, Any]:
    """Split a dataset into train, validation and test."""
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.partition import partition_dataset

    return partition_dataset(
        request.dataset_dir,
        splits=(request.train, request.val, request.test),
        seed=request.seed,
        stratify=request.stratify,
        mode=request.mode,
        group_by=request.group_by or None,
        background_label=request.background_label,
        metadata_filename=request.metadata_filename,
        verbose=request.verbose,
    )


@config_dataclass
class BuildRequest:
    """Build a training-ready dataset: extract clips, partition, write a manifest.

    Attributes:
        source: An annotated recording, or a folder of them.
        output_dir: The dataset folder to create.
        annotations: The annotation file when ``source`` is one recording.
        padding_ms: Padding around each clip (ms).
        bandpass: Band-pass each clip to its annotation's band.
        sample_rate: Resample clips to this rate.
        include: Keep only these labels.
        exclude: Drop these labels.
        min_duration: Drop clips shorter than this (s).
        train: Train fraction.
        val: Validation fraction.
        test: Test fraction.
        seed: Shuffle seed.
        partition: Split into train/val/test.
        name: Dataset name for the manifest.
        source_metadata: Catalog metadata.csv to join license and attribution from.
        attributions: Write ATTRIBUTIONS.md.
        verbose: Log progress.
    """

    source: PathStr
    output_dir: PathStr
    annotations: PathStr | None = None
    padding_ms: float = 0.0
    bandpass: bool = False
    sample_rate: int | None = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    min_duration: float | None = None
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 0
    partition: bool = True
    name: str = ""
    source_metadata: PathStr | None = None
    attributions: bool = True
    verbose: bool = True


@operation("dataset.build", request=BuildRequest, kind="job")
def build_op(request: BuildRequest, context: OperationContext) -> dict[str, Any]:
    """Build a training-ready dataset."""
    from bioamla.datasets.workflows import build_dataset

    return build_dataset(
        request.source,
        request.output_dir,
        annotations=request.annotations,
        padding_ms=request.padding_ms,
        bandpass=request.bandpass,
        sample_rate=request.sample_rate,
        include=request.include,
        exclude=request.exclude,
        min_duration=request.min_duration,
        splits=(request.train, request.val, request.test) if request.partition else None,
        seed=request.seed,
        name=request.name,
        source_metadata=request.source_metadata,
        attributions=request.attributions,
        verbose=request.verbose,
    )


@config_dataclass
class LicenseRequest:
    """A LICENSE or ATTRIBUTIONS.md from a dataset's metadata.

    Attributes:
        path: The dataset folder, or (with ``batch``) a folder of datasets.
        template: A file to put at the top.
        output: The file name.
        format: ``text`` (LICENSE) or ``md`` (ATTRIBUTIONS.md).
        metadata_filename: The metadata CSV's name.
        batch: Every dataset under ``path``.
    """

    path: PathStr
    template: PathStr | None = None
    output: str | None = None
    format: Literal["text", "md"] = "text"
    metadata_filename: str = "metadata.csv"
    batch: bool = False


@operation("dataset.license", request=LicenseRequest)
def license_op(request: LicenseRequest, context: OperationContext) -> dict[str, Any]:
    """Write a dataset's license or attributions file."""
    _plain_name(request.output, "output")
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.workflows import write_dataset_license

    return write_dataset_license(
        request.path,
        template=request.template,
        output=request.output,
        format=request.format,
        metadata_filename=request.metadata_filename,
        batch=request.batch,
    )


@config_dataclass
class AugmentRequest:
    """Augmented copies of every audio file in a folder.

    Attributes:
        input_dir: The folder.
        output_dir: Where the copies go.
        add_noise: Noise SNR range (dB), ``[min, max]``.
        time_stretch: Time-stretch rate range.
        pitch_shift: Pitch-shift range (semitones).
        gain: Gain range (dB).
        multiply: Copies per file.
        sample_rate: Output sample rate.
        recursive: Search subfolders.
        verbose: Log progress.
    """

    input_dir: PathStr
    output_dir: PathStr
    add_noise: tuple[float, float] | None = None
    time_stretch: tuple[float, float] | None = None
    pitch_shift: tuple[float, float] | None = None
    gain: tuple[float, float] | None = None
    multiply: int = 1
    sample_rate: int = 16000
    recursive: bool = True
    verbose: bool = True


@operation("dataset.augment", request=AugmentRequest, kind="job")
def augment_op(request: AugmentRequest, context: OperationContext) -> dict[str, Any]:
    """Write augmented copies of every audio file in a folder."""
    from bioamla.datasets.workflows import augment_dataset

    return augment_dataset(
        request.input_dir,
        request.output_dir,
        add_noise=request.add_noise,
        time_stretch=request.time_stretch,
        pitch_shift=request.pitch_shift,
        gain=request.gain,
        multiply=request.multiply,
        sample_rate=request.sample_rate,
        recursive=request.recursive,
        verbose=request.verbose,
    )


@config_dataclass
class RemixRequest:
    """A derived dataset from a recipe and/or selections.

    Attributes:
        output_dir: The new dataset's folder.
        recipe: A TOML recipe.
        source: Source dataset folder (overrides the recipe's).
        keep: Keep only these labels.
        drop: Drop these labels.
        name: Dataset name.
        label_column: The column holding labels.
        template: A file to put at the top of the generated LICENSE.
        metadata_filename: The metadata CSV's name.
        verbose: Log progress.
    """

    output_dir: PathStr
    recipe: PathStr | None = None
    source: PathStr | None = None
    keep: list[str] | None = None
    drop: list[str] | None = None
    name: str | None = None
    label_column: str | None = None
    template: PathStr | None = None
    metadata_filename: str = "metadata.csv"
    verbose: bool = True


@operation("dataset.remix", request=RemixRequest, kind="job")
def remix_op(request: RemixRequest, context: OperationContext) -> dict[str, Any]:
    """Build a derived dataset from a recipe and/or selections."""
    if request.recipe:
        from bioamla.datasets.remix import RemixRecipe

        recipe = RemixRecipe.from_toml(request.recipe)
        for source in [recipe.source, *(s.source for s in recipe.add + recipe.replace)]:
            if source:
                context.resolve_path(str(source))
    _plain_name(request.metadata_filename, "metadata_filename")
    from bioamla.datasets.workflows import remix_from

    return remix_from(
        request.output_dir,
        recipe=request.recipe,
        source=request.source,
        keep=request.keep,
        drop=request.drop,
        name=request.name,
        label_column=request.label_column,
        template=request.template,
        metadata_filename=request.metadata_filename,
        verbose=request.verbose,
    )
