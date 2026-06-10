"""HuggingFace Hub model and dataset publishing, dataset pulling, and cache mgmt.

Push local model/dataset folders to the HuggingFace Hub, pull a Hub audio
dataset down into the local labeled-folder + ``metadata.csv`` layout the rest of
the pipeline (partition/stats/``ast train``) consumes, and inspect/purge the
local HF cache that repeat pulls/loads populate. ``huggingface_hub`` and
``datasets`` are base dependencies, imported lazily so importing this module
stays light.

Upload/download failures raise :class:`~bioamla.exceptions.CatalogError`; a
missing path or unusable dataset raises :class:`~bioamla.exceptions.InvalidInputError`.
"""

import logging
from pathlib import Path
from typing import Any

from bioamla.catalogs._models import CachedRepo, PullResult, PurgeResult, PushResult
from bioamla.exceptions import CatalogError, InvalidInputError

logger = logging.getLogger(__name__)

# Column names to probe for a label when the dataset has no ClassLabel feature,
# in descending preference.
_LABEL_COLUMN_CANDIDATES = ("label", "category", "class", "species", "common_name", "target")


def _get_folder_size(path: str, limit: int | None = None) -> int:
    """Calculate the total size of a folder in bytes (optionally short-circuiting)."""
    total_size = 0
    for p in Path(path).glob("**/*"):
        if p.is_file():
            total_size += p.stat().st_size
            if limit is not None and total_size > limit:
                return total_size
    return total_size


def _count_files(path: str, limit: int | None = None) -> int:
    """Count the total number of files in a folder (optionally short-circuiting)."""
    count = 0
    for p in Path(path).glob("**/*"):
        if p.is_file():
            count += 1
            if limit is not None and count > limit:
                return count
    return count


def _is_large_folder(
    path: str,
    size_threshold_gb: float = 5.0,
    file_count_threshold: int = 1000,
) -> bool:
    """Return True if a folder should use ``upload_large_folder``."""
    size_threshold_bytes = int(size_threshold_gb * 1024 * 1024 * 1024)
    file_count = _count_files(path, limit=file_count_threshold)
    if file_count > file_count_threshold:
        return True
    folder_size = _get_folder_size(path, limit=size_threshold_bytes)
    return folder_size > size_threshold_bytes


def _get_hf_api():
    """Import and instantiate ``HfApi`` (lazy import)."""
    from huggingface_hub import HfApi

    return HfApi()


def _push_folder(
    path: str,
    repo_id: str,
    repo_type: str,
    private: bool,
    commit_message: str | None,
    url: str,
) -> PushResult:
    """Shared implementation for pushing a model or dataset folder."""
    if not Path(path).is_dir():
        raise InvalidInputError(f"Path '{path}' does not exist or is not a directory")

    api = _get_hf_api()
    default_message = f"Upload {repo_type}"
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        upload = api.upload_large_folder if _is_large_folder(path) else api.upload_folder
        upload(
            folder_path=path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or default_message,
        )
    except Exception as e:
        raise CatalogError(f"Push failed: {e}") from e

    return PushResult(
        repo_id=repo_id,
        repo_type=repo_type,
        url=url,
        files_uploaded=_count_files(path),
        total_size_bytes=_get_folder_size(path),
    )


def push_model(
    path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> PushResult:
    """Push a model folder to the HuggingFace Hub.

    Raises:
        InvalidInputError: if ``path`` is not a directory.
        CatalogError: on upload failure.
    """
    return _push_folder(
        path=path,
        repo_id=repo_id,
        repo_type="model",
        private=private,
        commit_message=commit_message,
        url=f"https://huggingface.co/{repo_id}",
    )


def push_dataset(
    path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str | None = None,
) -> PushResult:
    """Push a dataset folder to the HuggingFace Hub.

    Raises:
        InvalidInputError: if ``path`` is not a directory.
        CatalogError: on upload failure.
    """
    return _push_folder(
        path=path,
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        commit_message=commit_message,
        url=f"https://huggingface.co/datasets/{repo_id}",
    )


def _detect_audio_column(features: Any, audio_column: str | None) -> str:
    """Find the dataset's audio column (an ``Audio`` feature), or validate the override."""
    from datasets import Audio

    if audio_column is not None:
        if audio_column not in features:
            raise InvalidInputError(f"Audio column {audio_column!r} not found in dataset")
        return audio_column
    for name, feature in features.items():
        if isinstance(feature, Audio):
            return name
    raise InvalidInputError(
        "No audio column found in dataset (no Audio feature); pass audio_column explicitly"
    )


def _detect_label_column(features: Any, label_column: str | None) -> str:
    """Find a label column: an explicit override, a ClassLabel feature, or a known name."""
    from datasets import ClassLabel

    if label_column is not None:
        if label_column not in features:
            raise InvalidInputError(f"Label column {label_column!r} not found in dataset")
        return label_column
    for name, feature in features.items():
        if isinstance(feature, ClassLabel):
            return name
    for candidate in _LABEL_COLUMN_CANDIDATES:
        if candidate in features:
            return candidate
    raise InvalidInputError(
        "No label column found in dataset; pass label_column explicitly "
        f"(looked for a ClassLabel feature or one of {_LABEL_COLUMN_CANDIDATES})"
    )


def _extract_audio(cell: Any) -> tuple[Any, int, str | None]:
    """Return ``(mono_array, sample_rate, path)`` from a decoded audio cell.

    Supports both the legacy dict form (``{"array", "sampling_rate", "path"}``)
    and the ``torchcodec`` ``AudioDecoder`` returned by ``datasets`` >= 4.
    """
    if hasattr(cell, "get_all_samples"):
        samples = cell.get_all_samples()
        array = samples.data.numpy()
        if array.ndim == 2:
            array = array.mean(axis=0) if array.shape[0] > 1 else array[0]
        return array, int(samples.sample_rate), None
    if isinstance(cell, dict):
        return cell["array"], int(cell["sampling_rate"]), cell.get("path")
    raise InvalidInputError(f"Unrecognized audio cell type: {type(cell).__name__}")


def _label_to_str(value: Any, feature: Any) -> str:
    """Render a label cell as a folder-safe string (decoding ClassLabel ints)."""
    from datasets import ClassLabel

    if isinstance(feature, ClassLabel) and isinstance(value, int):
        return feature.int2str(value)
    return str(value)


def pull_dataset(
    repo_id: str,
    dest: str,
    *,
    split: str | None = None,
    config: str | None = None,
    audio_column: str | None = None,
    label_column: str | None = None,
    sample_rate: int | None = 16000,
    layout: str = "both",
    verbose: bool = True,
) -> PullResult:
    """Pull a HuggingFace audio dataset into the local labeled-folder layout.

    Downloads ``repo_id`` and materializes it to ``dest`` as label-named
    subdirectories (AudioFolder) and/or a ``metadata.csv`` — the same layout
    :func:`bioamla.datasets.extract_labeled_dataset` produces and that
    ``models ast train`` consumes directly.

    Args:
        repo_id: HuggingFace dataset id (e.g. ``"ashraq/esc-50"``).
        dest: Destination dataset directory (created if missing).
        split: A single split to pull (e.g. ``"train"``); all splits when omitted.
        config: Dataset config/subset name (e.g. ``"HSN"`` for ``BirdSet``).
        audio_column: Override the auto-detected audio column.
        label_column: Override the auto-detected label column.
        sample_rate: Resample clips to this rate (16000 for AST); ``None`` keeps
            the source rate.
        layout: ``"both"`` (label subdirs + metadata.csv), ``"audiofolder"``
            (label subdirs only), or ``"flat"`` (one dir + metadata.csv).
        verbose: Log progress.

    Returns:
        A :class:`~bioamla.catalogs._models.PullResult`.

    Raises:
        InvalidInputError: On bad ``layout`` or a dataset with no usable
            audio/label column.
        CatalogError: On download/materialization failure.
    """
    if layout not in ("both", "audiofolder", "flat"):
        raise InvalidInputError(f"Invalid layout: {layout!r} (expected both|audiofolder|flat)")

    from bioamla.audio import save_audio
    from bioamla.datasets._metadata import write_metadata_csv
    from bioamla.datasets.annotation_utils import create_label_map
    from datasets import Audio, Dataset, load_dataset

    try:
        loaded = load_dataset(repo_id, config, split=split)
    except Exception as e:
        raise CatalogError(f"Failed to load dataset {repo_id!r}: {e}") from e

    # Normalize to {split_name: Dataset}.
    splits: dict[str, Dataset] = {split: loaded} if isinstance(loaded, Dataset) else dict(loaded)

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    subdir_by_label = layout != "flat"
    write_csv = layout != "audiofolder"

    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    try:
        for split_name, ds in splits.items():
            audio_col = _detect_audio_column(ds.features, audio_column)
            label_col = _detect_label_column(ds.features, label_column)
            label_feature = ds.features[label_col]
            if sample_rate is not None:
                ds = ds.cast_column(audio_col, Audio(sampling_rate=sample_rate))

            written = 0
            for idx, example in enumerate(ds):
                array, sr, path = _extract_audio(example[audio_col])
                label = _label_to_str(example[label_col], label_feature)
                stem = Path(path).stem if path else f"{idx:06d}"
                fname = f"{idx:06d}_{stem}.wav"
                rel = f"{label}/{fname}" if subdir_by_label else fname
                out_path = dest_path / rel
                save_audio(str(out_path), array, sr, format="wav")
                rows.append({"file_name": rel, "label": label, "split": split_name})
                written += 1
            split_counts[split_name] = written
            if verbose:
                logger.info(f"{repo_id} [{split_name}]: wrote {written} clips")
    except (InvalidInputError, CatalogError):
        raise
    except Exception as e:
        raise CatalogError(f"Failed to materialize dataset {repo_id!r}: {e}") from e

    label_map = create_label_map([r["label"] for r in rows if r["label"]])
    for row in rows:
        row["target"] = label_map.get(row["label"], "")

    metadata_file = None
    if write_csv and rows:
        metadata_path = dest_path / "metadata.csv"
        write_metadata_csv(
            metadata_path, rows, {"file_name", "label", "target", "split"}, merge_existing=False
        )
        metadata_file = str(metadata_path)

    return PullResult(
        repo_id=repo_id,
        dest=str(dest_path),
        url=f"https://huggingface.co/datasets/{repo_id}",
        files_written=sum(split_counts.values()),
        labels=sorted(label_map, key=lambda label_value: label_map[label_value]),
        splits=split_counts,
        metadata_file=metadata_file,
    )


def scan_cache(*, models: bool = True, datasets: bool = True) -> list[CachedRepo]:
    """List repos in the local HuggingFace cache, optionally filtered by type.

    Datasets/models pulled or loaded from the Hub are cached in HF's own format
    so repeat grabs are fast; this reports what's taking up space.

    Args:
        models: Include cached models.
        datasets: Include cached datasets.

    Returns:
        Matching :class:`~bioamla.catalogs._models.CachedRepo` entries.
    """
    from huggingface_hub import scan_cache_dir

    repos: list[CachedRepo] = []
    for repo in scan_cache_dir().repos:
        if (repo.repo_type == "model" and models) or (repo.repo_type == "dataset" and datasets):
            repos.append(CachedRepo(repo.repo_id, repo.repo_type, repo.size_on_disk))
    return repos


def purge_cache(*, models: bool = True, datasets: bool = True) -> PurgeResult:
    """Delete repos from the local HuggingFace cache.

    Per-repo deletion failures are collected into ``PurgeResult.failures`` rather
    than aborting the whole purge.

    Args:
        models: Purge cached models.
        datasets: Purge cached datasets.

    Returns:
        A :class:`~bioamla.catalogs._models.PurgeResult` with the count deleted,
        bytes freed, and any per-repo failures.
    """
    import shutil

    from huggingface_hub import constants, scan_cache_dir

    cache_path = Path(constants.HF_HUB_CACHE)
    deleted = 0
    freed = 0
    failures: list[str] = []

    for repo in scan_cache_dir().repos:
        keep = (repo.repo_type == "model" and models) or (repo.repo_type == "dataset" and datasets)
        if not keep:
            continue
        try:
            for revision in repo.revisions:
                shutil.rmtree(revision.snapshot_path, ignore_errors=True)
            repo_dir = cache_path / f"{repo.repo_type}s--{repo.repo_id.replace('/', '--')}"
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
            deleted += 1
            freed += repo.size_on_disk
        except Exception as e:  # noqa: BLE001 - record per-repo failure, continue purge
            failures.append(f"{repo.repo_id}: {e}")

    return PurgeResult(deleted=deleted, freed_bytes=freed, failures=failures)
