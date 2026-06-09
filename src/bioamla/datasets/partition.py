"""Partition a dataset into train/val/test as a reproducible artifact.

Splits a dataset's ``metadata.csv`` into train/val/test, stratified by label and
grouped by source recording (so clips from one recording never leak across
splits). The split is written either as a populated ``split`` column or by
reorganizing files into ``train/val/test/<label>/`` subdirectories — the latter
is consumed natively by ``ast train`` (HF audiofolder recognizes ``val`` as the
validation split).
"""

from __future__ import annotations

import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bioamla.common.files import sanitize_filename
from bioamla.datasets._metadata import read_metadata_csv, write_metadata_csv
from bioamla.exceptions import DatasetError, NotFoundError

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def _primary_label(group_rows: list[dict]) -> str:
    """Most common non-empty label across a group's rows."""
    labels = [r.get("label", "") for r in group_rows if r.get("label")]
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def partition_dataset(
    dataset_dir: str,
    splits: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
    stratify: bool = True,
    mode: str = "subdirs",
    group_by: str | None = "source_file",
    background_label: str | None = None,
    metadata_filename: str = "metadata.csv",
    verbose: bool = True,
) -> dict[str, Any]:
    """Partition a dataset into train/val/test.

    Args:
        dataset_dir: Dataset directory containing ``metadata.csv``.
        splits: ``(train, val, test)`` fractions; must sum to 1.0 (val may be 0).
        seed: Reproducible shuffle seed.
        stratify: Balance each label across splits.
        mode: ``"subdirs"`` (reorganize into ``train/val/test/<label>/``) or
            ``"column"`` (populate the ``split`` column in place).
        group_by: Column whose value keeps rows together in one split (default
            ``source_file`` — prevents clip leakage). Falls back to per-row when
            the column is absent.
        background_label: If set, this label is partitioned as its own stratum so
            it appears in every split even when sparse.
        metadata_filename: Name of the metadata CSV.
        verbose: Log progress.

    Returns:
        Dict with ``splits`` (split->count), ``groups``, ``mode``,
        ``metadata_file``, ``dataset_dir``.

    Raises:
        NotFoundError: If the metadata CSV is missing.
        DatasetError: On bad arguments or empty metadata.
    """
    if mode not in ("subdirs", "column"):
        raise DatasetError(f"Invalid mode: {mode!r} (expected subdirs|column)")
    if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
        raise DatasetError("splits must be three fractions summing to 1.0")

    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / metadata_filename
    if not metadata_path.exists():
        raise NotFoundError(f"Metadata file not found: {metadata_path}")

    rows, _ = read_metadata_csv(metadata_path)
    if not rows:
        raise DatasetError(f"No rows in {metadata_path}")

    # Group rows so a whole group lands in one split (no leakage).
    groups: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if group_by and row.get(group_by):
            key = row[group_by]
        else:
            key = row.get("file_name") or f"__row_{idx}"
        groups[key].append(row)

    # Bucket groups into strata (by label, with background as its own stratum).
    strata: dict[str, list[str]] = defaultdict(list)
    for key, grp in groups.items():
        plabel = _primary_label(grp)
        if stratify or (background_label and plabel == background_label):
            stratum = plabel
        else:
            stratum = "__all__"
        strata[stratum].append(key)

    train_f, val_f, _test_f = splits
    assignment: dict[str, str] = {}
    for stratum, keys in strata.items():
        keys = sorted(keys)  # deterministic base order
        # Per-stratum RNG keyed by (seed, stratum) so assignment is independent
        # of dict iteration order and stable across runs.
        random.Random(f"{seed}:{stratum}").shuffle(keys)
        n = len(keys)
        n_train = min(int(round(n * train_f)), n)
        n_val = min(int(round(n * val_f)), n - n_train)
        for i, key in enumerate(keys):
            if i < n_train:
                assignment[key] = "train"
            elif i < n_train + n_val:
                assignment[key] = "val"
            else:
                assignment[key] = "test"

    split_counts: Counter = Counter()
    for key, grp in groups.items():
        split_name = assignment[key]
        for row in grp:
            row["split"] = split_name
            split_counts[split_name] += 1

    if mode == "subdirs":
        _reorganize_into_subdirs(dataset_path, groups, assignment)

    write_metadata_csv(metadata_path, rows, merge_existing=False)
    _refresh_manifest(dataset_path)

    if verbose:
        logger.info(f"Partitioned {len(rows)} rows / {len(groups)} groups: {dict(split_counts)}")

    return {
        "splits": dict(split_counts),
        "groups": len(groups),
        "mode": mode,
        "metadata_file": str(metadata_path),
        "dataset_dir": str(dataset_path),
    }


def _reorganize_into_subdirs(
    dataset_path: Path, groups: dict[str, list[dict]], assignment: dict[str, str]
) -> None:
    """Move clip files into ``<split>/<label>/`` and update each row's file_name."""
    touched_dirs: set[Path] = set()
    for key, grp in groups.items():
        split_name = assignment[key]
        for row in grp:
            old_rel = row.get("file_name", "")
            if not old_rel:
                continue
            old_path = dataset_path / old_rel
            label_dir = sanitize_filename(row.get("label") or "unknown")
            new_rel = f"{split_name}/{label_dir}/{Path(old_rel).name}"
            new_path = dataset_path / new_rel
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if old_path.exists() and old_path.resolve() != new_path.resolve():
                touched_dirs.add(old_path.parent)
                shutil.move(str(old_path), str(new_path))
            row["file_name"] = new_rel

    # Remove now-empty original label directories (but never the dataset root).
    for d in touched_dirs:
        if d != dataset_path and d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def _refresh_manifest(dataset_path: Path) -> None:
    """Rebuild dataset.json from the updated metadata, if a manifest exists."""
    manifest_path = dataset_path / "dataset.json"
    if not manifest_path.exists():
        return
    try:
        from bioamla.datasets.manifest import (
            build_manifest_from_metadata,
            load_dataset_manifest,
            save_dataset_manifest,
        )

        old = load_dataset_manifest(str(manifest_path))
        refreshed = build_manifest_from_metadata(
            str(dataset_path),
            name=old.name,
            kind="partitioned",
            created=old.created,
            sample_rate=old.sample_rate,
        )
        save_dataset_manifest(refreshed, str(manifest_path))
    except Exception as e:  # manifest refresh is best-effort
        logger.warning(f"Could not refresh manifest {manifest_path}: {e}")
