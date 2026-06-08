"""Compute simple statistics for a dataset from its metadata CSV."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from bioamla.exceptions import DatasetError, NotFoundError

logger = logging.getLogger(__name__)


def get_dataset_stats(
    dataset_path: str,
    metadata_filename: str = "metadata.csv",
) -> dict[str, Any]:
    """Compute statistics for a dataset from its metadata CSV.

    Args:
        dataset_path: Path to the dataset directory.
        metadata_filename: Name of the metadata CSV file.

    Returns:
        Dict with ``total_files``, ``categories`` (label->count), ``licenses``
        (license->count), ``num_categories`` and ``num_licenses``.

    Raises:
        NotFoundError: If the metadata CSV is missing.
        DatasetError: If the metadata cannot be read.
    """
    metadata_path = Path(dataset_path) / metadata_filename
    if not metadata_path.exists():
        raise NotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except OSError as e:
        raise DatasetError(f"Failed to read dataset metadata {metadata_path}: {e}") from e

    categories: Counter = Counter()
    licenses: Counter = Counter()

    for row in rows:
        if "category" in row:
            categories[row["category"]] += 1
        if "license" in row:
            licenses[row["license"]] += 1

    return {
        "total_files": len(rows),
        "categories": dict(categories),
        "licenses": dict(licenses),
        "num_categories": len(categories),
        "num_licenses": len(licenses),
    }
