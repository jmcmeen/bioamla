"""Dataset-level manifest (``dataset.json``).

A manifest records the things a ``metadata.csv`` doesn't capture on its own: the
label vocabulary (``label2id``/``id2label``, ordered to match what AST training
produces), per-class and per-split counts, provenance of the source collections,
and the sample rate. It is always *derived from* the metadata CSV and never
required by the merge/stats/train flows — purely additive.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bioamla.datasets._metadata import read_metadata_csv
from bioamla.datasets.annotation_utils import create_label_map
from bioamla.exceptions import DatasetError, NotFoundError

logger = logging.getLogger(__name__)

# Versioned format identifier written into every manifest.
BIOAMLA_DATASET_FORMAT = "bioamla-dataset/1"


@dataclass
class DatasetManifest:
    """Self-describing summary of a dataset directory."""

    format: str = BIOAMLA_DATASET_FORMAT
    name: str = ""
    created: str = ""
    kind: str = "labeled"  # "labeled" (clips) | "partitioned" (whole files)
    label2id: dict[str, int] = field(default_factory=dict)
    id2label: dict[int, str] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    sample_rate: int | None = None
    metadata_file: str = "metadata.csv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        filtered = {k: v for k, v in data.items() if k in known}
        # JSON object keys are strings; restore id2label's int keys.
        if "id2label" in filtered:
            filtered["id2label"] = {int(k): v for k, v in filtered["id2label"].items()}
        return cls(**filtered)


def build_manifest_from_metadata(
    dataset_dir: str,
    name: str = "",
    kind: str = "labeled",
    created: str = "",
    metadata_filename: str = "metadata.csv",
    sample_rate: int | None = None,
) -> DatasetManifest:
    """Derive a :class:`DatasetManifest` from a dataset's ``metadata.csv``.

    The label vocabulary is built with :func:`create_label_map` so it matches the
    ordering AST training assigns (sorted, unique), keeping a manifest consistent
    with a trained model's ``config.label2id``.

    Raises:
        NotFoundError: If the metadata CSV is missing.
    """
    metadata_path = Path(dataset_dir) / metadata_filename
    if not metadata_path.exists():
        raise NotFoundError(f"Metadata file not found: {metadata_path}")

    rows, _ = read_metadata_csv(metadata_path)

    labels = [(r.get("label") or r.get("category") or "") for r in rows]
    labels = [label_value for label_value in labels if label_value]
    label2id = create_label_map(sorted(set(labels)))
    id2label = {idx: label_value for label_value, idx in label2id.items()}

    class_counts: dict[str, int] = {}
    splits: dict[str, int] = {}
    sources_counter: dict[str, int] = {}
    for row in rows:
        label_value = row.get("label") or row.get("category")
        if label_value:
            class_counts[label_value] = class_counts.get(label_value, 0) + 1
        split_value = row.get("split")
        if split_value:
            splits[split_value] = splits.get(split_value, 0) + 1
        source_value = row.get("source")
        if source_value:
            sources_counter[source_value] = sources_counter.get(source_value, 0) + 1

    sources = [{"source": s, "files": n} for s, n in sorted(sources_counter.items())]

    return DatasetManifest(
        name=name,
        created=created,
        kind=kind,
        label2id=label2id,
        id2label=id2label,
        class_counts=class_counts,
        splits=splits,
        sources=sources,
        sample_rate=sample_rate,
        metadata_file=metadata_filename,
    )


def save_dataset_manifest(manifest: DatasetManifest, filepath: str) -> str:
    """Write a manifest to a JSON file.

    Raises:
        DatasetError: If the file cannot be written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str), encoding="utf-8")
    except OSError as e:
        raise DatasetError(f"Failed to write dataset manifest {filepath}: {e}") from e

    logger.info(f"Saved dataset manifest to {filepath}")
    return str(path)


def load_dataset_manifest(filepath: str) -> DatasetManifest:
    """Load a manifest from a JSON file.

    Raises:
        NotFoundError: If the file doesn't exist.
        DatasetError: If the file cannot be parsed.
    """
    path = Path(filepath)
    if not path.exists():
        raise NotFoundError(f"Dataset manifest not found: {filepath}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DatasetError(f"Failed to read dataset manifest {filepath}: {e}") from e

    return DatasetManifest.from_dict(data)
