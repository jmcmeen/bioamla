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
    # Recorder metadata summary, present only when metadata.csv carries the
    # recorder columns (see bioamla.datasets.add_recording_metadata).
    recorded_from: str | None = None
    recorded_to: str | None = None
    devices: list[dict[str, Any]] = field(default_factory=list)
    sites: list[dict[str, Any]] = field(default_factory=list)

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
    recording = summarize_recordings(rows)

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
        **recording,
    )


#: Coordinates in the manifest are rounded to this many decimals (0.01 deg, about
#: 1 km), coarse enough to publish without pinpointing a recorder.
SITE_DECIMALS = 2


def summarize_recordings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the recorder-metadata columns of metadata rows.

    Reads ``recorded_at``, ``device_model``, ``device_id``, ``latitude`` and
    ``longitude`` (as written by :func:`bioamla.datasets.add_recording_metadata`
    or ``bioamla batch audio metadata``). Returns the four manifest fields
    ``recorded_from``, ``recorded_to``, ``devices`` and ``sites``; all empty when
    the rows carry no recorder columns. Sites are distinct coordinates rounded to
    :data:`SITE_DECIMALS`.
    """
    timestamps: list[str] = []
    devices: dict[tuple[str, str], int] = {}
    sites: dict[tuple[float, float], int] = {}

    for row in rows:
        ts = (row.get("recorded_at") or "").strip()
        if ts:
            timestamps.append(ts)
        model = (row.get("device_model") or "").strip()
        device_id = (row.get("device_id") or "").strip()
        if model or device_id:
            devices[(model, device_id)] = devices.get((model, device_id), 0) + 1
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat not in (None, "") and lon not in (None, ""):
            try:
                key = (round(float(lat), SITE_DECIMALS), round(float(lon), SITE_DECIMALS))
            except (TypeError, ValueError):
                continue
            sites[key] = sites.get(key, 0) + 1

    # ISO 8601 strings sort chronologically within one offset; mixed offsets
    # are compared as text, which is good enough for a span.
    return {
        "recorded_from": min(timestamps) if timestamps else None,
        "recorded_to": max(timestamps) if timestamps else None,
        "devices": [
            {"device_model": m, "device_id": d, "files": n} for (m, d), n in sorted(devices.items())
        ],
        "sites": [
            {"latitude": lat, "longitude": lon, "files": n}
            for (lat, lon), n in sorted(sites.items())
        ],
    }


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


def build_dataset_card(manifest: DatasetManifest) -> str:
    """Render a HuggingFace dataset card (README.md) from a manifest.

    Produces YAML front-matter (tags + task category) followed by a human-readable
    summary of classes, counts, splits, and source provenance.
    """
    frontmatter = [
        "---",
        "tags:",
        "- bioacoustics",
        "- audio",
        "task_categories:",
        "- audio-classification",
        "---",
        "",
    ]

    total = sum(manifest.class_counts.values())
    body = [f"# {manifest.name or 'Audio dataset'}", ""]
    body.append(f"- **Kind:** {manifest.kind}")
    body.append(f"- **Total clips/files:** {total}")
    body.append(f"- **Classes:** {len(manifest.label2id)}")
    if manifest.sample_rate:
        body.append(f"- **Sample rate:** {manifest.sample_rate} Hz")
    if manifest.splits:
        split_str = ", ".join(f"{k}: {v}" for k, v in sorted(manifest.splits.items()))
        body.append(f"- **Splits:** {split_str}")
    body.append("")

    if manifest.class_counts:
        body += ["## Classes", "", "| label | id | count |", "|---|---|---|"]
        for label, idx in sorted(manifest.label2id.items(), key=lambda kv: kv[1]):
            body.append(f"| {label} | {idx} | {manifest.class_counts.get(label, 0)} |")
        body.append("")

    if manifest.sources:
        body += ["## Sources", ""]
        for src in manifest.sources:
            body.append(f"- {src.get('source', '?')}: {src.get('files', '?')} files")
        body.append("")

    if manifest.recorded_from or manifest.devices or manifest.sites:
        body += ["## Recording", ""]
        if manifest.recorded_from:
            body.append(f"- **Recorded:** {manifest.recorded_from} to {manifest.recorded_to}")
        if manifest.devices:
            names = ", ".join(
                " ".join(x for x in (d.get("device_model"), d.get("device_id")) if x) or "unknown"
                for d in manifest.devices
            )
            body.append(f"- **Recorders:** {names}")
        if manifest.sites:
            body.append(
                f"- **Sites:** {len(manifest.sites)} (coordinates rounded to "
                f"{10**-SITE_DECIMALS:g} deg)"
            )
        body.append("")

    body.append("_Generated by bioamla._")
    return "\n".join(frontmatter + body) + "\n"


def write_dataset_card(
    dataset_dir: str,
    metadata_filename: str = "metadata.csv",
    output_filename: str = "README.md",
) -> str | None:
    """Write a HuggingFace dataset card (README.md) for a dataset directory.

    Uses an existing ``dataset.json`` manifest if present, otherwise derives one
    from the metadata CSV. Returns the card path, or ``None`` when the directory
    has neither a manifest nor a metadata CSV to build from (so the caller can
    treat a non-dataset folder as a no-op).
    """
    ds_path = Path(dataset_dir)
    manifest_path = ds_path / "dataset.json"
    metadata_path = ds_path / metadata_filename

    if manifest_path.exists():
        manifest = load_dataset_manifest(str(manifest_path))
    elif metadata_path.exists():
        manifest = build_manifest_from_metadata(dataset_dir, metadata_filename=metadata_filename)
    else:
        return None

    card_path = ds_path / output_filename
    card_path.write_text(build_dataset_card(manifest), encoding="utf-8")
    return str(card_path)


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
