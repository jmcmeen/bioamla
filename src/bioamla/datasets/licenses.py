"""Generate license / attribution files from dataset metadata CSVs."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from bioamla.exceptions import InvalidInputError, LicenseGenerationError, NotFoundError

logger = logging.getLogger(__name__)


def _read_template_file(template_path: Path) -> str:
    """Read the content of a UTF-8 template file."""
    return Path(template_path).read_text(encoding="utf-8")


def _parse_license_csv(csv_path: Path) -> list[dict[str, str]]:
    """Parse a metadata CSV into normalized attribution records.

    Reads the canonical license/attribution fields with graceful fallback so a
    dataset from any catalog works: xeno-canto/macaulay populate ``license`` /
    ``attribution`` (canonical), while iNaturalist uses the ``attr_*`` block.
    Per row: ``attr_id <- attr_id|attribution``, ``attr_lic <- attr_lic|license``,
    ``attr_url <- attr_url``, plus optional context (``source``,
    ``scientific_name``, ``common_name``, ``date``). A row is kept when it has a
    ``file_name`` and at least one license/attribution signal.

    Raises:
        InvalidInputError: If the CSV is empty/headerless, has no ``file_name``
            column, or yields no attribution records.
    """
    records: list[dict[str, str]] = []

    with open(csv_path, encoding="utf-8", newline="") as f:
        sample = f.readline()
        f.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)

        if not reader.fieldnames:
            raise InvalidInputError("CSV file appears to be empty or invalid")
        if "file_name" not in reader.fieldnames:
            raise InvalidInputError("Missing required field in CSV: file_name")

        for row in reader:
            cleaned = {key: (str(value).strip() if value else "") for key, value in row.items()}
            file_name = cleaned.get("file_name", "")
            if not file_name:
                continue

            attr_id = cleaned.get("attr_id") or cleaned.get("attribution") or ""
            attr_lic = cleaned.get("attr_lic") or cleaned.get("license") or ""
            attr_url = cleaned.get("attr_url") or ""
            if not (attr_id or attr_lic or attr_url):
                continue  # no license/attribution signal — skip

            records.append(
                {
                    "file_name": file_name,
                    "attr_id": attr_id,
                    "attr_lic": attr_lic,
                    "attr_url": attr_url,
                    "attr_note": cleaned.get("attr_note", ""),
                    "source": cleaned.get("source", ""),
                    "scientific_name": cleaned.get("scientific_name", ""),
                    "common_name": cleaned.get("common_name", ""),
                    "date": cleaned.get("date", ""),
                }
            )

    if not records:
        raise InvalidInputError("No license/attribution data found in metadata file")

    return records


def _format_attribution(attribution: dict[str, str]) -> str:
    """Format a single attribution entry as a text block."""
    file_name = attribution["file_name"]
    formatted = f"File: {file_name}\n"
    formatted += "-" * (len(file_name) + 6) + "\n"

    if attribution.get("source"):
        formatted += f"Source: {attribution['source']}\n"
    species = attribution.get("scientific_name") or attribution.get("common_name")
    if species:
        formatted += f"Species: {species}\n"
    if attribution.get("attr_id"):
        formatted += f"Attribution: {attribution['attr_id']}\n"
    if attribution.get("attr_lic"):
        formatted += f"License: {attribution['attr_lic']}\n"
    if attribution.get("attr_url"):
        formatted += f"Source URL: {attribution['attr_url']}\n"
    if attribution.get("date"):
        formatted += f"Date: {attribution['date']}\n"
    if attribution.get("attr_note"):
        formatted += f"Notes: {attribution['attr_note']}\n"

    return formatted


def _generate_license_file(
    attributions: list[dict[str, str]], output_path: Path, template_content: str = ""
) -> dict[str, Any]:
    """Write the license file with all attributions and return summary stats."""
    with open(output_path, mode="w", encoding="utf-8") as f:
        if template_content:
            f.write(template_content)
            if not template_content.endswith("\n"):
                f.write("\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("FILE ATTRIBUTIONS\n")
            f.write("=" * 80 + "\n\n")
        else:
            f.write("LICENSE AND ATTRIBUTION FILE\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(attributions)}\n")
            f.write("=" * 80 + "\n\n")

        for i, attribution in enumerate(attributions, 1):
            f.write(f"{i}. ")
            f.write(_format_attribution(attribution))
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF ATTRIBUTIONS\n")
        f.write("=" * 80 + "\n")

    return {
        "output_path": str(output_path),
        "file_size": output_path.stat().st_size,
        "attributions_count": len(attributions),
    }


def _build_attributions_markdown(
    attributions: list[dict[str, str]], template_content: str = ""
) -> str:
    """Render attributions as Markdown, grouped by ``(source, license)``."""
    from collections import defaultdict

    lines: list[str] = []
    if template_content:
        lines.append(template_content.rstrip("\n"))
        lines.append("")

    lines.append("# Attributions")
    lines.append("")
    lines.append(
        f"_Generated by bioamla on {datetime.now().strftime('%Y-%m-%d')}. "
        f"{len(attributions)} file(s)._"
    )
    lines.append("")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for attr in attributions:
        key = (attr.get("source") or "Unspecified", attr.get("attr_lic") or "Unspecified")
        groups[key].append(attr)

    for source, lic in sorted(groups):
        lines.append(f"## {source} — {lic}")
        lines.append("")
        lines.append("| File | Attribution | License | Source URL |")
        lines.append("|---|---|---|---|")
        for attr in groups[(source, lic)]:
            url = attr.get("attr_url") or ""
            url_cell = f"[link]({url})" if url else ""
            lines.append(
                f"| {attr['file_name']} | {attr.get('attr_id') or ''} | "
                f"{attr.get('attr_lic') or ''} | {url_cell} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_license_for_dataset(
    dataset_path: Path,
    template_path: Path | None = None,
    output_filename: str | None = None,
    metadata_filename: str = "metadata.csv",
    format: str = "text",
) -> dict[str, Any]:
    """Generate a license/attribution file for a single dataset directory.

    Args:
        dataset_path: Path to the dataset directory.
        template_path: Optional template file to prepend.
        output_filename: Name for the output file. Defaults to ``LICENSE`` for
            ``format="text"`` and ``ATTRIBUTIONS.md`` for ``format="md"``.
        metadata_filename: Name of the metadata CSV file.
        format: ``"text"`` (plain LICENSE) or ``"md"`` (Markdown ATTRIBUTIONS).

    Returns:
        Dict with ``output_path``, ``file_size``, ``attributions_count`` and
        ``dataset_path``.

    Raises:
        NotFoundError: If the metadata CSV is missing.
        InvalidInputError: If the metadata is invalid or has no attributions.
        LicenseGenerationError: If the file cannot be written.
    """
    dataset_path = Path(dataset_path)
    csv_path = dataset_path / metadata_filename
    if not csv_path.exists():
        raise NotFoundError(f"Metadata file not found: {csv_path}")

    template_content = ""
    if template_path:
        template_content = _read_template_file(Path(template_path))

    attributions = _parse_license_csv(csv_path)
    valid_attributions = [attr for attr in attributions if attr["file_name"]]

    if not valid_attributions:
        raise InvalidInputError("No valid attributions found in metadata file")

    if output_filename is None:
        output_filename = "ATTRIBUTIONS.md" if format == "md" else "LICENSE"
    output_path = dataset_path / output_filename

    try:
        if format == "md":
            content = _build_attributions_markdown(valid_attributions, template_content)
            output_path.write_text(content, encoding="utf-8")
            stats = {
                "output_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "attributions_count": len(valid_attributions),
            }
        else:
            stats = _generate_license_file(valid_attributions, output_path, template_content)
    except OSError as e:
        raise LicenseGenerationError(f"Failed to write license file {output_path}: {e}") from e
    stats["dataset_path"] = str(dataset_path)

    return stats


def generate_licenses_for_directory(
    audio_dir: Path,
    template_path: Path | None = None,
    output_filename: str | None = None,
    metadata_filename: str = "metadata.csv",
    format: str = "text",
) -> dict[str, Any]:
    """Generate license files for every dataset subdirectory in a directory.

    Args:
        audio_dir: Directory containing dataset subdirectories.
        template_path: Optional template file to prepend.
        output_filename: Name for the output license files.
        metadata_filename: Name of the metadata CSV files.

    Returns:
        Dict with ``datasets_found``, ``datasets_processed``, ``datasets_failed``
        and a per-dataset ``results`` list.

    Raises:
        NotFoundError: If the audio directory doesn't exist.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise NotFoundError(f"Audio directory not found: {audio_dir}")

    datasets = []
    for item in sorted(audio_dir.iterdir()):
        if item.is_dir():
            csv_path = item / metadata_filename
            if csv_path.exists():
                datasets.append((item.name, item))

    results: list[dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for dataset_name, dataset_path in datasets:
        try:
            stats = generate_license_for_dataset(
                dataset_path=dataset_path,
                template_path=template_path,
                output_filename=output_filename,
                metadata_filename=metadata_filename,
                format=format,
            )
            stats["status"] = "success"
            stats["dataset_name"] = dataset_name
            results.append(stats)
            success_count += 1
        except Exception as e:  # noqa: BLE001 - record per-dataset failure, continue
            results.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_path": str(dataset_path),
                    "status": "failed",
                    "error": str(e),
                }
            )
            fail_count += 1

    return {
        "datasets_found": len(datasets),
        "datasets_processed": success_count,
        "datasets_failed": fail_count,
        "results": results,
    }
