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
    """Parse a metadata CSV into attribution records.

    Raises:
        InvalidInputError: If the CSV is empty/invalid or missing required fields.
    """
    required_fields = ["file_name", "attr_id", "attr_lic", "attr_url", "attr_note"]
    attributions: list[dict[str, str]] = []

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

        missing_fields = [field for field in required_fields if field not in reader.fieldnames]
        if missing_fields:
            raise InvalidInputError(f"Missing required fields in CSV: {', '.join(missing_fields)}")

        for row_num, row in enumerate(reader, start=2):
            cleaned_row = {key: str(value).strip() if value else "" for key, value in row.items()}
            if not cleaned_row.get("file_name"):
                continue
            attribution = {field: cleaned_row.get(field, "") for field in required_fields}
            attribution["row_number"] = str(row_num)
            attributions.append(attribution)

    return attributions


def _format_attribution(attribution: dict[str, str]) -> str:
    """Format a single attribution entry as a text block."""
    file_name = attribution["file_name"]
    formatted = f"File: {file_name}\n"
    formatted += "-" * (len(file_name) + 6) + "\n"

    if attribution["attr_id"]:
        formatted += f"Attribution ID: {attribution['attr_id']}\n"
    if attribution["attr_lic"]:
        formatted += f"License: {attribution['attr_lic']}\n"
    if attribution["attr_url"]:
        formatted += f"Source URL: {attribution['attr_url']}\n"
    if attribution["attr_note"]:
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


def generate_license_for_dataset(
    dataset_path: Path,
    template_path: Path | None = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv",
) -> dict[str, Any]:
    """Generate a license file for a single dataset directory.

    Args:
        dataset_path: Path to the dataset directory.
        template_path: Optional template file to prepend.
        output_filename: Name for the output license file.
        metadata_filename: Name of the metadata CSV file.

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

    output_path = dataset_path / output_filename
    try:
        stats = _generate_license_file(valid_attributions, output_path, template_content)
    except OSError as e:
        raise LicenseGenerationError(f"Failed to write license file {output_path}: {e}") from e
    stats["dataset_path"] = str(dataset_path)

    return stats


def generate_licenses_for_directory(
    audio_dir: Path,
    template_path: Path | None = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv",
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
