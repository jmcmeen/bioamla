"""
Attribution License File Generator

Parses a CSV file with attribution data and generates a formatted license file
with proper attributions for each file.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from bioamla.core.files import TextFile


def read_template_file(template_path: Path) -> str:
    """
    Read the content of a template file.

    Args:
        template_path: Path to the template file

    Returns:
        Content of the template file

    Raises:
        FileNotFoundError: If template file does not exist
    """
    with TextFile(template_path, mode='r', encoding='utf-8') as f:
        return f.read()


def parse_csv_file(csv_path: Path) -> list[dict[str, str]]:
    """
    Parse the CSV file and extract attribution data.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries containing attribution data

    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If required fields are missing
    """
    required_fields = ['file_name', 'attr_id', 'attr_lic', 'attr_url', 'attr_note']
    attributions = []

    with TextFile(csv_path, mode='r', encoding='utf-8', newline='') as f:
        sample = f.handle.readline()
        f.handle.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter

        reader = csv.DictReader(f.handle, delimiter=delimiter)

        if not reader.fieldnames:
            raise ValueError("CSV file appears to be empty or invalid")

        missing_fields = [field for field in required_fields if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(f"Missing required fields in CSV: {', '.join(missing_fields)}")

        for row_num, row in enumerate(reader, start=2):
            cleaned_row = {key: str(value).strip() if value else '' for key, value in row.items()}

            if not cleaned_row.get('file_name'):
                continue

            attribution = {field: cleaned_row.get(field, '') for field in required_fields}
            attribution['row_number'] = str(row_num)
            attributions.append(attribution)

    return attributions


def format_attribution(attribution: dict[str, str]) -> str:
    """
    Format a single attribution entry.

    Args:
        attribution: Attribution data dictionary

    Returns:
        Formatted attribution text
    """
    file_name = attribution['file_name']
    attr_id = attribution['attr_id']
    attr_lic = attribution['attr_lic']
    attr_url = attribution['attr_url']
    attr_note = attribution['attr_note']

    formatted = f"File: {file_name}\n"
    formatted += "-" * (len(file_name) + 6) + "\n"

    if attr_id:
        formatted += f"Attribution ID: {attr_id}\n"

    if attr_lic:
        formatted += f"License: {attr_lic}\n"

    if attr_url:
        formatted += f"Source URL: {attr_url}\n"

    if attr_note:
        formatted += f"Notes: {attr_note}\n"

    return formatted


def generate_license_file(
    attributions: list[dict[str, str]],
    output_path: Path,
    template_content: str = ""
) -> dict[str, str | int]:
    """
    Generate the license file with all attributions.

    Args:
        attributions: List of attribution dictionaries
        output_path: Path for the output license file
        template_content: Content from template file to prepend

    Returns:
        Dictionary with generation statistics

    Raises:
        OSError: If unable to write output file
    """
    with TextFile(output_path, mode='w', encoding='utf-8') as f:
        if template_content:
            f.write(template_content)
            if not template_content.endswith('\n'):
                f.write('\n')
            f.write('\n' + '=' * 80 + '\n')
            f.write('FILE ATTRIBUTIONS\n')
            f.write('=' * 80 + '\n\n')
        else:
            f.write('LICENSE AND ATTRIBUTION FILE\n')
            f.write('=' * 80 + '\n')
            f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Total files: {len(attributions)}\n')
            f.write('=' * 80 + '\n\n')

        for i, attribution in enumerate(attributions, 1):
            f.write(f"{i}. ")
            f.write(format_attribution(attribution))
            f.write('\n')

        f.write('=' * 80 + '\n')
        f.write('END OF ATTRIBUTIONS\n')
        f.write('=' * 80 + '\n')

    return {
        'output_path': str(output_path),
        'file_size': output_path.stat().st_size,
        'attributions_count': len(attributions)
    }


def validate_csv_structure(csv_path: Path) -> dict[str, bool | list[str] | int]:
    """
    Validate CSV structure.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with validation results including:
        - is_valid: Whether the CSV has required fields
        - field_names: List of field names found
        - row_count: Number of data rows
        - missing_fields: List of missing required fields
    """
    required_fields = ['file_name', 'attr_id', 'attr_lic', 'attr_url', 'attr_note']

    with TextFile(csv_path, mode='r', encoding='utf-8', newline='') as f:
        sample = f.handle.readline()
        f.handle.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter

        reader = csv.DictReader(f.handle, delimiter=delimiter)
        field_names = list(reader.fieldnames or [])

        rows = list(reader)
        row_count = len(rows)

        missing_fields = [field for field in required_fields if field not in field_names]

        return {
            'is_valid': len(missing_fields) == 0,
            'field_names': field_names,
            'row_count': row_count,
            'missing_fields': missing_fields
        }


def find_datasets(audio_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Find all datasets in the audio directory.

    A dataset is identified by a directory containing a metadata.csv file.

    Args:
        audio_dir: Path to the audio directory

    Returns:
        List of tuples (dataset_name, dataset_path, csv_path)

    Raises:
        FileNotFoundError: If audio directory does not exist
    """
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    datasets = []
    for item in sorted(audio_dir.iterdir()):
        if item.is_dir():
            csv_path = item / "metadata.csv"
            if csv_path.exists():
                datasets.append((item.name, item, csv_path))

    return datasets


def generate_license_for_dataset(
    dataset_path: Path,
    template_path: Optional[Path] = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv"
) -> dict[str, str | int]:
    """
    Generate a license file for a single dataset.

    Args:
        dataset_path: Path to the dataset directory
        template_path: Optional path to a template file to prepend
        output_filename: Name for the output license file
        metadata_filename: Name of the metadata CSV file

    Returns:
        Dictionary with generation statistics

    Raises:
        FileNotFoundError: If dataset or metadata file not found
        ValueError: If metadata CSV is invalid
    """
    csv_path = dataset_path / metadata_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    template_content = ""
    if template_path:
        template_content = read_template_file(template_path)

    attributions = parse_csv_file(csv_path)
    valid_attributions = [attr for attr in attributions if attr['file_name']]

    if not valid_attributions:
        raise ValueError("No valid attributions found in metadata file")

    output_path = dataset_path / output_filename
    stats = generate_license_file(valid_attributions, output_path, template_content)
    stats['dataset_path'] = str(dataset_path)

    return stats


def generate_licenses_for_directory(
    audio_dir: Path,
    template_path: Optional[Path] = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv"
) -> dict[str, int | list[dict[str, str | int]]]:
    """
    Generate license files for all datasets in a directory.

    Args:
        audio_dir: Path to the audio directory containing dataset subdirectories
        template_path: Optional path to a template file to prepend
        output_filename: Name for the output license files
        metadata_filename: Name of the metadata CSV files

    Returns:
        Dictionary with overall statistics including:
        - datasets_found: Number of datasets found
        - datasets_processed: Number successfully processed
        - datasets_failed: Number that failed
        - results: List of individual results

    Raises:
        FileNotFoundError: If audio directory does not exist
    """
    datasets = find_datasets(audio_dir)

    results = []
    success_count = 0
    fail_count = 0

    for dataset_name, dataset_path, _ in datasets:
        try:
            stats = generate_license_for_dataset(
                dataset_path=dataset_path,
                template_path=template_path,
                output_filename=output_filename,
                metadata_filename=metadata_filename
            )
            stats['status'] = 'success'
            stats['dataset_name'] = dataset_name
            results.append(stats)
            success_count += 1
        except Exception as e:
            results.append({
                'dataset_name': dataset_name,
                'dataset_path': str(dataset_path),
                'status': 'failed',
                'error': str(e)
            })
            fail_count += 1

    return {
        'datasets_found': len(datasets),
        'datasets_processed': success_count,
        'datasets_failed': fail_count,
        'results': results
    }
