"""
Dataset Management and Validation
=================================

This module provides utilities for managing and validating audio datasets.
It includes functions for counting audio files, validating metadata consistency,
and loading datasets for machine learning workflows.

These utilities are essential for ensuring data quality and consistency
in bioacoustic machine learning projects.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from bioamla.core.files.paths import sanitize_filename
from bioamla.core.globals import SUPPORTED_AUDIO_EXTENSIONS
from bioamla.core.logger import get_logger
from bioamla.core.metadata import (
    read_metadata_csv,
    write_metadata_csv,
)
from bioamla.core.services.species import find_species_name
from bioamla.core.utils import get_audio_files

logger = get_logger(__name__)


def count_audio_files(audio_folder_path: str) -> int:
    """
    Count the number of audio files in a directory.

    This function scans a directory and counts all files with supported
    audio extensions as defined in the global configuration.

    Args:
        audio_folder_path (str): Path to the directory containing audio files

    Returns:
        int: Number of audio files found in the directory
    """
    audio_files = get_audio_files(audio_folder_path, SUPPORTED_AUDIO_EXTENSIONS)
    return len(audio_files)


def validate_metadata(audio_folder_path: str, metadata_csv_filename: str = "metadata.csv") -> bool:
    """
    Validate that metadata CSV file matches the audio files in a directory.

    This function performs several validation checks to ensure consistency
    between audio files and their corresponding metadata:
    1. Number of audio files matches number of metadata entries
    2. All audio files are referenced in the metadata

    Args:
        audio_folder_path (str): Path to directory containing audio files
        metadata_csv_filename (str): Name of the metadata CSV file (default: 'metadata.csv')

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails (file count mismatch or missing references)
    """
    metadata_df = pd.read_csv(os.path.join(audio_folder_path, metadata_csv_filename))

    # Check that the audio folder contains the same number of files as the metadata.csv file
    num_audio_files = count_audio_files(audio_folder_path)
    num_metadata_files = len(metadata_df)
    if num_audio_files != num_metadata_files:
        raise ValueError(
            f"The number of audio files in the audio folder ({num_audio_files}) does not match the number of files in the metadata.csv file ({num_metadata_files})"
        )

    # Check that all audio files are in metadata
    audio_files = get_audio_files(audio_folder_path, SUPPORTED_AUDIO_EXTENSIONS)
    for audio_file in audio_files:
        if audio_file not in metadata_df["file_name"].tolist():
            raise ValueError(f"The audio file {audio_file} is not in the metadata.csv file")

    return True


def load_local_dataset(audio_folder_path: str):
    """
    Load a local audio dataset using Hugging Face datasets library.

    This function loads audio files from a local directory into a Hugging Face
    Dataset object for use in machine learning workflows. It performs basic
    validation to ensure the directory exists and contains audio files.

    Args:
        audio_folder_path (str): Path to directory containing audio files

    Returns:
        Dataset: Hugging Face Dataset object containing the audio data

    Raises:
        ValueError: If directory doesn't exist or contains no audio files
    """
    from datasets import load_dataset

    from bioamla.core.utils import directory_exists

    if not directory_exists(audio_folder_path):
        raise ValueError(f"The audio folder {audio_folder_path} does not exist")
    if count_audio_files(audio_folder_path) == 0:
        raise ValueError(f"The audio folder {audio_folder_path} is empty")

    dataset = load_dataset(audio_folder_path)

    return dataset


def merge_datasets(
    dataset_paths: List[str],
    output_dir: str,
    metadata_filename: str = "metadata.csv",
    skip_existing: bool = True,
    organize_by_category: bool = True,
    target_format: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Merge multiple audio datasets into a single dataset.

    This function combines audio files and metadata from multiple dataset
    directories into a single output directory. It handles:
    - Copying audio files into subdirectories based on category
    - Merging metadata.csv files
    - Detecting and handling duplicate files
    - Handling metadata field mismatches between datasets
    - Optional audio format conversion during merge

    Args:
        dataset_paths: List of paths to dataset directories to merge.
            Each directory should contain a metadata.csv file and audio files.
        output_dir: Path to the output directory for the merged dataset.
            Will be created if it doesn't exist.
        metadata_filename: Name of the metadata CSV file in each dataset
            (default: 'metadata.csv')
        skip_existing: If True, skip files that already exist in the output
            directory (default: True). If False, existing files will be
            overwritten.
        organize_by_category: If True, organize files into subdirectories
            based on the 'category' field in metadata (default: True).
            Directory names are derived from the category using sanitization.
        target_format: If specified, convert all audio files to this format
            during merge (e.g., 'wav', 'mp3', 'flac'). Files already in the
            target format are copied without conversion.
        verbose: If True, print progress information (default: True)

    Returns:
        dict: Summary statistics including:
            - total_files: Number of audio files in merged dataset
            - files_copied: Number of new files copied
            - files_skipped: Number of files skipped (already existed)
            - files_converted: Number of files converted (if target_format specified)
            - datasets_merged: Number of source datasets processed
            - output_dir: Path to output directory
            - metadata_file: Path to merged metadata CSV file

    Raises:
        ValueError: If no valid datasets are provided, output_dir is invalid,
            or target_format is not supported
        FileNotFoundError: If a dataset path doesn't exist

    Example:
        >>> stats = merge_datasets(
        ...     dataset_paths=["./birds_v1", "./birds_v2", "./frogs"],
        ...     output_dir="./merged_dataset",
        ...     target_format="wav"
        ... )
        >>> print(f"Merged {stats['datasets_merged']} datasets")
        >>> print(f"Total files: {stats['total_files']}")
    """
    if not dataset_paths:
        raise ValueError("At least one dataset path must be provided")

    # Validate target format if specified
    if target_format:
        target_format = target_format.lower().lstrip(".")
        if target_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported target format: {target_format}. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_files": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "files_converted": 0,
        "datasets_merged": 0,
        "output_dir": str(output_path.absolute()),
        "metadata_file": str(output_path / metadata_filename),
    }

    # Track all metadata rows and existing files
    all_metadata_rows: List[dict] = []
    existing_filenames: Set[str] = set()
    all_fieldnames: Set[str] = set()

    # Set of all known category names (used to find species for subspecies)
    all_categories: Set[str] = set()

    # Load existing metadata from output directory if it exists
    output_metadata_path = output_path / metadata_filename
    if output_metadata_path.exists():
        existing_rows, existing_fieldnames = read_metadata_csv(output_metadata_path)
        all_metadata_rows.extend(existing_rows)
        all_fieldnames.update(existing_fieldnames)
        existing_filenames.update(row.get("file_name", "") for row in existing_rows)
        # Collect all category names
        for row in existing_rows:
            category = row.get("category", "")
            if category:
                all_categories.add(category)
        if verbose:
            print(f"Found {len(existing_rows)} existing entries in output metadata.")

    # Process each source dataset
    for dataset_path in dataset_paths:
        source_path = Path(dataset_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        source_metadata_path = source_path / metadata_filename
        if not source_metadata_path.exists():
            if verbose:
                print(f"Warning: No {metadata_filename} found in {dataset_path}, skipping.")
            continue

        if verbose:
            print(f"Processing dataset: {dataset_path}")

        # Read source metadata
        source_rows, source_fieldnames = read_metadata_csv(source_metadata_path)
        all_fieldnames.update(source_fieldnames)

        # Collect all category names from source
        for row in source_rows:
            category = row.get("category", "")
            if category:
                all_categories.add(category)

        files_copied_from_source = 0
        files_skipped_from_source = 0

        files_converted_from_source = 0

        for row in source_rows:
            source_filename = row.get("file_name", "")
            if not source_filename:
                continue

            source_file_path = source_path / source_filename

            # Determine destination path based on organize_by_category setting
            if organize_by_category:
                category = row.get("category", "")
                # Check if this is a subspecies - find shortest matching species name
                dir_category = find_species_name(category, all_categories)
                if dir_category:
                    category_dir = sanitize_filename(dir_category)
                else:
                    category_dir = "unknown"
                # Use just the base filename, placed in category directory
                base_filename = Path(source_filename).name
                dest_filename = f"{category_dir}/{base_filename}"
            else:
                dest_filename = source_filename

            # Handle format conversion if target_format is specified
            source_ext = Path(dest_filename).suffix.lower().lstrip(".")
            needs_conversion = target_format and source_ext != target_format

            if needs_conversion:
                # Change destination filename extension to target format
                dest_filename = str(Path(dest_filename).with_suffix(f".{target_format}"))

            dest_file_path = output_path / dest_filename

            # Check if file already exists in output
            if dest_filename in existing_filenames:
                if skip_existing:
                    files_skipped_from_source += 1
                    continue
                # If not skipping, we'll overwrite but not add duplicate metadata

            # Create subdirectory if needed
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy or convert the audio file
            if source_file_path.exists():
                if needs_conversion:
                    # Convert the file
                    converter = _get_converter(source_ext, target_format)
                    if converter:
                        try:
                            converter(str(source_file_path), str(dest_file_path))
                            files_converted_from_source += 1

                            # Add to metadata with updated file_name and attr_note
                            if dest_filename not in existing_filenames:
                                updated_row = row.copy()
                                updated_row["file_name"] = dest_filename
                                updated_row["attr_note"] = "modified clip from original source"
                                all_metadata_rows.append(updated_row)
                                existing_filenames.add(dest_filename)

                            if verbose:
                                print(f"  Converted: {source_filename} -> {dest_filename}")
                        except Exception as e:
                            if verbose:
                                print(f"  Error converting {source_filename}: {e}")
                    else:
                        if verbose:
                            print(
                                f"  Warning: No converter for {source_ext} -> {target_format}, copying instead: {source_filename}"
                            )
                        shutil.copy2(source_file_path, dest_file_path)
                        files_copied_from_source += 1
                        if dest_filename not in existing_filenames:
                            updated_row = row.copy()
                            updated_row["file_name"] = dest_filename
                            all_metadata_rows.append(updated_row)
                            existing_filenames.add(dest_filename)
                else:
                    # Just copy the file
                    shutil.copy2(source_file_path, dest_file_path)
                    files_copied_from_source += 1

                    # Add to metadata with updated file_name
                    if dest_filename not in existing_filenames:
                        updated_row = row.copy()
                        updated_row["file_name"] = dest_filename
                        all_metadata_rows.append(updated_row)
                        existing_filenames.add(dest_filename)
            else:
                if verbose:
                    print(f"  Warning: Source file not found: {source_file_path}")

        stats["files_copied"] += files_copied_from_source
        stats["files_skipped"] += files_skipped_from_source
        stats["files_converted"] += files_converted_from_source
        stats["datasets_merged"] += 1

        if verbose:
            msg = f"  Copied: {files_copied_from_source}, Skipped: {files_skipped_from_source}"
            if target_format:
                msg += f", Converted: {files_converted_from_source}"
            print(msg)

    # Write merged metadata
    if all_metadata_rows:
        write_metadata_csv(
            output_metadata_path,
            all_metadata_rows,
            all_fieldnames,
            merge_existing=False,  # Already merged above
        )

    stats["total_files"] = len(all_metadata_rows)

    if verbose:
        print("\nMerge complete!")
        print(f"  Datasets merged: {stats['datasets_merged']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Files copied: {stats['files_copied']}")
        if target_format:
            print(f"  Files converted: {stats['files_converted']}")
        print(f"  Files skipped: {stats['files_skipped']}")
        print(f"  Output directory: {stats['output_dir']}")
        print(f"  Metadata file: {stats['metadata_file']}")

    return stats


# Supported audio formats for conversion
_SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "wma"}


def convert_audio_file(
    input_path: str,
    output_path: str,
    target_format: str = None,
) -> str:
    """
    Convert a single audio file to a different format.

    Args:
        input_path: Path to the input audio file
        output_path: Path for the output file (format inferred from extension if target_format not specified)
        target_format: Target audio format (optional, inferred from output_path if not provided)

    Returns:
        Path to the converted file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If target format is unsupported or no converter available
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    source_ext = input_path.suffix.lower().lstrip(".")

    # Determine target format
    if target_format:
        target_format = target_format.lower().lstrip(".")
    else:
        target_format = output_path.suffix.lower().lstrip(".")

    if target_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported target format: {target_format}. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    # If same format, just copy
    if source_ext == target_format:
        import shutil

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_path), str(output_path))
        return str(output_path)

    # Get converter
    converter = _get_converter(source_ext, target_format)
    if converter is None:
        raise ValueError(f"No converter available for {source_ext} -> {target_format}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert
    converter(str(input_path), str(output_path))
    return str(output_path)


def batch_convert_audio(
    input_dir: str,
    output_dir: str,
    target_format: str,
    recursive: bool = True,
    keep_original: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Convert all audio files in a directory to a different format.

    Args:
        input_dir: Path to the input directory
        output_dir: Path to the output directory
        target_format: Target audio format
        recursive: Search subdirectories (default: True)
        keep_original: Keep original files (default: True)
        verbose: Print progress information (default: True)

    Returns:
        Statistics dict with files_converted, files_skipped, files_failed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    target_format = target_format.lower().lstrip(".")
    if target_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported target format: {target_format}. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio files using glob
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"]
    audio_files = []
    for ext in audio_extensions:
        if recursive:
            audio_files.extend(input_dir.rglob(f"*{ext}"))
        else:
            audio_files.extend(input_dir.glob(f"*{ext}"))
    audio_files = [str(f) for f in audio_files]

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_dir}")
        return {
            "files_converted": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "output_dir": str(output_dir),
        }

    if verbose:
        print(f"Found {len(audio_files)} audio files to convert")

    stats = {
        "files_converted": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "output_dir": str(output_dir),
    }

    for audio_path in audio_files:
        audio_path = Path(audio_path)
        source_ext = audio_path.suffix.lower().lstrip(".")

        # Compute relative path for output
        try:
            rel_path = audio_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        # Change extension for output
        output_path = output_dir / rel_path.with_suffix(f".{target_format}")

        # Copy if already in target format (ensures files end up in output dir)
        if source_ext == target_format:
            stats["files_skipped"] += 1
            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy the file to output directory
            shutil.copy2(str(audio_path), str(output_path))
            if verbose:
                print(f"  Copied (same format): {audio_path.name}")
            continue

        try:
            convert_audio_file(str(audio_path), str(output_path), target_format)
            stats["files_converted"] += 1
            if verbose:
                print(f"  Converted: {output_path}")

            # Delete original if not keeping
            if not keep_original:
                audio_path.unlink()

        except Exception as e:
            stats["files_failed"] += 1
            if verbose:
                print(f"  Failed: {audio_path} - {e}")

    if verbose:
        print(
            f"Converted {stats['files_converted']} files, {stats['files_skipped']} skipped, {stats['files_failed']} failed"
        )

    return stats


def _get_converter(source_ext: str, target_format: str):
    """
    Get the appropriate converter function from bioamla.utils.

    Args:
        source_ext: Source file extension (without dot)
        target_format: Target format (without dot)

    Returns:
        Converter function or None if not available
    """
    from bioamla.core import utils as utils_module

    converter_name = f"{source_ext}_to_{target_format}"
    return getattr(utils_module, converter_name, None)


def convert_filetype(
    dataset_path: str,
    target_format: str,
    metadata_filename: str = "metadata.csv",
    keep_original: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Convert all audio files in a dataset to a specified format.

    This function parses the dataset metadata, converts each audio file to
    the target format using novus-pytils.audio conversion methods, and
    updates the metadata.csv file with the new filenames. The attr_note
    field is updated to indicate "modified clip from original source".

    Args:
        dataset_path: Path to the dataset directory containing audio files
            and a metadata.csv file.
        target_format: Target audio format (e.g., 'wav', 'mp3', 'flac', 'ogg',
            'm4a', 'aac', 'wma').
        metadata_filename: Name of the metadata CSV file (default: 'metadata.csv')
        keep_original: If True, keep original files after conversion
            (default: False). By default, original files are deleted.
        verbose: If True, print progress information (default: True)

    Returns:
        dict: Summary statistics including:
            - total_files: Total number of files in metadata
            - files_converted: Number of files successfully converted
            - files_skipped: Number of files skipped (already in target format)
            - files_failed: Number of files that failed conversion
            - target_format: The target format used

    Raises:
        ValueError: If dataset_path doesn't exist, target_format is invalid,
            or metadata file is not found
        FileNotFoundError: If the dataset path doesn't exist

    Example:
        >>> stats = convert_filetype(
        ...     dataset_path="./my_dataset",
        ...     target_format="mp3"
        ... )
        >>> print(f"Converted {stats['files_converted']} files to MP3")
    """
    target_format = target_format.lower().lstrip(".")

    if target_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported target format: {target_format}. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}"
        )

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    metadata_path = dataset_dir / metadata_filename
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    stats = {
        "total_files": 0,
        "files_converted": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "target_format": target_format,
    }

    # Read metadata
    rows, fieldnames = read_metadata_csv(metadata_path)
    stats["total_files"] = len(rows)

    if verbose:
        print(f"Converting {len(rows)} files to {target_format} format...")

    updated_rows = []
    for row in rows:
        file_name = row.get("file_name", "")
        if not file_name:
            updated_rows.append(row)
            continue

        source_path = dataset_dir / file_name
        source_ext = source_path.suffix.lower().lstrip(".")

        # Skip if already in target format
        if source_ext == target_format:
            stats["files_skipped"] += 1
            updated_rows.append(row)
            continue

        # Build target file_name
        target_file_name = str(Path(file_name).with_suffix(f".{target_format}"))
        target_path = dataset_dir / target_file_name

        # Get converter function
        converter = _get_converter(source_ext, target_format)
        if converter is None:
            if verbose:
                print(
                    f"  Warning: No converter for {source_ext} -> {target_format}, removing: {file_name}"
                )
            stats["files_failed"] += 1
            # Delete the source file since it can't be converted
            if source_path.exists():
                source_path.unlink()
            # Don't add to updated_rows - remove from metadata
            continue

        # Check if source file exists
        if not source_path.exists():
            if verbose:
                print(f"  Warning: Source file not found, removing from metadata: {file_name}")
            stats["files_failed"] += 1
            # Don't add to updated_rows - remove from metadata
            continue

        # Create target directory if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert the file
        try:
            converter(str(source_path), str(target_path))
            stats["files_converted"] += 1

            # Update row with new file_name and attr_note
            updated_row = row.copy()
            updated_row["file_name"] = target_file_name
            updated_row["attr_note"] = "modified clip from original source"
            updated_rows.append(updated_row)

            # Delete original unless keep_original is True
            if not keep_original and source_path.exists():
                source_path.unlink()

            if verbose:
                print(f"  Converted: {file_name} -> {target_file_name}")

        except Exception as e:
            if verbose:
                print(f"  Error converting {file_name}, removing: {e}")
            stats["files_failed"] += 1
            # Delete the source file since conversion failed
            if source_path.exists():
                source_path.unlink()
            # Don't add to updated_rows - remove from metadata

    # Write updated metadata
    write_metadata_csv(metadata_path, updated_rows, fieldnames, merge_existing=False)

    if verbose:
        print("\nConversion complete!")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Converted: {stats['files_converted']}")
        print(f"  Skipped (already {target_format}): {stats['files_skipped']}")
        print(f"  Failed: {stats['files_failed']}")

    return stats
