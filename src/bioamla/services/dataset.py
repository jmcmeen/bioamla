# services/dataset.py
"""
Dataset Management Service
==========================

This module provides comprehensive dataset management operations including:
- Merging multiple audio datasets
- Audio augmentation for training data expansion
- License/attribution file generation
- Dataset statistics and validation
- File format conversion

The service combines functionality previously split across core modules.
"""

import csv
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from bioamla.core.files import TextFile
from bioamla.models.dataset import (
    AugmentResult,
    BatchLicenseResult,
    LicenseResult,
    MergeResult,
)
from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult

logger = logging.getLogger(__name__)


# =============================================================================
# Augmentation Configuration
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""

    # Noise augmentation
    add_noise: bool = False
    noise_min_snr: float = 3.0
    noise_max_snr: float = 30.0
    noise_probability: float = 1.0

    # Time stretch augmentation
    time_stretch: bool = False
    time_stretch_min: float = 0.8
    time_stretch_max: float = 1.2
    time_stretch_probability: float = 1.0

    # Pitch shift augmentation
    pitch_shift: bool = False
    pitch_shift_min: float = -4.0
    pitch_shift_max: float = 4.0
    pitch_shift_probability: float = 1.0

    # Gain augmentation
    gain: bool = False
    gain_min_db: float = -12.0
    gain_max_db: float = 12.0
    gain_probability: float = 1.0

    # General settings
    sample_rate: int = 16000
    multiply: int = 1  # Number of augmented copies to create per file


# =============================================================================
# Augmentation Functions
# =============================================================================


def create_augmentation_pipeline(config: AugmentationConfig):
    """
    Create an audiomentations pipeline from configuration.

    Args:
        config: Augmentation configuration

    Returns:
        Compose pipeline or None if no augmentations are enabled
    """
    from audiomentations import (
        AddGaussianSNR,
        Compose,
        Gain,
        PitchShift,
        TimeStretch,
    )

    transforms = []

    if config.add_noise:
        transforms.append(
            AddGaussianSNR(
                min_snr_db=config.noise_min_snr,
                max_snr_db=config.noise_max_snr,
                p=config.noise_probability,
            )
        )

    if config.time_stretch:
        transforms.append(
            TimeStretch(
                min_rate=config.time_stretch_min,
                max_rate=config.time_stretch_max,
                p=config.time_stretch_probability,
            )
        )

    if config.pitch_shift:
        transforms.append(
            PitchShift(
                min_semitones=config.pitch_shift_min,
                max_semitones=config.pitch_shift_max,
                p=config.pitch_shift_probability,
            )
        )

    if config.gain:
        transforms.append(
            Gain(
                min_gain_db=config.gain_min_db,
                max_gain_db=config.gain_max_db,
                p=config.gain_probability,
            )
        )

    if not transforms:
        return None

    return Compose(transforms)


def augment_audio(audio: np.ndarray, sample_rate: int, pipeline) -> np.ndarray:
    """
    Apply augmentation pipeline to audio.

    Args:
        audio: Audio data as numpy array (1D)
        sample_rate: Sample rate of the audio
        pipeline: Audiomentations Compose pipeline

    Returns:
        Augmented audio as numpy array
    """
    # Ensure audio is float32 and 1D
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Apply augmentation
    augmented = pipeline(samples=audio, sample_rate=sample_rate)

    return augmented


def batch_augment(
    input_dir: str,
    output_dir: str,
    config: AugmentationConfig,
    recursive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Augment all audio files in a directory.

    Args:
        input_dir: Path to directory containing audio files
        output_dir: Path to directory for augmented output
        config: Augmentation configuration
        recursive: Whether to search subdirectories
        verbose: Whether to print progress messages

    Returns:
        dict: Statistics about the batch processing including:
            - files_processed: Number of files successfully processed
            - files_created: Total number of augmented files created
            - files_failed: Number of files that failed
            - output_dir: Path to output directory
    """
    from bioamla.adapters.pydub import save_audio
    from bioamla.core.torchaudio import load_waveform_tensor
    from bioamla.core.utils import get_files_by_extension

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pipeline = create_augmentation_pipeline(config)
    if pipeline is None:
        raise ValueError("No augmentations configured")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = get_files_by_extension(
        str(input_dir), extensions=audio_extensions, recursive=recursive
    )

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_dir}")
        return {
            "files_processed": 0,
            "files_created": 0,
            "files_failed": 0,
            "output_dir": str(output_dir),
        }

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Creating {config.multiply} augmented copies per file")

    files_processed = 0
    files_created = 0
    files_failed = 0

    for audio_path in audio_files:
        audio_path = Path(audio_path)

        # Preserve relative directory structure
        try:
            rel_path = audio_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        try:
            # Load audio once
            waveform, orig_sr = load_waveform_tensor(str(audio_path))
            audio = waveform.numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            else:
                audio = audio.squeeze()

            # Resample if needed
            if orig_sr != config.sample_rate:
                import torch
                import torchaudio

                waveform_tensor = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_sr, config.sample_rate)
                audio = resampler(waveform_tensor).squeeze().numpy()

            # Create multiple augmented versions
            for i in range(config.multiply):
                # Generate output filename with augmentation index
                stem = rel_path.stem
                suffix = ".wav"  # Always output as WAV
                if config.multiply > 1:
                    out_name = f"{stem}_aug{i + 1}{suffix}"
                else:
                    out_name = f"{stem}_aug{suffix}"

                out_path = output_dir / rel_path.parent / out_name

                # Apply augmentation
                augmented = augment_audio(audio, config.sample_rate, pipeline)

                # Save output
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(str(out_path), augmented, config.sample_rate)

                files_created += 1

                if verbose:
                    print(f"  Created: {out_path}")

            files_processed += 1

        except Exception as e:
            files_failed += 1
            if verbose:
                print(f"  Failed: {audio_path} - {e}")

    if verbose:
        print(
            f"Processed {files_processed} files, created {files_created} augmented files, {files_failed} failed"
        )

    return {
        "files_processed": files_processed,
        "files_created": files_created,
        "files_failed": files_failed,
        "output_dir": str(output_dir),
    }


# =============================================================================
# Dataset Merge Functions
# =============================================================================

# Supported audio formats for conversion
_SUPPORTED_FORMATS = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "wma"}


def _get_converter(source_ext: str, target_format: str):
    """Get the appropriate converter function."""
    from bioamla.core import utils as utils_module

    converter_name = f"{source_ext}_to_{target_format}"
    return getattr(utils_module, converter_name, None)


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
    directories into a single output directory.

    Args:
        dataset_paths: List of paths to dataset directories to merge
        output_dir: Path to the output directory for the merged dataset
        metadata_filename: Name of the metadata CSV file in each dataset
        skip_existing: If True, skip files that already exist
        organize_by_category: Organize files into subdirectories by category
        target_format: Convert all audio files to this format during merge
        verbose: Print progress information

    Returns:
        dict: Summary statistics
    """
    from bioamla.core.metadata import read_metadata_csv, write_metadata_csv
    from bioamla.core.files import sanitize_filename
    from bioamla.services.species import find_species_name

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
    all_categories: Set[str] = set()

    # Load existing metadata from output directory if it exists
    output_metadata_path = output_path / metadata_filename
    if output_metadata_path.exists():
        existing_rows, existing_fieldnames = read_metadata_csv(output_metadata_path)
        all_metadata_rows.extend(existing_rows)
        all_fieldnames.update(existing_fieldnames)
        existing_filenames.update(row.get("file_name", "") for row in existing_rows)
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
                dir_category = find_species_name(category, all_categories)
                if dir_category:
                    category_dir = sanitize_filename(dir_category)
                else:
                    category_dir = "unknown"
                base_filename = Path(source_filename).name
                dest_filename = f"{category_dir}/{base_filename}"
            else:
                dest_filename = source_filename

            # Handle format conversion if target_format is specified
            source_ext = Path(dest_filename).suffix.lower().lstrip(".")
            needs_conversion = target_format and source_ext != target_format

            if needs_conversion:
                dest_filename = str(Path(dest_filename).with_suffix(f".{target_format}"))

            dest_file_path = output_path / dest_filename

            # Check if file already exists in output
            if dest_filename in existing_filenames:
                if skip_existing:
                    files_skipped_from_source += 1
                    continue

            # Create subdirectory if needed
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy or convert the audio file
            if source_file_path.exists():
                if needs_conversion:
                    converter = _get_converter(source_ext, target_format)
                    if converter:
                        try:
                            converter(str(source_file_path), str(dest_file_path))
                            files_converted_from_source += 1

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
                                f"  Warning: No converter for {source_ext} -> {target_format}, copying: {source_filename}"
                            )
                        shutil.copy2(source_file_path, dest_file_path)
                        files_copied_from_source += 1
                        if dest_filename not in existing_filenames:
                            updated_row = row.copy()
                            updated_row["file_name"] = dest_filename
                            all_metadata_rows.append(updated_row)
                            existing_filenames.add(dest_filename)
                else:
                    shutil.copy2(source_file_path, dest_file_path)
                    files_copied_from_source += 1

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
            merge_existing=False,
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


# =============================================================================
# License Generation Functions
# =============================================================================


def _read_template_file(template_path: Path) -> str:
    """Read the content of a template file."""
    with TextFile(template_path, mode="r", encoding="utf-8") as f:
        return f.read()


def _parse_license_csv(csv_path: Path) -> list:
    """Parse the CSV file and extract attribution data."""
    required_fields = ["file_name", "attr_id", "attr_lic", "attr_url", "attr_note"]
    attributions = []

    with TextFile(csv_path, mode="r", encoding="utf-8", newline="") as f:
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
            cleaned_row = {key: str(value).strip() if value else "" for key, value in row.items()}

            if not cleaned_row.get("file_name"):
                continue

            attribution = {field: cleaned_row.get(field, "") for field in required_fields}
            attribution["row_number"] = str(row_num)
            attributions.append(attribution)

    return attributions


def _format_attribution(attribution: dict) -> str:
    """Format a single attribution entry."""
    file_name = attribution["file_name"]
    attr_id = attribution["attr_id"]
    attr_lic = attribution["attr_lic"]
    attr_url = attribution["attr_url"]
    attr_note = attribution["attr_note"]

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


def _generate_license_file(
    attributions: list, output_path: Path, template_content: str = ""
) -> dict:
    """Generate the license file with all attributions."""
    with TextFile(output_path, mode="w", encoding="utf-8") as f:
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
    template_path: Optional[Path] = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv",
) -> dict:
    """
    Generate a license file for a single dataset.

    Args:
        dataset_path: Path to the dataset directory
        template_path: Optional path to a template file to prepend
        output_filename: Name for the output license file
        metadata_filename: Name of the metadata CSV file

    Returns:
        Dictionary with generation statistics
    """
    csv_path = dataset_path / metadata_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    template_content = ""
    if template_path:
        template_content = _read_template_file(template_path)

    attributions = _parse_license_csv(csv_path)
    valid_attributions = [attr for attr in attributions if attr["file_name"]]

    if not valid_attributions:
        raise ValueError("No valid attributions found in metadata file")

    output_path = dataset_path / output_filename
    stats = _generate_license_file(valid_attributions, output_path, template_content)
    stats["dataset_path"] = str(dataset_path)

    return stats


def generate_licenses_for_directory(
    audio_dir: Path,
    template_path: Optional[Path] = None,
    output_filename: str = "LICENSE",
    metadata_filename: str = "metadata.csv",
) -> dict:
    """
    Generate license files for all datasets in a directory.

    Args:
        audio_dir: Path to directory containing dataset subdirectories
        template_path: Optional path to a template file to prepend
        output_filename: Name for the output license files
        metadata_filename: Name of the metadata CSV files

    Returns:
        Dictionary with overall statistics
    """
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Find datasets
    datasets = []
    for item in sorted(audio_dir.iterdir()):
        if item.is_dir():
            csv_path = item / metadata_filename
            if csv_path.exists():
                datasets.append((item.name, item, csv_path))

    results = []
    success_count = 0
    fail_count = 0

    for dataset_name, dataset_path, _ in datasets:
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
        except Exception as e:
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


# =============================================================================
# Dataset Service Class
# =============================================================================


class DatasetService(BaseService):
    """
    Service for dataset management operations.

    Provides ServiceResult-wrapped methods for dataset operations.
    All file I/O operations are delegated to the file repository.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the service.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)

    def merge(
        self,
        dataset_paths: List[str],
        output_dir: str,
        metadata_filename: str = "metadata.csv",
        skip_existing: bool = True,
        organize_by_category: bool = True,
        target_format: Optional[str] = None,
        verbose: bool = False,
    ) -> ServiceResult[MergeResult]:
        """
        Merge multiple audio datasets into a single dataset.

        Args:
            dataset_paths: Paths to datasets to merge
            output_dir: Output directory for merged dataset
            metadata_filename: Name of metadata CSV file in each dataset
            skip_existing: Skip files that already exist
            organize_by_category: Organize by category (True) or preserve structure
            target_format: Convert all audio files to this format
            verbose: Show progress output

        Returns:
            ServiceResult containing MergeResult
        """
        try:
            stats = merge_datasets(
                dataset_paths=dataset_paths,
                output_dir=output_dir,
                metadata_filename=metadata_filename,
                skip_existing=skip_existing,
                organize_by_category=organize_by_category,
                target_format=target_format,
                verbose=verbose,
            )

            result = MergeResult(
                datasets_merged=stats.get("datasets_merged", 0),
                total_files=stats.get("total_files", 0),
                files_copied=stats.get("files_copied", 0),
                files_skipped=stats.get("files_skipped", 0),
                files_converted=stats.get("files_converted", 0),
                output_dir=output_dir,
            )

            return ServiceResult.ok(
                data=result,
                message=f"Merged {result.datasets_merged} datasets, {result.total_files} files",
            )
        except Exception as e:
            return ServiceResult.fail(f"Dataset merge failed: {e}")

    def augment(
        self,
        input_dir: str,
        output_dir: str,
        add_noise: bool = False,
        noise_min_snr: float = 3.0,
        noise_max_snr: float = 30.0,
        time_stretch: bool = False,
        time_stretch_min: float = 0.8,
        time_stretch_max: float = 1.2,
        pitch_shift: bool = False,
        pitch_shift_min: float = -2.0,
        pitch_shift_max: float = 2.0,
        gain: bool = False,
        gain_min_db: float = -12.0,
        gain_max_db: float = 12.0,
        multiply: int = 1,
        sample_rate: int = 16000,
        recursive: bool = True,
        verbose: bool = False,
    ) -> ServiceResult[AugmentResult]:
        """
        Augment audio files to expand training datasets.

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for augmented files
            add_noise: Add Gaussian noise
            noise_min_snr: Minimum SNR for noise (dB)
            noise_max_snr: Maximum SNR for noise (dB)
            time_stretch: Apply time stretching
            time_stretch_min: Minimum stretch rate
            time_stretch_max: Maximum stretch rate
            pitch_shift: Apply pitch shifting
            pitch_shift_min: Minimum pitch shift (semitones)
            pitch_shift_max: Maximum pitch shift (semitones)
            gain: Apply gain adjustment
            gain_min_db: Minimum gain (dB)
            gain_max_db: Maximum gain (dB)
            multiply: Number of augmented copies per file
            sample_rate: Target sample rate
            recursive: Search subdirectories
            verbose: Show progress output

        Returns:
            ServiceResult containing AugmentResult
        """
        error = self._validate_input_path(input_dir)
        if error:
            return ServiceResult.fail(error)

        if not any([add_noise, time_stretch, pitch_shift, gain]):
            return ServiceResult.fail("At least one augmentation option must be enabled")

        try:
            config = AugmentationConfig(
                sample_rate=sample_rate,
                multiply=multiply,
                add_noise=add_noise,
                noise_min_snr=noise_min_snr,
                noise_max_snr=noise_max_snr,
                time_stretch=time_stretch,
                time_stretch_min=time_stretch_min,
                time_stretch_max=time_stretch_max,
                pitch_shift=pitch_shift,
                pitch_shift_min=pitch_shift_min,
                pitch_shift_max=pitch_shift_max,
                gain=gain,
                gain_min_db=gain_min_db,
                gain_max_db=gain_max_db,
            )

            stats = batch_augment(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                recursive=recursive,
                verbose=verbose,
            )

            result = AugmentResult(
                files_processed=stats.get("files_processed", 0),
                files_created=stats.get("files_created", 0),
                output_dir=stats.get("output_dir", output_dir),
            )

            return ServiceResult.ok(
                data=result,
                message=f"Created {result.files_created} augmented files",
            )
        except Exception as e:
            return ServiceResult.fail(f"Dataset augmentation failed: {e}")

    def generate_license(
        self,
        dataset_path: str,
        output_filename: str = "LICENSE",
        template_path: Optional[str] = None,
        metadata_filename: str = "metadata.csv",
    ) -> ServiceResult[LicenseResult]:
        """
        Generate license/attribution file from dataset metadata.

        Args:
            dataset_path: Path to dataset directory
            output_filename: Output filename for license
            template_path: Optional template file to prepend
            metadata_filename: Name of metadata CSV file

        Returns:
            ServiceResult containing LicenseResult
        """
        error = self._validate_input_path(dataset_path)
        if error:
            return ServiceResult.fail(error)

        try:
            path = Path(dataset_path)
            csv_path = path / metadata_filename

            if not self.file_repository.exists(csv_path):
                return ServiceResult.fail(f"Metadata file not found: {csv_path}")

            template = Path(template_path) if template_path else None

            stats = generate_license_for_dataset(
                dataset_path=path,
                template_path=template,
                output_filename=output_filename,
                metadata_filename=metadata_filename,
            )

            result = LicenseResult(
                output_path=stats.get("output_path", ""),
                attributions_count=stats.get("attributions_count", 0),
                file_size=stats.get("file_size", 0),
            )

            return ServiceResult.ok(
                data=result,
                message=f"Generated license with {result.attributions_count} attributions",
            )
        except Exception as e:
            return ServiceResult.fail(f"License generation failed: {e}")

    def generate_licenses_batch(
        self,
        directory: str,
        output_filename: str = "LICENSE",
        template_path: Optional[str] = None,
        metadata_filename: str = "metadata.csv",
    ) -> ServiceResult[BatchLicenseResult]:
        """
        Generate license files for all datasets in a directory.

        Args:
            directory: Directory containing multiple datasets
            output_filename: Output filename for each license
            template_path: Optional template file to prepend
            metadata_filename: Name of metadata CSV file

        Returns:
            ServiceResult containing BatchLicenseResult
        """
        error = self._validate_input_path(directory)
        if error:
            return ServiceResult.fail(error)

        try:
            path = Path(directory)
            template = Path(template_path) if template_path else None

            stats = generate_licenses_for_directory(
                audio_dir=path,
                template_path=template,
                output_filename=output_filename,
                metadata_filename=metadata_filename,
            )

            result = BatchLicenseResult(
                datasets_found=stats.get("datasets_found", 0),
                datasets_processed=stats.get("datasets_processed", 0),
                datasets_failed=stats.get("datasets_failed", 0),
                results=stats.get("results", []),
            )

            return ServiceResult.ok(
                data=result,
                message=f"Generated {result.datasets_processed} license files",
            )
        except Exception as e:
            return ServiceResult.fail(f"Batch license generation failed: {e}")

    def download(
        self,
        url: str,
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Download a file from URL.

        Args:
            url: URL to download from
            output_path: Local path to save file

        Returns:
            ServiceResult containing output path
        """
        try:
            from bioamla.core.utils import download_file

            download_file(url, output_path)

            return ServiceResult.ok(
                data=output_path,
                message=f"Downloaded to {output_path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Download failed: {e}")

    def extract_zip(
        self,
        file_path: str,
        output_path: str,
    ) -> ServiceResult[str]:
        """
        Extract a ZIP archive.

        Args:
            file_path: Path to ZIP file
            output_path: Directory to extract to

        Returns:
            ServiceResult containing output path
        """
        error = self._validate_input_path(file_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from bioamla.core.utils import extract_zip_file

            extract_zip_file(file_path, output_path)

            return ServiceResult.ok(
                data=output_path,
                message=f"Extracted to {output_path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Extraction failed: {e}")

    def create_zip(
        self,
        source_path: str,
        output_file: str,
    ) -> ServiceResult[str]:
        """
        Create a ZIP archive from a file or directory.

        Args:
            source_path: Path to file or directory to archive
            output_file: Output ZIP file path

        Returns:
            ServiceResult containing output path
        """
        error = self._validate_input_path(source_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from pathlib import Path as PathLib

            from bioamla.core.utils import create_zip_file, zip_directory

            if PathLib(source_path).is_dir():
                zip_directory(source_path, output_file)
            else:
                create_zip_file([source_path], output_file)

            return ServiceResult.ok(
                data=output_file,
                message=f"Created {output_file}",
            )
        except Exception as e:
            return ServiceResult.fail(f"ZIP creation failed: {e}")

    def get_stats(
        self,
        dataset_path: str,
        metadata_filename: str = "metadata.csv",
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Get statistics for a dataset.

        Args:
            dataset_path: Path to dataset directory
            metadata_filename: Name of metadata CSV file

        Returns:
            ServiceResult containing dataset statistics
        """
        error = self._validate_input_path(dataset_path)
        if error:
            return ServiceResult.fail(error)

        try:
            from collections import Counter
            from io import StringIO

            path = Path(dataset_path)
            metadata_path = path / metadata_filename

            if not self.file_repository.exists(metadata_path):
                return ServiceResult.fail(f"Metadata file not found: {metadata_path}")

            content = self.file_repository.read_text(metadata_path)
            buffer = StringIO(content)
            reader = csv.DictReader(buffer)
            rows = list(reader)

            # Compute stats
            total_files = len(rows)
            categories = Counter()
            licenses = Counter()

            for row in rows:
                if "category" in row:
                    categories[row["category"]] += 1
                if "license" in row:
                    licenses[row["license"]] += 1

            stats = {
                "total_files": total_files,
                "categories": dict(categories),
                "licenses": dict(licenses),
                "num_categories": len(categories),
                "num_licenses": len(licenses),
            }

            return ServiceResult.ok(
                data=stats,
                message=f"Dataset has {total_files} files in {len(categories)} categories",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to get dataset stats: {e}")
