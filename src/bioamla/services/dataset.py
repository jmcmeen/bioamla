# services/dataset.py
"""
Service for dataset management operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult, ToDictMixin


@dataclass
class MergeResult(ToDictMixin):
    """Result of dataset merge operation."""

    datasets_merged: int
    total_files: int
    files_copied: int
    files_skipped: int
    files_converted: int
    output_dir: str


@dataclass
class AugmentResult(ToDictMixin):
    """Result of dataset augmentation."""

    files_processed: int
    files_created: int
    output_dir: str


@dataclass
class LicenseResult(ToDictMixin):
    """Result of license generation."""

    output_path: str
    attributions_count: int
    file_size: int


@dataclass
class BatchLicenseResult(ToDictMixin):
    """Result of batch license generation."""

    datasets_found: int
    datasets_processed: int
    datasets_failed: int
    results: List[Dict[str, Any]] = field(default_factory=list)


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
            from bioamla.core.datasets.datasets import merge_datasets

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
            return ServiceResult.fail(
                "At least one augmentation option must be enabled"
            )

        try:
            from bioamla.core.augment import AugmentationConfig, batch_augment

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
            from bioamla.core.license import generate_license_for_dataset

            path = Path(dataset_path)
            csv_path = path / metadata_filename

            if not self.file_repository.exists(csv_path):
                return ServiceResult.fail(
                    f"Metadata file not found: {csv_path}"
                )

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
            from bioamla.core.license import generate_licenses_for_directory

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
            import csv
            from collections import Counter
            from io import StringIO

            path = Path(dataset_path)
            metadata_path = path / metadata_filename

            if not self.file_repository.exists(metadata_path):
                return ServiceResult.fail(
                    f"Metadata file not found: {metadata_path}"
                )

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
