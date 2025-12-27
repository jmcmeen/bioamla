"""CSV metadata handling for batch operations."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from bioamla.repository.protocol import FileRepositoryProtocol


@dataclass
class MetadataRow:
    """Single row from metadata CSV with file path and arbitrary metadata fields."""

    file_name: str  # Original relative path from CSV
    file_path: Path  # Resolved absolute path for processing
    metadata_fields: Dict[str, Any] = field(default_factory=dict)  # All other CSV columns
    output_path: Optional[Path] = None  # Updated path after processing


@dataclass
class CSVBatchContext:
    """Context for CSV-based batch processing."""

    csv_path: Path  # Input CSV location
    csv_dir: Path  # CSV directory (base for relative paths)
    output_dir: Optional[Path]  # Output directory if specified
    rows: List[MetadataRow] = field(default_factory=list)  # All CSV rows
    fieldnames: List[str] = field(default_factory=list)  # CSV column names (preserved order)
    new_fieldnames: List[str] = field(default_factory=list)  # New columns added during processing


class BatchCSVHandler:
    """Handles CSV I/O, path resolution, and result merging for batch operations."""

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize with file repository for consistent I/O.

        Args:
            file_repository: File repository for reading/writing CSV files
        """
        self.file_repository = file_repository

    def load_csv(self, csv_path: str, output_dir: Optional[str] = None) -> CSVBatchContext:
        """Load metadata CSV and resolve file paths.

        Args:
            csv_path: Path to metadata CSV file
            output_dir: Optional output directory for processed files

        Returns:
            CSVBatchContext with all rows and resolved paths

        Raises:
            ValueError: If CSV doesn't have file_name column
            FileNotFoundError: If CSV file doesn't exist
        """
        csv_path_obj = Path(csv_path).resolve()
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        csv_dir = csv_path_obj.parent
        output_dir_obj = Path(output_dir).resolve() if output_dir else None

        # Read CSV file
        csv_content = self.file_repository.read_text(csv_path_obj)
        reader = csv.DictReader(csv_content.splitlines())

        fieldnames = reader.fieldnames or []
        if "file_name" not in fieldnames:
            raise ValueError(f"CSV must have 'file_name' column. Found: {fieldnames}")

        # Parse rows and resolve paths
        rows: List[MetadataRow] = []
        for row_dict in reader:
            file_name = row_dict["file_name"]
            file_path = self.resolve_file_path(file_name, csv_dir)

            # Extract metadata fields (all columns except file_name)
            metadata_fields = {k: v for k, v in row_dict.items() if k != "file_name"}

            rows.append(
                MetadataRow(
                    file_name=file_name, file_path=file_path, metadata_fields=metadata_fields
                )
            )

        return CSVBatchContext(
            csv_path=csv_path_obj,
            csv_dir=csv_dir,
            output_dir=output_dir_obj,
            rows=rows,
            fieldnames=list(fieldnames),
        )

    def resolve_file_path(self, file_name: str, csv_dir: Path) -> Path:
        """Resolve file path relative to CSV directory.

        Args:
            file_name: Relative or absolute path from CSV
            csv_dir: Directory containing the CSV file

        Returns:
            Resolved absolute path
        """
        file_path = Path(file_name)

        # If already absolute, use as-is
        if file_path.is_absolute():
            return file_path

        # Otherwise, resolve relative to CSV directory
        return (csv_dir / file_path).resolve()

    def resolve_output_path(
        self,
        input_path: Path,
        csv_context: CSVBatchContext,
        new_extension: Optional[str] = None,
    ) -> Path:
        """Calculate output path for processed file.

        Args:
            input_path: Original input file path
            csv_context: CSV batch context with output directory info
            new_extension: New file extension (e.g., '.wav') if format changes

        Returns:
            Output file path

        Logic:
            WITH output_dir: output_dir / relative_structure / filename
            WITHOUT output_dir: Same location as input (in-place)
        """
        # Determine final extension
        if new_extension:
            output_stem = input_path.stem
            output_ext = new_extension if new_extension.startswith(".") else f".{new_extension}"
            output_name = f"{output_stem}{output_ext}"
        else:
            output_name = input_path.name

        if csv_context.output_dir:
            # WITH output_dir: preserve directory structure relative to CSV
            try:
                rel_to_csv = input_path.relative_to(csv_context.csv_dir)
                output_path = csv_context.output_dir / rel_to_csv.parent / output_name
            except ValueError:
                # If input_path is not relative to csv_dir, just use filename
                output_path = csv_context.output_dir / output_name
        else:
            # WITHOUT output_dir: in-place replacement
            output_path = input_path.parent / output_name

        return output_path

    def update_row_path(
        self, row: MetadataRow, new_path: Path, csv_context: CSVBatchContext
    ) -> None:
        """Update file_name in row to new path (relative to CSV if possible).

        Args:
            row: Metadata row to update
            new_path: New absolute path after processing
            csv_context: CSV batch context
        """
        row.output_path = new_path

        # Try to make path relative to output CSV location
        if csv_context.output_dir:
            # Output CSV will be in output_dir, make paths relative to it
            try:
                row.file_name = str(new_path.relative_to(csv_context.output_dir))
            except ValueError:
                # If can't make relative, use absolute
                row.file_name = str(new_path)
        else:
            # In-place mode: CSV stays in same location, make paths relative to csv_dir
            try:
                row.file_name = str(new_path.relative_to(csv_context.csv_dir))
            except ValueError:
                # If can't make relative, use absolute
                row.file_name = str(new_path)

    def merge_analysis_results(self, row: MetadataRow, results: Dict[str, Any]) -> None:
        """Merge analysis results into metadata_fields.

        Args:
            row: Metadata row to update
            results: Dictionary of result columns to add (e.g., {'aci': 0.85, 'adi': 0.72})
        """
        row.metadata_fields.update(results)

    def expand_row_for_segments(
        self,
        parent_row: MetadataRow,
        segments: List[Any],
        csv_context: CSVBatchContext,
    ) -> List[MetadataRow]:
        """Create multiple output rows from one input row (for segment operation).

        Args:
            parent_row: Original input row with parent file metadata
            segments: List of SegmentInfo objects from segment_file()
            csv_context: CSV batch context

        Returns:
            List of new MetadataRow objects (one per segment)
        """
        new_rows: List[MetadataRow] = []

        for seg_info in segments:
            # Calculate relative path for segment file
            try:
                if csv_context.output_dir:
                    rel_path = seg_info.segment_path.relative_to(csv_context.output_dir)
                else:
                    rel_path = seg_info.segment_path.relative_to(csv_context.csv_dir)
                file_name = str(rel_path)
            except ValueError:
                # If can't make relative, use absolute
                file_name = str(seg_info.segment_path)

            # Inherit all parent metadata
            segment_metadata = parent_row.metadata_fields.copy()

            # Add segment-specific fields
            segment_metadata.update(
                {
                    "parent_file": parent_row.file_name,
                    "segment_id": seg_info.segment_id,
                    "start_time": seg_info.start_time,
                    "end_time": seg_info.end_time,
                    "duration": seg_info.duration,
                }
            )

            new_rows.append(
                MetadataRow(
                    file_name=file_name,
                    file_path=seg_info.segment_path,
                    metadata_fields=segment_metadata,
                    output_path=seg_info.segment_path,
                )
            )

        return new_rows

    def write_csv(self, context: CSVBatchContext) -> Path:
        """Write updated metadata CSV to output location.

        Args:
            context: CSV batch context with all rows

        Returns:
            Path to written CSV file

        Logic:
            - Preserves all original columns
            - Adds new columns from analysis results
            - Updates file_name paths
            - Writes to output_dir if specified, else updates in-place
        """
        # Determine output CSV path
        if context.output_dir:
            output_csv_path = context.output_dir / context.csv_path.name
            # Ensure output directory exists
            self.file_repository.mkdir(str(context.output_dir), parents=True)
        else:
            output_csv_path = context.csv_path

        # Collect all fieldnames (original + new from analysis results)
        all_fieldnames = ["file_name"]  # file_name comes first
        all_fieldnames.extend([f for f in context.fieldnames if f != "file_name"])

        # Add any new fields from metadata
        new_fields = set()
        for row in context.rows:
            for key in row.metadata_fields.keys():
                if key not in all_fieldnames:
                    new_fields.add(key)

        all_fieldnames.extend(sorted(new_fields))

        # Write CSV
        rows_data = []
        for row in context.rows:
            row_dict = {"file_name": row.file_name}
            row_dict.update(row.metadata_fields)
            rows_data.append(row_dict)

        # Use file repository to write CSV
        import io

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(rows_data)

        self.file_repository.write_text(output_csv_path, csv_buffer.getvalue())

        return output_csv_path
