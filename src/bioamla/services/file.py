# services/file.py
"""
Service for file I/O operations used across CLI commands.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bioamla.repository.protocol import FileRepositoryProtocol

from .base import BaseService, ServiceResult


class FileService(BaseService):
    """
    Service for general file I/O operations.

    Provides ServiceResult-wrapped methods for common file operations
    used by CLI commands. All file I/O is delegated to the file repository.
    """

    def __init__(self, file_repository: FileRepositoryProtocol) -> None:
        """Initialize the service.

        Args:
            file_repository: File repository for all file I/O operations (required).
        """
        super().__init__(file_repository)

    def write_text(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Write text content to a file.

        Args:
            path: Path to the output file
            content: String content to write
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the output path on success
        """
        try:
            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)
            self.file_repository.write_text(path, content, encoding=encoding)

            return ServiceResult.ok(
                data=str(path),
                message=f"Wrote {len(content)} characters to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to write text file: {e}")

    def read_text(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Read text content from a file.

        Args:
            path: Path to the input file
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the file contents on success
        """
        try:
            path = Path(path)
            if not self.file_repository.exists(path):
                return ServiceResult.fail(f"File not found: {path}")

            content = self.file_repository.read_text(path, encoding=encoding)

            return ServiceResult.ok(
                data=content,
                message=f"Read {len(content)} characters from {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to read text file: {e}")

    def write_json(
        self,
        path: Union[str, Path],
        data: Any,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Write data to a JSON file.

        Args:
            path: Path to the output file
            data: Data to serialize to JSON
            indent: JSON indentation level
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the output path on success
        """
        try:
            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)

            content = json.dumps(data, indent=indent, default=str)
            self.file_repository.write_text(path, content, encoding=encoding)

            return ServiceResult.ok(
                data=str(path),
                message=f"Wrote JSON to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to write JSON file: {e}")

    def read_json(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> ServiceResult[Dict[str, Any]]:
        """
        Read data from a JSON file.

        Args:
            path: Path to the input file
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the parsed JSON data on success
        """
        try:
            path = Path(path)
            if not self.file_repository.exists(path):
                return ServiceResult.fail(f"File not found: {path}")

            content = self.file_repository.read_text(path, encoding=encoding)
            data = json.loads(content)

            return ServiceResult.ok(
                data=data,
                message=f"Read JSON from {path}",
            )
        except json.JSONDecodeError as e:
            return ServiceResult.fail(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            return ServiceResult.fail(f"Failed to read JSON file: {e}")

    def write_csv(
        self,
        path: Union[str, Path],
        rows: List[List[Any]],
        headers: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Write rows to a CSV file.

        Args:
            path: Path to the output file
            rows: List of rows (each row is a list of values)
            headers: Optional header row
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the output path on success
        """
        try:
            from io import StringIO

            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)

            # Write CSV to in-memory buffer
            buffer = StringIO()
            writer = csv.writer(buffer)
            if headers:
                writer.writerow(headers)
            writer.writerows(rows)

            # Write buffer contents to file via repository
            self.file_repository.write_text(path, buffer.getvalue(), encoding=encoding)

            row_count = len(rows)
            return ServiceResult.ok(
                data=str(path),
                message=f"Wrote {row_count} rows to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to write CSV file: {e}")

    def write_csv_dicts(
        self,
        path: Union[str, Path],
        rows: List[Dict[str, Any]],
        fieldnames: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Write dictionaries to a CSV file.

        Args:
            path: Path to the output file
            rows: List of dictionaries (each dict is a row)
            fieldnames: Column names (inferred from first row if not provided)
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the output path on success
        """
        try:
            from io import StringIO

            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)

            if not rows:
                return ServiceResult.fail("No rows to write")

            # Infer fieldnames from first row if not provided
            fields = fieldnames or list(rows[0].keys())

            # Write CSV to in-memory buffer
            buffer = StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

            # Write buffer contents to file via repository
            self.file_repository.write_text(path, buffer.getvalue(), encoding=encoding)

            row_count = len(rows)
            return ServiceResult.ok(
                data=str(path),
                message=f"Wrote {row_count} rows to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to write CSV file: {e}")

    def read_csv(
        self,
        path: Union[str, Path],
        has_header: bool = True,
        encoding: str = "utf-8",
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """
        Read a CSV file as a list of dictionaries.

        Args:
            path: Path to the input file
            has_header: Whether the CSV has a header row
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing list of row dictionaries on success
        """
        try:
            from io import StringIO

            path = Path(path)
            if not self.file_repository.exists(path):
                return ServiceResult.fail(f"File not found: {path}")

            # Read file contents via repository
            content = self.file_repository.read_text(path, encoding=encoding)

            # Parse CSV from in-memory buffer
            buffer = StringIO(content)
            if has_header:
                reader = csv.DictReader(buffer)
                rows = list(reader)
            else:
                reader = csv.reader(buffer)
                rows = [dict(enumerate(row)) for row in reader]

            return ServiceResult.ok(
                data=rows,
                message=f"Read {len(rows)} rows from {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to read CSV file: {e}")

    def ensure_directory(
        self,
        path: Union[str, Path],
    ) -> ServiceResult[str]:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            path: Path to the directory

        Returns:
            ServiceResult containing the directory path on success
        """
        try:
            path = Path(path)
            self.file_repository.mkdir(path, parents=True)

            return ServiceResult.ok(
                data=str(path),
                message=f"Directory exists: {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to create directory: {e}")

    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if a path exists.

        Args:
            path: Path to check

        Returns:
            True if the path exists, False otherwise
        """
        return self.file_repository.exists(path)

    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a file.

        Args:
            path: Path to check

        Returns:
            True if the path is a file, False otherwise
        """
        return self.file_repository.is_file(path)

    def is_directory(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a directory.

        Args:
            path: Path to check

        Returns:
            True if the path is a directory, False otherwise
        """
        return self.file_repository.is_dir(path)

    def append_text(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
    ) -> ServiceResult[str]:
        """
        Append text content to a file.

        Args:
            path: Path to the file
            content: String content to append
            encoding: Character encoding (default: utf-8)

        Returns:
            ServiceResult containing the file path on success
        """
        try:
            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)

            # Read existing content, append new content, then write back
            existing_content = ""
            if self.file_repository.exists(path):
                existing_content = self.file_repository.read_text(path, encoding=encoding)

            self.file_repository.write_text(path, existing_content + content, encoding=encoding)

            return ServiceResult.ok(
                data=str(path),
                message=f"Appended {len(content)} characters to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to append to file: {e}")

    def write_npy(
        self,
        path: Union[str, Path],
        array: Any,
    ) -> ServiceResult[str]:
        """
        Write a NumPy array to a .npy file.

        Args:
            path: Path to the output file
            array: NumPy array to save

        Returns:
            ServiceResult containing the output path on success
        """
        try:
            import io

            import numpy as np

            path = Path(path)
            self.file_repository.mkdir(path.parent, parents=True)

            # Write array to in-memory buffer
            buffer = io.BytesIO()
            np.save(buffer, array)

            # Write buffer contents to file via repository
            self.file_repository.write_bytes(path, buffer.getvalue())

            return ServiceResult.ok(
                data=str(path),
                message=f"Wrote NumPy array {array.shape} to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to write NumPy file: {e}")

    def read_npy(
        self,
        path: Union[str, Path],
    ) -> ServiceResult[Any]:
        """
        Read a NumPy array from a .npy file.

        Args:
            path: Path to the input file

        Returns:
            ServiceResult containing the NumPy array on success
        """
        try:
            import io

            import numpy as np

            path = Path(path)
            if not self.file_repository.exists(path):
                return ServiceResult.fail(f"File not found: {path}")

            # Read bytes from repository and load array
            content = self.file_repository.read_bytes(path)
            buffer = io.BytesIO(content)
            array = np.load(buffer)

            return ServiceResult.ok(
                data=array,
                message=f"Read NumPy array {array.shape} from {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to read NumPy file: {e}")
