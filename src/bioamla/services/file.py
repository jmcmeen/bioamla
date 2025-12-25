# services/file.py
"""
File Service
============

Service for file I/O operations used across CLI commands.

This service provides a consistent interface for reading and writing
text, JSON, and CSV files. It wraps the core file operations to provide
ServiceResult-based error handling.

Usage:
    from bioamla.services import FileService

    file_svc = FileService()

    # Write text
    result = file_svc.write_text("output.txt", "Hello, World!")

    # Read text
    result = file_svc.read_text("input.txt")
    if result.success:
        content = result.data

    # Write JSON
    result = file_svc.write_json("data.json", {"key": "value"})

    # Read JSON
    result = file_svc.read_json("data.json")
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseService, ServiceResult


class FileService(BaseService):
    """
    Service for general file I/O operations.

    Provides ServiceResult-wrapped methods for common file operations
    used by CLI commands.
    """

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
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

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
            if not path.exists():
                return ServiceResult.fail(f"File not found: {path}")

            with open(path, "r", encoding=encoding) as f:
                content = f.read()

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
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                json.dump(data, f, indent=indent, default=str)

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
            if not path.exists():
                return ServiceResult.fail(f"File not found: {path}")

            with open(path, "r", encoding=encoding) as f:
                data = json.load(f)

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
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding=encoding, newline="") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(rows)

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
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if not rows:
                return ServiceResult.fail("No rows to write")

            # Infer fieldnames from first row if not provided
            fields = fieldnames or list(rows[0].keys())

            with open(path, "w", encoding=encoding, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

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
            path = Path(path)
            if not path.exists():
                return ServiceResult.fail(f"File not found: {path}")

            with open(path, "r", encoding=encoding, newline="") as f:
                if has_header:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                else:
                    reader = csv.reader(f)
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
            path.mkdir(parents=True, exist_ok=True)

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
        return Path(path).exists()

    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a file.

        Args:
            path: Path to check

        Returns:
            True if the path is a file, False otherwise
        """
        return Path(path).is_file()

    def is_directory(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a directory.

        Args:
            path: Path to check

        Returns:
            True if the path is a directory, False otherwise
        """
        return Path(path).is_dir()

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
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding=encoding) as f:
                f.write(content)

            return ServiceResult.ok(
                data=str(path),
                message=f"Appended {len(content)} characters to {path}",
            )
        except Exception as e:
            return ServiceResult.fail(f"Failed to append to file: {e}")
