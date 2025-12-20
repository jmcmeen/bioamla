"""
Command History Logging
=======================

Persistent command history for bioamla projects.

This module provides functionality to log CLI commands to a project's
command history file. The history is stored in JSON Lines format for
efficient append-only operations.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from bioamla.core.files import TextFile
from bioamla.core.project import PROJECT_MARKER, find_project_root

LOGS_DIR = "logs"
LOG_FILENAME = "command_history.jsonl"


@dataclass
class CommandEntry:
    """A single command history entry."""

    timestamp: str
    command: str
    args: List[str]
    kwargs: Dict[str, Any]
    exit_code: int
    duration_seconds: float
    working_dir: str
    project_root: Optional[str] = None
    error_message: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "CommandEntry":
        """Deserialize from JSON string."""
        return cls(**json.loads(json_str))


class CommandLogger:
    """
    Manages command history logging to project log files.

    Uses JSON Lines format for append-only logging. Each line in the
    log file is a complete JSON object representing a command entry.

    Attributes:
        project_root: Root directory of the project
        log_path: Path to the command history file
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the command logger.

        Args:
            project_root: Root directory of the project.
                         If None, will attempt to find project root.
        """
        if project_root is not None:
            # Verify provided path is a valid project root
            if (project_root / PROJECT_MARKER).is_dir():
                self.project_root = project_root
            else:
                self.project_root = None
        else:
            self.project_root = find_project_root()

        self._log_path: Optional[Path] = None

        if self.project_root:
            self._log_path = self.project_root / PROJECT_MARKER / LOGS_DIR / LOG_FILENAME

    @property
    def log_path(self) -> Optional[Path]:
        """Get the path to the log file."""
        return self._log_path

    def is_available(self) -> bool:
        """
        Check if logging is available (i.e., in a project).

        Returns:
            True if logging is available, False otherwise
        """
        return self._log_path is not None

    def log_command(self, entry: CommandEntry) -> None:
        """
        Append a command entry to the log.

        Args:
            entry: Command entry to log
        """
        if not self.is_available():
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        with TextFile(self._log_path, mode="a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def get_history(
        self,
        limit: Optional[int] = None,
        command_filter: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[CommandEntry]:
        """
        Retrieve command history.

        Args:
            limit: Maximum number of entries to return
            command_filter: Filter by command name substring
            since: Only return entries after this timestamp

        Returns:
            List of matching command entries (newest first)
        """
        if not self.is_available() or not self._log_path.exists():
            return []

        entries = []
        for entry in self._iter_entries():
            if command_filter and command_filter not in entry.command:
                continue
            if since:
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                    if entry_time < since:
                        continue
                except ValueError:
                    pass
            entries.append(entry)

        entries.reverse()  # Newest first

        if limit:
            entries = entries[:limit]

        return entries

    def _iter_entries(self) -> Iterator[CommandEntry]:
        """Iterate over all log entries."""
        if not self._log_path or not self._log_path.exists():
            return

        with TextFile(self._log_path, mode="r", encoding="utf-8") as f:
            for line in f.handle:
                line = line.strip()
                if line:
                    try:
                        yield CommandEntry.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        # Skip malformed entries
                        continue

    def clear(self) -> int:
        """
        Clear all command history.

        Returns:
            Number of entries cleared
        """
        if not self.is_available() or not self._log_path.exists():
            return 0

        count = sum(1 for _ in self._iter_entries())
        self._log_path.unlink()
        return count

    def search(self, query: str) -> List[CommandEntry]:
        """
        Search command history.

        Args:
            query: Search string to match against command and args

        Returns:
            Matching entries (newest first)
        """
        results = []
        query_lower = query.lower()

        for entry in self._iter_entries():
            searchable = f"{entry.command} {' '.join(entry.args)}".lower()
            if query_lower in searchable:
                results.append(entry)

        results.reverse()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about command history.

        Returns:
            Dictionary with statistics
        """
        if not self.is_available() or not self._log_path.exists():
            return {
                "total_commands": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "command_counts": {},
            }

        total = 0
        successful = 0
        failed = 0
        command_counts: Dict[str, int] = {}

        for entry in self._iter_entries():
            total += 1
            if entry.exit_code == 0:
                successful += 1
            else:
                failed += 1

            command_counts[entry.command] = command_counts.get(entry.command, 0) + 1

        return {
            "total_commands": total,
            "successful_commands": successful,
            "failed_commands": failed,
            "command_counts": command_counts,
        }


def create_command_entry(
    command: str,
    args: Optional[List[str]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    exit_code: int = 0,
    duration_seconds: float = 0.0,
    working_dir: Optional[str] = None,
    error_message: Optional[str] = None,
) -> CommandEntry:
    """
    Create a command entry with current timestamp.

    Args:
        command: The command that was executed
        args: Positional arguments
        kwargs: Keyword arguments
        exit_code: Exit code of the command
        duration_seconds: Duration of command execution
        working_dir: Working directory when command was run
        error_message: Error message if command failed

    Returns:
        CommandEntry ready to be logged
    """
    project_root = find_project_root()

    return CommandEntry(
        timestamp=datetime.now().isoformat(),
        command=command,
        args=args or [],
        kwargs=kwargs or {},
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        working_dir=working_dir or str(Path.cwd()),
        project_root=str(project_root) if project_root else None,
        error_message=error_message,
    )
