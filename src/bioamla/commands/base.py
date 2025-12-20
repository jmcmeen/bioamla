# commands/base.py
"""
Command Pattern Infrastructure
==============================

Base classes for undoable commands and command management.

The command pattern encapsulates operations as objects, enabling:
- Undo/redo functionality
- Command history tracking
- Transaction-like operations
- Command composition

Usage:
    from bioamla.commands import Command, UndoManager

    class ResampleCommand(Command):
        def __init__(self, input_path: str, output_path: str, target_rate: int):
            self.input_path = input_path
            self.output_path = output_path
            self.target_rate = target_rate
            self._backup_path = None

        def execute(self) -> CommandResult:
            # Implementation
            ...

        def undo(self) -> None:
            # Restore from backup
            ...

    # Use with UndoManager for tracking
    manager = UndoManager()
    cmd = ResampleCommand("input.wav", "output.wav", 16000)
    manager.execute(cmd)
    manager.undo()  # Reverts the operation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID, uuid4

T = TypeVar("T")


class CommandStatus(Enum):
    """Status of a command execution."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNDONE = "undone"


@dataclass
class CommandResult(Generic[T]):
    """Result of a command execution."""

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T = None, message: str = None, **metadata) -> "CommandResult[T]":
        """Create a successful result."""
        return cls(success=True, data=data, message=message, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "CommandResult[T]":
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class CommandInfo:
    """Metadata about a command for logging and history."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: CommandStatus = CommandStatus.PENDING
    duration_seconds: float = 0.0
    result: Optional[CommandResult] = None


class Command(ABC):
    """
    Abstract base class for all commands.

    Commands encapsulate operations that can be executed and potentially undone.
    Each command should be self-contained and idempotent where possible.

    Subclasses must implement:
        - execute(): Perform the command operation
        - undo(): Reverse the command (optional, raise NotImplementedError if not undoable)
        - name: Human-readable command name
        - description: What the command does
    """

    def __init__(self):
        self._info = CommandInfo(name=self.name, description=self.description)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this command."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this command does."""
        ...

    @property
    def is_undoable(self) -> bool:
        """Whether this command can be undone."""
        return True

    @property
    def info(self) -> CommandInfo:
        """Get command metadata."""
        return self._info

    @abstractmethod
    def execute(self) -> CommandResult:
        """
        Execute the command.

        Returns:
            CommandResult with success status and any output data
        """
        ...

    def undo(self) -> None:
        """
        Undo the command.

        Override this method to implement undo functionality.
        Raise NotImplementedError if the command is not undoable.
        """
        raise NotImplementedError(f"{self.name} does not support undo")

    def validate(self) -> Optional[str]:
        """
        Validate command parameters before execution.

        Returns:
            None if valid, error message string if invalid
        """
        return None


class CompositeCommand(Command):
    """
    A command that executes multiple sub-commands in sequence.

    If any sub-command fails, previous commands are undone in reverse order.
    """

    def __init__(self, commands: List[Command]):
        self._commands = commands
        self._executed: List[Command] = []
        super().__init__()

    @property
    def name(self) -> str:
        return f"Composite({len(self._commands)} commands)"

    @property
    def description(self) -> str:
        names = [cmd.name for cmd in self._commands[:3]]
        if len(self._commands) > 3:
            names.append(f"... and {len(self._commands) - 3} more")
        return f"Execute: {', '.join(names)}"

    def execute(self) -> CommandResult:
        """Execute all sub-commands in order."""
        results = []
        for cmd in self._commands:
            result = cmd.execute()
            if not result.success:
                # Undo previously executed commands
                for executed_cmd in reversed(self._executed):
                    try:
                        executed_cmd.undo()
                    except NotImplementedError:
                        pass
                return CommandResult.fail(
                    f"Command '{cmd.name}' failed: {result.error}",
                    partial_results=results,
                )
            self._executed.append(cmd)
            results.append(result)

        return CommandResult.ok(
            data=results,
            message=f"Executed {len(self._commands)} commands successfully",
        )

    def undo(self) -> None:
        """Undo all executed sub-commands in reverse order."""
        for cmd in reversed(self._executed):
            cmd.undo()
        self._executed.clear()


class UndoManager:
    """
    Manages command execution with undo/redo support.

    Maintains a history of executed commands and allows undoing/redoing them.
    """

    def __init__(self, max_history: int = 100):
        self._history: List[Command] = []
        self._redo_stack: List[Command] = []
        self._max_history = max_history

    @property
    def can_undo(self) -> bool:
        """Check if there are commands to undo."""
        return len(self._history) > 0

    @property
    def can_redo(self) -> bool:
        """Check if there are commands to redo."""
        return len(self._redo_stack) > 0

    @property
    def history(self) -> List[CommandInfo]:
        """Get command history metadata."""
        return [cmd.info for cmd in self._history]

    def execute(self, command: Command) -> CommandResult:
        """
        Execute a command and add it to history.

        Args:
            command: The command to execute

        Returns:
            CommandResult from the command execution
        """
        # Validate first
        validation_error = command.validate()
        if validation_error:
            return CommandResult.fail(validation_error)

        # Execute
        command._info.status = CommandStatus.EXECUTING
        command._info.executed_at = datetime.utcnow()

        try:
            result = command.execute()
        except Exception as e:
            command._info.status = CommandStatus.FAILED
            return CommandResult.fail(str(e))

        command._info.completed_at = datetime.utcnow()
        command._info.duration_seconds = (
            command._info.completed_at - command._info.executed_at
        ).total_seconds()
        command._info.result = result

        if result.success:
            command._info.status = CommandStatus.COMPLETED
            # Clear redo stack on new command
            self._redo_stack.clear()
            # Add to history
            self._history.append(command)
            # Trim history if needed
            if len(self._history) > self._max_history:
                self._history.pop(0)
        else:
            command._info.status = CommandStatus.FAILED

        return result

    def undo(self) -> Optional[CommandResult]:
        """
        Undo the last executed command.

        Returns:
            CommandResult if undo was performed, None if no commands to undo
        """
        if not self.can_undo:
            return None

        command = self._history.pop()
        try:
            command.undo()
            command._info.status = CommandStatus.UNDONE
            self._redo_stack.append(command)
            return CommandResult.ok(message=f"Undid: {command.name}")
        except NotImplementedError:
            # Put it back if not undoable
            self._history.append(command)
            return CommandResult.fail(f"{command.name} cannot be undone")
        except Exception as e:
            return CommandResult.fail(f"Undo failed: {e}")

    def redo(self) -> Optional[CommandResult]:
        """
        Redo the last undone command.

        Returns:
            CommandResult if redo was performed, None if no commands to redo
        """
        if not self.can_redo:
            return None

        command = self._redo_stack.pop()
        return self.execute(command)

    def clear(self) -> None:
        """Clear all history and redo stack."""
        self._history.clear()
        self._redo_stack.clear()


# =============================================================================
# File Operation Commands (commonly need undo support)
# =============================================================================


class FileBackupMixin:
    """Mixin providing file backup/restore functionality for undoable file operations."""

    _backup_dir: Optional[Path] = None
    _backups: Dict[Path, Path] = {}

    def _ensure_backup_dir(self) -> Path:
        """Create a temporary backup directory."""
        if self._backup_dir is None:
            import tempfile

            self._backup_dir = Path(tempfile.mkdtemp(prefix="bioamla_backup_"))
        return self._backup_dir

    def _backup_file(self, filepath: Path) -> Optional[Path]:
        """Create a backup of a file."""
        if not filepath.exists():
            return None

        import shutil

        backup_dir = self._ensure_backup_dir()
        backup_path = backup_dir / f"{filepath.name}.{uuid4().hex[:8]}"
        shutil.copy2(filepath, backup_path)
        self._backups[filepath] = backup_path
        return backup_path

    def _restore_file(self, filepath: Path) -> bool:
        """Restore a file from backup."""
        import shutil

        backup_path = self._backups.get(filepath)
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, filepath)
            return True
        return False

    def _cleanup_backups(self) -> None:
        """Remove all backup files."""
        import shutil

        if self._backup_dir and self._backup_dir.exists():
            shutil.rmtree(self._backup_dir)
            self._backup_dir = None
            self._backups.clear()
