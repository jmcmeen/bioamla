# commands/__init__.py
"""
Commands Package
================

Command pattern implementation for undoable operations.

Components:
    - Command: Abstract base class for all commands
    - CompositeCommand: Execute multiple commands as a unit
    - UndoManager: Manage command history with undo/redo
    - CommandResult: Structured result from command execution
    - FileBackupMixin: Helper for file backup/restore operations

Usage:
    from bioamla.commands import Command, UndoManager, CommandResult

    class MyCommand(Command):
        @property
        def name(self) -> str:
            return "My Command"

        @property
        def description(self) -> str:
            return "Does something"

        def execute(self) -> CommandResult:
            # Do work
            return CommandResult.ok(data="result")

        def undo(self) -> None:
            # Reverse work
            pass

    manager = UndoManager()
    result = manager.execute(MyCommand())
    manager.undo()
"""
from .base import (
    Command,
    CommandInfo,
    CommandResult,
    CommandStatus,
    CompositeCommand,
    FileBackupMixin,
    UndoManager,
)

__all__ = [
    "Command",
    "CommandInfo",
    "CommandResult",
    "CommandStatus",
    "CompositeCommand",
    "FileBackupMixin",
    "UndoManager",
]
