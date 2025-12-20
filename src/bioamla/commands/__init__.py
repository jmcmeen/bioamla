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

Audio Commands:
    - ResampleCommand: Resample audio to different sample rate
    - NormalizeCommand: Normalize audio amplitude (peak or loudness)
    - FilterCommand: Apply frequency filters (lowpass, highpass, bandpass)
    - TrimCommand: Trim audio to time range
    - TrimSilenceCommand: Remove silence from audio
    - DenoiseCommand: Apply spectral noise reduction
    - GainCommand: Adjust audio volume
    - ConvertToMonoCommand: Convert stereo to mono
    - PipelineCommand: Execute a chain of audio operations
    - AudioProcessingPipeline: Builder for audio pipelines

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

Audio Command Example:
    from bioamla.commands import UndoManager, ResampleCommand, AudioProcessingPipeline

    manager = UndoManager()

    # Single operation
    cmd = ResampleCommand("input.wav", "output.wav", 16000)
    manager.execute(cmd)

    # Pipeline of operations
    pipeline = AudioProcessingPipeline("input.wav", "output.wav")
    pipeline.resample(16000).bandpass(500, 8000).normalize()
    manager.execute(pipeline.build())
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

from .audio import (
    AudioProcessingPipeline,
    ConvertToMonoCommand,
    DenoiseCommand,
    FilterCommand,
    GainCommand,
    NormalizeCommand,
    PipelineCommand,
    ResampleCommand,
    TrimCommand,
    TrimSilenceCommand,
)

__all__ = [
    # Base
    "Command",
    "CommandInfo",
    "CommandResult",
    "CommandStatus",
    "CompositeCommand",
    "FileBackupMixin",
    "UndoManager",
    # Audio
    "AudioProcessingPipeline",
    "ConvertToMonoCommand",
    "DenoiseCommand",
    "FilterCommand",
    "GainCommand",
    "NormalizeCommand",
    "PipelineCommand",
    "ResampleCommand",
    "TrimCommand",
    "TrimSilenceCommand",
]
