"""Audio file browser tab component for magpy-lite TUI."""

from pathlib import Path
from typing import List, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DirectoryTree, Label, Static

from bioamla.repository.local import LocalFileRepository
from bioamla.services.audio_playback import AudioPlayer
from bioamla.services.file import FileService


class AudioInfo(Static):
    """Widget to display currently playing audio information."""

    def __init__(self) -> None:
        """Initialize the audio info widget."""
        super().__init__()
        self._current_file: Optional[str] = None
        self._duration: float = 0.0

    def update_info(self, filepath: Optional[str], duration: float = 0.0) -> None:
        """Update the displayed audio information.

        Args:
            filepath: Path to the audio file
            duration: Duration in seconds
        """
        self._current_file = filepath
        self._duration = duration

        if filepath:
            filename = Path(filepath).name
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.update(f"[bold cyan]Now: [/bold cyan]{filename} [{minutes}:{seconds:02d}]")
        else:
            self.update("[dim]No file loaded[/dim]")


class PlaybackControls(Horizontal):
    """Widget containing playback control buttons."""

    def __init__(self) -> None:
        """Initialize playback controls."""
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the playback controls layout."""
        yield Button("Play", id="play-btn", variant="success")
        yield Button("Pause", id="pause-btn", variant="warning")
        yield Button("Stop", id="stop-btn", variant="error")


class AudioFileTree(DirectoryTree):
    """Custom directory tree that filters for audio files."""

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

    def filter_paths(self, paths: List[Path]) -> List[Path]:
        """Filter paths to only show directories and audio files.

        Args:
            paths: List of paths to filter

        Returns:
            Filtered list of paths
        """
        return [
            path
            for path in paths
            if path.is_dir() or path.suffix.lower() in self.AUDIO_EXTENSIONS
        ]


class BrowserTabContent(Vertical):
    """Browser tab for audio file navigation and playback."""

    BINDINGS = [
        ("space", "toggle_play", "Play/Pause"),
        ("s", "stop", "Stop"),
    ]

    def __init__(self, start_dir: Optional[str] = None) -> None:
        """Initialize the browser tab.

        Args:
            start_dir: Starting directory for file browser (defaults to home)
        """
        super().__init__()
        self._file_service = FileService(LocalFileRepository())
        self._audio_player = AudioPlayer()
        self._current_file: Optional[str] = None
        self._start_dir = Path(start_dir) if start_dir else Path.home()

    def compose(self) -> ComposeResult:
        """Compose the browser tab layout."""
        with Container(id="main-container"):
            with Vertical(id="file-tree-container"):
                yield Label("Audio Files")
                yield AudioFileTree(str(self._start_dir))

            with Vertical(id="control-panel"):
                yield AudioInfo()
                yield PlaybackControls()
                yield Label("Status: Ready", id="status-label")

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self._update_status("Ready to browse and play audio files")

    @on(DirectoryTree.FileSelected)
    def handle_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection in the directory tree.

        Args:
            event: File selection event
        """
        filepath = str(event.path)

        # Check if it's an audio file
        if not any(filepath.lower().endswith(ext) for ext in AudioFileTree.AUDIO_EXTENSIONS):
            self._update_status(f"Not an audio file: {Path(filepath).name}")
            return

        # Check if file exists
        if not self._file_service.exists(filepath):
            self._update_status(f"File not found: {filepath}")
            return

        # Load the audio file (but don't auto-play)
        try:
            self._current_file = filepath
            self._audio_player.load_file(filepath)
            duration = self._audio_player.duration

            # Update the info display
            audio_info = self.query_one(AudioInfo)
            audio_info.update_info(filepath, duration)

            self._update_status(f"Loaded: {Path(filepath).name}")

        except Exception as e:
            self._update_status(f"Error loading file: {str(e)}")

    @on(Button.Pressed, "#play-btn")
    def handle_play(self) -> None:
        """Handle play button press."""
        if self._current_file is None:
            self._update_status("No file loaded. Select a file to play.")
            return

        try:
            if self._audio_player.is_paused:
                self._audio_player.play()
                self._update_status("Resumed playback")
            else:
                self._audio_player.play()
                self._update_status(f"Playing: {Path(self._current_file).name}")
        except Exception as e:
            self._update_status(f"Error playing: {str(e)}")

    @on(Button.Pressed, "#pause-btn")
    def handle_pause(self) -> None:
        """Handle pause button press."""
        if not self._audio_player.is_playing:
            self._update_status("Nothing is playing")
            return

        try:
            self._audio_player.pause()
            self._update_status("Paused")
        except Exception as e:
            self._update_status(f"Error pausing: {str(e)}")

    @on(Button.Pressed, "#stop-btn")
    def handle_stop(self) -> None:
        """Handle stop button press."""
        try:
            self._audio_player.stop()
            self._update_status("Stopped")
        except Exception as e:
            self._update_status(f"Error stopping: {str(e)}")

    def action_toggle_play(self) -> None:
        """Toggle between play and pause."""
        if self._audio_player.is_playing:
            self.handle_pause()
        else:
            self.handle_play()

    def action_stop(self) -> None:
        """Stop playback."""
        self.handle_stop()

    def _update_status(self, message: str) -> None:
        """Update the status label.

        Args:
            message: Status message to display
        """
        status_label = self.query_one("#status-label", Label)
        status_label.update(f"Status: {message}")


# Legacy compatibility - import from magpy instead
def run_audio_browser(start_dir: Optional[str] = None) -> None:
    """Run the audio browser TUI.

    Args:
        start_dir: Starting directory for file browser (defaults to home)

    Note:
        This function is deprecated. Use magpy.run_magpy() instead.
    """
    from bioamla.tui.magpy import run_magpy

    run_magpy(start_dir)
