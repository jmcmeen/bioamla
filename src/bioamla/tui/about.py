"""About tab for magpy-lite TUI."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class AboutTabContent(Vertical):
    """About tab displaying application information, credits, and links."""

    def compose(self) -> ComposeResult:
        """Compose the about tab layout."""
        yield Static("[bold cyan]magpy-lite[/bold cyan]", id="app-title")
        yield Static("")  # Spacer

        yield Static("[bold]Description[/bold]")
        yield Static("Lightweight audio browser and player for bioacoustic analysis")
        yield Static("Interactive terminal UI for browsing and playing audio files")
        yield Static("")  # Spacer

        yield Static("[bold]Features[/bold]")
        yield Static("• Browse and filter audio files (WAV, MP3, FLAC, OGG, M4A, AAC, WMA)")
        yield Static("• Play, pause, and stop audio playback")
        yield Static("• Real-time playback information with duration")
        yield Static("• Keyboard shortcuts for efficient control")
        yield Static("")  # Spacer

        yield Static("[bold]Keyboard Shortcuts[/bold]")
        yield Static("[cyan]Q[/cyan] - Quit application")
        yield Static("[cyan]Space[/cyan] - Play/Pause (Browser tab)")
        yield Static("[cyan]S[/cyan] - Stop playback (Browser tab)")
        yield Static("")  # Spacer

        yield Static("[bold]Credits[/bold]")
        yield Static("Part of BioAMLA - Bioacoustic & Machine Learning Applications")
        yield Static("")  # Spacer

        yield Static("[bold]Links[/bold]")
        yield Static("Repository: https://github.com/jmcmeen/bioamla")
        yield Static("")  # Spacer

        yield Static("[bold]License[/bold]")
        yield Static("Open source software - see repository for license details")
        yield Static("")  # Spacer

        yield Static("[bold]Acknowledgments[/bold]")
        yield Static("Built with Textual framework by Textualize")
        yield Static("Audio playback via sounddevice library")
