"""Main tabbed application for magpy-lite TUI."""

from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from bioamla.tui.about import AboutTabContent
from bioamla.tui.audio_browser import BrowserTabContent


class MagpyLite(App):
    """Main magpy-lite application with tabbed interface."""

    CSS = """
    Screen {
        layout: vertical;
    }

    TabbedContent {
        height: 100%;
    }

    #main-container {
        height: 100%;
        layout: horizontal;
    }

    #file-tree-container {
        width: 60%;
        border: solid $accent;
        padding: 1;
    }

    #control-panel {
        width: 40%;
        layout: vertical;
        padding: 1;
    }

    AudioInfo {
        height: 3;
        background: $boost;
        padding: 1;
        margin-bottom: 1;
    }

    PlaybackControls {
        height: 3;
        margin-bottom: 1;
    }

    PlaybackControls Button {
        width: 1fr;
        margin-right: 1;
    }

    #status-label {
        height: 3;
        background: $panel;
        padding: 1;
        margin-bottom: 1;
    }

    DirectoryTree {
        height: 100%;
    }

    /* About tab styling */
    AboutTabContent {
        align: center top;
        padding: 3;
        overflow-y: auto;
    }

    AboutTabContent Static {
        width: 100%;
        max-width: 80;
        text-align: center;
    }

    #app-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #version {
        color: $text-muted;
        margin-bottom: 2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, start_dir: Optional[str] = None) -> None:
        """Initialize the magpy-lite application.

        Args:
            start_dir: Starting directory for file browser (defaults to home)
        """
        super().__init__()
        self._start_dir = start_dir

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        with TabbedContent(initial="browser"):
            with TabPane("Browser", id="browser"):
                yield BrowserTabContent(self._start_dir)
            with TabPane("About", id="about"):
                yield AboutTabContent()

        yield Footer()

    @on(TabbedContent.TabActivated)
    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Handle tab switching - pause playback when leaving Browser tab.

        Args:
            event: Tab activation event
        """
        # If switching TO a non-browser tab, pause playback in browser tab
        if event.pane.id == "about":
            # Get browser tab and pause its audio player if it exists and is playing
            try:
                browser_tabs = self.query(BrowserTabContent)
                if browser_tabs:
                    browser_tab = browser_tabs[0]
                    if hasattr(browser_tab, '_audio_player') and browser_tab._audio_player.is_playing:
                        browser_tab._audio_player.pause()
                        browser_tab._update_status("Paused (switched tabs)")
            except Exception as e:
                # Log error but don't prevent tab switch
                self.log(f"Error pausing playback: {e}")


def run_magpy(start_dir: Optional[str] = None) -> None:
    """Run the magpy-lite TUI application.

    Args:
        start_dir: Starting directory for file browser (defaults to home)
    """
    app = MagpyLite(start_dir)
    app.run()
