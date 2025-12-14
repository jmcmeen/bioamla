"""
Terminal User Interface for Dataset Exploration
================================================

This module provides a Textual-based TUI (Terminal User Interface) application
for interactively exploring audio datasets. Features include:

- File browser with sorting and filtering
- Dataset statistics display
- Category and split summaries
- Audio playback (using system audio players)
- Spectrogram generation and viewing
- Search functionality

The main entry point is the :func:`run_explorer` function, which launches the
:class:`DatasetExplorer` application.

Example:
    >>> from bioamla.core.tui import run_explorer
    >>> run_explorer("./my_dataset")
"""

import subprocess
import sys
import tempfile
from typing import List, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Rule,
    Select,
    Static,
)

from bioamla.core.explore import (
    AudioFileInfo,
    DatasetInfo,
    filter_audio_files,
    scan_directory,
    sort_audio_files,
)


class FileDetailScreen(ModalScreen[None]):
    """
    Modal screen showing detailed file information.

    Displays comprehensive metadata about a selected audio file including
    path, size, format, sample rate, duration, channels, and any associated
    metadata from the dataset CSV file.

    Attributes:
        file_info: The AudioFileInfo object containing file details.
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("p", "play", "Play Audio"),
        Binding("s", "spectrogram", "View Spectrogram"),
    ]

    def __init__(self, file_info: AudioFileInfo) -> None:
        """
        Initialize the file detail screen.

        Args:
            file_info: AudioFileInfo object containing the file details to display.
        """
        super().__init__()
        self.file_info = file_info

    def compose(self) -> ComposeResult:
        """Compose the file detail screen layout."""
        f = self.file_info
        with Container(id="file-detail-container"):
            yield Label(f"[bold]{f.filename}[/bold]", id="detail-title")
            yield Rule()
            with VerticalScroll():
                yield Static(f"[bold]Path:[/bold] {f.path}")
                yield Static(f"[bold]Size:[/bold] {f.size_human}")
                yield Static(f"[bold]Format:[/bold] {f.format or 'Unknown'}")
                yield Rule()
                yield Static(f"[bold]Sample Rate:[/bold] {f.sample_rate or 'Unknown'} Hz")
                yield Static(f"[bold]Duration:[/bold] {f.duration_human}")
                yield Static(f"[bold]Channels:[/bold] {f.num_channels or 'Unknown'}")
                yield Static(f"[bold]Frames:[/bold] {f.num_frames or 'Unknown'}")
                if f.category or f.split or f.attribution:
                    yield Rule()
                    yield Static("[bold]Metadata:[/bold]")
                    if f.category:
                        yield Static(f"  Category: {f.category}")
                    if f.split:
                        yield Static(f"  Split: {f.split}")
                    if f.target is not None:
                        yield Static(f"  Target: {f.target}")
                    if f.attribution:
                        yield Static(f"  Attribution: {f.attribution}")
            yield Rule()
            with Horizontal(id="detail-buttons"):
                yield Button("Play", id="btn-play", variant="primary")
                yield Button("Spectrogram", id="btn-spec", variant="default")
                yield Button("Close", id="btn-close", variant="default")

    def action_close(self) -> None:
        """Close the detail screen and return to the main view."""
        self.dismiss()

    def action_play(self) -> None:
        """Handle the play audio action."""
        self._play_audio()

    def action_spectrogram(self) -> None:
        """Handle the show spectrogram action."""
        self._show_spectrogram()

    @on(Button.Pressed, "#btn-close")
    def on_close_pressed(self) -> None:
        """Handle close button press event."""
        self.dismiss()

    @on(Button.Pressed, "#btn-play")
    def on_play_pressed(self) -> None:
        """Handle play button press event."""
        self._play_audio()

    @on(Button.Pressed, "#btn-spec")
    def on_spec_pressed(self) -> None:
        """Handle spectrogram button press event."""
        self._show_spectrogram()

    def _play_audio(self) -> None:
        """
        Attempt to play audio using the system audio player.

        Uses platform-specific audio players:
        - macOS: afplay
        - Linux: paplay, aplay, or ffplay (tries in order)
        - Windows: os.startfile (default system handler)

        Displays a notification on success or error.
        """
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["afplay", str(self.file_info.path)])
            elif sys.platform.startswith("linux"):
                # Try various Linux audio players
                for player in ["paplay", "aplay", "ffplay"]:
                    try:
                        subprocess.Popen(
                            [player, str(self.file_info.path)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        break
                    except FileNotFoundError:
                        continue
            elif sys.platform == "win32":
                import os
                os.startfile(str(self.file_info.path))
            self.notify("Playing audio...", severity="information")
        except Exception as e:
            self.notify(f"Could not play audio: {e}", severity="error")

    def _show_spectrogram(self) -> None:
        """
        Generate and display a spectrogram visualization.

        Creates a mel spectrogram image of the audio file using the
        bioamla.core.visualize module, saves it to a temporary file,
        and opens it with the system's default image viewer.

        Uses platform-specific image viewers:
        - macOS: open
        - Linux: xdg-open, feh, eog, or display (tries in order)
        - Windows: os.startfile (default system handler)

        Displays a notification on success or error.
        """
        try:
            from bioamla.core.visualize import generate_spectrogram

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                output_path = tmp.name

            generate_spectrogram(
                str(self.file_info.path),
                output_path,
                viz_type="mel",
            )

            # Try to open the image
            if sys.platform == "darwin":
                subprocess.Popen(["open", output_path])
            elif sys.platform.startswith("linux"):
                for viewer in ["xdg-open", "feh", "eog", "display"]:
                    try:
                        subprocess.Popen([viewer, output_path])
                        break
                    except FileNotFoundError:
                        continue
            elif sys.platform == "win32":
                import os
                os.startfile(output_path)

            self.notify("Spectrogram generated", severity="information")
        except Exception as e:
            self.notify(f"Could not generate spectrogram: {e}", severity="error")


class HelpScreen(ModalScreen[None]):
    """
    Modal screen showing keyboard shortcuts and help.

    Displays a comprehensive list of keyboard shortcuts and actions
    available in the Dataset Explorer application.
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        with Container(id="help-container"):
            yield Label("[bold]Bioamla Dataset Explorer[/bold]", id="help-title")
            yield Rule()
            yield Static("[bold]Navigation:[/bold]")
            yield Static("  ↑/↓, j/k    Navigate file list")
            yield Static("  Enter       View file details")
            yield Static("  Tab         Switch panels")
            yield Rule()
            yield Static("[bold]Actions:[/bold]")
            yield Static("  p           Play selected audio")
            yield Static("  s           Generate spectrogram")
            yield Static("  r           Refresh file list")
            yield Static("  /           Search files")
            yield Rule()
            yield Static("[bold]General:[/bold]")
            yield Static("  ?           Show this help")
            yield Static("  q           Quit application")
            yield Rule()
            yield Button("Close", id="btn-help-close", variant="primary")

    def action_close(self) -> None:
        """Close the help screen and return to the main view."""
        self.dismiss()

    @on(Button.Pressed, "#btn-help-close")
    def on_close_pressed(self) -> None:
        """Handle close button press event."""
        self.dismiss()


class DatasetExplorer(App):
    """
    Textual TUI application for exploring audio datasets.

    A full-featured terminal interface for browsing and analyzing audio files
    in a directory. Features include:

    - File browser with sorting and filtering by category
    - Dataset statistics panel (file count, total size, formats)
    - Category breakdown with quick filtering
    - Search functionality
    - Audio playback using system players
    - Spectrogram generation and viewing

    Attributes:
        directory: Path to the dataset directory being explored.
        audio_files: Currently displayed list of audio files (after filtering).
        dataset_info: Aggregated dataset statistics.
        selected_category: Currently selected category filter.
        search_term: Current search filter string.
        sort_by: Current sort field.
        is_loading: Whether data is currently being loaded.

    Example:
        >>> app = DatasetExplorer("./my_audio_dataset")
        >>> app.run()
    """

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    #sidebar {
        width: 30;
        min-width: 25;
        max-width: 40;
        border-right: solid $primary;
        padding: 1;
    }

    #content {
        width: 1fr;
        padding: 1;
    }

    #stats-panel {
        height: auto;
        max-height: 12;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #categories-panel {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #file-list {
        height: 1fr;
        border: solid $primary;
    }

    #search-bar {
        height: 3;
        margin-bottom: 1;
    }

    #search-input {
        width: 1fr;
    }

    #filter-bar {
        height: 3;
        margin-bottom: 1;
    }

    #sort-select {
        width: 20;
    }

    #category-filter {
        width: 20;
    }

    DataTable {
        height: 1fr;
    }

    #file-detail-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        margin: 4 4;
    }

    #detail-title {
        text-align: center;
        padding: 1;
    }

    #detail-buttons {
        height: 3;
        align: center middle;
    }

    #detail-buttons Button {
        margin: 0 1;
    }

    #help-container {
        width: 50;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        margin: 4 4;
    }

    #help-title {
        text-align: center;
        padding: 1;
    }

    .category-item {
        padding: 0 1;
    }

    .stat-label {
        color: $text-muted;
    }

    .stat-value {
        color: $text;
    }

    ProgressBar {
        margin: 1 0;
    }

    #loading-indicator {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "search", "Search"),
        Binding("p", "play", "Play"),
        Binding("s", "spectrogram", "Spectrogram"),
        Binding("escape", "clear_search", "Clear", show=False),
    ]

    directory: reactive[str] = reactive("")
    audio_files: reactive[List[AudioFileInfo]] = reactive([])
    dataset_info: reactive[Optional[DatasetInfo]] = reactive(None)
    selected_category: reactive[Optional[str]] = reactive(None)
    search_term: reactive[str] = reactive("")
    sort_by: reactive[str] = reactive("name")
    is_loading: reactive[bool] = reactive(False)

    def __init__(self, directory: str) -> None:
        """
        Initialize the dataset explorer application.

        Args:
            directory: Path to the directory containing audio files to explore.
        """
        super().__init__()
        self.directory = directory
        self._all_audio_files: List[AudioFileInfo] = []

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="sidebar"):
                yield Static("[bold]Dataset Info[/bold]", id="stats-title")
                with Container(id="stats-panel"):
                    yield Static("Loading...", id="stats-content")
                yield Static("[bold]Categories[/bold]", id="categories-title")
                with Container(id="categories-panel"):
                    yield OptionList(id="category-list")
            with Vertical(id="content"):
                with Horizontal(id="search-bar"):
                    yield Input(placeholder="Search files...", id="search-input")
                with Horizontal(id="filter-bar"):
                    yield Select(
                        [
                            ("Name", "name"),
                            ("Size", "size"),
                            ("Duration", "duration"),
                            ("Category", "category"),
                            ("Format", "format"),
                        ],
                        value="name",
                        id="sort-select",
                        prompt="Sort by",
                    )
                with Container(id="file-list"):
                    yield DataTable(id="files-table")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the application after mounting."""
        self.title = "Bioamla Dataset Explorer"
        self.sub_title = self.directory

        # Set up the data table
        table = self.query_one("#files-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("File", "Size", "Duration", "Format", "Category")

        # Load data
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        """
        Load audio files from the directory in a background thread.

        Scans the directory recursively for audio files, loading metadata
        and enriching with information from metadata.csv if present.
        Updates the UI via call_from_thread when complete.
        """
        self.is_loading = True
        try:
            files, info = scan_directory(
                self.directory,
                recursive=True,
                load_audio_metadata=True,
            )
            self._all_audio_files = files
            self.call_from_thread(self._update_ui, files, info)
        except Exception as e:
            self.call_from_thread(self.notify, f"Error loading data: {e}", severity="error")
        finally:
            self.is_loading = False

    def _update_ui(self, files: List[AudioFileInfo], info: DatasetInfo) -> None:
        """
        Update UI components with loaded data.

        Args:
            files: List of AudioFileInfo objects for all audio files found.
            info: DatasetInfo object with aggregated statistics.
        """
        self.dataset_info = info
        self.audio_files = files
        self._update_stats()
        self._update_categories()
        self._update_table()

    def _update_stats(self) -> None:
        """Update the stats panel with dataset information."""
        info = self.dataset_info
        if not info:
            return

        stats_content = self.query_one("#stats-content", Static)

        lines = [
            f"[bold]Path:[/bold] {info.path}",
            f"[bold]Files:[/bold] {info.total_files}",
            f"[bold]Size:[/bold] {info.total_size_human}",
            f"[bold]Metadata:[/bold] {'Yes' if info.has_metadata else 'No'}",
        ]

        if info.formats:
            fmt_str = ", ".join(f"{k}: {v}" for k, v in sorted(info.formats.items()))
            lines.append(f"[bold]Formats:[/bold] {fmt_str}")

        if info.splits:
            split_str = ", ".join(f"{k}: {v}" for k, v in sorted(info.splits.items()))
            lines.append(f"[bold]Splits:[/bold] {split_str}")

        stats_content.update("\n".join(lines))

    def _update_categories(self) -> None:
        """Update the categories sidebar list with counts."""
        category_list = self.query_one("#category-list", OptionList)
        category_list.clear_options()

        if not self.dataset_info:
            return

        # Add "All" option
        total = self.dataset_info.total_files
        category_list.add_option(f"All ({total})")

        # Add categories
        for cat, count in sorted(self.dataset_info.categories.items()):
            category_list.add_option(f"{cat} ({count})")

        # Add uncategorized if there are files without category
        categorized = sum(self.dataset_info.categories.values())
        uncategorized = total - categorized
        if uncategorized > 0:
            category_list.add_option(f"Uncategorized ({uncategorized})")

    def _update_table(self) -> None:
        """Update the file table with filtered and sorted audio files."""
        table = self.query_one("#files-table", DataTable)
        table.clear()

        # Apply filters and sorting
        files = self._all_audio_files

        if self.selected_category:
            if self.selected_category == "Uncategorized":
                files = [f for f in files if not f.category]
            else:
                files = [f for f in files if f.category == self.selected_category]

        if self.search_term:
            files = filter_audio_files(files, search_term=self.search_term)

        files = sort_audio_files(files, sort_by=self.sort_by)

        # Add rows
        for f in files:
            table.add_row(
                f.filename,
                f.size_human,
                f.duration_human,
                f.format or "-",
                f.category or "-",
                key=str(f.path),
            )

        self.audio_files = files

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        """
        Handle search input submission (Enter key).

        Args:
            event: The input submission event containing the search value.
        """
        self.search_term = event.value
        self._update_table()

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        """
        Handle search input change (live search).

        Args:
            event: The input change event containing the current search value.
        """
        self.search_term = event.value
        self._update_table()

    @on(Select.Changed, "#sort-select")
    def on_sort_changed(self, event: Select.Changed) -> None:
        """
        Handle sort selection change.

        Args:
            event: The select change event containing the new sort field.
        """
        self.sort_by = str(event.value)
        self._update_table()

    @on(OptionList.OptionSelected, "#category-list")
    def on_category_selected(self, event: OptionList.OptionSelected) -> None:
        """
        Handle category selection from the sidebar.

        Args:
            event: The option selected event containing the category choice.
        """
        option_text = str(event.option.prompt)
        # Extract category name (remove count)
        cat_name = option_text.rsplit(" (", 1)[0]

        if cat_name == "All":
            self.selected_category = None
        else:
            self.selected_category = cat_name

        self._update_table()

    @on(DataTable.RowSelected, "#files-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """
        Handle file row selection (Enter key on table row).

        Opens the file detail screen for the selected audio file.

        Args:
            event: The row selected event containing the row key.
        """
        if event.row_key:
            file_path = str(event.row_key.value)
            # Find the file info
            for f in self._all_audio_files:
                if str(f.path) == file_path:
                    self.push_screen(FileDetailScreen(f))
                    break

    def action_quit(self) -> None:
        """Quit the application and return to the shell."""
        self.exit()

    def action_help(self) -> None:
        """Display the help screen with keyboard shortcuts."""
        self.push_screen(HelpScreen())

    def action_refresh(self) -> None:
        """Refresh the file list by rescanning the directory."""
        self.load_data()
        self.notify("Refreshing...", severity="information")

    def action_search(self) -> None:
        """Focus the search input field."""
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        """Clear the search term and category filter."""
        search_input = self.query_one("#search-input", Input)
        search_input.value = ""
        self.search_term = ""
        self.selected_category = None
        self._update_table()

    def _get_selected_file(self) -> Optional[AudioFileInfo]:
        """
        Get the currently selected file from the data table.

        Returns:
            The AudioFileInfo for the selected row, or None if no selection.
        """
        table = self.query_one("#files-table", DataTable)
        if table.cursor_row is not None and table.row_count > 0:
            try:
                rows = list(table.ordered_rows)
                if 0 <= table.cursor_row < len(rows):
                    row_key = rows[table.cursor_row].key
                    file_path = str(row_key.value)
                    for f in self._all_audio_files:
                        if str(f.path) == file_path:
                            return f
            except (IndexError, AttributeError):
                pass
        return None

    def action_play(self) -> None:
        """Play the currently selected audio file."""
        file_info = self._get_selected_file()
        if file_info:
            screen = FileDetailScreen(file_info)
            screen._play_audio()

    def action_spectrogram(self) -> None:
        """Generate and display a spectrogram for the selected file."""
        file_info = self._get_selected_file()
        if file_info:
            screen = FileDetailScreen(file_info)
            screen._show_spectrogram()


def run_explorer(directory: str) -> None:
    """
    Run the dataset explorer TUI application.

    Args:
        directory: Path to the directory to explore
    """
    app = DatasetExplorer(directory)
    app.run()
