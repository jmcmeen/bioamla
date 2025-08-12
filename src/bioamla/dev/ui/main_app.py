"""
Main Textual application for the audio editor.
"""
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message
from textual import events
from typing import Optional
import asyncio
from pathlib import Path

from ..core.audio_editor import (
    AudioData, AudioProcessor, AudioFilters, 
    AnnotationManager, AudioPlayback, SpectrogramGenerator
)
from .audio_widgets import (
    WaveformDisplay, SpectrogramDisplay, AudioInfoPanel,
    TransportControls, FilterPanel, AnnotationPanel,
    FileOperationsPanel, StatusBar, ProgressDisplay
)


class AudioEditorApp(App):
    """Main audio editor application using Textual."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
        grid-rows: 1fr 3fr 1fr;
        grid-columns: 2fr 1fr;
    }
    
    #header {
        column-span: 2;
        height: 3;
    }
    
    #main_content {
        border: solid $primary;
        height: 100%;
    }
    
    #side_panel {
        border: solid $secondary;
        height: 100%;
    }
    
    #footer {
        column-span: 2;
        height: 3;
    }
    
    WaveformDisplay {
        border: solid $accent;
        margin: 1;
        padding: 1;
        min-height: 12;
    }
    
    SpectrogramDisplay {
        border: solid $accent;
        margin: 1;
        padding: 1;
        min-height: 18;
    }
    
    AudioInfoPanel {
        border: solid $success;
        margin: 1;
        padding: 1;
        min-height: 8;
    }
    
    TransportControls {
        margin: 1;
        height: 5;
    }
    
    .panel {
        border: solid $warning;
        margin: 1;
        padding: 1;
        min-height: 10;
    }
    
    StatusBar {
        dock: bottom;
        height: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("o", "open_file", "Open"),
        Binding("s", "save_file", "Save"),
        Binding("space", "play_pause", "Play/Pause"),
        Binding("escape", "stop", "Stop"),
        Binding("u", "undo", "Undo"),
        Binding("ctrl+z", "undo", "Undo"),
        Binding("f1", "show_help", "Help"),
    ]
    
    TITLE = "Audio Editor"
    SUB_TITLE = "Professional Audio Editing with Textual"
    
    # Reactive attributes
    current_audio: reactive[Optional[AudioData]] = reactive(None)
    is_playing: reactive[bool] = reactive(False)
    current_status: reactive[str] = reactive("Ready")
    
    def __init__(self, initial_file: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.initial_file = initial_file
        self.playback = AudioPlayback()
        self.waveform_display: Optional[WaveformDisplay] = None
        self.spectrogram_display: Optional[SpectrogramDisplay] = None
        self.info_panel: Optional[AudioInfoPanel] = None
        self.status_bar: Optional[StatusBar] = None
    
    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header(show_clock=True)
        
        with Container(id="header"):
            yield TransportControls(id="transport")
        
        with TabbedContent(id="main_content"):
            with TabPane("Waveform", id="waveform_tab"):
                yield WaveformDisplay(id="waveform")
            
            with TabPane("Spectrogram", id="spectrogram_tab"):
                yield SpectrogramDisplay(id="spectrogram")
        
        with Vertical(id="side_panel"):
            yield AudioInfoPanel(id="info_panel")
            yield FileOperationsPanel(classes="panel")
            yield FilterPanel(classes="panel")
            yield AnnotationPanel(classes="panel")
        
        with Container(id="footer"):
            yield StatusBar(id="status_bar")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the application."""
        # Get widget references
        self.waveform_display = self.query_one("#waveform", WaveformDisplay)
        self.spectrogram_display = self.query_one("#spectrogram", SpectrogramDisplay)
        self.info_panel = self.query_one("#info_panel", AudioInfoPanel)
        self.status_bar = self.query_one("#status_bar", StatusBar)
        
        # Load initial file if provided
        if self.initial_file:
            await self.load_audio_file(self.initial_file)
    
    def watch_current_audio(self, audio_data: Optional[AudioData]) -> None:
        """Update displays when audio data changes."""
        if self.waveform_display:
            self.waveform_display.audio_data = audio_data
        if self.spectrogram_display:
            self.spectrogram_display.audio_data = audio_data
        if self.info_panel:
            self.info_panel.audio_data = audio_data
    
    def watch_current_status(self, status: str) -> None:
        """Update status bar when status changes."""
        if self.status_bar:
            self.status_bar.status = status
    
    async def load_audio_file(self, filepath: str) -> bool:
        """Load an audio file."""
        try:
            self.current_status = f"Loading {filepath}..."
            
            # Load audio in a separate thread to avoid blocking UI
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, AudioProcessor.load_audio, filepath
            )
            
            self.current_audio = audio_data
            self.current_status = f"Loaded: {Path(filepath).name}"
            return True
            
        except Exception as e:
            self.current_status = f"Error loading file: {str(e)}"
            return False
    
    async def save_audio_file(self, filepath: str) -> bool:
        """Save the current audio file."""
        if not self.current_audio:
            self.current_status = "No audio to save"
            return False
        
        try:
            self.current_status = f"Saving {filepath}..."
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, AudioProcessor.save_audio, self.current_audio, filepath
            )
            
            self.current_status = f"Saved: {Path(filepath).name}"
            return True
            
        except Exception as e:
            self.current_status = f"Error saving file: {str(e)}"
            return False
    
    def play_audio(self) -> None:
        """Play the current audio."""
        if not self.current_audio:
            self.current_status = "No audio to play"
            return
        
        try:
            if self.waveform_display and self.waveform_display.selection_start is not None:
                start_time = self.waveform_display.selection_start
            else:
                start_time = 0.0
            
            self.playback.play(self.current_audio, start_time)
            self.is_playing = True
            self.current_status = "Playing audio..."
            
        except Exception as e:
            self.current_status = f"Playback error: {str(e)}"
    
    def stop_audio(self) -> None:
        """Stop audio playback."""
        self.playback.stop()
        self.is_playing = False
        self.current_status = "Stopped"
    
    def apply_filter(self, filter_type: str, **params) -> None:
        """Apply audio filter."""
        if not self.current_audio:
            self.current_status = "No audio loaded"
            return
        
        try:
            self.current_status = f"Applying {filter_type} filter..."
            
            if filter_type == "lowpass":
                AudioFilters.low_pass_filter(
                    self.current_audio, 
                    params.get('frequency', 1000)
                )
            elif filter_type == "highpass":
                AudioFilters.high_pass_filter(
                    self.current_audio,
                    params.get('frequency', 1000)
                )
            elif filter_type == "bandpass":
                AudioFilters.band_pass_filter(
                    self.current_audio,
                    params.get('low_frequency', 300),
                    params.get('high_frequency', 3000)
                )
            elif filter_type == "notch":
                AudioFilters.notch_filter(
                    self.current_audio,
                    params.get('frequency', 60)
                )
            
            # Trigger display updates
            self.current_audio = self.current_audio
            self.current_status = f"Applied {filter_type} filter"
            
        except Exception as e:
            self.current_status = f"Filter error: {str(e)}"
    
    def add_annotation(self, start: float, end: float, label: str, description: str = "") -> None:
        """Add annotation to current audio."""
        if not self.current_audio:
            self.current_status = "No audio loaded"
            return
        
        try:
            AnnotationManager.add_annotation(
                self.current_audio, start, end, label, description
            )
            self.current_status = f"Added annotation: {label}"
            
            # Update annotation panel
            annotation_panel = self.query_one(AnnotationPanel)
            annotation_panel.update_annotations(self.current_audio.annotations)
            
        except Exception as e:
            self.current_status = f"Annotation error: {str(e)}"
    
    def undo_last_action(self) -> None:
        """Undo the last audio operation."""
        if not self.current_audio:
            self.current_status = "No audio loaded"
            return
        
        if self.current_audio.undo():
            # Trigger display updates
            self.current_audio = self.current_audio
            self.current_status = "Undid last operation"
        else:
            self.current_status = "Nothing to undo"
    
    # Event handlers
    async def on_transport_controls_play_pressed(self, event: TransportControls.PlayPressed) -> None:
        """Handle play button press."""
        if self.is_playing:
            self.stop_audio()
        else:
            self.play_audio()
    
    async def on_transport_controls_stop_pressed(self, event: TransportControls.StopPressed) -> None:
        """Handle stop button press."""
        self.stop_audio()
    
    async def on_file_operations_panel_file_opened(self, event: FileOperationsPanel.FileOpened) -> None:
        """Handle file open request."""
        await self.load_audio_file(event.filepath)
    
    async def on_file_operations_panel_file_saved(self, event: FileOperationsPanel.FileSaved) -> None:
        """Handle file save request."""
        await self.save_audio_file(event.filepath)
    
    async def on_file_operations_panel_format_converted(self, event: FileOperationsPanel.FormatConverted) -> None:
        """Handle format conversion request."""
        try:
            self.current_status = "Converting format..."
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, AudioProcessor.convert_format, 
                event.input_path, event.output_path, event.format
            )
            self.current_status = f"Converted to {event.format}"
        except Exception as e:
            self.current_status = f"Conversion error: {str(e)}"
    
    async def on_filter_panel_filter_applied(self, event: FilterPanel.FilterApplied) -> None:
        """Handle filter application."""
        self.apply_filter(event.filter_type, **event.params)
    
    async def on_annotation_panel_annotation_added(self, event: AnnotationPanel.AnnotationAdded) -> None:
        """Handle annotation addition."""
        self.add_annotation(event.start, event.end, event.label, event.description)
    
    async def on_waveform_display_selection_changed(self, event: WaveformDisplay.SelectionChanged) -> None:
        """Handle waveform selection change."""
        self.current_status = f"Selected: {event.start:.2f}s - {event.end:.2f}s"
    
    # Key bindings
    async def action_open_file(self) -> None:
        """Open file action."""
        # In a real application, this would show a file dialog
        # For now, we'll use a simple input
        pass
    
    async def action_save_file(self) -> None:
        """Save file action."""
        if self.current_audio and self.current_audio.filepath:
            await self.save_audio_file(self.current_audio.filepath)
    
    async def action_play_pause(self) -> None:
        """Play/pause toggle action."""
        if self.is_playing:
            self.stop_audio()
        else:
            self.play_audio()
    
    async def action_stop(self) -> None:
        """Stop playback action."""
        self.stop_audio()
    
    async def action_undo(self) -> None:
        """Undo action."""
        self.undo_last_action()
    
    async def action_show_help(self) -> None:
        """Show help dialog."""
        self.current_status = "F1: Help | Q: Quit | Space: Play/Pause | S: Save | O: Open | U: Undo"


def run_audio_editor_ui(audio_file: Optional[str] = None) -> None:
    """Run the audio editor UI."""
    app = AudioEditorApp(initial_file=audio_file)
    app.run()