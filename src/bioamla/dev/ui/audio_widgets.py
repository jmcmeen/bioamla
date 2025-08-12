"""
Custom Textual widgets for the audio editor application.
"""
from textual.widgets import Static, Button, Input, Label, SelectionList, ProgressBar
from textual.containers import Horizontal, Vertical, Container
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.message import Message
from textual import events
from rich.text import Text
from rich.console import RenderableType
import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from ..core.audio_editor import AudioData, SpectrogramGenerator


class WaveformDisplay(Static):
    """Display audio waveform."""
    
    audio_data: reactive[Optional[AudioData]] = reactive(None)
    zoom_start: reactive[float] = reactive(0.0)
    zoom_end: reactive[float] = reactive(1.0)
    selection_start: reactive[Optional[float]] = reactive(None)
    selection_end: reactive[Optional[float]] = reactive(None)
    
    class SelectionChanged(Message):
        def __init__(self, start: float, end: float):
            self.start = start
            self.end = end
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height = 10
    
    def watch_audio_data(self, audio_data: Optional[AudioData]) -> None:
        """Update display when audio data changes."""
        if audio_data:
            self.zoom_end = audio_data.duration
        self.refresh()
    
    def render(self) -> RenderableType:
        """Render waveform display."""
        if not self.audio_data:
            return Text("No audio loaded", style="dim")
        
        # Simple ASCII waveform representation
        lines = []
        data = self.audio_data.data
        if data.ndim > 1:
            data = data[:, 0]  # Use first channel
        
        # Get data for current zoom range
        start_sample = int(self.zoom_start * self.audio_data.sample_rate)
        end_sample = int(self.zoom_end * self.audio_data.sample_rate)
        zoom_data = data[start_sample:end_sample]
        
        if len(zoom_data) == 0:
            return Text("No data in current view", style="dim")
        
        # Downsample for display
        display_width = min(80, self.size.width - 2)
        if len(zoom_data) > display_width:
            step = len(zoom_data) // display_width
            zoom_data = zoom_data[::step]
        
        # Create ASCII waveform
        height = 8
        max_val = np.max(np.abs(zoom_data)) if len(zoom_data) > 0 else 1.0
        if max_val == 0:
            max_val = 1.0
        
        for h in range(height):
            line = ""
            threshold = (height - h - 1) / height * max_val
            for sample in zoom_data[:display_width]:
                if abs(sample) >= threshold:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        # Add time markers
        lines.append("-" * display_width)
        time_line = f"{self.zoom_start:.2f}s"
        time_line += " " * (display_width - len(time_line) - len(f"{self.zoom_end:.2f}s"))
        time_line += f"{self.zoom_end:.2f}s"
        lines.append(time_line)
        
        # Show selection if any
        if self.selection_start is not None and self.selection_end is not None:
            sel_text = f"Selected: {self.selection_start:.2f}s - {self.selection_end:.2f}s"
            lines.append(sel_text)
        
        return Text("\n".join(lines), style="bold blue")
    
    def on_click(self, event: events.Click) -> None:
        """Handle click for selection."""
        if not self.audio_data:
            return
        
        # Calculate time position from click
        rel_x = event.x / self.size.width
        click_time = self.zoom_start + rel_x * (self.zoom_end - self.zoom_start)
        
        if self.selection_start is None:
            self.selection_start = click_time
        else:
            self.selection_end = click_time
            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start
            self.post_message(self.SelectionChanged(self.selection_start, self.selection_end))


class SpectrogramDisplay(Static):
    """Display audio spectrogram."""
    
    audio_data: reactive[Optional[AudioData]] = reactive(None)
    spectrogram_type: reactive[str] = reactive("linear")  # "linear" or "mel"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height = 15
    
    def watch_audio_data(self, audio_data: Optional[AudioData]) -> None:
        """Update display when audio data changes."""
        self.refresh()
    
    def render(self) -> RenderableType:
        """Render spectrogram display."""
        if not self.audio_data:
            return Text("No audio loaded", style="dim")
        
        try:
            # Generate spectrogram data
            if self.spectrogram_type == "mel":
                spec_data, times, freqs = SpectrogramGenerator.compute_mel_spectrogram(self.audio_data)
            else:
                spec_data, times, freqs = SpectrogramGenerator.compute_spectrogram(self.audio_data)
            
            # Create simple ASCII representation
            lines = []
            height = min(12, self.size.height - 3)
            width = min(60, self.size.width - 2)
            
            # Downsample spectrogram for display
            if spec_data.shape[1] > width:
                step = spec_data.shape[1] // width
                spec_data = spec_data[:, ::step]
            if spec_data.shape[0] > height:
                step = spec_data.shape[0] // height
                spec_data = spec_data[::step, :]
            
            # Normalize and convert to ASCII
            spec_data = spec_data[:height, :width]
            min_val, max_val = spec_data.min(), spec_data.max()
            if max_val > min_val:
                spec_data = (spec_data - min_val) / (max_val - min_val)
            
            chars = " ░▒▓█"
            for row in spec_data[::-1]:  # Flip vertically
                line = ""
                for val in row:
                    char_idx = min(int(val * len(chars)), len(chars) - 1)
                    line += chars[char_idx]
                lines.append(line)
            
            lines.append("-" * width)
            lines.append(f"0.0s{' ' * (width - 10)}{self.audio_data.duration:.1f}s")
            
            return Text("\n".join(lines), style="yellow")
        
        except Exception as e:
            return Text(f"Error generating spectrogram: {str(e)}", style="red")


class AudioInfoPanel(Static):
    """Display audio file information."""
    
    audio_data: reactive[Optional[AudioData]] = reactive(None)
    
    def watch_audio_data(self, audio_data: Optional[AudioData]) -> None:
        """Update display when audio data changes."""
        self.refresh()
    
    def render(self) -> RenderableType:
        """Render audio info."""
        if not self.audio_data:
            return Text("No audio loaded", style="dim")
        
        info_lines = [
            f"File: {self.audio_data.filepath or 'Untitled'}",
            f"Duration: {self.audio_data.duration:.2f}s",
            f"Sample Rate: {self.audio_data.sample_rate} Hz",
            f"Channels: {self.audio_data.channels}",
            f"Samples: {len(self.audio_data.data)}",
            f"Annotations: {len(self.audio_data.annotations)}"
        ]
        
        return Text("\n".join(info_lines), style="green")


class TransportControls(Container):
    """Audio transport controls (play, stop, etc.)."""
    
    class PlayPressed(Message):
        pass
    
    class StopPressed(Message):
        pass
    
    class RecordPressed(Message):
        pass
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("⏵ Play", id="play_btn", variant="primary")
            yield Button("⏹ Stop", id="stop_btn", variant="error")
            yield Button("⏺ Record", id="record_btn", variant="warning")
            yield Button("⏮ Previous", id="prev_btn")
            yield Button("⏭ Next", id="next_btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "play_btn":
            self.post_message(self.PlayPressed())
        elif event.button.id == "stop_btn":
            self.post_message(self.StopPressed())
        elif event.button.id == "record_btn":
            self.post_message(self.RecordPressed())


class FilterPanel(Container):
    """Audio filter controls."""
    
    class FilterApplied(Message):
        def __init__(self, filter_type: str, **params):
            self.filter_type = filter_type
            self.params = params
            super().__init__()
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Audio Filters")
            with Horizontal():
                yield Button("Low Pass", id="lowpass_btn")
                yield Button("High Pass", id="highpass_btn")
                yield Button("Band Pass", id="bandpass_btn")
                yield Button("Notch", id="notch_btn")
            with Horizontal():
                yield Label("Frequency:")
                yield Input(placeholder="1000", id="freq_input")
                yield Button("Apply", id="apply_filter_btn", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter button presses."""
        if event.button.id == "apply_filter_btn":
            freq_input = self.query_one("#freq_input", Input)
            try:
                freq = float(freq_input.value or "1000")
                # This would need to be extended based on selected filter type
                self.post_message(self.FilterApplied("lowpass", frequency=freq))
            except ValueError:
                pass


class AnnotationPanel(Container):
    """Annotation management panel."""
    
    class AnnotationAdded(Message):
        def __init__(self, start: float, end: float, label: str, description: str = ""):
            self.start = start
            self.end = end
            self.label = label
            self.description = description
            super().__init__()
    
    class AnnotationDeleted(Message):
        def __init__(self, annotation_id: int):
            self.annotation_id = annotation_id
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.annotations: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Annotations")
            with Horizontal():
                yield Input(placeholder="Start time", id="start_time_input")
                yield Input(placeholder="End time", id="end_time_input")
            with Horizontal():
                yield Input(placeholder="Label", id="label_input")
                yield Button("Add", id="add_annotation_btn", variant="success")
            yield SelectionList(*[], id="annotation_list")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle annotation button presses."""
        if event.button.id == "add_annotation_btn":
            start_input = self.query_one("#start_time_input", Input)
            end_input = self.query_one("#end_time_input", Input)
            label_input = self.query_one("#label_input", Input)
            
            try:
                start = float(start_input.value or "0")
                end = float(end_input.value or "1")
                label = label_input.value or "Untitled"
                
                self.post_message(self.AnnotationAdded(start, end, label))
                
                # Clear inputs
                start_input.value = ""
                end_input.value = ""
                label_input.value = ""
                
            except ValueError:
                pass
    
    def update_annotations(self, annotations: List[Dict[str, Any]]) -> None:
        """Update the annotation list display."""
        self.annotations = annotations
        annotation_list = self.query_one("#annotation_list", SelectionList)
        
        # Create list items
        items = []
        for ann in annotations:
            text = f"{ann['start_time']:.2f}s-{ann['end_time']:.2f}s: {ann['label']}"
            items.append((text, ann['id']))
        
        annotation_list.clear_options()
        for text, ann_id in items:
            annotation_list.add_option((text, ann_id))


class FileOperationsPanel(Container):
    """File operations panel."""
    
    class FileOpened(Message):
        def __init__(self, filepath: str):
            self.filepath = filepath
            super().__init__()
    
    class FileSaved(Message):
        def __init__(self, filepath: str):
            self.filepath = filepath
            super().__init__()
    
    class FormatConverted(Message):
        def __init__(self, input_path: str, output_path: str, format: str):
            self.input_path = input_path
            self.output_path = output_path
            self.format = format
            super().__init__()
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("File Operations")
            with Horizontal():
                yield Button("📁 Open", id="open_btn", variant="primary")
                yield Button("💾 Save", id="save_btn", variant="success")
                yield Button("💾 Save As", id="save_as_btn")
            with Horizontal():
                yield Input(placeholder="File path", id="file_path_input")
                yield Button("Convert", id="convert_btn", variant="warning")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle file operation button presses."""
        file_path_input = self.query_one("#file_path_input", Input)
        
        if event.button.id == "open_btn":
            if file_path_input.value:
                self.post_message(self.FileOpened(file_path_input.value))
        elif event.button.id == "save_btn":
            if file_path_input.value:
                self.post_message(self.FileSaved(file_path_input.value))
        elif event.button.id == "convert_btn":
            if file_path_input.value:
                # Simple format conversion - would need more sophisticated UI
                input_path = file_path_input.value
                output_path = input_path.rsplit('.', 1)[0] + '.wav'
                self.post_message(self.FormatConverted(input_path, output_path, 'wav'))


class StatusBar(Static):
    """Application status bar."""
    
    status: reactive[str] = reactive("Ready")
    
    def render(self) -> RenderableType:
        """Render status bar."""
        return Text(self.status, style="bold white on blue")


class ProgressDisplay(Container):
    """Progress display for long operations."""
    
    def compose(self) -> ComposeResult:
        yield Label("Processing...")
        yield ProgressBar(id="progress_bar")
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """Update progress display."""
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        progress_bar.progress = progress
        if message:
            label = self.query_one(Label)
            label.update(message)