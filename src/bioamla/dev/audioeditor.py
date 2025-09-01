"""
Audio Editor CLI - Both interactive UI and command-line scripting modes.
"""
import click
import sys
from pathlib import Path

from .core.audio_editor import (
    AudioProcessor, AudioFilters, 
    AnnotationManager, SpectrogramGenerator
)
from .ui.main_app import run_audio_editor_ui


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def main(ctx, verbose):
    """
    Audio Editor - Professional audio editing with both GUI and CLI modes.
    
    Run without subcommands to start the interactive UI.
    Use subcommands for command-line scripting.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument('audio_file', required=False)
def ui(audio_file):
    """Start the interactive Textual UI."""
    if audio_file and not Path(audio_file).exists():
        click.echo(f"Error: Audio file '{audio_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        run_audio_editor_ui(audio_file)
    except Exception as e:
        click.echo(f"Error starting UI: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--format', '-f', help='Output format (wav, mp3, flac, etc.)')
def convert(input_file, output_file, format):
    """Convert audio file format."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        click.echo(f"Converting {input_file} to {output_file}...")
        AudioProcessor.convert_format(input_file, output_file, format)
        click.echo("Conversion completed successfully.")
    except Exception as e:
        click.echo(f"Conversion failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--start', '-s', type=float, default=0.0, help='Start time in seconds')
@click.option('--end', '-e', type=float, help='End time in seconds')
@click.option('--duration', '-d', type=float, help='Duration in seconds (alternative to end)')
def trim(input_file, output_file, start, end, duration):
    """Trim audio file to specified time range."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        # Load audio
        audio_data = AudioProcessor.load_audio(input_file)
        
        # Calculate end time
        if duration is not None:
            end = start + duration
        elif end is None:
            end = audio_data.duration
        
        click.echo(f"Trimming {input_file} from {start}s to {end}s...")
        
        # Trim audio
        trimmed_audio = AudioProcessor.trim_audio(audio_data, start, end)
        
        # Save result
        AudioProcessor.save_audio(trimmed_audio, output_file)
        click.echo(f"Trimmed audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Trim operation failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--gain', '-g', type=float, default=0.0, help='Gain in dB')
@click.option('--normalize', '-n', is_flag=True, help='Normalize audio')
@click.option('--target-level', '-t', type=float, default=-3.0, help='Target level for normalization (dB)')
@click.option('--fade-in', type=float, default=0.0, help='Fade in duration (seconds)')
@click.option('--fade-out', type=float, default=0.0, help='Fade out duration (seconds)')
def process(input_file, output_file, gain, normalize, target_level, fade_in, fade_out):
    """Apply basic audio processing (gain, normalize, fade)."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        # Load audio
        audio_data = AudioProcessor.load_audio(input_file)
        click.echo(f"Processing {input_file}...")
        
        # Apply gain
        if gain != 0.0:
            click.echo(f"Applying gain: {gain} dB")
            AudioProcessor.apply_gain(audio_data, gain)
        
        # Normalize
        if normalize:
            click.echo(f"Normalizing to {target_level} dB")
            AudioProcessor.normalize(audio_data, target_level)
        
        # Apply fades
        if fade_in > 0.0 or fade_out > 0.0:
            click.echo(f"Applying fades: in={fade_in}s, out={fade_out}s")
            AudioProcessor.apply_fade(audio_data, fade_in, fade_out)
        
        # Save result
        AudioProcessor.save_audio(audio_data, output_file)
        click.echo(f"Processed audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Processing failed: {str(e)}", err=True)
        sys.exit(1)


@main.group()
def filter():
    """Audio filtering operations."""
    pass


@filter.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--frequency', '-f', type=float, required=True, help='Cutoff frequency (Hz)')
@click.option('--order', '-o', type=int, default=5, help='Filter order')
def lowpass(input_file, output_file, frequency, order):
    """Apply low-pass filter."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        audio_data = AudioProcessor.load_audio(input_file)
        click.echo(f"Applying low-pass filter (cutoff: {frequency} Hz, order: {order})")
        
        AudioFilters.low_pass_filter(audio_data, frequency, order)
        AudioProcessor.save_audio(audio_data, output_file)
        click.echo(f"Filtered audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Filter operation failed: {str(e)}", err=True)
        sys.exit(1)


@filter.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--frequency', '-f', type=float, required=True, help='Cutoff frequency (Hz)')
@click.option('--order', '-o', type=int, default=5, help='Filter order')
def highpass(input_file, output_file, frequency, order):
    """Apply high-pass filter."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        audio_data = AudioProcessor.load_audio(input_file)
        click.echo(f"Applying high-pass filter (cutoff: {frequency} Hz, order: {order})")
        
        AudioFilters.high_pass_filter(audio_data, frequency, order)
        AudioProcessor.save_audio(audio_data, output_file)
        click.echo(f"Filtered audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Filter operation failed: {str(e)}", err=True)
        sys.exit(1)


@filter.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--low', '-l', type=float, required=True, help='Low cutoff frequency (Hz)')
@click.option('--high', '-h', type=float, required=True, help='High cutoff frequency (Hz)')
@click.option('--order', '-o', type=int, default=5, help='Filter order')
def bandpass(input_file, output_file, low, high, order):
    """Apply band-pass filter."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        audio_data = AudioProcessor.load_audio(input_file)
        click.echo(f"Applying band-pass filter ({low} Hz - {high} Hz, order: {order})")
        
        AudioFilters.band_pass_filter(audio_data, low, high, order)
        AudioProcessor.save_audio(audio_data, output_file)
        click.echo(f"Filtered audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Filter operation failed: {str(e)}", err=True)
        sys.exit(1)


@filter.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--frequency', '-f', type=float, required=True, help='Notch frequency (Hz)')
@click.option('--quality', '-q', type=float, default=30, help='Quality factor')
def notch(input_file, output_file, frequency, quality):
    """Apply notch filter."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        audio_data = AudioProcessor.load_audio(input_file)
        click.echo(f"Applying notch filter (frequency: {frequency} Hz, Q: {quality})")
        
        AudioFilters.notch_filter(audio_data, frequency, quality)
        AudioProcessor.save_audio(audio_data, output_file)
        click.echo(f"Filtered audio saved to {output_file}")
        
    except Exception as e:
        click.echo(f"Filter operation failed: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file')
@click.option('--output', '-o', help='Output image file for spectrogram')
@click.option('--type', '-t', type=click.Choice(['linear', 'mel']), default='linear', 
              help='Spectrogram type')
@click.option('--show-info', is_flag=True, help='Show audio file information')
def analyze(input_file, output, type, show_info):
    """Analyze audio file and optionally generate spectrogram."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        # Load audio
        audio_data = AudioProcessor.load_audio(input_file)
        
        # Show basic info
        if show_info:
            click.echo("\nAudio File Information:")
            click.echo(f"File: {input_file}")
            click.echo(f"Duration: {audio_data.duration:.2f} seconds")
            click.echo(f"Sample Rate: {audio_data.sample_rate} Hz")
            click.echo(f"Channels: {audio_data.channels}")
            click.echo(f"Samples: {len(audio_data.data)}")
        
        # Generate spectrogram if output specified
        if output:
            import matplotlib.pyplot as plt
            
            click.echo(f"Generating {type} spectrogram...")
            
            if type == 'mel':
                spec_data, times, freqs = SpectrogramGenerator.compute_mel_spectrogram(audio_data)
                title = "Mel Spectrogram"
                ylabel = "Mel Frequency"
            else:
                spec_data, times, freqs = SpectrogramGenerator.compute_spectrogram(audio_data)
                title = "Linear Spectrogram"
                ylabel = "Frequency (Hz)"
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.imshow(spec_data, aspect='auto', origin='lower', 
                      extent=[times[0], times[-1], freqs[0], freqs[-1]])
            plt.colorbar(label='Magnitude (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel(ylabel)
            plt.title(f"{title} - {Path(input_file).name}")
            plt.tight_layout()
            
            plt.savefig(output, dpi=150, bbox_inches='tight')
            click.echo(f"Spectrogram saved to {output}")
        
    except Exception as e:
        click.echo(f"Analysis failed: {str(e)}", err=True)
        sys.exit(1)


@main.group()
def annotate():
    """Audio annotation operations."""
    pass


@annotate.command()
@click.argument('input_file')
@click.option('--start', '-s', type=float, required=True, help='Start time (seconds)')
@click.option('--end', '-e', type=float, required=True, help='End time (seconds)')
@click.option('--label', '-l', required=True, help='Annotation label')
@click.option('--description', '-d', default='', help='Annotation description')
@click.option('--output', '-o', help='Output annotation file (JSON)')
def add(input_file, start, end, label, description, output):
    """Add annotation to audio file."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        # Load audio
        audio_data = AudioProcessor.load_audio(input_file)
        
        # Add annotation
        AnnotationManager.add_annotation(audio_data, start, end, label, description)
        click.echo(f"Added annotation: {label} ({start}s - {end}s)")
        
        # Save annotations if output specified
        if output:
            AnnotationManager.export_annotations(audio_data, output)
            click.echo(f"Annotations saved to {output}")
        
    except Exception as e:
        click.echo(f"Annotation operation failed: {str(e)}", err=True)
        sys.exit(1)


@annotate.command()
@click.argument('input_file')
@click.argument('annotation_file')
def load(input_file, annotation_file):
    """Load annotations from file and display them."""
    if not Path(input_file).exists():
        click.echo(f"Error: Input file '{input_file}' not found.", err=True)
        sys.exit(1)
    
    if not Path(annotation_file).exists():
        click.echo(f"Error: Annotation file '{annotation_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        # Load audio and annotations
        audio_data = AudioProcessor.load_audio(input_file)
        AnnotationManager.import_annotations(audio_data, annotation_file)
        
        click.echo(f"Loaded {len(audio_data.annotations)} annotations:")
        for i, ann in enumerate(audio_data.annotations):
            click.echo(f"  {i+1}. {ann['start_time']:.2f}s - {ann['end_time']:.2f}s: {ann['label']}")
            if ann.get('description'):
                click.echo(f"      {ann['description']}")
        
    except Exception as e:
        click.echo(f"Failed to load annotations: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument('audio_file', required=False)
def run(audio_file):
    """
    Run the audio editor. 
    
    If no audio file is provided, starts the interactive UI.
    If an audio file is provided, loads it in the UI.
    """
    if audio_file and not Path(audio_file).exists():
        click.echo(f"Error: Audio file '{audio_file}' not found.", err=True)
        sys.exit(1)
    
    try:
        run_audio_editor_ui(audio_file)
    except KeyboardInterrupt:
        click.echo("\nAudio editor closed.")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


# Default behavior: if no subcommand is provided, start the UI
@main.result_callback()
@click.pass_context
def default_command(ctx, result, **kwargs):
    """Default behavior when no subcommand is specified."""
    if ctx.invoked_subcommand is None:
        # No subcommand was called, start the UI
        try:
            run_audio_editor_ui()
        except KeyboardInterrupt:
            click.echo("\nAudio editor closed.")
        except Exception as e:
            click.echo(f"Error starting audio editor: {str(e)}", err=True)
            sys.exit(1)


if __name__ == '__main__':
    main()