"""Audio file operations (info, convert, segment, visualize)."""

from pathlib import Path

import click

from bioamla.cli.options import frame_feature_options, picture_options, segment_options
from bioamla.exceptions import BioamlaError


@click.group()
def audio() -> None:
    """Audio file operations (info, convert, segment, visualize)."""
    pass


_RECORDING_LABELS = {
    "recorded_at": "Recorded at",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "device_make": "Device make",
    "device_model": "Device model",
    "device_id": "Device ID",
    "gain": "Gain",
    "temperature_c": "Temperature",
    "battery_voltage": "Battery",
    "metadata_sources": "Sources",
}


def _print_recording_metadata(meta_dict: dict) -> None:
    from bioamla.cli.console import print_kv

    for key, label in _RECORDING_LABELS.items():
        value = meta_dict.get(key)
        if value is None or value == "":
            continue
        if key == "temperature_c":
            value = f"{value} °C"
        elif key == "battery_voltage":
            value = f"{value} V"
        print_kv(label, value)


@audio.command("info")
@click.argument("path")
@click.option(
    "--metadata",
    "show_metadata",
    is_flag=True,
    help="Also show recorder metadata (GUANO, AudioMoth, WAMD, bext) if the WAV has any.",
)
def audio_info(path: str, show_metadata: bool) -> None:
    """Display audio file information (header only; the file is not decoded)."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    described = run_operation("audio.info", file=path, metadata=show_metadata)
    if json_mode():
        emit_result(described.to_dict())
        return

    info = described.info
    print_kv("File", path)
    print_kv("Format", info.format)
    print_kv("Duration", f"{info.duration:.2f}s")
    print_kv("Sample rate", f"{info.sample_rate} Hz")
    print_kv("Channels", info.channels)
    print_kv("Samples", info.samples)
    if info.bit_depth is not None:
        print_kv("Bit depth", info.bit_depth)
    if info.subtype:
        print_kv("Subtype", info.subtype)

    meta = described.recording_metadata
    if meta is not None and not meta.is_empty:
        from bioamla.cli.console import print_header

        print_header("\nRecorder metadata")
        _print_recording_metadata(meta.to_dict())


@audio.command("metadata")
@click.argument("path")
def audio_metadata(path: str) -> None:
    """Show recorder metadata embedded in a WAV file.

    Prints the flattened view (when, where, which device) that bioamla datasets use.
    With `bioamla --json`, the result holds the flattened view (``summary``) and
    everything riffy decoded, per standard (``standards``). For chunk-level
    inspection and editing use the `riffy` command directly.
    """
    from bioamla.cli.console import print_kv, print_warning
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    # Both reads are header-only. The flat summary is what datasets use;
    # "standards" has every decoded field, per standard.
    report = run_operation("audio.metadata", file=path)
    if json_mode():
        emit_result(report.to_dict())
        return
    meta = report.summary
    print_kv("File", path)
    if meta.is_empty:
        print_warning("No recorder metadata found (no GUANO, INFO, bext, AudioMoth or WAMD).")
        return
    _print_recording_metadata(meta.to_dict())


@audio.command("list")
@click.argument("path")
@click.option(
    "--recursive/--no-recursive",
    "-r",
    default=True,
    help="Search subdirectories (default: recursive)",
)
def audio_list(path: str, recursive: bool) -> None:
    """List audio files in a directory."""
    from bioamla.cli.console import DIM_STYLE, echo, print_header
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    listing = run_operation("audio.list", path=path, recursive=recursive)
    if emit_result(listing.to_dict()):
        return
    if not listing.files:
        echo("No audio files found", style=DIM_STYLE)
        return

    print_header(f"Found {listing.count} audio file(s):")
    for f in listing.files:
        echo(f"  {f}")


@audio.command("play")
@click.argument("path")
@click.option("--loop", "-l", is_flag=True, help="Loop continuously until interrupted")
def audio_play(path: str, loop: bool) -> None:
    """Play an audio file through the default output device.

    Blocks until playback finishes; press Ctrl-C to stop.
    """
    import time

    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import json_mode

    if json_mode():
        raise click.ClickException("audio play has no --json output (it plays sound)")

    player = run_operation("audio.play", file=path, loop=loop)

    echo(f"Playing {path} ({player.duration:.2f}s){' [loop]' if loop else ''} — Ctrl-C to stop")
    try:
        while player.is_playing:
            time.sleep(0.1)
    except KeyboardInterrupt:
        player.stop()
        echo("\nStopped.")
        return

    print_success("Done.")


@audio.command("features")
@click.argument("path")
@click.option(
    "--output", "-o", default=None, help="Write the per-frame table here (.csv or .parquet)"
)
@frame_feature_options
def audio_features(
    path: str,
    output: str | None,
    trajectory: bool,
    gate_db: float,
    no_denoise: bool,
    n_fft: int | None,
    hop: int | None,
    n_mels: int,
    n_mfcc: int,
    f_min: float,
    f_max: float | None,
) -> None:
    """Frame-level spectral descriptors and MFCCs, one row per frame.

    Each frame gets its level, spectral centroid, spread, flatness, crest,
    rolloff, flux, peak frequency, band edges and MFCCs; --trajectory adds its
    position on the recording's three main timbre axes. Without --output, a
    summary of each descriptor is shown; with --json and no --output, the
    per-frame records are in the result.

    \b
    Examples:
        bioamla audio features wren.wav -o wren.features.csv
        bioamla audio features wren.wav -o wren.parquet --trajectory
        bioamla --json audio features wren.wav --trajectory
    """
    from bioamla.audio import DESCRIPTORS
    from bioamla.cli.operations import run_operation
    from bioamla.cli.options import frame_feature_config
    from bioamla.cli.output import emit_file_result

    try:
        config = frame_feature_config(no_denoise, n_fft, hop, n_mels, n_mfcc, f_min, f_max)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    result = run_operation(
        "audio.features",
        file=path,
        output=output,
        config=config,
        trajectory=trajectory,
        gate_db=gate_db,
    )

    summary = result.to_dict()
    extra = {k: v for k, v in summary.items() if k not in ("file", "output_path")}
    if output is None:
        extra["frames"] = result.table.to_dict("records")
    if emit_file_result(path, output=result.output_path, **extra):
        return

    from bioamla.cli.console import print_kv, print_success, print_table

    print_kv("File", path)
    print_kv("Frames", f"{summary['n_frames']} ({summary['duration']:.2f}s)")
    if result.trajectory is not None:
        ratio = ", ".join(f"{v:.0%}" for v in summary["explained_variance_ratio"])
        print_kv("Trajectory", f"{int(result.trajectory.active.sum())} frames, variance {ratio}")
    if output is not None:
        print_success(f"Per-frame table saved to: {result.output_path}")
        return
    table = result.table
    rows = []
    for name in DESCRIPTORS:
        q = table[name].quantile([0.1, 0.5, 0.9])
        rows.append([name, *(f"{v:.4g}" for v in q)])
    print_table("Descriptors over frames", ["descriptor", "p10", "median", "p90"], rows)
    print_kv("Hint", "use --output to save every frame (.csv or .parquet)")


@audio.command("convert")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", default=None, type=int, help="Target sample rate")
@click.option(
    "--channels",
    "-c",
    default=None,
    type=int,
    help="Target number of channels: 1 or 2 (default: keep the file's own)",
)
@click.option(
    "--bit-depth",
    "-b",
    default=None,
    type=click.Choice(["8", "16", "24", "32"]),
    help="Bits per sample: WAV 8, 16, 24 or 32; FLAC 8, 16 or 24 (default: the source's "
    "when the output holds it, else 16)",
)
@click.option(
    "--format",
    "-f",
    default=None,
    type=click.Choice(["wav", "mp3", "flac", "ogg"]),
    help="Output format",
)
def audio_convert(
    input_path: str,
    output_path: str,
    sample_rate: int,
    channels: int,
    bit_depth: str | None,
    format: str,
) -> None:
    """Convert audio file format or properties."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation(
        "audio.convert",
        file=input_path,
        output=output_path,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=int(bit_depth) if bit_depth else None,
        format=format,
    )
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Converted: {output_path}")


@audio.command("segment")
@click.argument("input_path")
@click.argument("output_dir")
@segment_options
@click.option(
    "--prefix", "-p", default=None, help="Prefix for output filenames (default: input filename)"
)
def audio_segment(
    input_path: str,
    output_dir: str,
    duration: float,
    overlap: float,
    out_format: str,
    keep_short: bool,
    prefix: str,
) -> None:
    """Segment audio file into fixed-duration clips, named PREFIX_seg0000.FORMAT, ...

    A last clip shorter than the duration is left out unless --keep-short.
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    result = run_operation(
        "audio.segment",
        file=input_path,
        output_dir=output_dir,
        duration=duration,
        overlap=overlap,
        format=out_format,
        prefix=prefix,
        include_partial=keep_short,
    )
    if emit_file_result(input_path, output=output_dir, segments=result.count):
        return
    print_success(f"Created {result.count} segments in {output_dir}")


@audio.command("trim")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--start", "-s", default=0.0, type=float, help="Start time in seconds")
@click.option("--end", "-e", default=None, type=float, help="End time in seconds")
@click.option(
    "--duration-seconds", "-d", "duration", default=None, type=float, help="Duration in seconds"
)
def audio_trim(
    input_path: str, output_path: str, start: float, end: float, duration: float
) -> None:
    """Trim audio file to specified time range."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation(
        "audio.trim", file=input_path, output=output_path, start=start, end=end, duration=duration
    )
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Trimmed audio saved to: {output_path}")


@audio.command("normalize")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--target-db", "-t", default=-3.0, type=float, help="Target peak level in dB")
@click.option("--method", "-m", type=click.Choice(["peak", "rms"]), default="peak", help="Method")
def audio_normalize(input_path: str, output_path: str, target_db: float, method: str) -> None:
    """Normalize audio amplitude."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation(
        "audio.normalize", file=input_path, output=output_path, target_db=target_db, method=method
    )
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Normalized audio saved to: {output_path}")


@audio.command("resample")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--sample-rate", "-r", required=True, type=int, help="Target sample rate in Hz")
def audio_resample(input_path: str, output_path: str, sample_rate: int) -> None:
    """Resample audio to a different sample rate."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.resample", file=input_path, output=output_path, sample_rate=sample_rate)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Resampled audio saved to: {output_path}")


@audio.command("filter")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--lowpass", default=None, type=float, help="Lowpass cutoff frequency in Hz")
@click.option("--highpass", default=None, type=float, help="Highpass cutoff frequency in Hz")
@click.option("--bandpass-low", default=None, type=float, help="Bandpass low frequency in Hz")
@click.option("--bandpass-high", default=None, type=float, help="Bandpass high frequency in Hz")
@click.option("--order", default=5, type=int, help="Filter order (default: 5)")
def audio_filter(
    input_path: str,
    output_path: str,
    lowpass: float,
    highpass: float,
    bandpass_low: float,
    bandpass_high: float,
    order: int,
) -> None:
    """Apply frequency filter to audio file."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    result = run_operation(
        "audio.filter",
        file=input_path,
        output=output_path,
        lowpass=lowpass,
        highpass=highpass,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        order=order,
    )
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Applied {result.description} to: {output_path}")


@audio.command("denoise")
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--strength", default=1.0, type=float, help="Noise reduction strength (0-2, default: 1.0)"
)
def audio_denoise(input_path: str, output_path: str, strength: float) -> None:
    """Apply spectral noise reduction to audio file."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.denoise", file=input_path, output=output_path, strength=strength)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Denoised audio saved to: {output_path}")


@audio.command("pitch-shift")
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--steps", "-n", required=True, type=float, help="Semitones to shift (positive raises pitch)"
)
def audio_pitch_shift(input_path: str, output_path: str, steps: float) -> None:
    """Shift pitch up/down without changing duration."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.pitch-shift", file=input_path, output=output_path, steps=steps)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Pitch-shifted audio ({steps:+g} semitones) saved to: {output_path}")


@audio.command("time-stretch")
@click.argument("input_path")
@click.argument("output_path")
@click.option(
    "--rate", "-r", required=True, type=float, help="Stretch factor (>1 faster, <1 slower)"
)
def audio_time_stretch(input_path: str, output_path: str, rate: float) -> None:
    """Time-stretch audio without changing pitch."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.time-stretch", file=input_path, output=output_path, rate=rate)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Time-stretched audio (rate {rate:g}) saved to: {output_path}")


@audio.command("add-noise")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--snr-db", required=True, type=float, help="Target SNR in dB (lower = more noise)")
@click.option("--seed", default=None, type=int, help="RNG seed for reproducible noise")
def audio_add_noise(input_path: str, output_path: str, snr_db: float, seed: int) -> None:
    """Add Gaussian white noise at a target SNR."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.add-noise", file=input_path, output=output_path, snr_db=snr_db, seed=seed)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Noisy audio (SNR {snr_db:g} dB) saved to: {output_path}")


@audio.command("gain")
@click.argument("input_path")
@click.argument("output_path")
@click.option("--gain-db", required=True, type=float, help="Gain in dB (positive amplifies)")
def audio_gain(input_path: str, output_path: str, gain_db: float) -> None:
    """Apply a fixed gain (dB) to audio."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    run_operation("audio.gain", file=input_path, output=output_path, gain_db=gain_db)
    if emit_file_result(input_path, output=output_path):
        return
    print_success(f"Gain-adjusted audio ({gain_db:+g} dB) saved to: {output_path}")


@audio.command("visualize")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output image file path")
@click.option(
    "--type",
    "-t",
    "viz_type",
    default="mel",
    type=click.Choice(["mel", "stft", "mfcc", "waveform"]),
    help="Visualization type",
)
@picture_options
@click.option(
    "--box",
    "boxes",
    multiple=True,
    metavar="START,END[,LOW,HIGH][,LABEL]",
    help="Outline a region: seconds, and optionally its band in Hz and a label "
    "(repeatable). Without a band it spans the full height.",
)
@click.option(
    "--annotations",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Outline every annotation in this file (Raven .txt, bioamla .json, CSV, Arbimon), "
    "labelled",
)
def audio_visualize(
    path: str,
    output: str,
    viz_type: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_mfcc: int,
    cmap: str,
    dpi: int,
    legend: bool,
    boxes: tuple[str, ...],
    annotations: str | None,
) -> None:
    """Generate audio visualization (spectrogram, waveform, MFCC) for a single file.

    \b
    Examples:
        # Mel spectrogram (default), auto-named <stem>_mel.png
        bioamla audio visualize rec.wav
        # Linear STFT spectrogram to a chosen path
        bioamla audio visualize rec.wav --type stft -o spec.png
        # Waveform plot
        bioamla audio visualize rec.wav --type waveform -o wave.png
        # Outline detections, and one more region with its band
        bioamla audio visualize rec.wav --annotations det.csv --box 1.5,2.0,1000,4000,frog
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    output_path = output or f"{Path(path).stem}_{viz_type}.png"
    run_operation(
        "audio.visualize",
        file=path,
        output=output_path,
        viz_type=viz_type,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        cmap=cmap,
        dpi=dpi,
        legend=legend,
        boxes=[_parse_box(b) for b in boxes],
        annotations=annotations,
    )
    if emit_file_result(path, output=output_path):
        return
    print_success(f"Visualization saved to: {output_path}")


def _parse_box(text: str) -> dict:
    """``START,END[,LOW,HIGH][,LABEL]`` as a box request (the box itself is checked
    when it's drawn). Empty band edges (``1,2,,,frog``) mean no band."""
    parts = [p.strip() for p in text.split(",", 4)]  # a label may hold commas
    try:
        start, end = float(parts[0]), float(parts[1])
        rest = parts[2:]
        band = [float(p) if p else None for p in rest[:2]] if len(rest) >= 2 else []
    except (ValueError, IndexError):
        band = None
    if band is None or len(parts) < 2:
        raise click.BadParameter(
            f"{text!r}: give START,END[,LOW,HIGH][,LABEL] (seconds, Hz)", param_hint="--box"
        )
    box: dict = {"start": start, "end": end}
    if len(rest) == 1:  # START,END,LABEL
        box["label"] = rest[0]
    elif rest:
        box.update(low_freq=band[0], high_freq=band[1])
        if len(rest) > 2:
            box["label"] = rest[2]
    return box


@audio.command("video")
@click.argument("path")
@click.option("--output", "-o", default=None, help="Output video path (default: <stem>.mp4)")
@click.option(
    "--type",
    "-t",
    "viz_type",
    default="mel",
    type=click.Choice(["mel", "stft", "waveform", "trajectory"]),
    help="Spectrogram type, a waveform-only video, or the MFCC-PCA timbre trajectory",
)
@click.option(
    "--motion",
    "-m",
    default="scroll",
    type=click.Choice(["scroll", "page"]),
    help="scroll: picture moves past a fixed playhead; page: playhead moves across a held picture",
)
@click.option("--start", "-s", default=0.0, type=float, help="Start at this many seconds")
@click.option("--end", "-e", default=None, type=float, help="Stop at this many seconds")
@click.option("--fps", default=30, type=int, help="Frames per second")
@click.option(
    "--window",
    "-w",
    default=4.0,
    type=float,
    help="Seconds of audio across the frame; 0 = whole recording (page mode)",
)
@click.option("--size", default="1280x720", help="Frame size as WIDTHxHEIGHT")
@click.option(
    "--cmap", default=None, help="Colormap name (default: magma; plasma for -t trajectory)"
)
@click.option("--n-fft", default=2048, type=int, help="FFT window size")
@click.option("--hop-length", default=512, type=int, help="Hop length")
@click.option("--n-mels", default=128, type=int, help="Number of mel bands")
@click.option("--fmax", default=None, type=float, help="Highest frequency shown (default: Nyquist)")
@click.option("--db-range", default=80.0, type=float, help="Dynamic range in dB")
@click.option(
    "--playhead", default=0.5, type=float, help="Playhead position across the frame, 0 to 1"
)
@click.option("--waveform/--no-waveform", default=True, help="Scrolling waveform strip")
@click.option("--axes/--no-axes", default=True, help="Frequency ticks and time readout")
@click.option("--audio/--no-audio", "include_audio", default=True, help="Mux the audio track")
@click.option(
    "--trail",
    default=None,
    type=float,
    help="Trajectory: fade out path older than this many seconds (default: keep it all)",
)
@click.option(
    "--orbit/--no-orbit", default=True, help="Trajectory: turn the view slowly (default: on)"
)
@click.option(
    "--spectrogram/--no-spectrogram",
    default=True,
    help="Trajectory: scrolling mel spectrogram strip along the bottom (default: on)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress the progress bar")
def audio_video(
    path: str,
    output: str | None,
    viz_type: str,
    motion: str,
    start: float,
    end: float | None,
    fps: int,
    window: float,
    size: str,
    cmap: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmax: float | None,
    db_range: float,
    playhead: float,
    waveform: bool,
    axes: bool,
    include_audio: bool,
    trail: float | None,
    orbit: bool,
    spectrogram: bool,
    quiet: bool,
) -> None:
    """Render a spectrogram, waveform or trajectory video of a file, in sync with its audio.

    Needs the ffmpeg command on PATH.

    \b
    Examples:
        # Scrolling mel spectrogram with waveform strip, auto-named <stem>.mp4
        bioamla audio video wren.wav
        # Linear STFT, 8 seconds across the frame, 1080p, no waveform
        bioamla audio video wren.wav -t stft -w 8 --size 1920x1080 --no-waveform
        # Playhead at the right edge, so only what has played is shown
        bioamla audio video wren.wav --playhead 1
        # Whole recording held still, playhead sweeping across it
        bioamla audio video wren.wav -m page -w 0
        # Waveform-only video, 10-second pages
        bioamla audio video wren.wav -t waveform -m page -w 10
        # Just 1:30 to 2:00 of a long recording
        bioamla audio video night.wav -s 90 -e 120

        # MFCC-PCA timbre trajectory, path fading after 4 seconds
        bioamla audio video wren.wav -t trajectory --trail 4

    With -t trajectory, --window sets the spectrogram strip's span and --cmap
    colours the path (default plasma); the spectrogram-only options don't apply.
    """
    from bioamla.cli.console import ProgressBar, print_success
    from bioamla.cli.output import emit_file_result

    try:
        width, height = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise click.BadParameter(
            "expected WIDTHxHEIGHT, e.g. 1280x720", param_hint="--size"
        ) from None
    output_path = output or f"{Path(path).stem}.mp4"

    ctx = click.get_current_context()
    given = {
        name
        for name in ctx.params
        if ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
    }
    if viz_type == "trajectory":
        _reject_options(given, _SPECTROGRAM_ONLY, "don't apply to -t trajectory")
        if window <= 0:
            raise click.BadParameter(
                "-t trajectory needs a positive span for its spectrogram strip",
                param_hint="--window",
            )
    else:
        _reject_options(given, ("trail", "orbit", "spectrogram"), "only apply to -t trajectory")

    from bioamla.cli.operations import run_operation

    bar = ProgressBar(description="Rendering frames", disable=quiet)
    with bar:
        result = run_operation(
            "audio.video",
            on_progress=bar.on_progress,
            file=path,
            output=output_path,
            viz_type=viz_type,
            motion=motion,
            start=start,
            end=end,
            fps=fps,
            window=window,
            width=width,
            height=height,
            cmap=cmap,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=fmax,
            db_range=db_range,
            playhead=playhead,
            waveform=waveform,
            axes=axes,
            include_audio=include_audio,
            trail=trail,
            orbit=orbit,
            spectrogram=spectrogram,
        )

    extra = {}
    if viz_type == "trajectory":
        extra = {
            "active_frames": result.active_frames,
            "explained_variance_ratio": result.explained_variance_ratio,
        }
    if emit_file_result(
        path,
        output=result.output_path,
        frames=result.frames,
        width=result.width,
        height=result.height,
        duration=result.duration,
        **extra,
    ):
        return
    detail = f"{result.frames} frames, {result.width}x{result.height}, {result.duration:.1f}s"
    if viz_type == "trajectory":
        detail += "; axes " + ", ".join(f"{v:.0%}" for v in result.explained_variance_ratio)
    print_success(f"Video saved to: {result.output_path} ({detail})")


_SPECTROGRAM_ONLY = (
    "motion",
    "n_fft",
    "hop_length",
    "n_mels",
    "fmax",
    "db_range",
    "playhead",
    "waveform",
)


def _reject_options(given: set[str], names: tuple[str, ...], reason: str) -> None:
    """UsageError naming the options in ``names`` that were set on the command line."""
    misplaced = sorted(given & set(names))
    if not misplaced:
        return
    flags = ", ".join("--" + n.replace("_", "-") for n in misplaced)
    if len(misplaced) == 1:  # "doesn't apply" / "only applies"
        reason = reason.replace("don't apply", "doesn't apply").replace(
            "only apply", "only applies"
        )
    raise click.UsageError(f"{flags} {reason}")
