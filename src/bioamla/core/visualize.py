"""
Audio Visualization Utilities
=============================

This module provides functions for generating spectrograms and other audio
visualizations using matplotlib and librosa.

Supported visualization types:
- stft: Short-Time Fourier Transform spectrogram
- mel: Mel spectrogram (default)
- mfcc: Mel-frequency cepstral coefficients
- waveform: Time-domain waveform plot

Features:
- Configurable FFT size (256-8192)
- Window function selection (hann, hamming, blackman, etc.)
- Configurable hop length and overlap
- dB scaling with adjustable min/max limits
- Multiple export formats (PNG, JPEG)
"""
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from bioamla.core.audio.torchaudio import load_waveform_tensor, resample_waveform_tensor

VisualizationType = Literal["stft", "mel", "mfcc", "waveform"]
WindowType = Literal["hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser"]


def generate_spectrogram(
    audio_path: str,
    output_path: str,
    viz_type: VisualizationType = "mel",
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    window: WindowType = "hann",
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = "magma",
    title: Optional[str] = None,
    db_min: Optional[float] = None,
    db_max: Optional[float] = None,
    dpi: int = 150,
    format: Optional[str] = None,
) -> str:
    """
    Generate a spectrogram visualization from an audio file.

    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the output image
        viz_type: Type of visualization ('stft', 'mel', 'mfcc', or 'waveform')
        sample_rate: Target sample rate for processing
        n_mels: Number of mel bands (for mel spectrogram)
        n_mfcc: Number of MFCCs to compute (for mfcc visualization)
        hop_length: Number of samples between successive frames
        n_fft: FFT window size (256-8192 recommended)
        window: Window function ('hann', 'hamming', 'blackman', 'bartlett',
                'rectangular', or 'kaiser')
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for spectrogram visualizations
        title: Optional title for the plot (defaults to filename)
        db_min: Minimum dB value for scaling (clips values below this)
        db_max: Maximum dB value for scaling (clips values above this)
        dpi: Resolution for output image (dots per inch)
        format: Output format ('png', 'jpg', 'jpeg'). If None, inferred from extension.

    Returns:
        str: Path to the saved output image

    Raises:
        FileNotFoundError: If the audio file does not exist
        ValueError: If an invalid visualization type or window is specified
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    valid_types = ("stft", "mel", "mfcc", "waveform")
    if viz_type not in valid_types:
        raise ValueError(f"Invalid visualization type: {viz_type}. Must be one of {valid_types}")

    valid_windows = ("hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser")
    if window not in valid_windows:
        raise ValueError(f"Invalid window type: {window}. Must be one of {valid_windows}")

    # Load and resample audio
    waveform, orig_sr = load_waveform_tensor(str(audio_path))
    if orig_sr != sample_rate:
        waveform = resample_waveform_tensor(waveform, orig_sr, sample_rate)

    # Convert to numpy and mono
    audio = waveform.numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio.squeeze()

    # Get window function
    win_func = _get_window_function(window, n_fft)

    # Generate the appropriate plot
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = audio_path.name

    if viz_type == "stft":
        _plot_stft_spectrogram(
            audio, sample_rate, ax, hop_length=hop_length,
            n_fft=n_fft, window=win_func, cmap=cmap, title=title,
            db_min=db_min, db_max=db_max
        )
    elif viz_type == "mel":
        _plot_mel_spectrogram(
            audio, sample_rate, ax, n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, window=win_func, cmap=cmap, title=title,
            db_min=db_min, db_max=db_max
        )
    elif viz_type == "mfcc":
        _plot_mfcc(
            audio, sample_rate, ax, n_mfcc=n_mfcc, hop_length=hop_length,
            n_fft=n_fft, window=win_func, cmap=cmap, title=title
        )
    elif viz_type == "waveform":
        _plot_waveform(audio, sample_rate, ax, title=title)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    if format is None:
        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            format = "jpeg"
        else:
            format = "png"

    # Save with appropriate settings for format
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if format == "jpeg":
        save_kwargs["format"] = "jpeg"
        save_kwargs["pil_kwargs"] = {"quality": 95}
    else:
        save_kwargs["format"] = "png"

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

    return str(output_path)


def _get_window_function(window: str, n_fft: int) -> np.ndarray:
    """
    Get a window function array for STFT.

    Args:
        window: Name of the window function
        n_fft: FFT window size

    Returns:
        numpy array containing the window function
    """
    if window == "hann":
        return np.hanning(n_fft)
    elif window == "hamming":
        return np.hamming(n_fft)
    elif window == "blackman":
        return np.blackman(n_fft)
    elif window == "bartlett":
        return np.bartlett(n_fft)
    elif window == "rectangular":
        return np.ones(n_fft)
    elif window == "kaiser":
        # Kaiser window with beta=14 for good sidelobe suppression
        return np.kaiser(n_fft, beta=14)
    else:
        # Default to Hann
        return np.hanning(n_fft)


def _plot_stft_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    ax: plt.Axes,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
    db_min: Optional[float] = None,
    db_max: Optional[float] = None,
) -> None:
    """Plot an STFT spectrogram."""
    # Compute STFT
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Apply dB limits if specified
    if db_min is not None or db_max is not None:
        vmin = db_min if db_min is not None else stft_db.min()
        vmax = db_max if db_max is not None else stft_db.max()
        stft_db = np.clip(stft_db, vmin, vmax)
    else:
        vmin, vmax = None, None

    img = librosa.display.specshow(
        stft_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="hz", ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f"STFT Spectrogram - {title}")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def _plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    ax: plt.Axes,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
    db_min: Optional[float] = None,
    db_max: Optional[float] = None,
) -> None:
    """Plot a mel spectrogram."""
    # Compute STFT first with custom window
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)

    # Convert to mel scale
    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, np.abs(stft) ** 2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Apply dB limits if specified
    if db_min is not None or db_max is not None:
        vmin = db_min if db_min is not None else mel_spec_db.min()
        vmax = db_max if db_max is not None else mel_spec_db.max()
        mel_spec_db = np.clip(mel_spec_db, vmin, vmax)
    else:
        vmin, vmax = None, None

    img = librosa.display.specshow(
        mel_spec_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    ax.set_title(f"Mel Spectrogram - {title}")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")


def _plot_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    ax: plt.Axes,
    n_mfcc: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
) -> None:
    """Plot MFCCs."""
    # Compute STFT first with custom window
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=window)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sample_rate)

    # Compute MFCCs from mel spectrogram
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    img = librosa.display.specshow(
        mfccs, sr=sample_rate, hop_length=hop_length,
        x_axis="time", ax=ax, cmap=cmap
    )
    ax.set_title(f"MFCC - {title}")
    ax.set_ylabel("MFCC Coefficient")
    plt.colorbar(img, ax=ax)


def _plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    ax: plt.Axes,
    title: str,
) -> None:
    """Plot a time-domain waveform."""
    times = np.arange(len(audio)) / sample_rate
    ax.plot(times, audio, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform - {title}")
    ax.set_xlim([0, times[-1]])


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: WindowType = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of an audio signal.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        n_fft: FFT window size (256-8192 recommended)
        hop_length: Number of samples between successive frames
        window: Window function name

    Returns:
        Tuple of (frequencies, times, stft_magnitude) where:
            - frequencies: Array of frequency bin centers
            - times: Array of time frame centers
            - stft_magnitude: 2D array of STFT magnitudes (frequencies x time)
    """
    win_func = _get_window_function(window, n_fft)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=win_func)

    # Get frequency and time arrays
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.times_like(stft, sr=sample_rate, hop_length=hop_length)

    return frequencies, times, np.abs(stft)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    window: WindowType = "hann",
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a mel spectrogram from an audio signal.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        n_fft: FFT window size (256-8192 recommended)
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        window: Window function name
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency for mel filterbank (default: sample_rate/2)

    Returns:
        Tuple of (times, mel_spectrogram) where:
            - times: Array of time frame centers
            - mel_spectrogram: 2D array of mel spectrogram values (mels x time)
    """
    if fmax is None:
        fmax = sample_rate / 2

    win_func = _get_window_function(window, n_fft)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, window=win_func)

    # Create mel filterbank with frequency limits
    mel_filter = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel_spec = np.dot(mel_filter, np.abs(stft) ** 2)

    times = librosa.times_like(stft, sr=sample_rate, hop_length=hop_length)

    return times, mel_spec


def spectrogram_to_db(
    spectrogram: np.ndarray,
    ref: Union[float, str] = "max",
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    """
    Convert a spectrogram to decibel (dB) scale.

    Args:
        spectrogram: Power or amplitude spectrogram
        ref: Reference value for dB computation. Can be 'max' to use the
             maximum value, or a float reference value.
        amin: Minimum amplitude threshold (prevents log of zero)
        top_db: Maximum dynamic range in dB. Values below (ref_db - top_db)
                are clipped. Set to None for no clipping.

    Returns:
        Spectrogram in dB scale
    """
    if ref == "max":
        ref_value = np.max(spectrogram)
    else:
        ref_value = float(ref)

    # Convert to dB
    spec_db = 10.0 * np.log10(np.maximum(amin, spectrogram))
    ref_db = 10.0 * np.log10(np.maximum(amin, ref_value))
    spec_db = spec_db - ref_db

    # Apply top_db clipping
    if top_db is not None:
        spec_db = np.maximum(spec_db, -top_db)

    return spec_db


def spectrogram_to_image(
    spectrogram: np.ndarray,
    output_path: str,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 150,
    format: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Frequency",
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> str:
    """
    Export a spectrogram array to an image file.

    Args:
        spectrogram: 2D spectrogram array (frequency x time)
        output_path: Path to save the output image
        cmap: Colormap for visualization
        figsize: Figure size as (width, height) in inches
        dpi: Resolution for output image (dots per inch)
        format: Output format ('png', 'jpg', 'jpeg'). If None, inferred from extension.
        title: Optional title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        colorbar: Whether to include a colorbar
        colorbar_label: Label for the colorbar
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling

    Returns:
        str: Path to the saved output image
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectrogram
    img = ax.imshow(
        spectrogram, aspect="auto", origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if colorbar:
        cbar = plt.colorbar(img, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    if format is None:
        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            format = "jpeg"
        else:
            format = "png"

    # Save with appropriate settings
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if format == "jpeg":
        save_kwargs["format"] = "jpeg"
        save_kwargs["pil_kwargs"] = {"quality": 95}
    else:
        save_kwargs["format"] = "png"

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)

    return str(output_path)


def batch_generate_spectrograms(
    input_dir: str,
    output_dir: str,
    viz_type: VisualizationType = "mel",
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    window: WindowType = "hann",
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = "magma",
    db_min: Optional[float] = None,
    db_max: Optional[float] = None,
    dpi: int = 150,
    format: str = "png",
    recursive: bool = True,
    verbose: bool = True,
    use_rich_progress: bool = False,
) -> dict:
    """
    Generate spectrograms for all audio files in a directory.

    Args:
        input_dir: Path to directory containing audio files
        output_dir: Path to directory for output images
        viz_type: Type of visualization ('stft', 'mel', 'mfcc', or 'waveform')
        sample_rate: Target sample rate for processing
        n_mels: Number of mel bands (for mel spectrogram)
        n_mfcc: Number of MFCCs to compute (for mfcc visualization)
        hop_length: Number of samples between successive frames
        n_fft: FFT window size (256-8192 recommended)
        window: Window function name
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for spectrogram visualizations
        db_min: Minimum dB value for scaling
        db_max: Maximum dB value for scaling
        dpi: Resolution for output images
        format: Output format ('png' or 'jpg')
        recursive: Whether to search subdirectories
        verbose: Whether to print progress messages
        use_rich_progress: Use Rich progress bar instead of simple output

    Returns:
        dict: Statistics about the batch processing including:
            - files_processed: Number of files successfully processed
            - files_failed: Number of files that failed
            - output_dir: Path to output directory
    """
    from bioamla.core.utils import get_files_by_extension

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = get_files_by_extension(
        str(input_dir), extensions=audio_extensions, recursive=recursive
    )

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_dir}")
        return {"files_processed": 0, "files_failed": 0, "output_dir": str(output_dir)}

    if verbose and not use_rich_progress:
        print(f"Found {len(audio_files)} audio files to process")

    files_processed = 0
    files_failed = 0

    # Determine output extension
    out_ext = ".jpg" if format.lower() in ("jpg", "jpeg") else ".png"

    # Use Rich progress bar if requested
    if use_rich_progress and verbose:
        from bioamla.core.progress import ProgressBar, print_error, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Generating spectrograms",
        ) as progress:
            for audio_path in audio_files:
                audio_path = Path(audio_path)
                try:
                    rel_path = audio_path.relative_to(input_dir)
                except ValueError:
                    rel_path = audio_path.name

                output_path = output_dir / rel_path.with_suffix(out_ext)

                try:
                    generate_spectrogram(
                        audio_path=str(audio_path),
                        output_path=str(output_path),
                        viz_type=viz_type,
                        sample_rate=sample_rate,
                        n_mels=n_mels,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                        n_fft=n_fft,
                        window=window,
                        figsize=figsize,
                        cmap=cmap,
                        db_min=db_min,
                        db_max=db_max,
                        dpi=dpi,
                        format=format,
                    )
                    files_processed += 1
                except Exception as e:
                    files_failed += 1

                progress.advance()

        if files_failed == 0:
            print_success(f"Generated {files_processed} spectrograms")
        else:
            print_error(f"Processed {files_processed} files, {files_failed} failed")
    else:
        # Simple output mode
        for audio_path in audio_files:
            audio_path = Path(audio_path)
            try:
                rel_path = audio_path.relative_to(input_dir)
            except ValueError:
                rel_path = audio_path.name

            output_path = output_dir / rel_path.with_suffix(out_ext)

            try:
                generate_spectrogram(
                    audio_path=str(audio_path),
                    output_path=str(output_path),
                    viz_type=viz_type,
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    n_mfcc=n_mfcc,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    window=window,
                    figsize=figsize,
                    cmap=cmap,
                    db_min=db_min,
                    db_max=db_max,
                    dpi=dpi,
                    format=format,
                )
                files_processed += 1
                if verbose:
                    print(f"  Generated: {output_path}")
            except Exception as e:
                files_failed += 1
                if verbose:
                    print(f"  Failed: {audio_path} - {e}")

        if verbose:
            print(f"Processed {files_processed} files, {files_failed} failed")

    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "output_dir": str(output_dir),
    }
