"""
Audio Visualization
===================

Functions for generating spectrograms and other audio visualizations using
matplotlib and librosa, folded from ``core/visualize.py``.

Supported visualization types:
- ``stft``: Short-Time Fourier Transform spectrogram
- ``mel``: Mel spectrogram (default)
- ``mfcc``: Mel-frequency cepstral coefficients
- ``waveform``: Time-domain waveform plot

Visualization runs on the slim install (librosa + matplotlib, CPU only). When a
CUDA GPU and torch are available, the STFT can be computed on the GPU for a
speedup via the optional ``backend`` argument (``"auto"`` uses the GPU when
present and falls back to librosa otherwise). See :mod:`bioamla.viz._backend`.

Heavy/optional backends are imported lazily:
- ``matplotlib`` is imported inside the rendering functions (and the ``Agg``
  non-interactive backend is selected there for thread-safety).
- ``torch`` is imported lazily only when the GPU backend is selected; it is
  never required to generate a spectrogram.
"""

from pathlib import Path
from typing import Literal

import librosa
import numpy as np

from bioamla.exceptions import AudioLoadError, NotFoundError, ProcessingError
from bioamla.viz._backend import Backend, mel_power, select_backend, stft_magnitude

VisualizationType = Literal["stft", "mel", "mfcc", "waveform"]
WindowType = Literal["hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser"]


def _get_pyplot():  # noqa: ANN202 - returns matplotlib.pyplot module
    """Import matplotlib.pyplot with the Agg backend selected for thread-safety."""
    import sys

    import matplotlib

    # Select a non-interactive backend for headless/parallel use, but only if
    # pyplot has not been imported yet (matplotlib.use() raises otherwise).
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_audio_for_viz(audio_path: str, sample_rate: int) -> np.ndarray:
    """
    Load and resample an audio file to a mono numpy array for plotting.

    Uses the slim-core numpy loader (soundfile/pydub) — no torch required.

    Raises:
        AudioLoadError: If the audio cannot be loaded.
    """
    from bioamla.audio import load_audio
    from bioamla.audio.processing import resample_audio

    audio, orig_sr = load_audio(str(audio_path))  # mono float32 numpy

    if orig_sr != sample_rate:
        try:
            audio = resample_audio(audio, orig_sr, sample_rate)
        except Exception as e:
            raise AudioLoadError(f"Failed to resample audio: {e}") from e

    return audio


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
    figsize: tuple[int, int] = (10, 4),
    cmap: str = "magma",
    title: str | None = None,
    db_min: float | None = None,
    db_max: float | None = None,
    dpi: int = 150,
    format: str | None = None,
    show_colorbar: bool = True,
    backend: str = "auto",
) -> str:
    """
    Generate a spectrogram visualization from an audio file.

    Args:
        audio_path: Path to the input audio file.
        output_path: Path to save the output image.
        viz_type: Type of visualization ('stft', 'mel', 'mfcc', or 'waveform').
        sample_rate: Target sample rate for processing.
        n_mels: Number of mel bands (for mel spectrogram).
        n_mfcc: Number of MFCCs to compute (for mfcc visualization).
        hop_length: Number of samples between successive frames.
        n_fft: FFT window size (256-8192 recommended).
        window: Window function name.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap for spectrogram visualizations.
        title: Optional title for the plot (defaults to filename).
        db_min: Minimum dB value for scaling.
        db_max: Maximum dB value for scaling.
        dpi: Resolution for output image (dots per inch).
        format: Output format ('png', 'jpg', 'jpeg'); inferred from extension if None.
        show_colorbar: Whether to show axes/title/colorbar (default: True).
        backend: Spectrogram compute backend — 'auto' (GPU/torch if available,
            else librosa), 'librosa' (CPU), or 'torch' (force GPU/torch).

    Returns:
        Path to the saved output image (as a string).

    Raises:
        NotFoundError: If the audio file does not exist.
        ValueError: If an invalid visualization type or window is specified.
        AudioLoadError: If the audio cannot be loaded.
        DependencyError: If backend='torch' but torch is not installed.
        ProcessingError: If the image cannot be written.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise NotFoundError(f"Audio file not found: {audio_path}")

    valid_types = ("stft", "mel", "mfcc", "waveform")
    if viz_type not in valid_types:
        raise ValueError(f"Invalid visualization type: {viz_type}. Must be one of {valid_types}")

    valid_windows = ("hann", "hamming", "blackman", "bartlett", "rectangular", "kaiser")
    if window not in valid_windows:
        raise ValueError(f"Invalid window type: {window}. Must be one of {valid_windows}")

    resolved_backend = select_backend(backend)

    plt = _get_pyplot()

    # Load and resample audio
    audio = _load_audio_for_viz(str(audio_path), sample_rate)

    # Get window function
    win_func = _get_window_function(window, n_fft)

    # Generate the appropriate plot
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = audio_path.name

    if viz_type == "stft":
        _plot_stft_spectrogram(
            audio,
            sample_rate,
            ax,
            hop_length=hop_length,
            n_fft=n_fft,
            window=win_func,
            cmap=cmap,
            title=title,
            db_min=db_min,
            db_max=db_max,
            show_legend=show_colorbar,
            backend=resolved_backend,
        )
    elif viz_type == "mel":
        _plot_mel_spectrogram(
            audio,
            sample_rate,
            ax,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            window=win_func,
            cmap=cmap,
            title=title,
            db_min=db_min,
            db_max=db_max,
            show_legend=show_colorbar,
            backend=resolved_backend,
        )
    elif viz_type == "mfcc":
        _plot_mfcc(
            audio,
            sample_rate,
            ax,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            n_fft=n_fft,
            window=win_func,
            cmap=cmap,
            title=title,
            show_legend=show_colorbar,
            backend=resolved_backend,
        )
    elif viz_type == "waveform":
        _plot_waveform(audio, sample_rate, ax, title=title, show_legend=show_colorbar)

    # Only use tight_layout when legend is shown, otherwise use full canvas
    if show_colorbar:
        plt.tight_layout()
    else:
        # Remove all margins for borderless output
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.tight_layout(pad=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    if format is None:
        ext = output_path.suffix.lower()
        format = "jpeg" if ext in (".jpg", ".jpeg") else "png"

    # Save with appropriate settings for format
    save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if not show_colorbar:
        save_kwargs["pad_inches"] = 0

    if format == "jpeg":
        save_kwargs["format"] = "jpeg"
        save_kwargs["pil_kwargs"] = {"quality": 95}
    else:
        save_kwargs["format"] = "png"

    try:
        fig.savefig(output_path, **save_kwargs)
    except Exception as e:
        raise ProcessingError(f"Failed to save visualization: {e}") from e
    finally:
        plt.close(fig)

    return str(output_path)


def _get_window_function(window: str, n_fft: int) -> np.ndarray:
    """Get a window function array for STFT."""
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
        return np.hanning(n_fft)


def _plot_stft_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    ax,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
    db_min: float | None = None,
    db_max: float | None = None,
    show_legend: bool = True,
    backend: Backend = "librosa",
) -> None:
    """Plot an STFT spectrogram."""
    import librosa.display

    mag = stft_magnitude(audio, n_fft, hop_length, window, backend)
    stft_db = librosa.amplitude_to_db(mag, ref=np.max)

    if db_min is not None or db_max is not None:
        vmin = db_min if db_min is not None else stft_db.min()
        vmax = db_max if db_max is not None else stft_db.max()
        stft_db = np.clip(stft_db, vmin, vmax)
    else:
        vmin, vmax = None, None

    img = librosa.display.specshow(
        stft_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time" if show_legend else None,
        y_axis="hz" if show_legend else None,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if show_legend:
        ax.set_title(f"STFT Spectrogram - {title}")
        ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
    else:
        ax.axis("off")


def _plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    ax,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
    db_min: float | None = None,
    db_max: float | None = None,
    show_legend: bool = True,
    backend: Backend = "librosa",
) -> None:
    """Plot a mel spectrogram."""
    import librosa.display

    mel_spec = mel_power(audio, sample_rate, n_fft, hop_length, n_mels, window, backend=backend)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if db_min is not None or db_max is not None:
        vmin = db_min if db_min is not None else mel_spec_db.min()
        vmax = db_max if db_max is not None else mel_spec_db.max()
        mel_spec_db = np.clip(mel_spec_db, vmin, vmax)
    else:
        vmin, vmax = None, None

    img = librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time" if show_legend else None,
        y_axis="mel" if show_legend else None,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if show_legend:
        ax.set_title(f"Mel Spectrogram - {title}")
        ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
    else:
        ax.axis("off")


def _plot_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    ax,
    n_mfcc: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    cmap: str,
    title: str,
    show_legend: bool = True,
    backend: Backend = "librosa",
) -> None:
    """Plot MFCCs."""
    import librosa.display

    mag = stft_magnitude(audio, n_fft, hop_length, window, backend)
    mel_spec = librosa.feature.melspectrogram(S=mag**2, sr=sample_rate)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)

    img = librosa.display.specshow(
        mfccs,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time" if show_legend else None,
        ax=ax,
        cmap=cmap,
    )

    if show_legend:
        ax.set_title(f"MFCC - {title}")
        ax.set_ylabel("MFCC Coefficient")
        ax.figure.colorbar(img, ax=ax)
    else:
        ax.axis("off")


def _plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    ax,
    title: str,
    show_legend: bool = True,
) -> None:
    """Plot a time-domain waveform."""
    times = np.arange(len(audio)) / sample_rate
    ax.plot(times, audio, linewidth=0.5)

    if show_legend:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform - {title}")
        ax.set_xlim([0, times[-1]])
    else:
        ax.axis("off")


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: WindowType = "hann",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of an audio signal.

    Args:
        audio: Audio samples as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size (256-8192 recommended).
        hop_length: Number of samples between successive frames.
        window: Window function name.
        backend: Compute backend — 'auto' (GPU/torch if available, else librosa),
            'librosa' (CPU), or 'torch' (force GPU/torch).

    Returns:
        Tuple of (frequencies, times, stft_magnitude).
    """
    win_func = _get_window_function(window, n_fft)
    mag = stft_magnitude(audio, n_fft, hop_length, win_func, select_backend(backend))

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.times_like(mag, sr=sample_rate, hop_length=hop_length)

    return frequencies, times, mag


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    window: WindowType = "hann",
    fmin: float = 0.0,
    fmax: float | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a mel spectrogram from an audio signal.

    Args:
        audio: Audio samples as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size (256-8192 recommended).
        hop_length: Number of samples between successive frames.
        n_mels: Number of mel bands.
        window: Window function name.
        fmin: Minimum frequency for the mel filterbank.
        fmax: Maximum frequency for the mel filterbank (default: sample_rate/2).
        backend: Compute backend — 'auto' (GPU/torch if available, else librosa),
            'librosa' (CPU), or 'torch' (force GPU/torch).

    Returns:
        Tuple of (times, mel_spectrogram).
    """
    if fmax is None:
        fmax = sample_rate / 2

    win_func = _get_window_function(window, n_fft)
    mel_spec = mel_power(
        audio, sample_rate, n_fft, hop_length, n_mels, win_func, fmin, fmax, select_backend(backend)
    )

    times = librosa.times_like(mel_spec, sr=sample_rate, hop_length=hop_length)

    return times, mel_spec


def spectrogram_to_db(
    spectrogram: np.ndarray,
    ref: float | str = "max",
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    """
    Convert a spectrogram to decibel (dB) scale.

    Args:
        spectrogram: Power or amplitude spectrogram.
        ref: Reference value for dB computation ('max' or a float).
        amin: Minimum amplitude threshold (prevents log of zero).
        top_db: Maximum dynamic range in dB (None disables clipping).

    Returns:
        Spectrogram in dB scale.
    """
    if ref == "max":
        ref_value = np.max(spectrogram)
    else:
        ref_value = float(ref)

    spec_db = 10.0 * np.log10(np.maximum(amin, spectrogram))
    ref_db = 10.0 * np.log10(np.maximum(amin, ref_value))
    spec_db = spec_db - ref_db

    if top_db is not None:
        spec_db = np.maximum(spec_db, -top_db)

    return spec_db


def spectrogram_to_image(
    spectrogram: np.ndarray,
    output_path: str,
    cmap: str = "magma",
    figsize: tuple[int, int] = (10, 4),
    dpi: int = 150,
    format: str | None = None,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Frequency",
    colorbar: bool = True,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    """
    Export a spectrogram array to an image file.

    Args:
        spectrogram: 2D spectrogram array (frequency x time).
        output_path: Path to save the output image.
        cmap: Colormap for visualization.
        figsize: Figure size as (width, height) in inches.
        dpi: Resolution for output image (dots per inch).
        format: Output format ('png', 'jpg', 'jpeg'); inferred from extension if None.
        title: Optional title for the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        colorbar: Whether to include a colorbar.
        colorbar_label: Label for the colorbar.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.

    Returns:
        Path to the saved output image (as a string).

    Raises:
        ProcessingError: If the image cannot be written.
    """
    plt = _get_pyplot()

    fig, ax = plt.subplots(figsize=figsize)

    img = ax.imshow(spectrogram, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

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

    if format is None:
        ext = output_path.suffix.lower()
        format = "jpeg" if ext in (".jpg", ".jpeg") else "png"

    save_kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if format == "jpeg":
        save_kwargs["format"] = "jpeg"
        save_kwargs["pil_kwargs"] = {"quality": 95}
    else:
        save_kwargs["format"] = "png"

    try:
        fig.savefig(output_path, **save_kwargs)
    except Exception as e:
        raise ProcessingError(f"Failed to save visualization: {e}") from e
    finally:
        plt.close(fig)

    return str(output_path)


__all__ = [
    "VisualizationType",
    "WindowType",
    "generate_spectrogram",
    "compute_stft",
    "compute_mel_spectrogram",
    "spectrogram_to_db",
    "spectrogram_to_image",
]
