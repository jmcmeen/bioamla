"""
Audio Visualization Utilities
=============================

This module provides functions for generating spectrograms and other audio
visualizations using matplotlib and librosa.

Supported visualization types:
- mel: Mel spectrogram (default)
- mfcc: Mel-frequency cepstral coefficients
- waveform: Time-domain waveform plot
"""
from pathlib import Path
from typing import Literal, Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from bioamla.torchaudio import load_waveform_tensor, resample_waveform_tensor

VisualizationType = Literal["mel", "mfcc", "waveform"]


def generate_spectrogram(
    audio_path: str,
    output_path: str,
    viz_type: VisualizationType = "mel",
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    figsize: tuple = (10, 4),
    cmap: str = "magma",
    title: Optional[str] = None,
) -> str:
    """
    Generate a spectrogram visualization from an audio file.

    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the output image
        viz_type: Type of visualization ('mel', 'mfcc', or 'waveform')
        sample_rate: Target sample rate for processing
        n_mels: Number of mel bands (for mel spectrogram)
        n_mfcc: Number of MFCCs to compute (for mfcc visualization)
        hop_length: Number of samples between successive frames
        n_fft: FFT window size
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for spectrogram visualizations
        title: Optional title for the plot (defaults to filename)

    Returns:
        str: Path to the saved output image

    Raises:
        FileNotFoundError: If the audio file does not exist
        ValueError: If an invalid visualization type is specified
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if viz_type not in ("mel", "mfcc", "waveform"):
        raise ValueError(f"Invalid visualization type: {viz_type}. Must be 'mel', 'mfcc', or 'waveform'")

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

    # Generate the appropriate plot
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = audio_path.name

    if viz_type == "mel":
        _plot_mel_spectrogram(
            audio, sample_rate, ax, n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, cmap=cmap, title=title
        )
    elif viz_type == "mfcc":
        _plot_mfcc(
            audio, sample_rate, ax, n_mfcc=n_mfcc, hop_length=hop_length,
            n_fft=n_fft, cmap=cmap, title=title
        )
    elif viz_type == "waveform":
        _plot_waveform(audio, sample_rate, ax, title=title)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def _plot_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    ax: plt.Axes,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    cmap: str,
    title: str,
) -> None:
    """Plot a mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        mel_spec_db, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax, cmap=cmap
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
    cmap: str,
    title: str,
) -> None:
    """Plot MFCCs."""
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
    )

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


def batch_generate_spectrograms(
    input_dir: str,
    output_dir: str,
    viz_type: VisualizationType = "mel",
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    figsize: tuple = (10, 4),
    cmap: str = "magma",
    recursive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Generate spectrograms for all audio files in a directory.

    Args:
        input_dir: Path to directory containing audio files
        output_dir: Path to directory for output images
        viz_type: Type of visualization ('mel', 'mfcc', or 'waveform')
        sample_rate: Target sample rate for processing
        n_mels: Number of mel bands (for mel spectrogram)
        n_mfcc: Number of MFCCs to compute (for mfcc visualization)
        hop_length: Number of samples between successive frames
        n_fft: FFT window size
        figsize: Figure size as (width, height) in inches
        cmap: Colormap for spectrogram visualizations
        recursive: Whether to search subdirectories
        verbose: Whether to print progress messages

    Returns:
        dict: Statistics about the batch processing including:
            - files_processed: Number of files successfully processed
            - files_failed: Number of files that failed
            - output_dir: Path to output directory
    """
    from novus_pytils.files import get_files_by_extension

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

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")

    files_processed = 0
    files_failed = 0

    for audio_path in audio_files:
        audio_path = Path(audio_path)
        # Preserve relative directory structure
        try:
            rel_path = audio_path.relative_to(input_dir)
        except ValueError:
            rel_path = audio_path.name

        output_path = output_dir / rel_path.with_suffix(".png")

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
                figsize=figsize,
                cmap=cmap,
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
