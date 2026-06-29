"""
Batch Spectrogram Generation
============================

Batch wrapper around :func:`bioamla.viz.core.generate_spectrogram`, folded from
``core/visualize.py``. The original CLI->core inversion (importing ``ProgressBar``
and ``print_success``/``print_error`` from the CLI console module) has been
removed: progress is reported via an optional callback / simple prints, and file
discovery uses pathlib instead of the file repository.
"""

from collections.abc import Callable
from pathlib import Path

from bioamla.audio.discovery import list_audio_files
from bioamla.exceptions import NotFoundError
from bioamla.viz.core import (
    VisualizationType,
    WindowType,
    generate_spectrogram,
)


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
    figsize: tuple[int, int] = (10, 4),
    cmap: str = "magma",
    db_min: float | None = None,
    db_max: float | None = None,
    dpi: int = 150,
    format: str = "png",
    recursive: bool = True,
    verbose: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Generate spectrograms for all audio files in a directory.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory for output images.
        viz_type: Type of visualization ('stft', 'mel', 'mfcc', or 'waveform').
        sample_rate: Target sample rate for processing.
        n_mels: Number of mel bands (for mel spectrogram).
        n_mfcc: Number of MFCCs to compute (for mfcc visualization).
        hop_length: Number of samples between successive frames.
        n_fft: FFT window size (256-8192 recommended).
        window: Window function name.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap for spectrogram visualizations.
        db_min: Minimum dB value for scaling.
        db_max: Maximum dB value for scaling.
        dpi: Resolution for output images.
        format: Output format ('png' or 'jpg').
        recursive: Whether to search subdirectories.
        verbose: Whether to print progress messages.
        on_progress: Optional ``(completed, total)`` progress callback.

    Returns:
        Statistics dict with ``files_processed``, ``files_failed``, ``output_dir``.

    Raises:
        NotFoundError: If the input directory does not exist.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        raise NotFoundError(f"Input directory not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list_audio_files(str(in_dir), recursive=recursive)

    if not audio_files:
        if verbose:
            print(f"No audio files found in {in_dir}")
        return {"files_processed": 0, "files_failed": 0, "output_dir": str(out_dir)}

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")

    files_processed = 0
    files_failed = 0
    total = len(audio_files)

    # Determine output extension
    out_ext = ".jpg" if format.lower() in ("jpg", "jpeg") else ".png"

    for idx, audio_path in enumerate(audio_files, start=1):
        audio_path = Path(audio_path)
        try:
            rel_path = audio_path.relative_to(in_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        output_path = out_dir / rel_path.with_suffix(out_ext)

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

        if on_progress:
            on_progress(idx, total)

    if verbose:
        print(f"Processed {files_processed} files, {files_failed} failed")

    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "output_dir": str(out_dir),
    }


__all__ = ["batch_generate_spectrograms"]
