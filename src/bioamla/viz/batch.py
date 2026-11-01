"""
Batch Spectrogram Generation
============================

Batch wrapper around :func:`bioamla.viz.core.generate_spectrogram`, folded from
``core/visualize.py``. The original CLI->core inversion (importing ``ProgressBar``
and ``print_success``/``print_error`` from the CLI console module) has been
removed: progress is reported via an optional callback / simple prints, and file
discovery uses pathlib instead of the file repository.
"""

import logging
from pathlib import Path

from bioamla.audio.discovery import list_audio_files
from bioamla.batch import output_stems, run_batch
from bioamla.exceptions import NotFoundError
from bioamla.protocols import OnProgress
from bioamla.viz.core import (
    VisualizationType,
    WindowType,
    generate_spectrogram,
)

logger = logging.getLogger(__name__)


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
    show_colorbar: bool = True,
    recursive: bool = True,
    verbose: bool = True,
    on_progress: OnProgress | None = None,
    max_workers: int = 1,
    continue_on_error: bool = True,
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
        show_colorbar: Show axes, title and colorbar (as
            :func:`~bioamla.viz.generate_spectrogram`); ``False`` gives the bare image.
        recursive: Whether to search subdirectories.
        verbose: Whether to log progress at INFO level.
        on_progress: Optional ``(completed, total)`` progress callback.
        max_workers: Worker threads; 1 runs sequentially.
        continue_on_error: Keep going after a file fails (else raise its error).

    Returns:
        Statistics dict with ``files_processed``, ``files_failed``, ``output_dir``,
        ``output_files`` and ``errors`` (one ``"<file>: <error>"`` per failure).

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
            logger.info(f"No audio files found in {in_dir}")
        return {
            "files_processed": 0,
            "files_failed": 0,
            "output_dir": str(out_dir),
            "output_files": [],
            "errors": [],
        }

    if verbose:
        logger.info(f"Found {len(audio_files)} audio files to process")

    # Determine output extension
    out_ext = ".jpg" if format.lower() in ("jpg", "jpeg") else ".png"

    files = [str(f) for f in audio_files]
    stems = output_stems(files, in_dir)

    def _generate(audio_path: str) -> Path:
        stem = stems[audio_path]
        output_path = out_dir / stem.parent / f"{stem.name}{out_ext}"
        generate_spectrogram(
            audio_path=audio_path,
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
            show_colorbar=show_colorbar,
        )
        if verbose:
            logger.info(f"  Generated: {output_path}")
        return output_path

    result = run_batch(
        files,
        _generate,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
        operation="spectrogram",
    )

    return {
        "files_processed": result.successful,
        "files_failed": result.failed,
        "output_dir": str(out_dir),
        "output_files": result.output_files,
        "errors": result.errors,
    }


__all__ = ["batch_generate_spectrograms"]
