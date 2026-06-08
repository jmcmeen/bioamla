"""Audio augmentation for expanding training datasets.

Wraps the ``audiomentations`` pipeline (noise/time-stretch/pitch-shift/gain) and
provides a batch helper that walks an input directory and writes augmented WAVs.
All heavy dependencies (audiomentations, torch, torchaudio) are imported lazily;
on a slim install the augmentation functions raise
:class:`~bioamla.exceptions.DependencyError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioamla.exceptions import AugmentationError, DependencyError, NotFoundError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Augmentation Configuration
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for the audio augmentation pipeline."""

    # Noise augmentation
    add_noise: bool = False
    noise_min_snr: float = 3.0
    noise_max_snr: float = 30.0
    noise_probability: float = 1.0

    # Time stretch augmentation
    time_stretch: bool = False
    time_stretch_min: float = 0.8
    time_stretch_max: float = 1.2
    time_stretch_probability: float = 1.0

    # Pitch shift augmentation
    pitch_shift: bool = False
    pitch_shift_min: float = -4.0
    pitch_shift_max: float = 4.0
    pitch_shift_probability: float = 1.0

    # Gain augmentation
    gain: bool = False
    gain_min_db: float = -12.0
    gain_max_db: float = 12.0
    gain_probability: float = 1.0

    # General settings
    sample_rate: int = 16000
    multiply: int = 1  # Number of augmented copies to create per file


# =============================================================================
# Augmentation Functions
# =============================================================================


def create_augmentation_pipeline(config: AugmentationConfig) -> Any:
    """Create an ``audiomentations`` Compose pipeline from a config.

    Returns:
        A ``Compose`` pipeline, or None if no augmentations are enabled.

    Raises:
        DependencyError: If ``audiomentations`` is not installed.
    """
    try:
        from audiomentations import (
            AddGaussianSNR,
            Compose,
            Gain,
            PitchShift,
            TimeStretch,
        )
    except ImportError as e:
        raise DependencyError(
            "Audio augmentation requires audiomentations — install bioamla[augment]"
        ) from e

    transforms = []

    if config.add_noise:
        transforms.append(
            AddGaussianSNR(
                min_snr_db=config.noise_min_snr,
                max_snr_db=config.noise_max_snr,
                p=config.noise_probability,
            )
        )

    if config.time_stretch:
        transforms.append(
            TimeStretch(
                min_rate=config.time_stretch_min,
                max_rate=config.time_stretch_max,
                p=config.time_stretch_probability,
            )
        )

    if config.pitch_shift:
        transforms.append(
            PitchShift(
                min_semitones=config.pitch_shift_min,
                max_semitones=config.pitch_shift_max,
                p=config.pitch_shift_probability,
            )
        )

    if config.gain:
        transforms.append(
            Gain(
                min_gain_db=config.gain_min_db,
                max_gain_db=config.gain_max_db,
                p=config.gain_probability,
            )
        )

    if not transforms:
        return None

    return Compose(transforms)


def augment_audio(audio: np.ndarray, sample_rate: int, pipeline: Any) -> np.ndarray:
    """Apply an augmentation pipeline to a 1-D float audio array.

    Args:
        audio: Audio data as a numpy array.
        sample_rate: Sample rate of the audio.
        pipeline: An audiomentations Compose pipeline.

    Returns:
        Augmented audio as a numpy array.
    """
    import numpy as np

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    return pipeline(samples=audio, sample_rate=sample_rate)


def batch_augment(
    input_dir: str,
    output_dir: str,
    config: AugmentationConfig,
    recursive: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Augment all audio files in a directory.

    Args:
        input_dir: Directory containing audio files.
        output_dir: Directory for augmented output (created if missing).
        config: Augmentation configuration.
        recursive: Whether to search subdirectories.
        verbose: Whether to print progress messages.

    Returns:
        Dict with keys ``files_processed``, ``files_created``, ``files_failed``,
        and ``output_dir``.

    Raises:
        NotFoundError: If the input directory doesn't exist.
        AugmentationError: If no augmentations are enabled.
        DependencyError: If augmentation dependencies are missing.
    """
    from bioamla.adapters.pydub import save_audio
    from bioamla.common.files import get_files_by_extension

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise NotFoundError(f"Input directory not found: {input_dir}")

    pipeline = create_augmentation_pipeline(config)
    if pipeline is None:
        raise AugmentationError("No augmentations configured")

    output_path.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = get_files_by_extension(
        str(input_path), extensions=audio_extensions, recursive=recursive
    )

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_path}")
        return {
            "files_processed": 0,
            "files_created": 0,
            "files_failed": 0,
            "output_dir": str(output_path),
        }

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Creating {config.multiply} augmented copies per file")

    files_processed = 0
    files_created = 0
    files_failed = 0

    for raw_path in audio_files:
        audio_path = Path(raw_path)

        try:
            rel_path = audio_path.relative_to(input_path)
        except ValueError:
            rel_path = Path(audio_path.name)

        try:
            audio, orig_sr = _load_waveform(str(audio_path))

            if orig_sr != config.sample_rate:
                audio = _resample(audio, orig_sr, config.sample_rate)

            for i in range(config.multiply):
                stem = rel_path.stem
                suffix = ".wav"  # Always output as WAV
                if config.multiply > 1:
                    out_name = f"{stem}_aug{i + 1}{suffix}"
                else:
                    out_name = f"{stem}_aug{suffix}"

                out_path = output_path / rel_path.parent / out_name
                augmented = augment_audio(audio, config.sample_rate, pipeline)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(str(out_path), augmented, config.sample_rate)
                files_created += 1

                if verbose:
                    print(f"  Created: {out_path}")

            files_processed += 1
        except Exception as e:  # noqa: BLE001 - report per-file failures, continue batch
            files_failed += 1
            if verbose:
                print(f"  Failed: {audio_path} - {e}")

    if verbose:
        print(
            f"Processed {files_processed} files, created {files_created} "
            f"augmented files, {files_failed} failed"
        )

    return {
        "files_processed": files_processed,
        "files_created": files_created,
        "files_failed": files_failed,
        "output_dir": str(output_path),
    }


def _load_waveform(path: str) -> tuple[np.ndarray, int]:
    """Load a mono float waveform via torchaudio (lazy import)."""
    try:
        from bioamla.audio.torchaudio import load_waveform_tensor
    except ImportError as e:
        raise DependencyError(
            "Audio augmentation requires torchaudio — install bioamla[augment]"
        ) from e

    waveform, orig_sr = load_waveform_tensor(path)
    audio = waveform.numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio.squeeze()
    return audio, orig_sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono waveform with torchaudio (lazy import)."""
    try:
        import torch
        import torchaudio
    except ImportError as e:
        raise DependencyError(
            "Resampling requires torch and torchaudio — install bioamla[augment]"
        ) from e

    waveform_tensor = torch.from_numpy(audio).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform_tensor).squeeze().numpy()
