"""Audio augmentation for expanding training datasets.

Wraps the ``audiomentations`` pipeline (noise/time-stretch/pitch-shift/gain) and
provides a batch helper that walks an input directory and writes augmented WAVs.
All heavy dependencies (audiomentations, torch, torchaudio) are imported lazily
for fast startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bioamla.exceptions import AugmentationError, NotFoundError

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

    # Gain transition (smooth gain ramp; mainly used for on-the-fly training aug)
    gain_transition: bool = False
    gain_transition_min_duration: float = 0.01
    gain_transition_max_duration: float = 0.3
    gain_transition_probability: float = 0.5

    # Clipping distortion (mainly used for on-the-fly training aug)
    clipping_distortion: bool = False
    clipping_min_percentile: int = 0
    clipping_max_percentile: int = 30
    clipping_probability: float = 0.5

    # Pipeline-level controls. ``pipeline_probability`` is the chance the whole
    # Compose is applied to a sample (1.0 = always, the dataset-synthesis
    # default); ``shuffle`` randomizes transform order each call (used by
    # on-the-fly training augmentation).
    pipeline_probability: float = 1.0
    shuffle: bool = False

    # General settings
    sample_rate: int = 16000
    multiply: int = 1  # Number of augmented copies to create per file


# =============================================================================
# Augmentation Functions
# =============================================================================


def create_augmentation_pipeline(config: AugmentationConfig) -> Any:
    """Create an ``audiomentations`` Compose pipeline from a config.

    This is the single builder shared by both augmentation use cases: synthetic
    dataset generation (``dataset augment`` / :func:`batch_augment`, which leaves
    ``pipeline_probability=1.0`` and ``shuffle=False``) and on-the-fly training
    augmentation (``models ast train``, which enables ``gain_transition`` /
    ``clipping_distortion`` and sets a compose-level ``pipeline_probability`` with
    ``shuffle=True``).

    Returns:
        A ``Compose`` pipeline, or None if no augmentations are enabled.
    """
    from audiomentations import (
        AddGaussianSNR,
        ClippingDistortion,
        Compose,
        Gain,
        GainTransition,
        PitchShift,
        TimeStretch,
    )

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

    if config.gain_transition:
        transforms.append(
            GainTransition(
                min_gain_db=config.gain_min_db,
                max_gain_db=config.gain_max_db,
                min_duration=config.gain_transition_min_duration,
                max_duration=config.gain_transition_max_duration,
                duration_unit="fraction",
                p=config.gain_transition_probability,
            )
        )

    if config.clipping_distortion:
        transforms.append(
            ClippingDistortion(
                min_percentile_threshold=config.clipping_min_percentile,
                max_percentile_threshold=config.clipping_max_percentile,
                p=config.clipping_probability,
            )
        )

    if not transforms:
        return None

    return Compose(transforms, p=config.pipeline_probability, shuffle=config.shuffle)


# Transform attributes that are bookkeeping rather than tunable parameters; they
# add noise to a human-readable summary, so leave them out.
_DESCRIBE_SKIP_ATTRS = frozenset({"p", "parameters", "are_parameters_frozen"})


def describe_augmentation_pipeline(pipeline: Any) -> list[str]:
    """Summarize each transform in an ``audiomentations`` Compose pipeline.

    Introspects the *built* pipeline (rather than re-deriving from the config) so
    the description always reflects exactly what is applied — same source of truth
    as training. Each line is ``Name(p=...): k=v, ...`` for one transform.

    Args:
        pipeline: A ``Compose`` pipeline from :func:`create_augmentation_pipeline`,
            or ``None``.

    Returns:
        One description string per transform; empty list if ``pipeline`` is None.
    """
    if pipeline is None:
        return []

    lines: list[str] = []
    for transform in pipeline.transforms:
        params = {
            key: value
            for key, value in vars(transform).items()
            if key not in _DESCRIBE_SKIP_ATTRS and not key.startswith("_") and value is not None
        }
        param_str = ", ".join(f"{key}={value}" for key, value in params.items())
        prob = getattr(transform, "p", None)
        lines.append(f"{type(transform).__name__}(p={prob}): {param_str}")
    return lines


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
    """
    from bioamla.audio import save_audio
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
    from bioamla.audio.torchaudio import load_waveform_tensor

    waveform, orig_sr = load_waveform_tensor(path)
    audio = waveform.numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio.squeeze()
    return audio, orig_sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono waveform with torchaudio (lazy import)."""
    import torch
    import torchaudio

    waveform_tensor = torch.from_numpy(audio).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform_tensor).squeeze().numpy()
