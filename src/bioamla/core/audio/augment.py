"""
Audio Augmentation Utilities
============================

This module provides functions for augmenting audio files using the
audiomentations library. Augmentation is useful for expanding training
datasets and improving model robustness.

Supported augmentations:
- add-noise: Add Gaussian noise with specified SNR
- time-stretch: Time stretching (speed up/slow down without pitch change)
- pitch-shift: Pitch shifting (change pitch without speed change)
- gain: Random gain adjustment
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from audiomentations import (
    AddGaussianSNR,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

from bioamla.core.audio.torchaudio import load_waveform_tensor


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""

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


def create_augmentation_pipeline(config: AugmentationConfig) -> Optional[Compose]:
    """
    Create an audiomentations pipeline from configuration.

    Args:
        config: Augmentation configuration

    Returns:
        Compose pipeline or None if no augmentations are enabled
    """
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


def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    pipeline: Compose,
) -> np.ndarray:
    """
    Apply augmentation pipeline to audio.

    Args:
        audio: Audio data as numpy array (1D)
        sample_rate: Sample rate of the audio
        pipeline: Audiomentations Compose pipeline

    Returns:
        Augmented audio as numpy array
    """
    # Ensure audio is float32 and 1D
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Apply augmentation
    augmented = pipeline(samples=audio, sample_rate=sample_rate)

    return augmented


def augment_file(
    input_path: str,
    output_path: str,
    config: AugmentationConfig,
) -> str:
    """
    Augment a single audio file and save the result.

    Args:
        input_path: Path to input audio file
        output_path: Path to save augmented audio
        config: Augmentation configuration

    Returns:
        Path to the saved augmented file

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If no augmentations are configured
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pipeline = create_augmentation_pipeline(config)
    if pipeline is None:
        raise ValueError("No augmentations configured")

    # Load audio
    waveform, orig_sr = load_waveform_tensor(str(input_path))

    # Convert to numpy and mono
    audio = waveform.numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    else:
        audio = audio.squeeze()

    # Resample if needed
    if orig_sr != config.sample_rate:
        import torch
        import torchaudio

        waveform_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, config.sample_rate)
        audio = resampler(waveform_tensor).squeeze().numpy()

    # Apply augmentation
    augmented = augment_audio(audio, config.sample_rate, pipeline)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), augmented, config.sample_rate)

    return str(output_path)


def batch_augment(
    input_dir: str,
    output_dir: str,
    config: AugmentationConfig,
    recursive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Augment all audio files in a directory.

    Args:
        input_dir: Path to directory containing audio files
        output_dir: Path to directory for augmented output
        config: Augmentation configuration
        recursive: Whether to search subdirectories
        verbose: Whether to print progress messages

    Returns:
        dict: Statistics about the batch processing including:
            - files_processed: Number of files successfully processed
            - files_created: Total number of augmented files created
            - files_failed: Number of files that failed
            - output_dir: Path to output directory
    """
    from bioamla.core.utils import get_files_by_extension

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pipeline = create_augmentation_pipeline(config)
    if pipeline is None:
        raise ValueError("No augmentations configured")

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    audio_files = get_files_by_extension(
        str(input_dir), extensions=audio_extensions, recursive=recursive
    )

    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_dir}")
        return {
            "files_processed": 0,
            "files_created": 0,
            "files_failed": 0,
            "output_dir": str(output_dir),
        }

    if verbose:
        print(f"Found {len(audio_files)} audio files to process")
        print(f"Creating {config.multiply} augmented copies per file")

    files_processed = 0
    files_created = 0
    files_failed = 0

    for audio_path in audio_files:
        audio_path = Path(audio_path)

        # Preserve relative directory structure
        try:
            rel_path = audio_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(audio_path.name)

        try:
            # Load audio once
            waveform, orig_sr = load_waveform_tensor(str(audio_path))
            audio = waveform.numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            else:
                audio = audio.squeeze()

            # Resample if needed
            if orig_sr != config.sample_rate:
                import torch
                import torchaudio

                waveform_tensor = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_sr, config.sample_rate)
                audio = resampler(waveform_tensor).squeeze().numpy()

            # Create multiple augmented versions
            for i in range(config.multiply):
                # Generate output filename with augmentation index
                stem = rel_path.stem
                suffix = ".wav"  # Always output as WAV
                if config.multiply > 1:
                    out_name = f"{stem}_aug{i+1}{suffix}"
                else:
                    out_name = f"{stem}_aug{suffix}"

                out_path = output_dir / rel_path.parent / out_name

                # Apply augmentation
                augmented = augment_audio(audio, config.sample_rate, pipeline)

                # Save output
                out_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_path), augmented, config.sample_rate)

                files_created += 1

                if verbose:
                    print(f"  Created: {out_path}")

            files_processed += 1

        except Exception as e:
            files_failed += 1
            if verbose:
                print(f"  Failed: {audio_path} - {e}")

    if verbose:
        print(f"Processed {files_processed} files, created {files_created} augmented files, {files_failed} failed")

    return {
        "files_processed": files_processed,
        "files_created": files_created,
        "files_failed": files_failed,
        "output_dir": str(output_dir),
    }
