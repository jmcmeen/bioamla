"""Mel-spectrogram preprocessing and SpecAugment for AST training.

Generates mel spectrograms from audio files or samples (via librosa, a core
dependency) with optional augmentation — SpecAugment-style time/frequency
masking plus audio-domain gain/noise. The only heavy dependency is ``torch``,
imported lazily and required solely for :meth:`BioamlaPreprocessor.to_tensor`
and fixed-size resizing; mel generation and augmentation run on the slim core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import librosa
import numpy as np

if TYPE_CHECKING:
    import torch


def _require_torch():
    """Import and return the torch module."""
    import torch

    return torch


@dataclass
class AugmentationConfig:
    """Configuration for spectrogram augmentation.

    Attributes:
        time_mask: Enable time masking (SpecAugment).
        frequency_mask: Enable frequency masking (SpecAugment).
        time_mask_max_masks: Maximum number of time masks.
        time_mask_max_length: Maximum length of time mask (fraction of total).
        frequency_mask_max_masks: Maximum number of frequency masks.
        frequency_mask_max_length: Maximum length of frequency mask (fraction of total).
        random_gain: Enable random gain adjustment.
        gain_range_db: Gain range in dB (min, max).
        add_noise: Enable adding background noise.
    """

    time_mask: bool = False
    frequency_mask: bool = False
    time_mask_max_masks: int = 2
    time_mask_max_length: float = 0.1
    frequency_mask_max_masks: int = 2
    frequency_mask_max_length: float = 0.1
    random_gain: bool = False
    gain_range_db: tuple[float, float] = field(default_factory=lambda: (-6.0, 6.0))
    add_noise: bool = False


class BioamlaPreprocessor:
    """Mel-spectrogram preprocessing for AST, with optional SpecAugment.

    Generates mel spectrograms from audio files or raw samples using librosa.
    When augmentation is enabled (training), applies audio-domain gain/noise and
    spectrogram-domain time/frequency masking. ``process_samples`` never
    augments (inference path).

    Example:
        >>> preprocessor = BioamlaPreprocessor(sample_duration=3.0, sample_rate=16000)
        >>> spectrogram = preprocessor.process_file("audio.wav")
        >>> spectrogram.shape
        (128, 94)  # (n_mels, time_frames)

        >>> # With augmentation for training
        >>> from bioamla.ml import AugmentationConfig
        >>> aug_config = AugmentationConfig(time_mask=True, frequency_mask=True)
        >>> preprocessor.enable_augmentation(aug_config)
        >>> augmented_spec = preprocessor.process_file("audio.wav")
    """

    def __init__(
        self,
        sample_duration: float = 3.0,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            sample_duration: Duration of audio clips in seconds.
            sample_rate: Target sample rate in Hz.
            n_mels: Number of mel bands.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            f_min: Minimum frequency for mel filterbank.
            f_max: Maximum frequency (None = Nyquist).
            height: Output spectrogram height (None = n_mels).
            width: Output spectrogram width (None = computed from duration).
        """
        self.sample_duration = sample_duration
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.height = height
        self.width = width

        self._augmentation_config: AugmentationConfig | None = None

    def enable_augmentation(self, config: AugmentationConfig) -> None:
        """Enable augmentation with the given configuration."""
        self._augmentation_config = config

    def disable_augmentation(self) -> None:
        """Disable all augmentations."""
        self._augmentation_config = None

    def process_file(
        self,
        filepath: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> np.ndarray:
        """Process an audio file to generate a mel spectrogram.

        Applies augmentation when enabled (audio-domain on samples, then
        spectrogram-domain masking on the mel).

        Args:
            filepath: Path to audio file.
            start_time: Optional start time in seconds (default: 0).
            end_time: Optional end time in seconds (default: start + sample_duration).

        Returns:
            Mel spectrogram as 2D numpy array (frequency x time).
        """
        from bioamla.audio import load_audio

        samples, sr = load_audio(filepath)  # mono float32 numpy, no torch
        if sr != self.sample_rate:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=self.sample_rate)

        # Slice the requested window before fitting to the target duration.
        start = start_time or 0.0
        start_idx = int(start * self.sample_rate)
        if end_time is not None:
            samples = samples[start_idx : int(end_time * self.sample_rate)]
        else:
            samples = samples[start_idx:]

        samples = self._ensure_duration(samples)

        cfg = self._augmentation_config
        if cfg is not None:
            samples = self._augment_audio(samples, cfg)

        spectrogram = self._mel_spectrogram(samples)

        if cfg is not None:
            spectrogram = self._augment_spectrogram(spectrogram, cfg)

        if self.height or self.width:
            spectrogram = self._resize_spectrogram(spectrogram)

        return spectrogram

    def process_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Process raw audio samples to generate a mel spectrogram.

        Augmentation is not applied when processing samples directly.

        Args:
            samples: Audio samples as 1D numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Mel spectrogram as 2D numpy array (frequency x time).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if sample_rate != self.sample_rate:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=self.sample_rate)

        samples = self._ensure_duration(samples)
        spectrogram = self._mel_spectrogram(samples)

        if self.height or self.width:
            spectrogram = self._resize_spectrogram(spectrogram)

        return spectrogram

    def _mel_spectrogram(self, samples: np.ndarray) -> np.ndarray:
        """Compute a mel power spectrogram via librosa."""
        return librosa.feature.melspectrogram(
            y=samples,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=2.0,
        )

    def _ensure_duration(self, samples: np.ndarray) -> np.ndarray:
        """Trim or zero-pad samples to exactly ``sample_duration`` seconds."""
        target = int(self.sample_duration * self.sample_rate)
        n = len(samples)
        if n > target:
            return samples[:target]
        if n < target:
            return np.pad(samples, (0, target - n), mode="constant")
        return samples

    def _augment_audio(self, samples: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
        """Apply audio-domain augmentation (random gain, additive noise)."""
        rng = np.random.default_rng()
        out = samples.astype(np.float32, copy=True)

        if cfg.random_gain:
            gain_db = rng.uniform(*cfg.gain_range_db)
            out = out * float(10.0 ** (gain_db / 20.0))

        if cfg.add_noise:
            rms = float(np.sqrt(np.mean(out**2))) or 1.0
            out = out + rng.normal(0.0, 0.01 * rms, size=out.shape).astype(np.float32)

        return out

    def _augment_spectrogram(self, spec: np.ndarray, cfg: AugmentationConfig) -> np.ndarray:
        """Apply SpecAugment-style time/frequency masking (sets bands to 0)."""
        rng = np.random.default_rng()
        out = spec.copy()
        n_mels, n_frames = out.shape

        if cfg.frequency_mask:
            for _ in range(rng.integers(1, cfg.frequency_mask_max_masks + 1)):
                width = int(rng.uniform(0, cfg.frequency_mask_max_length) * n_mels)
                if width <= 0:
                    continue
                f0 = int(rng.integers(0, max(1, n_mels - width)))
                out[f0 : f0 + width, :] = 0.0

        if cfg.time_mask:
            for _ in range(rng.integers(1, cfg.time_mask_max_masks + 1)):
                width = int(rng.uniform(0, cfg.time_mask_max_length) * n_frames)
                if width <= 0:
                    continue
                t0 = int(rng.integers(0, max(1, n_frames - width)))
                out[:, t0 : t0 + width] = 0.0

        return out

    def _resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Resize spectrogram to (height, width) via torch bilinear interpolation."""
        target_height = self.height or spectrogram.shape[0]
        target_width = self.width or spectrogram.shape[1]

        if spectrogram.shape == (target_height, target_width):
            return spectrogram

        torch = _require_torch()
        spec_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            spec_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze().numpy()

    def to_tensor(
        self,
        spectrogram: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Convert spectrogram to a PyTorch tensor.

        Args:
            spectrogram: Input spectrogram as numpy array.
            normalize: Normalize to [0, 1] range.

        Returns:
            Spectrogram as a PyTorch tensor with a leading channel dim.
        """
        torch = _require_torch()
        tensor = torch.from_numpy(spectrogram).float()

        if normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val - min_val > 1e-8:
                tensor = (tensor - min_val) / (max_val - min_val)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    @property
    def augmentation_enabled(self) -> bool:
        """Check if augmentation is enabled."""
        return self._augmentation_config is not None

    @property
    def config(self) -> dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            "sample_duration": self.sample_duration,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "height": self.height,
            "width": self.width,
            "augmentation_enabled": self.augmentation_enabled,
        }
