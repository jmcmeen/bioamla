from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from opensoundscape import Audio as OSSAudio
from opensoundscape import MelSpectrogram as OSSMelSpectrogram
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.sample import AudioSample


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
    """Preprocessing adapter using OpenSoundscape for mel spectrogram generation.

    This class provides a bioamla-compatible interface for generating
    mel spectrograms from audio files or samples, with optional augmentation
    for training via the SpectrogramPreprocessor pipeline.

    The adapter isolates OpenSoundscape dependencies from the core bioamla code,
    enabling future replacement if needed.

    For file-based processing with augmentation (training), uses SpectrogramPreprocessor.
    For samples processing (inference), uses MelSpectrogram directly.

    Example:
        >>> preprocessor = BioamlaPreprocessor(sample_duration=3.0, sample_rate=16000)
        >>> spectrogram = preprocessor.process_file("audio.wav")
        >>> spectrogram.shape
        (128, 313)  # (n_mels, time_frames)

        >>> # With augmentation for training
        >>> from bioamla.adapters.opensoundscape.preprocessing import AugmentationConfig
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
        f_max: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
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

        self._augmentation_config: Optional[AugmentationConfig] = None
        self._preprocessor: Optional[SpectrogramPreprocessor] = None
        self._init_preprocessor()

    def _init_preprocessor(self) -> None:
        """Initialize the underlying OpenSoundscape preprocessor."""
        self._preprocessor = SpectrogramPreprocessor(
            sample_duration=self.sample_duration,
        )

        # Configure spectrogram parameters
        self._preprocessor.pipeline.to_spec.set(
            window_samples=self.n_fft,
            overlap_fraction=1 - (self.hop_length / self.n_fft),
        )

        # Disable all augmentations by default
        self._disable_all_augmentations()

    def _disable_all_augmentations(self) -> None:
        """Disable all augmentation actions in the pipeline."""
        pipeline = self._preprocessor.pipeline

        # Audio-domain augmentations
        if hasattr(pipeline, "audio_time_mask"):
            pipeline.audio_time_mask.bypass = True
        if hasattr(pipeline, "audio_random_gain"):
            pipeline.audio_random_gain.bypass = True
        if hasattr(pipeline, "audio_add_noise"):
            pipeline.audio_add_noise.bypass = True
        if hasattr(pipeline, "random_wrap_audio"):
            pipeline.random_wrap_audio.bypass = True
        if hasattr(pipeline, "random_trim_audio"):
            pipeline.random_trim_audio.bypass = True
        if hasattr(pipeline, "overlay"):
            pipeline.overlay.bypass = True
        if hasattr(pipeline, "add_noise"):
            pipeline.add_noise.bypass = True
        if hasattr(pipeline, "random_affine"):
            pipeline.random_affine.bypass = True

        # Spectrogram-domain augmentations and filters
        if hasattr(pipeline, "time_mask"):
            pipeline.time_mask.bypass = True
        if hasattr(pipeline, "frequency_mask"):
            pipeline.frequency_mask.bypass = True
        if hasattr(pipeline, "bandpass"):
            pipeline.bandpass.bypass = True

    def enable_augmentation(self, config: AugmentationConfig) -> None:
        """Enable augmentation with the given configuration.

        Args:
            config: Augmentation configuration.
        """
        self._augmentation_config = config
        pipeline = self._preprocessor.pipeline

        # SpecAugment time masking
        if config.time_mask and hasattr(pipeline, "time_mask"):
            pipeline.time_mask.bypass = False
            pipeline.time_mask.set(
                max_masks=config.time_mask_max_masks,
                max_width=config.time_mask_max_length,
            )

        # SpecAugment frequency masking
        if config.frequency_mask and hasattr(pipeline, "frequency_mask"):
            pipeline.frequency_mask.bypass = False
            pipeline.frequency_mask.set(
                max_masks=config.frequency_mask_max_masks,
                max_width=config.frequency_mask_max_length,
            )

        # Random gain
        if config.random_gain and hasattr(pipeline, "audio_random_gain"):
            pipeline.audio_random_gain.bypass = False
            pipeline.audio_random_gain.set(gain_range=config.gain_range_db)

        # Background noise
        if config.add_noise and hasattr(pipeline, "add_noise"):
            pipeline.add_noise.bypass = False

    def disable_augmentation(self) -> None:
        """Disable all augmentations."""
        self._augmentation_config = None
        self._disable_all_augmentations()

    def process_file(
        self,
        filepath: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Process an audio file to generate a mel spectrogram.

        Uses SpectrogramPreprocessor pipeline for file-based processing,
        which supports augmentation when enabled.

        Args:
            filepath: Path to audio file.
            start_time: Optional start time in seconds (default: 0).
            end_time: Optional end time in seconds.

        Returns:
            Mel spectrogram as 2D numpy array (frequency x time).
        """
        # Create AudioSample for the preprocessor
        start = start_time or 0.0
        sample = AudioSample(
            source=filepath,
            start_time=start,
            duration=self.sample_duration,
        )

        # Process through OSS preprocessor pipeline
        result = self._preprocessor.forward(sample)

        # The result is an AudioSample with .data containing the tensor
        spectrogram = result.data

        # Convert to numpy if it's a tensor
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.numpy()

        # Remove batch/channel dimensions if present
        while spectrogram.ndim > 2:
            spectrogram = spectrogram.squeeze(0)

        # Resize if needed
        if self.height or self.width:
            spectrogram = self._resize_spectrogram(spectrogram)

        return spectrogram

    def process_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Process raw audio samples to generate a mel spectrogram.

        Uses MelSpectrogram.from_audio() directly for sample-based processing.
        Note: Augmentation is not applied when processing samples directly.

        Args:
            samples: Audio samples as 1D numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Mel spectrogram as 2D numpy array (frequency x time).
        """
        # Create OSS Audio from samples
        audio = OSSAudio(samples, sample_rate)

        # Resample if needed
        if sample_rate != self.sample_rate:
            audio = audio.resample(self.sample_rate)

        # Ensure correct duration
        audio = self._ensure_duration(audio)

        # Generate mel spectrogram directly
        mel = OSSMelSpectrogram.from_audio(
            audio,
            n_mels=self.n_mels,
            window_samples=self.n_fft,
            overlap_fraction=1 - (self.hop_length / self.n_fft),
        )

        spectrogram = np.array(mel.spectrogram)

        # Resize if needed
        if self.height or self.width:
            spectrogram = self._resize_spectrogram(spectrogram)

        return spectrogram

    def _ensure_duration(self, audio: OSSAudio) -> OSSAudio:
        """Ensure audio is exactly the target duration.

        Args:
            audio: Input audio.

        Returns:
            Audio trimmed or padded to target duration.
        """
        target_samples = int(self.sample_duration * self.sample_rate)

        samples = audio.samples
        current_samples = len(samples)

        if current_samples > target_samples:
            # Trim to target duration
            samples = samples[:target_samples]
        elif current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            samples = np.pad(samples, (0, padding), mode="constant")

        return OSSAudio(samples, self.sample_rate)

    def _resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Resize spectrogram to target dimensions.

        Args:
            spectrogram: Input spectrogram.

        Returns:
            Resized spectrogram.
        """
        target_height = self.height or spectrogram.shape[0]
        target_width = self.width or spectrogram.shape[1]

        if spectrogram.shape == (target_height, target_width):
            return spectrogram

        # Use torch for interpolation
        spec_tensor = torch.from_numpy(spectrogram).float()
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

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
        """Convert spectrogram to PyTorch tensor.

        Args:
            spectrogram: Input spectrogram as numpy array.
            normalize: Normalize to [0, 1] range.

        Returns:
            Spectrogram as PyTorch tensor.
        """
        tensor = torch.from_numpy(spectrogram).float()

        if normalize:
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val - min_val > 1e-8:
                tensor = (tensor - min_val) / (max_val - min_val)

        # Add channel dimension if needed
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
