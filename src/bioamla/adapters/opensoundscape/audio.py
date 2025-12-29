from typing import Optional

import numpy as np
from opensoundscape import Audio as OSSAudio


class AudioAdapter:
    """Adapter providing bioamla-compatible interface over OpenSoundscape Audio.

    This class wraps the opensoundscape.Audio class and provides methods
    that return numpy arrays (not torch tensors) for compatibility with
    the bioamla services layer.

    Example:
        >>> adapter = AudioAdapter.from_file("audio.wav", sample_rate=16000)
        >>> resampled = adapter.resample(8000)
        >>> filtered = adapter.bandpass(500, 5000)
        >>> samples = filtered.to_samples()
    """

    def __init__(self, oss_audio: OSSAudio) -> None:
        """Initialize adapter with an OpenSoundscape Audio object.

        Args:
            oss_audio: An opensoundscape.Audio instance to wrap.
        """
        self._audio = oss_audio

    @classmethod
    def from_file(
        cls, path: str, sample_rate: Optional[int] = None
    ) -> "AudioAdapter":
        """Load audio from a file.

        Args:
            path: Path to the audio file.
            sample_rate: Target sample rate. If None, uses the file's native rate.

        Returns:
            AudioAdapter instance wrapping the loaded audio.
        """
        return cls(OSSAudio.from_file(path, sample_rate=sample_rate))

    @classmethod
    def from_samples(
        cls, samples: np.ndarray, sample_rate: int
    ) -> "AudioAdapter":
        """Create adapter from a numpy array of samples.

        Args:
            samples: Audio samples as a 1D numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            AudioAdapter instance wrapping the audio data.
        """
        return cls(OSSAudio(samples, sample_rate))

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the audio in Hz."""
        return self._audio.sample_rate

    @property
    def duration(self) -> float:
        """Get the duration of the audio in seconds."""
        return self._audio.duration

    @property
    def samples(self) -> np.ndarray:
        """Get the audio samples as a numpy array."""
        return self._audio.samples

    def to_samples(self) -> np.ndarray:
        """Return the audio samples as a numpy array.

        Returns:
            Audio samples as a 1D numpy array.
        """
        return self._audio.samples

    def resample(self, target_sr: int) -> "AudioAdapter":
        """Resample audio to a target sample rate.

        Args:
            target_sr: Target sample rate in Hz.

        Returns:
            New AudioAdapter with resampled audio.
        """
        return AudioAdapter(self._audio.resample(target_sr))

    def trim(self, start: float, end: float) -> "AudioAdapter":
        """Trim audio to a time range.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            New AudioAdapter with trimmed audio.
        """
        return AudioAdapter(self._audio.trim(start, end))

    def bandpass(
        self, low_f: float, high_f: float, order: int = 4
    ) -> "AudioAdapter":
        """Apply bandpass filter to audio.

        Args:
            low_f: Low cutoff frequency in Hz.
            high_f: High cutoff frequency in Hz.
            order: Butterworth filter order (steepness of cutoff).

        Returns:
            New AudioAdapter with filtered audio.
        """
        return AudioAdapter(self._audio.bandpass(low_f, high_f, order))

    def lowpass(self, cutoff_f: float, order: int = 4) -> "AudioAdapter":
        """Apply lowpass filter to audio.

        Args:
            cutoff_f: Cutoff frequency in Hz.
            order: Butterworth filter order (steepness of cutoff).

        Returns:
            New AudioAdapter with filtered audio.
        """
        return AudioAdapter(self._audio.lowpass(cutoff_f, order))

    def highpass(self, cutoff_f: float, order: int = 4) -> "AudioAdapter":
        """Apply highpass filter to audio.

        Args:
            cutoff_f: Cutoff frequency in Hz.
            order: Butterworth filter order (steepness of cutoff).

        Returns:
            New AudioAdapter with filtered audio.
        """
        return AudioAdapter(self._audio.highpass(cutoff_f, order))

    def normalize(self, peak_level: float = 1.0) -> "AudioAdapter":
        """Normalize audio to a peak level.

        Args:
            peak_level: Target peak level (0.0 to 1.0).

        Returns:
            New AudioAdapter with normalized audio.
        """
        # OpenSoundscape Audio doesn't have a direct normalize method,
        # so we implement it manually
        samples = self._audio.samples
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            normalized = samples * (peak_level / max_val)
        else:
            normalized = samples
        return AudioAdapter.from_samples(normalized, self._audio.sample_rate)
