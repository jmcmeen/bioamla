from typing import Optional

import numpy as np
from opensoundscape import MelSpectrogram as OSSMelSpectrogram
from opensoundscape import Spectrogram as OSSSpectrogram

from bioamla.adapters.opensoundscape.audio import AudioAdapter


class SpectrogramAdapter:
    """Adapter for generating spectrograms using OpenSoundscape.

    This class provides methods to generate linear and mel spectrograms
    from audio data, returning numpy arrays for compatibility with
    the bioamla services layer.

    Example:
        >>> adapter = SpectrogramAdapter()
        >>> audio = AudioAdapter.from_file("audio.wav")
        >>> mel_spec = adapter.to_mel_spectrogram(audio)
        >>> linear_spec = adapter.to_spectrogram(audio)
    """

    def to_mel_spectrogram(
        self,
        audio: AudioAdapter,
        n_mels: int = 128,
        window_samples: int = 512,
        overlap_fraction: float = 0.5,
        fft_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a mel spectrogram from audio.

        Args:
            audio: AudioAdapter instance containing the audio data.
            n_mels: Number of mel bands.
            window_samples: Number of audio samples per spectrogram window.
            overlap_fraction: Fractional temporal overlap between windows.
            fft_size: FFT size (if None, defaults based on window_samples).

        Returns:
            Mel spectrogram as a 2D numpy array (n_mels x time).
        """
        # Create OSS Audio from adapter's internal audio
        oss_audio = audio._audio

        mel = OSSMelSpectrogram.from_audio(
            oss_audio,
            n_mels=n_mels,
            window_samples=window_samples,
            overlap_fraction=overlap_fraction,
            fft_size=fft_size,
        )

        return np.array(mel.spectrogram)

    def to_spectrogram(
        self,
        audio: AudioAdapter,
        window_samples: int = 512,
        overlap_fraction: float = 0.5,
    ) -> np.ndarray:
        """Generate a linear spectrogram from audio.

        Args:
            audio: AudioAdapter instance containing the audio data.
            window_samples: Number of samples per window.
            overlap_fraction: Fraction of overlap between windows.

        Returns:
            Linear spectrogram as a 2D numpy array (frequency x time).
        """
        oss_audio = audio._audio

        spec = OSSSpectrogram.from_audio(
            oss_audio,
            window_samples=window_samples,
            overlap_fraction=overlap_fraction,
        )

        return np.array(spec.spectrogram)

    @staticmethod
    def mel_from_file(
        path: str,
        sample_rate: Optional[int] = None,
        n_mels: int = 128,
        window_samples: int = 512,
        overlap_fraction: float = 0.5,
        fft_size: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a mel spectrogram directly from a file.

        Convenience method that loads audio and generates spectrogram in one step.

        Args:
            path: Path to the audio file.
            sample_rate: Target sample rate. If None, uses file's native rate.
            n_mels: Number of mel bands.
            window_samples: Number of audio samples per spectrogram window.
            overlap_fraction: Fractional temporal overlap between windows.
            fft_size: FFT size (if None, defaults based on window_samples).

        Returns:
            Mel spectrogram as a 2D numpy array.
        """
        audio = AudioAdapter.from_file(path, sample_rate=sample_rate)
        adapter = SpectrogramAdapter()
        return adapter.to_mel_spectrogram(
            audio,
            n_mels=n_mels,
            window_samples=window_samples,
            overlap_fraction=overlap_fraction,
            fft_size=fft_size,
        )
