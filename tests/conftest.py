"""Shared fixtures for domain tests."""

import functools
import shutil

import numpy as np
import pytest

from bioamla.audio import AudioData


@functools.cache
def _torchcodec_loadable() -> bool:
    """Return True if torchcodec's libtorchcodec/libav backend can load.

    ``torchaudio.load`` (which ``load_waveform_tensor`` and HuggingFace dataset
    decoding delegate to) decodes through torchcodec, which dlopens the FFmpeg
    ``libav*`` shared libraries. Importing ``torchcodec`` triggers that load, so
    a failed import means audio decoding is unavailable on this machine.
    """
    try:
        import torchcodec  # noqa: F401  (import triggers the libtorchcodec/libav load)
    except Exception:
        return False
    return True


@functools.cache
def _ffmpeg_cli_available() -> bool:
    """Return True if the ``ffmpeg`` CLI is on PATH (pydub uses it to encode)."""
    return shutil.which("ffmpeg") is not None


@pytest.fixture
def requires_torchcodec() -> None:
    """Skip unless the torchcodec/FFmpeg audio *decode* backend is available.

    Apply with ``@pytest.mark.usefixtures("requires_torchcodec")`` to tests that
    decode a real audio file (``load_waveform_tensor``, HuggingFace dataset
    materialization, AST inference/training). See the README "System
    dependencies" section for the FFmpeg requirement.
    """
    if not _torchcodec_loadable():
        pytest.skip(
            "requires the torchcodec/FFmpeg decode backend (libav 4-8); "
            "see README → System dependencies"
        )


@pytest.fixture
def requires_ffmpeg_cli() -> None:
    """Skip unless the ``ffmpeg`` CLI is available for non-WAV *encoding*.

    pydub shells out to ``ffmpeg`` to encode FLAC/MP3/OGG/M4A (WAV is written
    natively, so WAV-only tests don't need this). Apply with
    ``@pytest.mark.usefixtures("requires_ffmpeg_cli")``.
    """
    if not _ffmpeg_cli_available():
        pytest.skip(
            "requires the ffmpeg CLI for non-WAV encoding; see README → System dependencies"
        )


@pytest.fixture
def sample_audio_data() -> AudioData:
    """Create sample audio data for testing.

    Returns a 1-second, 16kHz mono audio signal with a 440Hz sine wave.
    """
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        source_path="/test/audio.wav",
        metadata={"test": True},
    )


@pytest.fixture
def sample_audio_data_44100() -> AudioData:
    """Create sample audio data at 44100 Hz for resampling tests."""
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        source_path="/test/audio_44100.wav",
    )


@pytest.fixture
def sample_audio_stereo() -> AudioData:
    """Create stereo audio data for channel conversion tests."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * frequency * t)
    right = 0.5 * np.sin(2 * np.pi * (frequency * 1.5) * t)  # Different frequency
    samples = np.stack([left, right], axis=-1)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=2,
        source_path="/test/stereo.wav",
    )


@pytest.fixture
def sample_audio_with_noise() -> AudioData:
    """Create audio data with noise for denoise tests."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
    samples = signal + noise

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        source_path="/test/noisy.wav",
    )


@pytest.fixture
def sample_audio_3s() -> AudioData:
    """Create 3-second audio data for segmentation tests."""
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * frequency * t)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        source_path="/test/audio_3s.wav",
    )


@pytest.fixture
def sample_audio_with_silence() -> AudioData:
    """Create audio data with silence regions for trim tests."""
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = np.zeros_like(t)

    # Add signal only in middle section (0.5s - 1.5s)
    start_idx = int(0.5 * sample_rate)
    end_idx = int(1.5 * sample_rate)
    t_signal = t[start_idx:end_idx] - t[start_idx]
    samples[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * frequency * t_signal)

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        source_path="/test/with_silence.wav",
    )


@pytest.fixture
def test_audio_path(tmp_path) -> str:
    """Create a temporary test audio file and return its path.

    Uses scipy to create a valid WAV file.
    """
    import scipy.io.wavfile as wav

    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    audio_path = tmp_path / "test_audio.wav"
    wav.write(str(audio_path), sample_rate, samples)

    return str(audio_path)


@pytest.fixture
def test_audio_path_3s(tmp_path) -> str:
    """Create a 3-second temporary test audio file."""
    import scipy.io.wavfile as wav

    sample_rate = 16000
    duration = 3.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    audio_path = tmp_path / "test_audio_3s.wav"
    wav.write(str(audio_path), sample_rate, samples)

    return str(audio_path)


@pytest.fixture
def test_audio_dir(tmp_path) -> str:
    """Create a directory with multiple test audio files."""
    import scipy.io.wavfile as wav

    audio_dir = tmp_path / "audio_files"
    audio_dir.mkdir()

    sample_rate = 16000
    duration = 1.0

    for i, freq in enumerate([440.0, 880.0, 1320.0]):
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        audio_path = audio_dir / f"audio_{i}.wav"
        wav.write(str(audio_path), sample_rate, samples)

    return str(audio_dir)
