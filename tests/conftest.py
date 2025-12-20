# tests/conftest.py
"""
Global pytest fixtures for bioamla tests.
"""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def tmp_audio_file(tmp_path):
    """Create a temporary WAV file for testing."""
    import soundfile as sf

    # Generate 1 second of silence at 16kHz
    samples = np.zeros(16000, dtype=np.float32)
    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), samples, 16000)
    return str(audio_path)


@pytest.fixture
def tmp_audio_file_with_signal(tmp_path):
    """Create a temporary WAV file with a sine wave signal."""
    import soundfile as sf

    # Generate 1 second of 440Hz sine wave at 16kHz
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio_path = tmp_path / "test_signal.wav"
    sf.write(str(audio_path), samples, sr)
    return str(audio_path)


@pytest.fixture
def sample_audio_data():
    """Return an AudioData object with synthetic samples."""
    from bioamla.controllers.audio_file import AudioData

    # 1 second of 440Hz sine wave at 16kHz
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    return AudioData(
        samples=samples,
        sample_rate=sr,
        source_path="/fake/path/audio.wav",
    )


@pytest.fixture
def mock_storage(mocker):
    """Mock the storage backend for run tracking."""
    mock = mocker.patch("bioamla.core.storage.get_storage")
    storage_instance = Mock()
    storage_instance.create_run.return_value = "test_run_id"
    storage_instance.update_run.return_value = True
    storage_instance.complete_run.return_value = True
    storage_instance.fail_run.return_value = True
    storage_instance.is_available.return_value = True
    mock.return_value = storage_instance
    return storage_instance


@pytest.fixture
def tmp_dir_with_audio_files(tmp_path):
    """Create a temporary directory with multiple audio files."""
    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create 3 test audio files
    for i in range(3):
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        sf.write(str(audio_dir / f"test_{i}.wav"), samples, 16000)

    return str(audio_dir)
