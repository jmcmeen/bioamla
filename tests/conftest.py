"""
Pytest Configuration and Fixtures
==================================

This module provides shared fixtures and configuration for the bioamla test suite.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_metadata_rows():
    """Sample metadata rows for testing."""
    return [
        {
            "filename": "audio1.wav",
            "split": "train",
            "target": "1",
            "category": "species_a",
            "attr_id": "user1",
            "attr_lic": "CC-BY",
            "attr_url": "https://example.com/1",
            "attr_note": ""
        },
        {
            "filename": "audio2.wav",
            "split": "train",
            "target": "2",
            "category": "species_b",
            "attr_id": "user2",
            "attr_lic": "CC-BY-NC",
            "attr_url": "https://example.com/2",
            "attr_note": ""
        },
        {
            "filename": "audio3.wav",
            "split": "test",
            "target": "1",
            "category": "species_a",
            "attr_id": "user3",
            "attr_lic": "CC0",
            "attr_url": "https://example.com/3",
            "attr_note": "test note"
        }
    ]


@pytest.fixture
def sample_metadata_with_inat():
    """Sample metadata with iNaturalist optional fields."""
    return [
        {
            "filename": "inat_123_sound_456.mp3",
            "split": "train",
            "target": "1",
            "category": "Lithobates catesbeianus",
            "attr_id": "naturalist1",
            "attr_lic": "CC-BY",
            "attr_url": "https://inaturalist.org/sounds/456",
            "attr_note": "",
            "observation_id": "123",
            "sound_id": "456",
            "common_name": "American Bullfrog",
            "taxon_id": "12345",
            "observed_on": "2024-01-15",
            "location": "37.7749,-122.4194",
            "place_guess": "San Francisco, CA",
            "observer": "naturalist1",
            "quality_grade": "research",
            "observation_url": "https://inaturalist.org/observations/123"
        }
    ]


@pytest.fixture
def metadata_csv_file(temp_dir, sample_metadata_rows):
    """Create a temporary metadata CSV file."""
    import csv

    csv_path = temp_dir / "metadata.csv"
    fieldnames = list(sample_metadata_rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_metadata_rows)

    return csv_path


@pytest.fixture
def mock_audio_file(temp_dir):
    """Create a mock audio file for testing."""
    # Create a minimal WAV file header
    import struct

    audio_path = temp_dir / "test_audio.wav"

    # Minimal WAV file with 1 second of silence at 16kHz, 16-bit mono
    sample_rate = 16000
    duration = 1  # seconds
    num_samples = sample_rate * duration
    bits_per_sample = 16
    num_channels = 1

    # WAV file parameters
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    file_size = 36 + data_size

    with open(audio_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")

        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Chunk size
        f.write(struct.pack("<H", 1))   # Audio format (PCM)
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        # Write silence (zeros)
        f.write(b"\x00" * data_size)

    return audio_path


# Skip markers for conditional tests
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: marks tests that require CUDA"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and environment."""
    import torch

    skip_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "requires_cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)
