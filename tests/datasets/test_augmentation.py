"""Coverage tests for :mod:`bioamla.datasets.augmentation`.

Builds small real audiomentations pipelines and exercises ``augment_audio`` and
the ``batch_augment`` directory walker on tiny synthetic WAVs.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io.wavfile as wav

from bioamla.datasets.augmentation import (
    AugmentationConfig,
    augment_audio,
    batch_augment,
    create_augmentation_pipeline,
    describe_augmentation_pipeline,
)
from bioamla.exceptions import AugmentationError, NotFoundError

pytest.importorskip("audiomentations", reason="audiomentations not installed")


def _write_wav(path, sample_rate: int = 16000, duration: float = 0.5) -> None:
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    samples = (0.4 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    wav.write(str(path), sample_rate, samples)


class TestCreatePipeline:
    def test_no_transforms_returns_none(self) -> None:
        assert create_augmentation_pipeline(AugmentationConfig()) is None

    def test_all_transforms_enabled(self) -> None:
        config = AugmentationConfig(
            add_noise=True,
            time_stretch=True,
            pitch_shift=True,
            gain=True,
            gain_transition=True,
            clipping_distortion=True,
            shuffle=True,
            pipeline_probability=0.9,
        )
        pipeline = create_augmentation_pipeline(config)
        assert pipeline is not None
        assert len(pipeline.transforms) == 6


class TestDescribePipeline:
    def test_none_pipeline_returns_empty(self) -> None:
        assert describe_augmentation_pipeline(None) == []

    def test_one_line_per_transform_with_params(self) -> None:
        config = AugmentationConfig(
            add_noise=True,
            noise_min_snr=5.0,
            noise_max_snr=25.0,
            noise_probability=0.7,
            gain=True,
        )
        pipeline = create_augmentation_pipeline(config)
        lines = describe_augmentation_pipeline(pipeline)
        assert len(lines) == len(pipeline.transforms) == 2
        # Each line names its transform, its probability, and its tunable params.
        noise_line = next(line for line in lines if line.startswith("AddGaussianSNR"))
        assert "p=0.7" in noise_line
        assert "min_snr_db=5.0" in noise_line
        assert "max_snr_db=25.0" in noise_line
        # Bookkeeping attributes are not surfaced.
        assert "are_parameters_frozen" not in noise_line
        assert "parameters" not in noise_line


class TestAugmentAudio:
    def test_applies_pipeline(self) -> None:
        config = AugmentationConfig(gain=True, gain_probability=1.0)
        pipeline = create_augmentation_pipeline(config)
        audio = np.sin(np.linspace(0, 10, 8000)).astype(np.float32)
        out = augment_audio(audio, 16000, pipeline)
        assert out.shape == audio.shape

    def test_casts_non_float32_and_downmixes(self) -> None:
        config = AugmentationConfig(gain=True)
        pipeline = create_augmentation_pipeline(config)
        # 2-D float64 input -> cast to float32 and averaged to mono.
        stereo = np.random.rand(2, 8000).astype(np.float64)
        out = augment_audio(stereo, 16000, pipeline)
        assert out.ndim == 1
        assert out.dtype == np.float32


class TestBatchAugment:
    def test_missing_input_dir_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError, match="Input directory not found"):
            batch_augment(
                str(tmp_path / "nope"),
                str(tmp_path / "out"),
                AugmentationConfig(gain=True),
            )

    def test_no_augmentations_raises(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        with pytest.raises(AugmentationError, match="No augmentations configured"):
            batch_augment(str(in_dir), str(tmp_path / "out"), AugmentationConfig())

    def test_empty_directory_returns_zero_stats(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        stats = batch_augment(
            str(in_dir),
            str(tmp_path / "out"),
            AugmentationConfig(gain=True),
            verbose=True,
        )
        assert stats["files_processed"] == 0
        assert stats["files_created"] == 0

    def test_single_copy(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "a.wav")
        out_dir = tmp_path / "out"

        stats = batch_augment(
            str(in_dir),
            str(out_dir),
            AugmentationConfig(gain=True, sample_rate=16000, multiply=1),
            verbose=False,
        )
        assert stats["files_processed"] == 1
        assert stats["files_created"] == 1
        assert (out_dir / "a_aug.wav").exists()

    def test_multiple_copies_with_resample(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # File at 44100 forces the resample branch (config sample_rate=16000).
        _write_wav(in_dir / "b.wav", sample_rate=44100)
        out_dir = tmp_path / "out"

        stats = batch_augment(
            str(in_dir),
            str(out_dir),
            AugmentationConfig(gain=True, sample_rate=16000, multiply=2),
            verbose=True,
        )
        assert stats["files_processed"] == 1
        assert stats["files_created"] == 2
        assert (out_dir / "b_aug1.wav").exists()
        assert (out_dir / "b_aug2.wav").exists()

    def test_per_file_failure_is_counted(self, tmp_path) -> None:
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        # A non-audio file with a .wav extension triggers a load failure.
        (in_dir / "broken.wav").write_bytes(b"not audio")
        out_dir = tmp_path / "out"

        stats = batch_augment(
            str(in_dir),
            str(out_dir),
            AugmentationConfig(gain=True, sample_rate=16000),
            verbose=True,
        )
        assert stats["files_failed"] == 1
        assert stats["files_created"] == 0
