"""Coverage tests for bioamla.audio.batch."""

import pytest

from bioamla.audio.batch import (
    batch_convert_files,
    batch_resample_files,
    batch_transform_files,
    segment_audio_file,
)
from bioamla.batch import BatchResult, SegmentInfo
from bioamla.exceptions import InvalidInputError, NotFoundError


class TestBatchTransformFiles:
    def test_identity_transform(self, test_audio_dir: str, tmp_path) -> None:
        result = batch_transform_files(test_audio_dir, str(tmp_path / "out"), lambda a, sr: a)
        assert isinstance(result, BatchResult)
        assert result.successful == 3
        assert result.failed == 0

    def test_with_resample(self, test_audio_dir: str, tmp_path) -> None:
        result = batch_transform_files(
            test_audio_dir, str(tmp_path / "out"), lambda a, sr: a, sample_rate=8000
        )
        assert result.successful == 3

    def test_missing_input_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_transform_files(str(tmp_path / "nope"), str(tmp_path / "out"), lambda a, sr: a)

    def test_error_collected(self, test_audio_dir: str, tmp_path) -> None:
        def boom(_a, _sr):
            raise RuntimeError("fail")

        result = batch_transform_files(test_audio_dir, str(tmp_path / "out"), boom)
        assert result.failed == 3
        assert len(result.errors) == 3


class TestBatchResampleFiles:
    def test_resamples(self, test_audio_dir: str, tmp_path) -> None:
        result = batch_resample_files(test_audio_dir, str(tmp_path / "out"), 8000)
        assert result.successful == 3


class TestBatchConvertFiles:
    def test_converts(self, test_audio_dir: str, tmp_path) -> None:
        result = batch_convert_files(test_audio_dir, str(tmp_path / "out"), target_format="wav")
        assert result.successful == 3

    def test_missing_input_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_convert_files(str(tmp_path / "nope"), str(tmp_path / "out"))


class TestSegmentAudioFile:
    def test_segments(self, test_audio_path_3s: str, tmp_path) -> None:
        segments = segment_audio_file(test_audio_path_3s, str(tmp_path / "segs"), duration=1.0)
        assert len(segments) == 3
        assert all(isinstance(s, SegmentInfo) for s in segments)
        assert all(s.segment_path.exists() for s in segments)

    def test_with_overlap_and_prefix(self, test_audio_path_3s: str, tmp_path) -> None:
        segments = segment_audio_file(
            test_audio_path_3s,
            str(tmp_path / "segs"),
            duration=1.0,
            overlap=0.5,
            prefix="clip",
        )
        assert len(segments) > 3
        assert segments[0].segment_path.name.startswith("clip")

    def test_invalid_duration_raises(self, test_audio_path_3s: str, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            segment_audio_file(test_audio_path_3s, str(tmp_path / "s"), duration=0)

    def test_overlap_ge_duration_raises(self, test_audio_path_3s: str, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            segment_audio_file(test_audio_path_3s, str(tmp_path / "s"), duration=1.0, overlap=1.0)
