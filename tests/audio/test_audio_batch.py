"""Coverage tests for bioamla.audio.batch."""

import csv
import shutil
from pathlib import Path

import pytest

from bioamla.audio.batch import (
    batch_convert_files,
    batch_info_files,
    batch_resample_files,
    batch_segment_files,
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


class TestDistinctOutputs:
    def test_same_stem_different_extension(self, test_audio_dir: str, tmp_path) -> None:
        src = Path(test_audio_dir)
        # Two inputs whose names differ only in extension (the decoder reads
        # the WAV header, whatever the suffix says).
        shutil.copy(src / "audio_0.wav", src / "clip.wav")
        shutil.copy(src / "audio_0.wav", src / "clip.flac")
        out = tmp_path / "out"

        result = batch_transform_files(test_audio_dir, str(out), lambda a, sr: a, max_workers=2)

        assert result.failed == 0
        assert {"clip_wav.wav", "clip_flac.wav"} <= {p.name for p in out.glob("*.wav")}


class TestBatchInfoFiles:
    def test_rows_csv_and_progress(self, test_audio_dir: str, tmp_path) -> None:
        calls: list[tuple[int, int]] = []
        result = batch_info_files(
            test_audio_dir,
            tmp_path / "out",
            max_workers=3,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        rows = result.metadata["rows"]
        assert result.successful == 3
        assert [Path(r["file_name"]).name for r in rows] == [
            "audio_0.wav",
            "audio_1.wav",
            "audio_2.wav",
        ]
        assert rows[0]["sample_rate"] == 16000
        assert calls[-1] == (3, 3)
        with open(result.output_files[0], newline="") as f:
            assert len(list(csv.DictReader(f))) == 3

    def test_no_output_dir_writes_nothing(self, test_audio_dir: str) -> None:
        result = batch_info_files(test_audio_dir)
        assert result.output_files == []
        assert len(result.metadata["rows"]) == 3

    def test_missing_input_raises(self, tmp_path) -> None:
        with pytest.raises(NotFoundError):
            batch_info_files(tmp_path / "nope")


class TestSegmentCleanup:
    def test_failed_write_removes_partial_segments(
        self, test_audio_path_3s: str, tmp_path, monkeypatch
    ) -> None:
        import bioamla.audio.batch as audio_batch

        real_save = audio_batch.save_audio
        calls = {"n": 0}

        def flaky_save(path, chunk, sr):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("disk full")
            return real_save(path, chunk, sr)

        monkeypatch.setattr(audio_batch, "save_audio", flaky_save)
        out = tmp_path / "segments"
        with pytest.raises(OSError, match="disk full"):
            segment_audio_file(test_audio_path_3s, str(out), duration=1.0)
        assert list(out.glob("*.wav")) == []


class TestBatchSegmentFiles:
    def test_creates_output_dir_with_no_inputs(self, tmp_path) -> None:
        (tmp_path / "in").mkdir()
        out = tmp_path / "out"
        result = batch_segment_files(tmp_path / "in", out, duration=1.0)
        assert result.total_files == 0
        assert out.is_dir()

    def test_segments_in_input_order(self, test_audio_dir: str, tmp_path) -> None:
        calls: list[tuple[int, int]] = []
        result = batch_segment_files(
            test_audio_dir,
            tmp_path / "out",
            duration=0.5,
            max_workers=3,
            on_progress=lambda done, total: calls.append((done, total)),
        )
        names = [Path(p).name for p in result.output_files]
        assert names == [f"audio_{i}_seg{n:04d}.wav" for i in range(3) for n in range(2)]
        assert calls[-1] == (3, 3)

    @pytest.mark.parametrize(("duration", "overlap"), [(0, 0), (1.0, 1.0)])
    def test_invalid_args_raise(self, test_audio_dir: str, tmp_path, duration, overlap) -> None:
        with pytest.raises(InvalidInputError):
            batch_segment_files(test_audio_dir, tmp_path, duration=duration, overlap=overlap)
