"""Tests for the generic batch runner (:func:`bioamla.batch.run_batch`)."""

import pytest

from bioamla.batch import discover_files, run_batch

from ._helpers import fail_on_odd, square


class TestRunBatchSequential:
    def test_collects_successful_results(self):
        result = run_batch([1, 2, 3], square)
        assert result.total_files == 3
        assert result.successful == 3
        assert result.failed == 0
        # Non-None return values are captured as strings.
        assert sorted(result.output_files) == ["1", "4", "9"]

    def test_per_item_failures_collected(self):
        result = run_batch([1, 2, 3, 4], fail_on_odd, continue_on_error=True)
        assert result.total_files == 4
        assert result.successful == 2  # 2 and 4
        assert result.failed == 2  # 1 and 3
        assert len(result.errors) == 2
        assert any("odd value: 1" in e for e in result.errors)

    def test_continue_on_error_false_reraises(self):
        with pytest.raises(ValueError):
            run_batch([1, 3], fail_on_odd, continue_on_error=False)

    def test_progress_callback_invoked(self):
        seen = []
        run_batch([1, 2, 3], square, on_progress=lambda c, t: seen.append((c, t)))
        assert seen == [(1, 3), (2, 3), (3, 3)]


class TestRunBatchParallel:
    def test_parallel_collects_results(self):
        result = run_batch([1, 2, 3, 4], square, max_workers=2)
        assert result.successful == 4
        assert result.failed == 0
        # future.result() return values are collected even in parallel mode.
        assert sorted(int(v) for v in result.output_files) == [1, 4, 9, 16]

    def test_parallel_collects_failures(self):
        result = run_batch([1, 2, 3, 4], fail_on_odd, max_workers=2, continue_on_error=True)
        assert result.successful == 2
        assert result.failed == 2
        assert len(result.errors) == 2


class TestDiscoverFiles:
    def test_discovers_audio_files(self, test_audio_dir):
        files = discover_files(test_audio_dir, recursive=True)
        assert len(files) == 3
        assert all(f.suffix == ".wav" for f in files)

    def test_missing_dir_returns_empty(self, tmp_path):
        assert discover_files(tmp_path / "nope") == []

    def test_file_filter_applied(self, test_audio_dir):
        files = discover_files(
            test_audio_dir, file_filter=lambda p: p.name == "audio_0.wav"
        )
        assert [f.name for f in files] == ["audio_0.wav"]
