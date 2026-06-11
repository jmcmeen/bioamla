"""Tests for bioamla.common.progress."""

import pytest

from bioamla.common.progress import (
    BatchProcessResult,
    BatchProgress,
    process_batch,
)


class TestBatchProgress:
    def test_percent_and_remaining(self):
        p = BatchProgress(total=4, completed=1)
        assert p.percent == 25.0
        assert p.remaining == 3

    def test_percent_zero_total(self):
        p = BatchProgress(total=0)
        assert p.percent == 0
        assert p.remaining == 0

    def test_defaults(self):
        p = BatchProgress(total=2)
        assert p.completed == 0
        assert p.current_file is None
        assert p.errors == []


class TestProcessBatch:
    def test_all_success(self):
        result = process_batch([1, 2, 3], processor=lambda x: x * 2)
        assert isinstance(result, BatchProcessResult)
        assert result.results == [2, 4, 6]
        assert result.errors == []

    def test_empty_items(self):
        result = process_batch([], processor=lambda x: x)
        assert result.results == []
        assert result.errors == []

    def test_progress_callback_invoked(self):
        calls = []

        def on_progress(p: BatchProgress) -> None:
            calls.append((p.completed, p.current_file))

        process_batch([10, 20], processor=lambda x: x, on_progress=on_progress)
        # initial call + (before+after) per item = 1 + 2*2 = 5
        assert len(calls) == 5
        # completion reaches total
        assert calls[-1][0] == 2

    def test_continue_on_error_collects(self):
        def proc(x):
            if x == 2:
                raise ValueError("boom")
            return x

        result = process_batch([1, 2, 3], processor=proc, continue_on_error=True)
        assert result.results == [1, 3]
        assert len(result.errors) == 1
        item, err = result.errors[0]
        assert item == 2
        assert isinstance(err, ValueError)

    def test_reraise_on_error(self):
        def proc(x):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            process_batch([1], processor=proc, continue_on_error=False)

    def test_error_recorded_in_progress(self):
        seen = []

        def on_progress(p: BatchProgress) -> None:
            seen.append(list(p.errors))

        def proc(x):
            raise ValueError("x")

        process_batch([1], processor=proc, on_progress=on_progress)
        # the final progress callback should include the recorded error string
        assert any(errs for errs in seen)
