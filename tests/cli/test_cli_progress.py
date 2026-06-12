"""Tests for bioamla.cli.progress (Rich-based helpers)."""

from __future__ import annotations

import bioamla.cli.progress as progress
from bioamla.cli.progress import (
    BatchProcessor,
    ProgressBar,
    confirm,
    is_terminal,
    print_error,
    print_info,
    print_panel,
    print_success,
    print_summary,
    print_table,
    print_warning,
    status,
    track,
)


def test_progressbar_context_and_advance():
    with ProgressBar(total=3, description="Test", disable=True) as pb:
        assert pb.completed == 0
        pb.advance()
        pb.advance(2)
    assert pb.completed == 3


def test_progressbar_update_and_set_total():
    with ProgressBar(total=10, disable=True) as pb:
        pb.update(completed=5, description="halfway")
        assert pb.completed == 5
        pb.set_total(20)
        assert pb.total == 20
        # update with no kwargs is a no-op
        pb.update()


def test_progressbar_columns_without_time():
    with ProgressBar(total=2, disable=True, show_time=False) as pb:
        pb.advance()
        assert pb.completed == 1


def test_progressbar_methods_before_enter_are_safe():
    pb = ProgressBar(total=5, disable=True)
    # No active progress yet; these should not raise
    pb.advance()
    pb.update(completed=1)
    pb.set_total(10)
    assert pb.completed == 0


def test_track_iterates_all_items():
    items = [1, 2, 3, 4]
    result = list(track(items, description="iter", disable=True))
    assert result == items


def test_track_with_generator_no_len():
    gen = (x for x in range(3))
    result = list(track(gen, disable=True))
    assert result == [0, 1, 2]


def test_status_context_manager():
    with status("loading"):
        pass


def test_print_helpers_run(capsys):
    print_success("ok")
    print_error("bad")
    print_warning("warn")
    print_info("info")
    out = capsys.readouterr().out
    assert "ok" in out


def test_print_table(capsys):
    print_table("Title", ["a", "b"], [[1, 2], [3, 4]])
    out = capsys.readouterr().out
    assert "Title" in out


def test_print_panel(capsys):
    print_panel("hello", title="T")
    assert "hello" in capsys.readouterr().out


def test_print_summary_floats_and_ints(capsys):
    print_summary("Stats", {"count": 5, "ratio": 0.123456})
    out = capsys.readouterr().out
    assert "Stats" in out


def test_confirm_yes(monkeypatch):
    monkeypatch.setattr(progress.console, "input", lambda prompt: "yes")
    assert confirm("ok?") is True


def test_confirm_no(monkeypatch):
    monkeypatch.setattr(progress.console, "input", lambda prompt: "n")
    assert confirm("ok?", default=True) is False


def test_confirm_default_on_empty(monkeypatch):
    monkeypatch.setattr(progress.console, "input", lambda prompt: "")
    assert confirm("ok?", default=True) is True
    assert confirm("ok?", default=False) is False


def test_is_terminal_returns_bool():
    assert isinstance(is_terminal(), bool)


def test_batch_processor_all_success():
    bp = BatchProcessor([1, 2, 3], description="proc", verbose=False)
    results = bp.run(lambda x: x * 2)
    assert results == [2, 4, 6]
    assert bp.success_count == 3
    assert bp.error_count == 0


def test_batch_processor_with_errors():
    def proc(x):
        if x == 2:
            raise ValueError("bad")
        return x

    bp = BatchProcessor([1, 2, 3], verbose=False)
    results = bp.run(proc)
    assert results == [1, 3]
    assert bp.success_count == 2
    assert bp.error_count == 1
    assert bp.errors[0][0] == 2


def test_batch_processor_stop_on_error():
    def proc(x):
        raise RuntimeError("boom")

    bp = BatchProcessor([1, 2, 3], verbose=False, stop_on_error=True)
    bp.run(proc)
    assert bp.error_count == 1


def test_batch_processor_verbose_summary(capsys):
    bp = BatchProcessor([1, 2], description="p", verbose=True)
    bp.run(lambda x: x)
    out = capsys.readouterr().out
    assert "successfully" in out


def test_batch_processor_verbose_with_error(capsys):
    def proc(x):
        raise ValueError("nope")

    bp = BatchProcessor([1], verbose=True)
    bp.run(proc)
    out = capsys.readouterr().out
    assert "failed" in out
