"""Machine-readable progress: BIOAMLA_PROGRESS=jsonl."""

import json

import pytest

from bioamla.cli.console import ProgressBar, machine_progress, track


def events(err: str) -> list[dict]:
    return [json.loads(line) for line in err.splitlines() if line.startswith('{"event"')]


@pytest.fixture
def jsonl(monkeypatch):
    monkeypatch.setenv("BIOAMLA_PROGRESS", "jsonl")


def test_off_by_default(capsys) -> None:
    assert not machine_progress()
    with ProgressBar(total=3) as bar:
        bar.advance()
    assert events(capsys.readouterr().err) == []


def test_start_progress_end(jsonl, capsys) -> None:
    with ProgressBar(description="Working") as bar:
        bar.on_progress(1, 2)
        bar.on_progress(2, 2)
    got = events(capsys.readouterr().err)
    assert got[0] == {"event": "start", "description": "Working", "done": 0, "total": None}
    assert got[-2] == {"event": "progress", "description": "Working", "done": 2, "total": 2}
    assert got[-1] == {"event": "end", "description": "Working", "done": 2, "total": 2, "ok": True}


def test_throttled_but_final_count_always_sent(jsonl, capsys) -> None:
    with ProgressBar(total=10_000) as bar:
        for _ in range(10_000):
            bar.advance()
    got = [e for e in events(capsys.readouterr().err) if e["event"] == "progress"]
    assert len(got) <= 102  # one per percent at most, not one per update
    assert got[-1]["done"] == 10_000


def test_description_change_is_always_sent(jsonl, capsys) -> None:
    with ProgressBar(total=10) as bar:
        bar.update(completed=1, description="Loading")
        bar.update(completed=2, description="Saving")
    descriptions = [e["description"] for e in events(capsys.readouterr().err)]
    assert "Loading" in descriptions and "Saving" in descriptions


def test_failure_is_reported(jsonl, capsys) -> None:
    with pytest.raises(RuntimeError), ProgressBar(total=5) as bar:
        bar.advance()
        raise RuntimeError("boom")
    assert events(capsys.readouterr().err)[-1]["ok"] is False


def test_quiet_still_sends_events(jsonl, capsys) -> None:
    with ProgressBar(total=3, disable=True) as bar:  # what --quiet does
        bar.advance()
    assert [e["event"] for e in events(capsys.readouterr().err)] == ["start", "progress", "end"]


def test_every_item_of_a_small_batch_is_reported(jsonl, capsys) -> None:
    with ProgressBar(description="Files") as bar:
        for i in range(1, 6):  # all within a few microseconds
            bar.on_progress(i, 5)
    done = [e["done"] for e in events(capsys.readouterr().err) if e["event"] == "progress"]
    assert done == [1, 2, 3, 4, 5]


def test_no_duplicates_after_the_final_count(jsonl, capsys) -> None:
    with ProgressBar(total=2) as bar:
        bar.on_progress(2, 2)
        bar.on_progress(2, 2)
        bar.update(completed=2)
    progress = [e for e in events(capsys.readouterr().err) if e["event"] == "progress"]
    assert len(progress) == 1


def test_a_closed_stream_doesnt_fail_the_run(jsonl, monkeypatch) -> None:
    class Closed:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    monkeypatch.setattr("sys.stderr", Closed())
    with ProgressBar(total=3) as bar:
        for _ in range(3):
            bar.advance()  # no exception


def test_stopping_a_track_loop_early_counts_the_item_in_hand(jsonl, capsys) -> None:
    loop = track([1, 2, 3, 4], description="Items")
    for item in loop:
        if item == 2:
            break
    loop.close()
    got = events(capsys.readouterr().err)
    assert got[-2]["event"] == "progress" and got[-2]["done"] == 2
    assert got[-1] == {"event": "end", "description": "Items", "done": 2, "total": 4, "ok": False}


def test_an_error_in_a_track_loop_is_not_ok(jsonl, capsys) -> None:
    loop = track([1, 2, 3], description="Items")
    with pytest.raises(RuntimeError):
        for item in loop:
            if item == 2:
                raise RuntimeError("bad item")
    loop.close()
    assert events(capsys.readouterr().err)[-1]["ok"] is False


def test_finishing_a_track_loop_is_ok(jsonl, capsys) -> None:
    for _ in track([1, 2], description="Items"):
        pass
    assert events(capsys.readouterr().err)[-1]["ok"] is True


def test_the_final_count_is_sent_before_end(jsonl, capsys, monkeypatch) -> None:
    with ProgressBar(total=1000) as bar:
        bar.advance()
        monkeypatch.setattr("bioamla.cli.console._JSONL_INTERVAL", 3600.0)
        bar.advance(3)  # within the same percent and interval: held back
    got = events(capsys.readouterr().err)
    assert [e["done"] for e in got if e["event"] == "progress"] == [1, 4]
    assert got[-1]["event"] == "end" and got[-1]["done"] == 4


def test_no_events_outside_the_block(jsonl, capsys) -> None:
    bar = ProgressBar(total=5)
    bar.advance(2)  # before the block: counted, not sent
    with bar:
        bar.advance()
    bar.advance()  # after it: counted, not sent
    got = events(capsys.readouterr().err)
    assert got[0] == {"event": "start", "description": "Processing", "done": 2, "total": 5}
    assert [e["event"] for e in got] == ["start", "progress", "end"]
    assert bar.completed == 4


def test_counting_goes_on_after_stderr_closes(jsonl, monkeypatch) -> None:
    class Closed:
        def write(self, _):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

    with ProgressBar(total=5) as bar:
        monkeypatch.setattr("sys.stderr", Closed())
        bar.advance(2)
        bar.update(completed=4)
        bar.advance()
    assert bar.completed == 5


def test_track_emits_events(jsonl, capsys) -> None:
    assert list(track([1, 2, 3], description="Items")) == [1, 2, 3]
    got = events(capsys.readouterr().err)
    assert got[0]["total"] == 3 and got[-1]["done"] == 3


def test_nothing_is_drawn(jsonl, capsys) -> None:
    with ProgressBar(total=2) as bar:
        bar.advance(2)
    err = capsys.readouterr().err
    assert all(line.startswith('{"event"') for line in err.splitlines() if line.strip())
