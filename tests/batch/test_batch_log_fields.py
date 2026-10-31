"""The batch engine attaches structured fields to its log records."""

import logging
from pathlib import Path

import pytest

from bioamla.batch import CSVBatchContext, MetadataRow, run_batch, run_csv_batch


def _fields(record: logging.LogRecord) -> dict:
    return {
        k: getattr(record, k)
        for k in ("operation", "file", "index", "total", "status", "error")
        if hasattr(record, k)
    }


def _fail_on_b(path: str) -> str:
    if path.endswith("b.wav"):
        raise ValueError("bad file\nwith a second line")
    return path


@pytest.mark.parametrize("workers", [1, 2])
def test_run_batch_records(caplog: pytest.LogCaptureFixture, workers: int) -> None:
    caplog.set_level(logging.DEBUG, logger="bioamla.batch")
    run_batch(["a.wav", "b.wav"], _fail_on_b, max_workers=workers, operation="segment")
    records = [_fields(r) for r in caplog.records if r.name == "bioamla.batch"]

    processed = [r for r in records if r.get("file") == "a.wav"]
    assert processed and processed[0]["operation"] == "segment" and processed[0]["total"] == 2
    failed = next(r for r in records if "error" in r)
    assert failed["file"] == "b.wav"
    assert failed["error"] == "bad file"  # first line only
    summary = records[-1]
    assert summary == {"operation": "segment", "total": 2, "status": "failed"}
    assert all(hasattr(r, "elapsed_s") for r in caplog.records if r.name == "bioamla.batch")


def test_run_batch_indexes_and_finishes(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="bioamla.batch")
    run_batch(["a.wav", "c.wav"], str, operation="info")
    records = [_fields(r) for r in caplog.records]
    assert [r.get("index") for r in records[:2]] == [1, 2]
    assert records[-1]["status"] == "finished"


def test_non_path_items_have_no_file_field(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="bioamla.batch")
    run_batch([1, 2], str)
    assert not any(hasattr(r, "file") for r in caplog.records)
    assert caplog.records[0].operation == "batch"


def test_run_csv_batch_records(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    present = tmp_path / "a.wav"
    present.write_bytes(b"")
    rows = [
        MetadataRow(file_name="a.wav", file_path=present, metadata_fields={}),
        MetadataRow(file_name="gone.wav", file_path=tmp_path / "gone.wav", metadata_fields={}),
    ]
    context = CSVBatchContext(
        csv_path=tmp_path / "m.csv",
        csv_dir=tmp_path,
        output_dir=None,
        rows=rows,
        fieldnames=["file_name"],
    )
    caplog.set_level(logging.DEBUG, logger="bioamla.batch")
    run_csv_batch(context, lambda row: None, operation="batch audio info")
    records = [_fields(r) for r in caplog.records]
    assert records[0]["file"] == str(present)
    assert records[1]["file"] == str(tmp_path / "gone.wav") and "error" in records[1]
    assert records[-1]["operation"] == "batch audio info"
