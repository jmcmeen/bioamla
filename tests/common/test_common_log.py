"""Structured log fields and the JSON formatter."""

import json
import logging

import pytest

from bioamla.common.log import LOG_FIELDS, JsonFormatter, log_fields


def _record(**extra) -> logging.LogRecord:
    logger = logging.getLogger("bioamla.test")
    return logger.makeRecord(
        "bioamla.test", logging.INFO, __file__, 1, "hello %s", ("x",), None, extra=extra
    )


def test_log_fields_drops_none_and_wraps_in_extra() -> None:
    assert log_fields(file="a.wav", index=None) == {"extra": {"file": "a.wav"}}


def test_log_fields_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="fiel"):
        log_fields(fiel="a.wav")


def test_json_formatter_emits_message_and_fields() -> None:
    out = json.loads(
        JsonFormatter().format(_record(**log_fields(file="a.wav", index=2, total=5)["extra"]))
    )
    assert out["message"] == "hello x"
    assert out["level"] == "INFO"
    assert out["logger"] == "bioamla.test"
    assert (out["file"], out["index"], out["total"]) == ("a.wav", 2, 5)
    assert out["time"].endswith("+00:00")


def test_json_formatter_skips_absent_fields_and_stringifies_values() -> None:
    from pathlib import Path

    out = json.loads(JsonFormatter().format(_record(file=Path("a.wav"))))
    assert out["file"] == "a.wav"
    assert not set(LOG_FIELDS) - {"file"} & set(out)


def test_json_formatter_includes_exception() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in out["exception"]
