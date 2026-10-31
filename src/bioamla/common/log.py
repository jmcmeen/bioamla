"""Structured fields on bioamla's log records, and a JSON formatter for them.

Library code logs through :mod:`logging`. Records about a file, a batch or a
training run also carry structured fields (passed with ``extra=``), so a consumer
such as a GUI, a batch runner or a log aggregator can read them without parsing
the message text. :data:`LOG_FIELDS` is the contract: these names, with these
meanings, are what bioamla attaches. A record carries only the fields that apply.

The library never configures handlers. An application that wants the fields as
JSON lines attaches :class:`JsonFormatter` to a handler::

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.getLogger("bioamla").addHandler(handler)

The ``bioamla`` command does this with ``bioamla --log-format json``.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from bioamla.exceptions import InvalidInputError

#: Structured fields bioamla attaches to log records: name -> meaning.
LOG_FIELDS: dict[str, str] = {
    "operation": (
        "What is running: a batch function's name (e.g. 'segment', 'indices'), "
        "a CLI command path for CLI-driven batches, or 'train'."
    ),
    "file": "Path of the file the record is about.",
    "index": "1-based position of this item (a file, or a training step) within the run.",
    "total": "Number of items (files, or training steps) in the run.",
    "run_id": "Identifier of a training run, the same on every record of that run.",
    "epoch": "Fractional training epoch reached.",
    "elapsed_s": "Seconds since the operation started.",
    "status": "Outcome, on the last record of a run: 'finished', 'cancelled' or 'failed'.",
    "error": "Error message, on the record for an item that failed.",
}


def log_fields(**fields: Any) -> dict[str, dict[str, Any]]:
    """Build the ``extra=`` argument for a log call, checked against :data:`LOG_FIELDS`.

    ``None`` values are dropped, so callers can pass optional fields directly.

    Example:
        >>> logger.info("Segmented %s", path, **log_fields(file=str(path), index=3, total=10))

    Raises:
        InvalidInputError: If a name isn't in :data:`LOG_FIELDS` (a bug in the caller).
    """
    unknown = sorted(set(fields) - set(LOG_FIELDS))
    if unknown:
        raise InvalidInputError(f"Unknown log fields {unknown}; add them to LOG_FIELDS first")
    return {"extra": {k: v for k, v in fields.items() if v is not None}}


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line.

    Keys: ``time`` (UTC, ISO 8601), ``level``, ``logger``, ``message``, then every
    :data:`LOG_FIELDS` field present on the record, then ``exception`` when the
    record has exception info. Values that aren't JSON types (e.g. paths) are
    written as strings.
    """

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in LOG_FIELDS:
            if name in record.__dict__:
                out[name] = record.__dict__[name]
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)
