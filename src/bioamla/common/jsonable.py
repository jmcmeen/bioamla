"""Convert results to JSON types, for ``--json`` output and the server's responses."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import math
from pathlib import PurePath
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert a result into JSON types.

    Dataclasses (their ``to_dict()`` when they have one), mappings and sequences
    are converted recursively. Paths, datetimes and enums become strings or their
    values, numpy scalars and arrays become numbers and lists, and NaN and
    infinity become ``None``. Anything else becomes ``str(value)``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_jsonable(to_dict())
        # Field by field (not dataclasses.asdict, which deep-copies and skips the
        # to_dict() of nested dataclasses).
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    tolist = getattr(value, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        return to_jsonable(tolist())
    return str(value)


__all__ = ["to_jsonable"]
