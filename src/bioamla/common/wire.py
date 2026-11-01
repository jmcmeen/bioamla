"""Results as JSON that turns back into the same Python objects.

``to_jsonable`` (and each result's ``to_dict()``) make a result readable JSON, but the
shape is for people and scripts: types are gone. The wire form keeps them, so a client
of a bioamla server gets back the very objects a local run returns (a command can then
print a remote result exactly as it prints a local one).

Supported: ``None``, booleans, numbers (including NaN and infinity), strings, lists,
tuples, dicts, :class:`pathlib.Path`, dates and datetimes, NumPy arrays and scalars,
pandas DataFrames (with their column dtypes and index), and dataclasses and enums
defined in ``bioamla``. Rebuilding only ever imports
``bioamla`` modules and only constructs dataclasses and enums from them.

Example:
    >>> from bioamla.common.wire import from_wire, to_wire
    >>> from_wire(to_wire(result)) == result
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import importlib
import json
import math
from pathlib import Path, PurePath
from typing import Any

from bioamla.exceptions import BioamlaError, InvalidInputError

_TAG = "__wire__"


class WireError(InvalidInputError):
    """A value can't be put on the wire, or wire data can't be rebuilt."""


def _type_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _is_bioamla(module: str) -> bool:
    return module == "bioamla" or module.startswith("bioamla.")


def _bioamla_type(name: str, kind: str) -> type:
    """The class ``name`` (``module:QualName``), if it is a dataclass (``kind``
    ``"dataclass"``) or an enum (``"enum"``) defined in bioamla. Nothing else is ever
    looked up, so wire data can't reach other classes (e.g. one a bioamla module
    imports)."""
    module, _, qualname = name.partition(":")
    if not _is_bioamla(module) or not qualname:
        raise WireError(f"Not a bioamla type: {name!r}")
    try:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
            if not isinstance(obj, type) or not _is_bioamla(obj.__module__):
                raise WireError(f"Not a bioamla type: {name!r}")
    except (ImportError, AttributeError) as e:
        raise WireError(f"Unknown type {name!r}: {e}") from e
    except WireError:
        raise
    except BioamlaError as e:  # e.g. the module needs an extra that isn't installed
        raise WireError(f"Can't load {name!r} here: {e}") from e
    wanted = dataclasses.is_dataclass(obj) if kind == "dataclass" else issubclass(obj, enum.Enum)
    if not wanted or f"{obj.__module__}:{obj.__qualname__}" != name:
        raise WireError(f"Not a bioamla {kind}: {name!r}")
    return obj


def to_wire(value: Any) -> Any:
    """``value`` as JSON-ready data that :func:`from_wire` turns back into it.

    Raises:
        WireError: If it holds something unsupported (see the module docstring).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        cls = type(value)
        if not cls.__module__.startswith("bioamla"):
            raise WireError(f"Can't send a {_type_name(cls)}")
        return {_TAG: "enum", "type": _type_name(cls), "value": to_wire(value.value)}
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_TAG: "float", "value": repr(value)}
    if isinstance(value, PurePath):
        return {_TAG: "path", "value": str(value)}
    if isinstance(value, _dt.datetime):
        return {_TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, _dt.date):
        return {_TAG: "date", "value": value.isoformat()}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [to_wire(v) for v in value]}
    if isinstance(value, list):
        return [to_wire(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and _TAG not in value:
            return {k: to_wire(v) for k, v in value.items()}
        return {_TAG: "dict", "items": [[to_wire(k), to_wire(v)] for k, v in value.items()]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        if not cls.__module__.startswith("bioamla"):
            raise WireError(f"Can't send a {_type_name(cls)}")
        fields = {f.name: to_wire(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {_TAG: "dataclass", "type": _type_name(cls), "fields": fields}
    pandas = _loaded("pandas")
    if pandas is not None and isinstance(value, pandas.DataFrame):
        return {
            _TAG: "dataframe",
            "columns": [to_wire(c) for c in value.columns],
            "dtypes": [str(t) for t in value.dtypes],
            "index": to_wire(value.index.tolist()),
            "data": to_wire(value.to_numpy(dtype=object).tolist()),
        }
    numpy = _numpy()
    if numpy is not None:
        if isinstance(value, numpy.ndarray):
            if value.dtype.hasobject:
                raise WireError("Can't send an object array")
            import base64

            raw = numpy.ascontiguousarray(value).tobytes()
            return {_TAG: "ndarray", "dtype": value.dtype.str, "shape": list(value.shape),
                    "b64": base64.b64encode(raw).decode("ascii")}  # fmt: skip
        if isinstance(value, numpy.generic):
            return {_TAG: "npscalar", "dtype": value.dtype.str, "value": to_wire(value.item())}
    raise WireError(f"Can't send a {type(value).__module__}.{type(value).__qualname__}")


def from_wire(data: Any) -> Any:
    """The object :func:`to_wire` made ``data`` from.

    Raises:
        WireError: If ``data`` names a type outside ``bioamla``, or can't be rebuilt.
    """
    if isinstance(data, list):
        return [from_wire(v) for v in data]
    if not isinstance(data, dict):
        return data
    tag = data.get(_TAG)
    if tag is None:
        return {k: from_wire(v) for k, v in data.items()}
    try:
        if tag == "float":
            return float(data["value"])
        if tag == "path":
            return Path(data["value"])
        if tag == "datetime":
            return _dt.datetime.fromisoformat(data["value"])
        if tag == "date":
            return _dt.date.fromisoformat(data["value"])
        if tag == "tuple":
            return tuple(from_wire(v) for v in data["items"])
        if tag == "dict":
            return {_hashable(from_wire(k)): from_wire(v) for k, v in data["items"]}
        if tag == "enum":
            return _bioamla_type(data["type"], "enum")(from_wire(data["value"]))
        if tag == "dataclass":
            return _dataclass(data)
        if tag == "dataframe":
            import pandas

            columns = [from_wire(c) for c in data["columns"]]
            frame = pandas.DataFrame(
                from_wire(data["data"]), columns=columns, index=from_wire(data["index"])
            )
            if frame.empty:
                frame = pandas.DataFrame(columns=columns)
            return frame.astype(dict(zip(columns, data["dtypes"], strict=True)))
        if tag in ("ndarray", "npscalar"):
            import numpy

            if tag == "npscalar":
                return numpy.dtype(data["dtype"]).type(from_wire(data["value"]))
            import base64

            dtype = numpy.dtype(data["dtype"])
            if dtype.hasobject:
                raise WireError("Object arrays aren't sent")
            raw = base64.b64decode(data["b64"], validate=True)
            return numpy.frombuffer(raw, dtype=dtype).reshape(data["shape"]).copy()
    except WireError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise WireError(f"Malformed {tag} on the wire: {e}") from e
    raise WireError(f"Unknown wire tag {tag!r}")


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _dataclass(data: dict[str, Any]) -> Any:
    cls = _bioamla_type(data["type"], "dataclass")
    values = data["fields"]
    # Set the fields as the server's object had them, without running __init__ or
    # __post_init__ again (including init=False fields and frozen classes).
    obj = cls.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.name in values:
            value = from_wire(values[f.name])
        elif f.default is not dataclasses.MISSING:
            value = f.default
        elif f.default_factory is not dataclasses.MISSING:
            value = f.default_factory()
        else:
            raise WireError(f"{data['type']} is missing its {f.name!r}")
        object.__setattr__(obj, f.name, value)
    return obj


def _loaded(module: str) -> Any:
    import sys

    return sys.modules.get(module)  # a value of its type can only exist if it's loaded


def _numpy() -> Any:
    return _loaded("numpy")


def dumps(value: Any) -> str:
    """``value`` on the wire, as a JSON string."""
    return json.dumps(to_wire(value))


__all__ = ["WireError", "dumps", "from_wire", "to_wire"]
