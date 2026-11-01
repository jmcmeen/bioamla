"""
bioamla.operations — the index of everything bioamla can do, shared by its front ends.

An **operation** is a named unit of work (``detect.energy``, ``audio.features``, ...)
with a typed request, a function and a typed result. The CLI builds requests from
command options; the server builds them from HTTP requests; both call :func:`run`.

Operations are thin adapters over the **public library**: each maps its request
onto calls to documented domain functions (e.g. :func:`bioamla.detect.detect_file`)
and returns their result. Anything an operation can do, a script can do by calling
the library directly. See
[Workspaces, remote work and agent skills](../architecture/workspaces-and-remote.md).

Operations are declared in their domain packages with :func:`operation` and loaded
on demand, so listing or running one never imports unrelated domains.

Example:
    >>> from bioamla import operations
    >>> result = operations.run("detect.energy", {"file": "rec.wav", "threshold_db": -25})
    >>> result.count
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from bioamla.exceptions import BioamlaError, DependencyError, NotFoundError

#: Name prefix -> the module that declares those operations; the longest matching
#: prefix wins, so ``batch.audio`` and ``batch.detect`` can load separately (one
#: domain's missing extra doesn't hide another's). Each domain adds its module here
#: when its commands are converted.
OPERATION_MODULES: dict[str, str] = {
    "batch.audio": "bioamla.audio.batch_operations",
    "batch.cluster": "bioamla.cluster.batch_operations",
    "batch.detect": "bioamla.detect.batch_operations",
    "batch.index": "bioamla.indices.batch_operations",
    "batch.models": "bioamla.ml.batch_operations",
    "annotation": "bioamla.datasets.annotation_operations",
    "audio": "bioamla.audio.operations",
    "cluster": "bioamla.cluster.operations",
    "dataset": "bioamla.datasets.dataset_operations",
    "detect": "bioamla.detect.operations",
    "indices": "bioamla.indices.operations",
    "models": "bioamla.ml.operations",
}

OperationKind = Literal["quick", "job"]


class OperationCancelled(BioamlaError):
    """The operation stopped because its caller asked it to."""


class PathField:
    """Marks a request field that names a file or folder; see :data:`PathStr`.

    The field also accepts a :class:`~pathlib.Path` (stored as a string), and in the
    request's JSON Schema it gets ``"format": "path"``.
    """

    def __repr__(self) -> str:
        return "PathField()"

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        from pathlib import PurePath

        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            lambda v: str(v) if isinstance(v, PurePath) else v, handler(source)
        )

    def __get_pydantic_json_schema__(self, core_schema: Any, handler: Any) -> dict[str, Any]:
        schema = handler(core_schema)
        schema["format"] = "path"
        return schema


#: The type of a request field naming a file or folder. Use it as ``PathStr``,
#: ``PathStr | None`` or ``list[PathStr]`` on a request's own fields: each path goes
#: through :meth:`OperationContext.resolve_path` before the operation runs (on the server,
#: that confines it to the allowed folders). Registering a request that hides a path
#: anywhere else (a nested config, a dict) is an error.
PathStr = Annotated[str, PathField()]


@dataclass
class OperationContext:
    """What a front end supplies to an operation run.

    Attributes:
        workspace: The workspace the operation runs in, if any.
        on_progress: ``on_progress(done, total)`` for long operations.
        cancelled: Returns True when the caller wants the operation to stop.
        path_check: Maps a request path to the path to use, raising if it isn't
            allowed. The default (local runs) allows anything and returns it as given.
    """

    workspace: Any = None
    on_progress: Callable[[int, int], None] | None = None
    cancelled: Callable[[], bool] = field(default=lambda: False)
    path_check: Callable[[str], str] = field(default=lambda p: p)

    def resolve_path(self, path: str | None) -> str | None:
        """``path`` after the context's check (``None`` stays ``None``)."""
        return None if path is None else self.path_check(path)

    def check_cancelled(self) -> None:
        """Raise :class:`OperationCancelled` if the caller asked to stop."""
        if self.cancelled():
            raise OperationCancelled("Cancelled")

    def progress(self, done: int, total: int) -> None:
        """Report progress (a no-op without ``on_progress``), then honour a cancel."""
        if self.on_progress is not None:
            self.on_progress(done, total)
        self.check_cancelled()


@dataclass(frozen=True)
class Operation:
    """A registered operation.

    Attributes:
        name: Dotted name, e.g. ``detect.energy``.
        request: The request class (a validated config with a JSON Schema).
        func: ``func(request, context) -> result``.
        kind: ``quick`` (answer directly) or ``job`` (long; run in the background).
        remote: Whether a server may run it (False for e.g. local playback).
        summary: One-line description.
    """

    name: str
    request: type
    func: Callable[[Any, OperationContext], Any]
    kind: OperationKind = "quick"
    remote: bool = True
    summary: str = ""

    def path_fields(self) -> tuple[str, ...]:
        """The request's fields that name files or folders."""
        return path_fields(self.request)


_REGISTRY: dict[str, Operation] = {}


def operation(
    name: str, *, request: type, kind: OperationKind = "quick", remote: bool = True
) -> Callable[[Callable], Callable]:
    """Register the decorated function as the operation ``name``.

    The function takes ``(request, context)`` and returns the result; it should be a
    thin adapter over public library functions.
    """

    def register(func: Callable) -> Callable:
        existing = _REGISTRY.get(name)
        if existing is not None and _qualname(existing.func) != _qualname(func):
            raise ValueError(
                f"Operation {name!r} is already registered by {_qualname(existing.func)}"
            )
        path_fields(request)  # rejects paths it couldn't check
        summary = next(iter((func.__doc__ or "").strip().splitlines()), "")
        _REGISTRY[name] = Operation(name, request, func, kind, remote, summary)
        return func

    return register


def _qualname(func: Callable) -> str:
    return f"{func.__module__}.{func.__qualname__}"


def _module_for(name: str) -> str | None:
    """The module for ``name``'s longest registered prefix (``batch.audio.info`` ->
    ``batch.audio`` before ``batch``); a name may itself be registered (``batch.index``)."""
    parts = name.split(".")
    for n in range(len(parts), 0, -1):
        module = OPERATION_MODULES.get(".".join(parts[:n]))
        if module is not None:
            return module
    return None


def get(name: str) -> Operation:
    """The operation called ``name``, loading its domain on first use.

    Raises:
        NotFoundError: If there is no such operation.
        DependencyError: If its domain needs an extra that isn't installed.
    """
    if name not in _REGISTRY:
        module = _module_for(name)
        if module is None:
            raise NotFoundError(f"No operation {name!r}")
        _load(module)
    if name not in _REGISTRY:
        raise NotFoundError(f"No operation {name!r}")
    return _REGISTRY[name]


def _load(module: str) -> None:
    from bioamla._extras import extra_hint

    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:  # a domain that imports an extra's module directly
        hint = extra_hint(e)
        if hint is None:  # not an optional dependency: a real bug, so let it show
            raise
        raise DependencyError(f"{e.name} is not installed. {hint}") from e


def all_operations() -> tuple[list[Operation], dict[str, str]]:
    """Every operation that can load here, by name, and the domains that can't
    (prefix -> why), e.g. because an extra isn't installed."""
    unavailable: dict[str, str] = {}
    for prefix, module in OPERATION_MODULES.items():
        try:
            _load(module)
        except DependencyError as e:
            unavailable[prefix] = str(e)
    return sorted(_REGISTRY.values(), key=lambda op: op.name), unavailable


@functools.cache
def path_fields(request_type: type) -> tuple[str, ...]:
    """Names of the fields of ``request_type`` that name files or folders.

    Raises:
        TypeError: If a :data:`PathStr` sits where :func:`run` can't check it (in a
            nested config, a dict, ...).
    """
    hints = typing.get_type_hints(request_type, include_extras=True)
    names = []
    for name, hint in hints.items():
        kind = _path_kind(hint)
        if kind is not None:
            names.append(name)
        elif _mentions_path(hint):
            raise TypeError(
                f"{request_type.__name__}.{name}: a path must be a PathStr, PathStr | None "
                "or list[PathStr] field of the request itself, so it can be checked"
            )
    return tuple(names)


def _is_path_str(hint: Any) -> bool:
    return typing.get_origin(hint) is Annotated and any(
        isinstance(m, PathField) for m in typing.get_args(hint)[1:]
    )


def _path_kind(hint: Any) -> str | None:
    """``"one"`` for PathStr (optionally ``| None``), ``"many"`` for list/tuple of them."""
    if _is_path_str(hint):
        return "one"
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        kinds = {_path_kind(a) for a in args if a is not type(None)}
        return kinds.pop() if len(kinds) == 1 else None
    if origin in (list, tuple) and args and all(_is_path_str(a) for a in args if a is not ...):
        return "many"
    return None


def _mentions_path(hint: Any, seen: frozenset = frozenset()) -> bool:
    if _is_path_str(hint):
        return True
    if dataclasses.is_dataclass(hint) and isinstance(hint, type) and hint not in seen:
        nested = typing.get_type_hints(hint, include_extras=True).values()
        return any(_mentions_path(h, seen | {hint}) for h in nested)
    return any(_mentions_path(a, seen) for a in typing.get_args(hint))


def build_request(op: Operation, data: Any) -> Any:
    """``data`` (a dict, or already a request) as ``op``'s validated request.

    Raises:
        ConfigValidationError: If it doesn't fit the request type.
    """
    if isinstance(data, op.request):
        return data
    from bioamla.common.config import validate_config

    return validate_config(op.request, dict(data), source=f"operation {op.name}")


def run(name: str, request: Any, context: OperationContext | None = None) -> Any:
    """Run the operation ``name`` on ``request`` (a request object or a dict).

    Every path field goes through ``context.resolve_path`` first.

    Raises:
        NotFoundError: If there's no such operation.
        ConfigValidationError: If the request is invalid.
        BioamlaError: Whatever the operation raises.
    """
    op = get(name)
    context = context or OperationContext()
    req = build_request(op, request)
    resolved: dict[str, Any] = {}
    for f in op.path_fields():
        value = getattr(req, f)
        if isinstance(value, (list, tuple)):
            resolved[f] = type(value)(context.resolve_path(v) for v in value)
        elif value is not None:
            resolved[f] = context.resolve_path(value)
    if resolved:
        req = dataclasses.replace(req, **resolved)
    return op.func(req, context)


__all__ = [
    "OPERATION_MODULES",
    "Operation",
    "OperationCancelled",
    "OperationContext",
    "PathField",
    "PathStr",
    "all_operations",
    "build_request",
    "get",
    "operation",
    "path_fields",
    "run",
]
