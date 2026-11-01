"""The operations registry, and detect.energy run without the CLI."""

import subprocess
import sys

import pytest

from bioamla import operations
from bioamla.exceptions import ConfigValidationError, NotFoundError


def test_unknown_operation():
    with pytest.raises(NotFoundError):
        operations.get("nope.thing")
    with pytest.raises(NotFoundError):
        operations.get("detect.nope")


def test_registering_and_path_fields():
    from bioamla.common.config import config_dataclass

    @config_dataclass
    class Req:
        src: operations.PathStr
        dest: operations.PathStr | None = None
        n: int = 1

    seen = []

    @operations.operation("test.echo", request=Req, kind="job", remote=False)
    def echo(request, context):
        """Echo the request."""
        seen.append(context)
        return request

    try:
        op = operations.get("test.echo")
        assert (op.kind, op.remote, op.summary) == ("job", False, "Echo the request.")
        assert op.path_fields() == ("src", "dest")

        ctx = operations.OperationContext(path_check=lambda p: "/root/" + p)
        out = operations.run("test.echo", {"src": "a.wav", "n": "3"}, ctx)
        assert (out.src, out.dest, out.n) == ("/root/a.wav", None, 3)
        assert seen == [ctx]

        with pytest.raises(ConfigValidationError, match="unknown field"):
            operations.run("test.echo", {"src": "a.wav", "bogus": 1})
    finally:
        operations._REGISTRY.pop("test.echo", None)


def _register(name, request, doc="Do it."):
    def func(request, context):
        return request

    func.__doc__ = doc
    func.__qualname__ = f"func_{name}"
    return operations.operation(name, request=request)(func)


@pytest.fixture
def cleanup():
    names = []
    yield names
    for n in names:
        operations._REGISTRY.pop(n, None)


def test_paths_accept_pathlib_and_lists(cleanup, tmp_path):
    from bioamla.common.config import config_dataclass

    @config_dataclass
    class Req:
        one: operations.PathStr
        many: list[operations.PathStr]

    cleanup.append("test.paths")
    _register("test.paths", Req)
    assert operations.get("test.paths").path_fields() == ("one", "many")
    ctx = operations.OperationContext(path_check=lambda p: p.upper())
    out = operations.run("test.paths", {"one": tmp_path / "a", "many": ["b", tmp_path]}, ctx)
    assert out.one == str(tmp_path / "a").upper()
    assert out.many == ["B", str(tmp_path).upper()]


def test_hidden_paths_are_rejected_at_registration():
    from bioamla.common.config import config_dataclass

    @config_dataclass
    class Inner:
        out: operations.PathStr

    @config_dataclass
    class Outer:
        inner: Inner

    @config_dataclass
    class Mapped:
        files: dict[str, operations.PathStr]

    for req in (Outer, Mapped):
        with pytest.raises(TypeError, match="a path must be"):
            _register("test.hidden", req)
    assert "test.hidden" not in operations._REGISTRY


def test_duplicate_name_is_an_error(cleanup):
    from bioamla.common.config import config_dataclass

    @config_dataclass
    class Req:
        n: int = 0

    cleanup.append("test.dup")
    operations.operation("test.dup", request=Req)(lambda r, c: 1)
    with pytest.raises(ValueError, match="already registered"):

        @operations.operation("test.dup", request=Req)
        def other(request, context):
            return 2


def test_blank_docstring_gives_empty_summary(cleanup):
    from bioamla.common.config import config_dataclass

    @config_dataclass
    class Req:
        n: int = 0

    cleanup.append("test.blank")
    _register("test.blank", Req, doc="\n    ")
    assert operations.get("test.blank").summary == ""


def test_path_check_can_refuse():
    pytest.importorskip("librosa")

    def refuse(path):
        raise PermissionError(path)

    ctx = operations.OperationContext(path_check=refuse)
    with pytest.raises(PermissionError):
        operations.run("detect.energy", {"file": "/etc/passwd"}, ctx)


def test_cancel_and_progress():
    calls = []
    ctx = operations.OperationContext(
        on_progress=lambda d, t: calls.append((d, t)), cancelled=lambda: len(calls) >= 2
    )
    ctx.progress(1, 3)
    with pytest.raises(operations.OperationCancelled):
        ctx.progress(2, 3)
    assert calls == [(1, 3), (2, 3)]


def test_request_schema_marks_paths():
    pytest.importorskip("librosa")
    from bioamla.common.config import config_schema

    props = config_schema(operations.get("detect.energy").request)["properties"]
    assert props["file"]["format"] == "path"
    assert {"format": "path", "type": "string"} in props["output"]["anyOf"]
    assert "format" not in props["low_freq"]


def test_all_operations_lists_detect():
    pytest.importorskip("librosa")
    ops, unavailable = operations.all_operations()
    assert "detect.energy" in [op.name for op in ops]
    assert unavailable == {}


def test_missing_extra_is_unavailable(monkeypatch):
    monkeypatch.setitem(operations.OPERATION_MODULES, "ghost", "bioamla._no_such_ops")
    from bioamla.exceptions import DependencyError

    def boom(name):
        if name == "bioamla._no_such_ops":
            raise DependencyError("needs bioamla[ghost]")
        return __import__(name)

    monkeypatch.setattr(operations.importlib, "import_module", boom)
    with pytest.raises(DependencyError, match="ghost"):
        operations.get("ghost.op")
    _, unavailable = operations.all_operations()
    assert unavailable == {"ghost": "needs bioamla[ghost]"}


def test_missing_optional_module_is_unavailable(monkeypatch):
    monkeypatch.setitem(operations.OPERATION_MODULES, "ghost", "bioamla._no_such_ops")

    def boom(name):
        if name == "bioamla._no_such_ops":
            raise ModuleNotFoundError("No module named 'umap'", name="umap")
        return __import__(name)

    monkeypatch.setattr(operations.importlib, "import_module", boom)
    _, unavailable = operations.all_operations()
    assert "[cluster]" in unavailable["ghost"]


def test_missing_non_optional_module_is_a_bug(monkeypatch):
    monkeypatch.setitem(operations.OPERATION_MODULES, "ghost", "bioamla._no_such_ops")
    with pytest.raises(ModuleNotFoundError):
        operations.get("ghost.op")


def test_importing_registry_is_light():
    code = (
        "import sys, bioamla.operations\n"
        "print(any(m.startswith(('bioamla.detect', 'numpy', 'pydantic')) for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_detect_energy_operation(test_audio_path, tmp_path):
    pytest.importorskip("librosa")
    out = tmp_path / "found.json"
    found = operations.run(
        "detect.energy",
        {"file": test_audio_path, "low_freq": 200, "high_freq": 1000, "output": str(out)},
    )
    assert found.count > 0 and found.output == str(out) and out.exists()
