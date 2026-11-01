"""Operations for ``system``: thin adapters over :mod:`bioamla.system` and the config
schemas. Installing system packages is local-only.

``system commands`` has no operation: it describes the command-line interface (from
its Click tree), which a server doesn't load; :func:`bioamla.operations.all_operations`
is the interface-free listing.
"""

from __future__ import annotations

from typing import Any

from bioamla.common.config import config_dataclass
from bioamla.operations import OperationContext, PathStr, operation


@config_dataclass
class NoRequest:
    """No arguments."""


@operation("system.version", request=NoRequest)
def version_op(request: NoRequest, context: OperationContext) -> Any:
    """bioamla, Python, platform, PyTorch and CUDA versions."""
    from bioamla.system import util

    return util.get_version()


@operation("system.devices", request=NoRequest)
def devices_op(request: NoRequest, context: OperationContext) -> Any:
    """Compute devices (CUDA, MPS, CPU)."""
    from bioamla.system import util

    return util.get_device_info()


@operation("system.deps", request=NoRequest)
def deps_op(request: NoRequest, context: OperationContext) -> Any:
    """Check the system libraries bioamla uses (FFmpeg, libsndfile, PortAudio)."""
    from bioamla.system import dependency

    return dependency.check_all()


@operation("system.deps-install", request=NoRequest, kind="job", remote=False)
def deps_install_op(request: NoRequest, context: OperationContext) -> dict[str, Any]:
    """Install missing system libraries with the OS package manager (local only)."""
    from bioamla.system import install_missing

    return install_missing()


@config_dataclass
class SchemaRequest:
    """A config's JSON Schema, or (without ``name``) the configs that have one.

    Attributes:
        name: The config (see ``configs``).
        output: A file to write the schema to.
    """

    name: str | None = None
    output: PathStr | None = None


@operation("system.schema", request=SchemaRequest)
def schema_op(request: SchemaRequest, context: OperationContext) -> dict[str, Any]:
    """A config's JSON Schema, or the list of configs."""
    from bioamla.common.config import config_schema_report

    return config_schema_report(request.name, request.output)


@config_dataclass
class ExportSchemasRequest:
    """Every config's JSON Schema, written to a folder.

    Attributes:
        output_dir: The folder.
    """

    output_dir: PathStr


@operation("system.export-schemas", request=ExportSchemasRequest)
def export_schemas_op(request: ExportSchemasRequest, context: OperationContext) -> dict[str, Any]:
    """Write every config's JSON Schema to a folder."""
    from bioamla.common.config import export_config_schemas

    return export_config_schemas(request.output_dir)
