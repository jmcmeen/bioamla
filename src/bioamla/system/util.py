"""System information utilities (version, compute devices).

Folds the former ``services/util.py`` into plain raising functions and
dataclasses. Device info requires ``torch`` (the ``[ml]`` extra) and raises
:class:`~bioamla.exceptions.DependencyError` when it is not installed.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

from bioamla.exceptions import DependencyError


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    name: str
    device_type: str  # "cuda", "mps", "cpu"
    device_id: str | None = None
    memory_gb: float | None = None


@dataclass
class DevicesData:
    """Available compute devices."""

    devices: list[DeviceInfo] = field(default_factory=list)
    cuda_available: bool = False
    mps_available: bool = False


@dataclass
class VersionData:
    """Version information for bioamla and key dependencies."""

    bioamla_version: str
    python_version: str
    platform: str
    pytorch_version: str | None = None
    cuda_version: str | None = None


def get_version() -> VersionData:
    """Return version information for bioamla and key dependencies."""
    from bioamla import __version__

    pytorch_version = None
    cuda_version = None

    try:
        import torch

        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
    except ImportError:
        pass

    return VersionData(
        bioamla_version=__version__,
        python_version=sys.version,
        platform=platform.platform(),
        pytorch_version=pytorch_version,
        cuda_version=cuda_version,
    )


def get_device_info() -> DevicesData:
    """Return available compute devices (CUDA, MPS, CPU).

    Raises:
        DependencyError: If ``torch`` is not installed.
    """
    try:
        import torch
    except ImportError as e:
        raise DependencyError("device info requires torch — install bioamla[ml]") from e

    devices: list[DeviceInfo] = []
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            devices.append(
                DeviceInfo(
                    name=props.name,
                    device_type="cuda",
                    device_id=f"cuda:{i}",
                    memory_gb=round(memory_gb, 1),
                )
            )

    if mps_available:
        devices.append(
            DeviceInfo(
                name="Apple Metal Performance Shaders",
                device_type="mps",
                device_id="mps",
            )
        )

    devices.append(DeviceInfo(name="CPU", device_type="cpu", device_id="cpu"))

    return DevicesData(
        devices=devices,
        cuda_available=cuda_available,
        mps_available=mps_available,
    )


__all__ = [
    "DeviceInfo",
    "DevicesData",
    "VersionData",
    "get_version",
    "get_device_info",
]
