# models/util.py
"""
Data models for utility operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .base import ToDictMixin


@dataclass
class DeviceInfo(ToDictMixin):
    """Information about a compute device."""

    name: str
    device_type: str  # "cuda", "mps", "cpu"
    device_id: Optional[str] = None
    memory_gb: Optional[float] = None


@dataclass
class DevicesData(ToDictMixin):
    """Data returned by get_devices()."""

    devices: List[DeviceInfo] = field(default_factory=list)
    cuda_available: bool = False
    mps_available: bool = False


@dataclass
class VersionData(ToDictMixin):
    """Data returned by get_version()."""

    bioamla_version: str
    python_version: str
    platform: str
    pytorch_version: Optional[str] = None
    cuda_version: Optional[str] = None
