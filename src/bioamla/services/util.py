"""
Service for system information and utility operations.
"""

import platform
import sys

from bioamla.models.util import DeviceInfo, DevicesData, VersionData

from .base import BaseService, ServiceResult


class UtilityService(BaseService):
    """Service for system information and utility operations."""

    def __init__(self) -> None:
        super().__init__()

    def get_device_info(self) -> ServiceResult[DevicesData]:
        """
        Get available compute devices.

        Returns:
            ServiceResult containing DevicesData with available devices.
        """
        import torch

        devices = []
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

        devices.append(
            DeviceInfo(
                name="CPU",
                device_type="cpu",
                device_id="cpu",
            )
        )

        data = DevicesData(
            devices=devices,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )

        return ServiceResult.ok(data=data)

    def get_version(self) -> ServiceResult[VersionData]:
        """
        Get version information for bioamla and dependencies.

        Returns:
            ServiceResult containing VersionData with version info.
        """
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

        data = VersionData(
            bioamla_version=__version__,
            python_version=sys.version,
            platform=platform.platform(),
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
        )

        return ServiceResult.ok(data=data)
