"""
Device Management
=================

Centralized device management for PyTorch operations: device detection and
model placement.

PyTorch ships in the base install but is imported lazily inside each function so
this module imports fast.

Usage:
    from bioamla.ml.device import get_device, move_to_device

    device = get_device()
    model = move_to_device(model)
"""

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = logging.getLogger(__name__)


def _require_torch():
    """Import and return the torch module."""
    import torch

    return torch


def get_device(prefer_cuda: bool = True) -> "torch.device":
    """
    Get the best available device for computation.

    Args:
        prefer_cuda: If True (default), prefer CUDA if available.

    Returns:
        torch.device: The selected device (cuda or cpu)
    """
    torch = _require_torch()
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device")

    return device


def get_device_string(prefer_cuda: bool = True) -> str:
    """
    Get the device as a string (for use with device_map="auto" etc.).

    Args:
        prefer_cuda: If True (default), prefer CUDA if available.

    Returns:
        str: "cuda" or "cpu"
    """
    torch = _require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def move_to_device(
    model: "nn.Module", device: Union[str, "torch.device"] | None = None
) -> "nn.Module":
    """
    Move a model to the specified device.

    Args:
        model: PyTorch model to move
        device: Target device. If None, uses get_device() to select.

    Returns:
        The model on the target device
    """
    torch = _require_torch()
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    logger.debug(f"Moved model to {device}")
    return model


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    torch = _require_torch()
    return torch.cuda.is_available()


def get_device_count() -> int:
    """Get the number of available CUDA devices."""
    torch = _require_torch()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_current_device_index() -> int | None:
    """Get the current CUDA device index, or None if not available."""
    torch = _require_torch()
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None


def get_device_name(device_index: int = 0) -> str | None:
    """
    Get the name of a CUDA device.

    Args:
        device_index: The device index (default 0)

    Returns:
        Device name string, or None if not available
    """
    torch = _require_torch()
    if torch.cuda.is_available() and device_index < torch.cuda.device_count():
        return torch.cuda.get_device_name(device_index)
    return None


def get_device_info() -> dict:
    """
    Get comprehensive information about available devices.

    Returns:
        dict: Device information including:
            - cuda_available: Whether CUDA is available
            - current_device: Current device index (if CUDA)
            - device_count: Number of CUDA devices
            - devices: List of device info dicts
    """
    torch = _require_torch()
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "current_device": get_current_device_index(),
        "device_count": get_device_count(),
        "devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info["devices"].append({"index": i, "name": torch.cuda.get_device_name(i)})

    return device_info


class DeviceContext:
    """
    Context manager for temporarily using a specific device.

    Example:
        with DeviceContext("cuda:1"):
            # Operations here use cuda:1
            pass
        # Back to original device
    """

    def __init__(self, device: Union[str, "torch.device"]):
        torch = _require_torch()
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.original_device: int | None = None

    def __enter__(self):
        torch = _require_torch()
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.original_device = torch.cuda.current_device()
            if self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch = _require_torch()
        if self.original_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.original_device)
        return False


__all__ = [
    "get_device",
    "get_device_string",
    "move_to_device",
    "is_cuda_available",
    "get_device_count",
    "get_current_device_index",
    "get_device_name",
    "get_device_info",
    "DeviceContext",
]
