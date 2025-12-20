"""
Diagnostic information controller for the bioamla package.

This module provides comprehensive diagnostic functionality to retrieve
information about the bioamla package environment, installed dependencies,
and hardware configuration. It serves as a centralized location for
system introspection and troubleshooting support.

The module offers tools to examine package versions, CUDA device availability,
and other runtime environment details that are crucial for debugging and
ensuring proper bioamla functionality across different deployment scenarios.
"""

import importlib.metadata
from typing import Any, Dict

import torch


def get_bioamla_version() -> str:
    """
    Get the current version of the bioamla package.

    Retrieves the version information from the package metadata using
    importlib.metadata. This provides the exact version string as defined
    in the package's setup configuration.

    Returns:
        str: The version string of the currently installed bioamla package

    Raises:
        importlib.metadata.PackageNotFoundError: If the bioamla package
                                                 is not properly installed
                                                 or not found in the Python
                                                 environment.
    """
    return importlib.metadata.version("bioamla")


def get_package_versions() -> Dict[str, str]:
    """
    Get a comprehensive dictionary of all installed packages and their versions.

    Scans the current Python environment to retrieve version information for
    all installed packages. This is useful for debugging dependency issues,
    creating reproducible environments, or generating system reports.

    Returns:
        Dict[str, str]: A dictionary mapping package names to their version
                       strings. Keys are package names (e.g., "numpy") and
                       values are version strings (e.g., "1.21.0").

    Example:
        >>> packages = get_package_versions()
        >>> print(f"NumPy version: {packages.get('numpy', 'Not installed')}")
        NumPy version: 1.21.0
        >>> print(f"Total packages: {len(packages)}")
        Total packages: 245

    Note:
        The returned dictionary includes all packages in the current Python
        environment, not just bioamla dependencies. For large environments,
        this may return hundreds of packages.
    """
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive information about available CUDA devices and GPU configuration.

    Examines the current system's GPU configuration through PyTorch's CUDA
    interface. This includes CUDA availability status, device count, current
    active device, and detailed information about each available GPU.

    The function is essential for verifying that bioamla can leverage GPU
    acceleration for audio processing tasks and machine learning operations.

    Returns:
        Dict[str, Any]: A dictionary containing comprehensive device information:
            - cuda_available (bool): Whether CUDA is available on the system
            - current_device (Optional[int]): Index of the current active device,
                                            None if CUDA is not available
            - device_count (int): Total number of available CUDA devices
            - devices (List[Dict[str, Any]]): List of device information dicts,
                                            each containing:
                - index (int): Device index number
                - name (str): GPU device name/model
    Note:
        This function requires PyTorch to be installed with CUDA support.
        If PyTorch was installed with CPU-only support, cuda_available
        will be False even if CUDA devices are physically present.
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info["devices"].append({"index": i, "name": torch.cuda.get_device_name(i)})

    return device_info
