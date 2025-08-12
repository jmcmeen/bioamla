# src/bioamla/controllers/diagnostics.py
# This module provides functions to retrieve diagnostic information about the bioamla package.

def get_bioamla_version() -> str:
    """
    Returns the current version of the bioamla package.
    """
    import importlib.metadata
    return importlib.metadata.version('bioamla')

def get_package_versions() -> dict:
    """
    Returns a dictionary of all installed packages and their versions.
    """
    import importlib.metadata
    return {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}

def get_device_info() -> dict:
    """
    Returns information about the available CUDA devices.
    """
    import torch
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info['devices'].append({
                'index': i,
                'name': torch.cuda.get_device_name(i)
            })
    
    return device_info

