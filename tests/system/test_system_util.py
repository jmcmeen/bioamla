"""Tests for bioamla.system.util (version + device info)."""

from __future__ import annotations

import types

import pytest

from bioamla.system.util import (
    DeviceInfo,
    DevicesData,
    VersionData,
    get_device_info,
    get_version,
)

torch = pytest.importorskip("torch")


def test_get_version_returns_versiondata():
    info = get_version()
    assert isinstance(info, VersionData)
    assert info.bioamla_version
    assert info.python_version
    assert info.platform
    # torch is installed in this environment
    assert info.pytorch_version is not None


def test_get_version_handles_missing_torch(monkeypatch):
    """When torch import fails, pytorch/cuda versions stay None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    info = get_version()
    assert info.pytorch_version is None
    assert info.cuda_version is None


def test_get_device_info_always_has_cpu():
    data = get_device_info()
    assert isinstance(data, DevicesData)
    assert any(d.device_type == "cpu" for d in data.devices)
    cpu = [d for d in data.devices if d.device_type == "cpu"][0]
    assert isinstance(cpu, DeviceInfo)
    assert cpu.device_id == "cpu"


def test_get_device_info_cpu_only(monkeypatch):
    """No CUDA, no MPS -> only CPU device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Make mps report unavailable if present
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    data = get_device_info()
    assert data.cuda_available is False
    assert data.mps_available is False
    assert len(data.devices) == 1
    assert data.devices[0].device_type == "cpu"


def test_get_device_info_cuda_branch(monkeypatch):
    """Patch CUDA to be available with one fake device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    fake_props = types.SimpleNamespace(name="FakeGPU", total_memory=8 * (1024**3))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: fake_props)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    data = get_device_info()
    assert data.cuda_available is True
    cuda_devs = [d for d in data.devices if d.device_type == "cuda"]
    assert len(cuda_devs) == 1
    assert cuda_devs[0].name == "FakeGPU"
    assert cuda_devs[0].device_id == "cuda:0"
    assert cuda_devs[0].memory_gb == 8.0
    # CPU is still appended
    assert data.devices[-1].device_type == "cpu"


def test_get_device_info_mps_branch(monkeypatch):
    """Patch MPS to be available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    fake_mps = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch.backends, "mps", fake_mps, raising=False)

    data = get_device_info()
    assert data.mps_available is True
    mps_devs = [d for d in data.devices if d.device_type == "mps"]
    assert len(mps_devs) == 1
    assert mps_devs[0].device_id == "mps"
    assert data.devices[-1].device_type == "cpu"
