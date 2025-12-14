"""
Unit tests for bioamla.core.device module.
"""

import pytest
import torch

from bioamla.core.device import (
    DeviceContext,
    get_current_device_index,
    get_device,
    get_device_count,
    get_device_info,
    get_device_name,
    get_device_string,
    is_cuda_available,
    move_to_device,
)


class TestGetDevice:
    """Tests for get_device function."""

    def test_returns_torch_device(self):
        """Test that function returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_cpu_when_cuda_not_preferred(self):
        """Test that CPU is returned when CUDA not preferred."""
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"

    @pytest.mark.requires_cuda
    def test_returns_cuda_when_available(self):
        """Test that CUDA is returned when available and preferred."""
        device = get_device(prefer_cuda=True)
        assert device.type == "cuda"


class TestGetDeviceString:
    """Tests for get_device_string function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = get_device_string()
        assert isinstance(result, str)
        assert result in ["cuda", "cpu"]

    def test_returns_cpu_when_not_preferred(self):
        """Test that 'cpu' is returned when CUDA not preferred."""
        result = get_device_string(prefer_cuda=False)
        assert result == "cpu"


class TestMoveToDevice:
    """Tests for move_to_device function."""

    def test_moves_model_to_cpu(self):
        """Test moving a model to CPU."""
        model = torch.nn.Linear(10, 5)
        moved = move_to_device(model, device="cpu")

        assert next(moved.parameters()).device.type == "cpu"

    def test_moves_model_to_default_device(self):
        """Test moving a model to default device."""
        model = torch.nn.Linear(10, 5)
        moved = move_to_device(model)

        # Should be on some valid device
        device_type = next(moved.parameters()).device.type
        assert device_type in ["cpu", "cuda"]

    @pytest.mark.requires_cuda
    def test_moves_model_to_cuda(self):
        """Test moving a model to CUDA."""
        model = torch.nn.Linear(10, 5)
        moved = move_to_device(model, device="cuda")

        assert next(moved.parameters()).device.type == "cuda"


class TestIsCudaAvailable:
    """Tests for is_cuda_available function."""

    def test_returns_bool(self):
        """Test that function returns a boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_matches_torch(self):
        """Test that result matches torch.cuda.is_available()."""
        assert is_cuda_available() == torch.cuda.is_available()


class TestGetDeviceCount:
    """Tests for get_device_count function."""

    def test_returns_int(self):
        """Test that function returns an integer."""
        result = get_device_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_zero_when_no_cuda(self):
        """Test that returns 0 when CUDA not available."""
        if not torch.cuda.is_available():
            assert get_device_count() == 0


class TestGetCurrentDeviceIndex:
    """Tests for get_current_device_index function."""

    def test_none_when_no_cuda(self):
        """Test that returns None when CUDA not available."""
        if not torch.cuda.is_available():
            assert get_current_device_index() is None

    @pytest.mark.requires_cuda
    def test_returns_int_when_cuda(self):
        """Test that returns int when CUDA available."""
        result = get_current_device_index()
        assert isinstance(result, int)
        assert result >= 0


class TestGetDeviceName:
    """Tests for get_device_name function."""

    def test_none_when_no_cuda(self):
        """Test that returns None when CUDA not available."""
        if not torch.cuda.is_available():
            assert get_device_name() is None

    @pytest.mark.requires_cuda
    def test_returns_string_when_cuda(self):
        """Test that returns string when CUDA available."""
        result = get_device_name()
        assert isinstance(result, str)


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        info = get_device_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        """Test that dict contains required keys."""
        info = get_device_info()

        assert "cuda_available" in info
        assert "current_device" in info
        assert "device_count" in info
        assert "devices" in info

    def test_devices_is_list(self):
        """Test that devices is a list."""
        info = get_device_info()
        assert isinstance(info["devices"], list)


class TestDeviceContext:
    """Tests for DeviceContext context manager."""

    def test_context_manager_protocol(self):
        """Test that DeviceContext works as context manager."""
        with DeviceContext("cpu") as ctx:
            assert ctx is not None

    def test_exits_cleanly(self):
        """Test that context manager exits without error."""
        with DeviceContext("cpu"):
            pass
        # Should not raise

    @pytest.mark.requires_cuda
    def test_switches_device(self):
        """Test that context manager switches CUDA device."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 CUDA devices")

        original = torch.cuda.current_device()

        with DeviceContext("cuda:1"):
            assert torch.cuda.current_device() == 1

        assert torch.cuda.current_device() == original
