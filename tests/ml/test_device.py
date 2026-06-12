"""Coverage tests for bioamla.ml.device.

Device selection is exercised by patching ``torch.cuda`` so the cuda/cpu
branches run regardless of the host hardware. No GPU required.
"""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from bioamla.ml import device as dev


class TestGetDevice:
    def test_cpu_when_cuda_unavailable(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            d = dev.get_device()
            assert d.type == "cpu"

    def test_cuda_when_available(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_name", return_value="FakeGPU"),
        ):
            d = dev.get_device()
            assert d.type == "cuda"

    def test_prefer_cuda_false_forces_cpu(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            d = dev.get_device(prefer_cuda=False)
            assert d.type == "cpu"


class TestGetDeviceString:
    def test_cpu_string(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert dev.get_device_string() == "cpu"

    def test_cuda_string(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert dev.get_device_string() == "cuda"


class TestMoveToDevice:
    def test_move_with_explicit_string(self) -> None:
        model = MagicMock()
        out = dev.move_to_device(model, device="cpu")
        model.to.assert_called_once()
        assert out is model.to.return_value

    def test_move_with_none_uses_get_device(self) -> None:
        model = MagicMock()
        with patch.object(torch.cuda, "is_available", return_value=False):
            dev.move_to_device(model, device=None)
        model.to.assert_called_once()

    def test_move_with_torch_device(self) -> None:
        model = MagicMock()
        dev.move_to_device(model, device=torch.device("cpu"))
        model.to.assert_called_once()


class TestCudaQueries:
    def test_is_cuda_available(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=True):
            assert dev.is_cuda_available() is True
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert dev.is_cuda_available() is False

    def test_device_count_zero_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert dev.get_device_count() == 0

    def test_device_count_with_cuda(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
        ):
            assert dev.get_device_count() == 2

    def test_current_device_index_none_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert dev.get_current_device_index() is None

    def test_current_device_index_with_cuda(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=1),
        ):
            assert dev.get_current_device_index() == 1

    def test_device_name_none_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert dev.get_device_name(0) is None

    def test_device_name_with_cuda(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(torch.cuda, "get_device_name", return_value="FakeGPU"),
        ):
            assert dev.get_device_name(0) == "FakeGPU"

    def test_device_name_out_of_range(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
        ):
            assert dev.get_device_name(5) is None


class TestDeviceInfo:
    def test_info_without_cuda(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            info = dev.get_device_info()
        assert info["cuda_available"] is False
        assert info["device_count"] == 0
        assert info["devices"] == []

    def test_info_with_cuda(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_name", side_effect=lambda i: f"GPU{i}"),
        ):
            info = dev.get_device_info()
        assert info["cuda_available"] is True
        assert info["device_count"] == 2
        assert info["devices"] == [
            {"index": 0, "name": "GPU0"},
            {"index": 1, "name": "GPU1"},
        ]


class TestDeviceContext:
    def test_context_cpu_noop(self) -> None:
        with dev.DeviceContext("cpu") as ctx:
            assert ctx.device.type == "cpu"
            assert ctx.original_device is None

    def test_context_accepts_torch_device(self) -> None:
        c = dev.DeviceContext(torch.device("cpu"))
        assert c.device.type == "cpu"

    def test_context_cuda_sets_and_restores(self) -> None:
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=3),
            patch.object(torch.cuda, "set_device") as set_dev,
        ):
            with dev.DeviceContext("cuda:1") as ctx:
                assert ctx.original_device == 3
            # entered (set to 1) and exited (restored to 3)
            assert set_dev.call_count == 2
