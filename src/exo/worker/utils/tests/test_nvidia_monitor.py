"""
Tests for the NVIDIA GPU monitoring module.

These tests use mocking to simulate NVIDIA GPU presence since the tests
may run on systems without NVIDIA GPUs (e.g., macOS, Linux without NVIDIA).
"""

from unittest.mock import MagicMock, patch

import pytest

from exo.worker.utils.nvidia_monitor import (
    NvidiaGpuInfo,
    NvidiaMetrics,
    NvidiaMonitorError,
    _get_gpu_info,
    get_gpu_count,
    get_metrics,
    is_nvidia_available,
    shutdown,
)


class TestNvidiaGpuInfo:
    """Tests for the NvidiaGpuInfo model."""

    def test_gpu_info_creation(self):
        """Test creating a NvidiaGpuInfo instance."""
        gpu_info = NvidiaGpuInfo(
            index=0,
            name="NVIDIA GeForce RTX 4090",
            uuid="GPU-12345678-1234-1234-1234-123456789012",
            memory_total_mb=24576,
            memory_used_mb=1024,
            memory_free_mb=23552,
            gpu_utilization=45.0,
            memory_utilization=4.2,
            temperature=55.0,
            power_draw=150.0,
            power_limit=450.0,
        )
        assert gpu_info.index == 0
        assert gpu_info.name == "NVIDIA GeForce RTX 4090"
        assert gpu_info.memory_total_mb == 24576
        assert gpu_info.memory_free_mb == 23552
        assert gpu_info.gpu_utilization == 45.0

    def test_gpu_info_extra_fields_ignored(self):
        """Test that extra fields are ignored (model_config)."""
        gpu_info = NvidiaGpuInfo(
            index=0,
            name="Test GPU",
            uuid="UUID-123",
            memory_total_mb=8192,
            memory_used_mb=512,
            memory_free_mb=7680,
            gpu_utilization=10.0,
            memory_utilization=6.25,
            temperature=40.0,
            power_draw=50.0,
            power_limit=200.0,
            extra_field="should be ignored",  # type: ignore
        )
        assert not hasattr(gpu_info, "extra_field")


class TestNvidiaMetrics:
    """Tests for the NvidiaMetrics model."""

    def test_metrics_creation(self):
        """Test creating a NvidiaMetrics instance."""
        gpu = NvidiaGpuInfo(
            index=0,
            name="Test GPU",
            uuid="UUID-123",
            memory_total_mb=8192,
            memory_used_mb=512,
            memory_free_mb=7680,
            gpu_utilization=10.0,
            memory_utilization=6.25,
            temperature=40.0,
            power_draw=50.0,
            power_limit=200.0,
        )
        metrics = NvidiaMetrics(
            gpu_count=1,
            gpus=[gpu],
            gpu_usage=10.0,
            gpu_temp=40.0,
            gpu_power=50.0,
            gpu_memory_total_mb=8192,
            gpu_memory_used_mb=512,
            gpu_memory_free_mb=7680,
            driver_version="535.154.05",
            cuda_version="12.2",
        )
        assert metrics.gpu_count == 1
        assert len(metrics.gpus) == 1
        assert metrics.driver_version == "535.154.05"
        assert metrics.cuda_version == "12.2"

    def test_metrics_multiple_gpus(self):
        """Test creating metrics with multiple GPUs."""
        gpus = [
            NvidiaGpuInfo(
                index=i,
                name=f"GPU {i}",
                uuid=f"UUID-{i}",
                memory_total_mb=24576,
                memory_used_mb=1024 * (i + 1),
                memory_free_mb=24576 - 1024 * (i + 1),
                gpu_utilization=20.0 * (i + 1),
                memory_utilization=4.0 * (i + 1),
                temperature=50.0 + i * 5,
                power_draw=100.0 + i * 50,
                power_limit=450.0,
            )
            for i in range(4)
        ]

        total_memory = sum(g.memory_total_mb for g in gpus)
        used_memory = sum(g.memory_used_mb for g in gpus)
        free_memory = sum(g.memory_free_mb for g in gpus)

        metrics = NvidiaMetrics(
            gpu_count=4,
            gpus=gpus,
            gpu_usage=50.0,  # average
            gpu_temp=57.5,  # average
            gpu_power=400.0,  # total
            gpu_memory_total_mb=total_memory,
            gpu_memory_used_mb=used_memory,
            gpu_memory_free_mb=free_memory,
            driver_version="535.154.05",
            cuda_version="12.2",
        )
        assert metrics.gpu_count == 4
        assert metrics.gpu_memory_total_mb == 24576 * 4


class TestIsNvidiaAvailable:
    """Tests for is_nvidia_available function."""

    @patch("exo.worker.utils.nvidia_monitor.platform")
    def test_not_available_on_darwin(self, mock_platform: MagicMock):
        """Test that NVIDIA is not available on macOS."""
        mock_platform.system.return_value = "Darwin"
        assert is_nvidia_available() is False

    @patch("exo.worker.utils.nvidia_monitor._ensure_nvml_initialized")
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    @patch("exo.worker.utils.nvidia_monitor.platform")
    def test_available_on_linux_with_gpu(
        self,
        mock_platform: MagicMock,
        mock_get_pynvml: MagicMock,
        mock_ensure_init: MagicMock,
    ):
        """Test that NVIDIA is available on Linux with GPUs."""
        mock_platform.system.return_value = "Linux"
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_get_pynvml.return_value = mock_pynvml
        assert is_nvidia_available() is True

    @patch("exo.worker.utils.nvidia_monitor._ensure_nvml_initialized")
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    @patch("exo.worker.utils.nvidia_monitor.platform")
    def test_not_available_on_linux_without_gpu(
        self,
        mock_platform: MagicMock,
        mock_get_pynvml: MagicMock,
        mock_ensure_init: MagicMock,
    ):
        """Test that NVIDIA is not available on Linux without GPUs."""
        mock_platform.system.return_value = "Linux"
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_get_pynvml.return_value = mock_pynvml
        assert is_nvidia_available() is False

    @patch("exo.worker.utils.nvidia_monitor._ensure_nvml_initialized")
    @patch("exo.worker.utils.nvidia_monitor.platform")
    def test_not_available_on_nvml_error(
        self,
        mock_platform: MagicMock,
        mock_ensure_init: MagicMock,
    ):
        """Test that NVIDIA is not available when NVML fails."""
        mock_platform.system.return_value = "Linux"
        mock_ensure_init.side_effect = NvidiaMonitorError("NVML init failed")
        assert is_nvidia_available() is False


class TestGetGpuCount:
    """Tests for get_gpu_count function."""

    @patch("exo.worker.utils.nvidia_monitor._ensure_nvml_initialized")
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    def test_get_gpu_count(
        self,
        mock_get_pynvml: MagicMock,
        mock_ensure_init: MagicMock,
    ):
        """Test getting GPU count."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 4
        mock_get_pynvml.return_value = mock_pynvml
        assert get_gpu_count() == 4


class TestGetMetrics:
    """Tests for get_metrics function."""

    def _create_mock_pynvml(self, gpu_count: int = 1) -> MagicMock:
        """Create a mock pynvml module with reasonable defaults."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = gpu_count
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535.154.05"
        mock_pynvml.nvmlSystemGetCudaDriverVersion_v2.return_value = 12020  # 12.2

        # Mock handle
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 4090"
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345678"

        # Memory info
        mock_memory = MagicMock()
        mock_memory.total = 24576 * 1024 * 1024  # 24GB in bytes
        mock_memory.used = 2048 * 1024 * 1024  # 2GB in bytes
        mock_memory.free = 22528 * 1024 * 1024  # 22GB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        # Utilization
        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_util.memory = 8
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        # Temperature
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 55
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        # Power
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 200000  # 200W in mW
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 450000  # 450W

        # NVMLError exception class
        mock_pynvml.NVMLError = Exception

        return mock_pynvml

    @patch("exo.worker.utils.nvidia_monitor._nvml_initialized", True)
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    def test_get_metrics_single_gpu(self, mock_get_pynvml: MagicMock):
        """Test getting metrics from a single GPU."""
        mock_pynvml = self._create_mock_pynvml(gpu_count=1)
        mock_get_pynvml.return_value = mock_pynvml

        metrics = get_metrics()

        assert metrics.gpu_count == 1
        assert len(metrics.gpus) == 1
        assert metrics.driver_version == "535.154.05"
        assert metrics.cuda_version == "12.2"
        assert metrics.gpu_memory_total_mb == 24576
        assert metrics.gpu_usage == 50.0
        assert metrics.gpu_temp == 55.0

    @patch("exo.worker.utils.nvidia_monitor._nvml_initialized", True)
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    def test_get_metrics_no_gpus_raises_error(self, mock_get_pynvml: MagicMock):
        """Test that get_metrics raises error when no GPUs found."""
        mock_pynvml = self._create_mock_pynvml(gpu_count=0)
        mock_get_pynvml.return_value = mock_pynvml

        with pytest.raises(NvidiaMonitorError, match="No NVIDIA GPUs found"):
            get_metrics()


class TestShutdown:
    """Tests for shutdown function."""

    @patch("exo.worker.utils.nvidia_monitor._nvml_initialized", True)
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    def test_shutdown_calls_nvml_shutdown(self, mock_get_pynvml: MagicMock):
        """Test that shutdown calls nvmlShutdown."""
        mock_pynvml = MagicMock()
        mock_get_pynvml.return_value = mock_pynvml

        shutdown()

        mock_pynvml.nvmlShutdown.assert_called_once()

    @patch("exo.worker.utils.nvidia_monitor._nvml_initialized", False)
    @patch("exo.worker.utils.nvidia_monitor._get_pynvml")
    def test_shutdown_noop_when_not_initialized(self, mock_get_pynvml: MagicMock):
        """Test that shutdown does nothing when not initialized."""
        mock_pynvml = MagicMock()
        mock_get_pynvml.return_value = mock_pynvml

        shutdown()

        mock_pynvml.nvmlShutdown.assert_not_called()
