"""Tests for Windows platform support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.utils.profile import (
    BaselineMetrics,
    _get_baseline_metrics,
    get_metrics_async,
)
from exo.worker.utils.system_info import (
    _get_windows_chip,
    _get_windows_model,
    get_friendly_name,
    get_model_and_chip,
)


class TestWindowsSystemInfo:
    """Tests for Windows system identity functions."""

    @pytest.mark.asyncio
    async def test_get_windows_model_success(self):
        """Test successful model retrieval on Windows."""
        mock_process = MagicMock()
        mock_process.stdout = b"Model=Dell XPS 15 9520\r\n"

        with patch(
            "exo.worker.utils.system_info.run_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            result = await _get_windows_model()
            assert result == "Dell XPS 15 9520"

    @pytest.mark.asyncio
    async def test_get_windows_model_fallback(self):
        """Test fallback when wmic fails."""
        with patch(
            "exo.worker.utils.system_info.run_process",
            new_callable=AsyncMock,
            side_effect=OSError("wmic not found"),
        ):
            result = await _get_windows_model()
            assert result == "Windows PC"

    @pytest.mark.asyncio
    async def test_get_windows_model_ignores_placeholder(self):
        """Test that placeholder values are ignored."""
        mock_process = MagicMock()
        mock_process.stdout = b"Model=To Be Filled By O.E.M.\r\n"

        with patch(
            "exo.worker.utils.system_info.run_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            result = await _get_windows_model()
            assert result == "Windows PC"

    @pytest.mark.asyncio
    async def test_get_windows_chip_success(self):
        """Test successful CPU retrieval on Windows."""
        mock_process = MagicMock()
        mock_process.stdout = b"Name=Intel(R) Core(TM) i9-12900K @ 3.20GHz\r\n"

        with patch(
            "exo.worker.utils.system_info.run_process",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            result = await _get_windows_chip()
            assert result == "Intel(R) Core(TM) i9-12900K @ 3.20GHz"

    @pytest.mark.asyncio
    async def test_get_windows_chip_fallback(self):
        """Test fallback when wmic fails."""
        with patch(
            "exo.worker.utils.system_info.run_process",
            new_callable=AsyncMock,
            side_effect=OSError("wmic not found"),
        ):
            result = await _get_windows_chip()
            assert result == "Unknown CPU"


class TestGetModelAndChipWindows:
    """Tests for get_model_and_chip on Windows."""

    @pytest.mark.asyncio
    async def test_windows_uses_wmic(self):
        """On Windows, uses wmic to get model and chip."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "win32"),
            patch(
                "exo.worker.utils.system_info._get_windows_model",
                new_callable=AsyncMock,
                return_value="Dell XPS 15",
            ),
            patch(
                "exo.worker.utils.system_info._get_windows_chip",
                new_callable=AsyncMock,
                return_value="Intel Core i9",
            ),
        ):
            model, chip = await get_model_and_chip()
            assert model == "Dell XPS 15"
            assert chip == "Intel Core i9"


class TestGetFriendlyNameWindows:
    """Tests for get_friendly_name on Windows."""

    @pytest.mark.asyncio
    async def test_windows_returns_hostname(self):
        """On Windows, returns socket.gethostname."""
        with (
            patch("exo.worker.utils.system_info.sys.platform", "win32"),
            patch(
                "exo.worker.utils.system_info.socket.gethostname",
                return_value="DESKTOP-ABC123",
            ),
        ):
            result = await get_friendly_name()
            assert result == "DESKTOP-ABC123"


class TestWindowsMetrics:
    """Tests for Windows metrics collection."""

    @pytest.mark.asyncio
    async def test_windows_with_nvidia_returns_nvidia_metrics(self):
        """On Windows with NVIDIA, returns NvidiaMetrics."""
        from exo.worker.utils.nvidia_monitor import NvidiaGpuInfo, NvidiaMetrics

        mock_nvidia_metrics = NvidiaMetrics(
            gpu_count=1,
            gpus=[
                NvidiaGpuInfo(
                    index=0,
                    name="RTX 4090",
                    uuid="GPU-123",
                    memory_total_mb=24576,
                    memory_used_mb=1024,
                    memory_free_mb=23552,
                    gpu_utilization=50.0,
                    memory_utilization=4.0,
                    temperature=55.0,
                    power_draw=200.0,
                    power_limit=450.0,
                )
            ],
            gpu_usage=50.0,
            gpu_temp=55.0,
            gpu_power=200.0,
            gpu_memory_total_mb=24576,
            gpu_memory_used_mb=1024,
            gpu_memory_free_mb=23552,
            driver_version="535.154.05",
            cuda_version="12.2",
        )

        with (
            patch("exo.worker.utils.profile.platform.system", return_value="windows"),
            patch("exo.worker.utils.profile.is_nvidia_available", return_value=True),
            patch(
                "exo.worker.utils.profile.nvidia_get_metrics_async",
                new_callable=AsyncMock,
                return_value=mock_nvidia_metrics,
            ),
        ):
            result = await get_metrics_async()
            assert isinstance(result, NvidiaMetrics)
            assert result.gpu_count == 1

    @pytest.mark.asyncio
    async def test_windows_without_nvidia_returns_baseline(self):
        """On Windows without NVIDIA, returns BaselineMetrics."""
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="windows"),
            patch("exo.worker.utils.profile.is_nvidia_available", return_value=False),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=45.0),
        ):
            result = await get_metrics_async()
            assert isinstance(result, BaselineMetrics)
            assert result.cpu_usage == 45.0
            assert result.cpu_temp is None  # No temperature on Windows


class TestBaselineMetrics:
    """Tests for BaselineMetrics model."""

    def test_creation(self):
        """Test creating a BaselineMetrics instance."""
        metrics = BaselineMetrics(cpu_usage=50.0, cpu_temp=65.0)
        assert metrics.cpu_usage == 50.0
        assert metrics.cpu_temp == 65.0

    def test_creation_without_temp(self):
        """Test creating metrics without temperature."""
        metrics = BaselineMetrics(cpu_usage=30.0, cpu_temp=None)
        assert metrics.cpu_usage == 30.0
        assert metrics.cpu_temp is None

    def test_is_frozen(self):
        """Test that the model is frozen."""
        metrics = BaselineMetrics(cpu_usage=10.0, cpu_temp=40.0)
        with pytest.raises(Exception):
            metrics.cpu_usage = 20.0  # type: ignore[misc]


class TestGetBaselineMetrics:
    """Tests for _get_baseline_metrics function."""

    def test_linux_includes_temperature(self):
        """On Linux, includes temperature if available."""
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=35.0),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=55.0),
        ):
            metrics = _get_baseline_metrics()
            assert metrics.cpu_usage == 35.0
            assert metrics.cpu_temp == 55.0

    def test_windows_no_temperature(self):
        """On Windows, temperature is None."""
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="windows"),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=40.0),
        ):
            metrics = _get_baseline_metrics()
            assert metrics.cpu_usage == 40.0
            assert metrics.cpu_temp is None
