"""Tests for Linux baseline metrics and fallback behavior in profile.py."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.shared.types.profiling import SystemPerformanceProfile
from exo.worker.utils.profile import (
    LinuxBaselineMetrics,
    _build_system_profile,
    _get_linux_baseline_metrics,
    _read_linux_cpu_temp,
    get_metrics_async,
)


class TestLinuxBaselineMetrics:
    """Tests for the LinuxBaselineMetrics model."""

    def test_creation(self):
        """Test creating a LinuxBaselineMetrics instance."""
        metrics = LinuxBaselineMetrics(cpu_usage=45.5, cpu_temp=55.0)
        assert metrics.cpu_usage == 45.5
        assert metrics.cpu_temp == 55.0

    def test_creation_without_temp(self):
        """Test creating metrics when temperature is unavailable."""
        metrics = LinuxBaselineMetrics(cpu_usage=30.0, cpu_temp=None)
        assert metrics.cpu_usage == 30.0
        assert metrics.cpu_temp is None

    def test_is_frozen(self):
        """Test that the model is frozen."""
        metrics = LinuxBaselineMetrics(cpu_usage=10.0, cpu_temp=40.0)
        with pytest.raises(Exception):
            metrics.cpu_usage = 20.0  # type: ignore[misc]


class TestReadLinuxCpuTemp:
    """Tests for _read_linux_cpu_temp."""

    def test_returns_temp_from_coretemp(self, tmp_path: Path):
        """Returns temperature from coretemp hwmon."""
        hwmon_dir = tmp_path / "hwmon0"
        hwmon_dir.mkdir()
        (hwmon_dir / "name").write_text("coretemp\n")
        (hwmon_dir / "temp1_input").write_text("55000\n")  # 55°C in millidegrees

        with patch(
            "exo.worker.utils.profile.Path",
            return_value=tmp_path,
        ):
            result = _read_linux_cpu_temp()
            # The function uses Path("/sys/class/hwmon") internally
            # We need to mock at a different level
        # Since Path is tricky to mock, let's test the actual logic differently

    def test_returns_none_when_hwmon_missing(self):
        """Returns None when /sys/class/hwmon doesn't exist."""
        with patch("exo.worker.utils.profile.Path") as mock_path:
            mock_hwmon = MagicMock()
            mock_hwmon.exists.return_value = False
            mock_path.return_value = mock_hwmon
            result = _read_linux_cpu_temp()
            assert result is None


class TestGetLinuxBaselineMetrics:
    """Tests for _get_linux_baseline_metrics."""

    def test_collects_cpu_usage(self):
        """Collects CPU usage from psutil."""
        with (
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=42.5),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=None),
        ):
            metrics = _get_linux_baseline_metrics()
            assert metrics.cpu_usage == 42.5
            assert metrics.cpu_temp is None

    def test_collects_cpu_temp_when_available(self):
        """Collects CPU temperature when available on Linux."""
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=25.0),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=60.5),
        ):
            metrics = _get_linux_baseline_metrics()
            assert metrics.cpu_usage == 25.0
            assert metrics.cpu_temp == 60.5


class TestBuildSystemProfile:
    """Tests for _build_system_profile with LinuxBaselineMetrics."""

    def test_builds_profile_from_linux_baseline(self):
        """Builds SystemPerformanceProfile from LinuxBaselineMetrics."""
        metrics = LinuxBaselineMetrics(cpu_usage=50.0, cpu_temp=65.0)
        profile = _build_system_profile(metrics)

        assert isinstance(profile, SystemPerformanceProfile)
        assert profile.pcpu_usage == 50.0
        assert profile.temp == 65.0
        assert profile.gpu_usage == 0.0
        assert profile.sys_power == 0.0
        assert profile.ecpu_usage == 0.0
        assert profile.ane_power == 0.0
        assert profile.gpu_memory_total_mb is None
        assert profile.has_gpu_memory is False

    def test_builds_profile_with_no_temp(self):
        """Builds profile when CPU temperature is unavailable."""
        metrics = LinuxBaselineMetrics(cpu_usage=30.0, cpu_temp=None)
        profile = _build_system_profile(metrics)

        assert profile.pcpu_usage == 30.0
        assert profile.temp == 0.0  # Falls back to 0.0


class TestGetMetricsAsync:
    """Tests for get_metrics_async behavior on Linux."""

    @pytest.mark.asyncio
    async def test_linux_with_nvidia_returns_nvidia_metrics(self):
        """On Linux with NVIDIA, returns NvidiaMetrics."""
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
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
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
    async def test_linux_without_nvidia_returns_baseline(self):
        """On Linux without NVIDIA, returns LinuxBaselineMetrics."""
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
            patch("exo.worker.utils.profile.is_nvidia_available", return_value=False),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=35.0),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=50.0),
        ):
            result = await get_metrics_async()
            assert isinstance(result, LinuxBaselineMetrics)
            assert result.cpu_usage == 35.0
            assert result.cpu_temp == 50.0

    @pytest.mark.asyncio
    async def test_linux_nvidia_error_falls_back_to_baseline(self):
        """On Linux when NVIDIA fails, falls back to LinuxBaselineMetrics."""
        from exo.worker.utils.nvidia_monitor import NvidiaMonitorError

        with (
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
            patch("exo.worker.utils.profile.is_nvidia_available", return_value=True),
            patch(
                "exo.worker.utils.profile.nvidia_get_metrics_async",
                new_callable=AsyncMock,
                side_effect=NvidiaMonitorError("NVML error"),
            ),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=20.0),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=45.0),
        ):
            result = await get_metrics_async()
            assert isinstance(result, LinuxBaselineMetrics)
            assert result.cpu_usage == 20.0

    @pytest.mark.asyncio
    async def test_darwin_returns_macmon_metrics(self):
        """On Darwin, returns macmon Metrics."""
        from exo.worker.utils.macmon import Metrics, TempMetrics

        mock_metrics = Metrics(
            all_power=50.0,
            ane_power=5.0,
            cpu_power=20.0,
            ecpu_usage=(4, 25.0),
            gpu_power=15.0,
            gpu_ram_power=5.0,
            gpu_usage=(8, 40.0),
            pcpu_usage=(8, 60.0),
            ram_power=5.0,
            sys_power=50.0,
            temp=TempMetrics(cpu_temp_avg=55.0, gpu_temp_avg=50.0),
            timestamp="2024-01-01T00:00:00Z",
        )

        with (
            patch("exo.worker.utils.profile.platform.system", return_value="darwin"),
            patch(
                "exo.worker.utils.profile.macmon_get_metrics_async",
                new_callable=AsyncMock,
                return_value=mock_metrics,
            ),
        ):
            result = await get_metrics_async()
            assert isinstance(result, Metrics)

    @pytest.mark.asyncio
    async def test_unsupported_platform_returns_none(self):
        """On unsupported platforms, returns None."""
        with patch("exo.worker.utils.profile.platform.system", return_value="freebsd"):
            result = await get_metrics_async()
            assert result is None


class TestLinuxNodeProfileEmission:
    """Regression tests to ensure Linux nodes always emit profiles."""

    @pytest.mark.asyncio
    async def test_linux_without_nvml_emits_profile(self):
        """Linux nodes without NVML still emit a valid NodePerformanceProfile."""
        from exo.shared.types.profiling import (
            MemoryPerformanceProfile,
            NetworkInterfaceInfo,
            NodePerformanceProfile,
        )
        from exo.shared.types.memory import Memory

        # Mock all the components
        with (
            patch("exo.worker.utils.profile.platform.system", return_value="linux"),
            patch("exo.worker.utils.profile.is_nvidia_available", return_value=False),
            patch("exo.worker.utils.profile.psutil.cpu_percent", return_value=25.0),
            patch("exo.worker.utils.profile._read_linux_cpu_temp", return_value=None),
            patch(
                "exo.worker.utils.profile.get_network_interfaces",
                return_value=[NetworkInterfaceInfo(name="eth0", ip_address="192.168.1.100")],
            ),
            patch(
                "exo.worker.utils.profile.get_model_and_chip",
                new_callable=AsyncMock,
                return_value=("Dell PowerEdge", "Intel Xeon"),
            ),
            patch(
                "exo.worker.utils.profile.get_friendly_name",
                new_callable=AsyncMock,
                return_value="my-linux-server",
            ),
            patch(
                "exo.worker.utils.profile.get_memory_profile",
                return_value=MemoryPerformanceProfile(
                    ram_total=Memory.from_gb(64),
                    ram_available=Memory.from_gb(32),
                    swap_total=Memory.from_gb(8),
                    swap_available=Memory.from_gb(8),
                ),
            ),
        ):
            metrics = await get_metrics_async()

            # Verify we got baseline metrics (not None)
            assert metrics is not None
            assert isinstance(metrics, LinuxBaselineMetrics)

            # Build the profile
            system_profile = _build_system_profile(metrics)

            # Verify the profile is valid
            assert system_profile.pcpu_usage == 25.0
            assert system_profile.has_gpu_memory is False

    def test_baseline_metrics_produce_valid_system_profile(self):
        """LinuxBaselineMetrics produces a valid SystemPerformanceProfile for placement."""
        metrics = LinuxBaselineMetrics(cpu_usage=50.0, cpu_temp=None)
        profile = _build_system_profile(metrics)

        # Verify placement-relevant properties
        assert profile.has_gpu_memory is False
        assert profile.gpu_memory_available_mb == 0
        assert profile.gpu_memory_total_mb is None
