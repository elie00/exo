import asyncio
import os
import platform
from pathlib import Path
from typing import Any, Callable, Coroutine, Union

import anyio
import psutil
from loguru import logger
from pydantic import BaseModel, ConfigDict

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    GpuDeviceInfo,
    GpuLinkType,
    GpuTopology,
    GpuTopologyLink,
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .nvidia_monitor import (
    NvidiaGpuInfo,
    NvidiaGpuLink,
    NvidiaMetrics,
    NvidiaMonitorError,
    NvidiaTopology,
    is_nvidia_available,
)
from .nvidia_monitor import (
    get_metrics_async as nvidia_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


class BaselineMetrics(BaseModel):
    """Baseline metrics for systems without dedicated GPU monitoring (Linux/Windows without NVIDIA)."""

    cpu_usage: float  # percentage 0-100
    cpu_temp: float | None  # Celsius, None if unavailable

    model_config = ConfigDict(frozen=True)


# Alias for backward compatibility
LinuxBaselineMetrics = BaselineMetrics


def _read_linux_cpu_temp() -> float | None:
    """
    Read CPU temperature from Linux hwmon.

    Returns the temperature in Celsius, or None if unavailable.
    """
    hwmon_base = Path("/sys/class/hwmon")
    if not hwmon_base.exists():
        return None

    try:
        for hwmon_dir in hwmon_base.iterdir():
            name_file = hwmon_dir / "name"
            if name_file.exists():
                name = name_file.read_text().strip()
                if name in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                    temp_file = hwmon_dir / "temp1_input"
                    if temp_file.exists():
                        millidegrees = int(temp_file.read_text().strip())
                        return millidegrees / 1000.0
    except (OSError, ValueError):
        pass

    return None


def _get_baseline_metrics() -> BaselineMetrics:
    """Collect baseline metrics (CPU usage, optional temp) for Linux/Windows."""
    cpu_usage = psutil.cpu_percent(interval=None)
    # Temperature reading only works on Linux via hwmon
    cpu_temp = _read_linux_cpu_temp() if platform.system().lower() == "linux" else None
    return BaselineMetrics(cpu_usage=cpu_usage, cpu_temp=cpu_temp)


# Alias for backward compatibility
_get_linux_baseline_metrics = _get_baseline_metrics


# Union type for all possible metrics types
AnyMetrics = Union[Metrics, NvidiaMetrics, BaselineMetrics]


async def get_metrics_async() -> AnyMetrics | None:
    """
    Return platform-specific metrics.

    On macOS, returns macmon Metrics.
    On Linux/Windows, returns NvidiaMetrics if available, otherwise BaselineMetrics.
    Returns None only on unsupported platforms.
    """
    system = platform.system().lower()

    if system == "darwin":
        return await macmon_get_metrics_async()

    if system in ("linux", "windows"):
        if is_nvidia_available():
            try:
                return await nvidia_get_metrics_async()
            except NvidiaMonitorError as e:
                logger.debug(f"NVIDIA monitoring failed, falling back to baseline: {e}")
        return _get_baseline_metrics()

    return None


def get_memory_profile() -> MemoryPerformanceProfile:
    """Construct a MemoryPerformanceProfile using psutil"""
    override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
    override_memory: int | None = (
        Memory.from_mb(int(override_memory_env)).in_bytes
        if override_memory_env
        else None
    )

    return MemoryPerformanceProfile.from_psutil(override_memory=override_memory)


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
) -> None:
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls
    """
    while True:
        try:
            mem = get_memory_profile()
            await callback(mem)
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


def _convert_nvidia_gpu_info(gpu: NvidiaGpuInfo) -> GpuDeviceInfo:
    """Convert NvidiaGpuInfo to GpuDeviceInfo for the profiling types."""
    return GpuDeviceInfo(
        index=gpu.index,
        uuid=gpu.uuid,
        name=gpu.name,
        memory_total_mb=gpu.memory_total_mb,
        memory_used_mb=gpu.memory_used_mb,
        memory_free_mb=gpu.memory_free_mb,
        utilization=gpu.gpu_utilization,
        temperature=gpu.temperature,
        power_draw=gpu.power_draw,
        power_limit=gpu.power_limit,
        pcie_gen=gpu.pcie_gen,
        pcie_width=gpu.pcie_width,
    )


def _convert_nvidia_link(link: NvidiaGpuLink) -> GpuTopologyLink:
    """Convert NvidiaGpuLink to GpuTopologyLink for the profiling types."""
    link_type: GpuLinkType = "unknown"
    if link.link_type == "nvlink":
        link_type = "nvlink"
    elif link.link_type == "pcie":
        link_type = "pcie"
    elif link.link_type == "pcie_switch":
        link_type = "pcie_switch"
    elif link.link_type == "system":
        link_type = "system"

    return GpuTopologyLink(
        gpu_index_a=link.gpu_index_a,
        gpu_index_b=link.gpu_index_b,
        link_type=link_type,
        nvlink_count=link.nvlink_count,
        p2p_supported=link.p2p_supported,
        bandwidth_gbps=None,  # Not directly available from NVML
    )


def _convert_nvidia_topology(topology: NvidiaTopology) -> GpuTopology:
    """Convert NvidiaTopology to GpuTopology for the profiling types."""
    return GpuTopology(
        gpu_count=topology.gpu_count,
        links=[_convert_nvidia_link(link) for link in topology.links],
    )


def _build_system_profile(metrics: AnyMetrics) -> SystemPerformanceProfile:
    """Build a SystemPerformanceProfile from macOS, NVIDIA, or Linux baseline metrics."""
    if isinstance(metrics, NvidiaMetrics):
        # Convert per-GPU info
        gpus = [_convert_nvidia_gpu_info(gpu) for gpu in metrics.gpus]

        # Convert topology if available
        gpu_topology: GpuTopology | None = None
        if metrics.topology is not None:
            gpu_topology = _convert_nvidia_topology(metrics.topology)

        return SystemPerformanceProfile(
            gpu_usage=metrics.gpu_usage,
            temp=metrics.gpu_temp,
            sys_power=metrics.gpu_power,
            pcpu_usage=0.0,
            ecpu_usage=0.0,
            ane_power=0.0,
            gpu_memory_total_mb=metrics.gpu_memory_total_mb,
            gpu_memory_used_mb=metrics.gpu_memory_used_mb,
            gpu_count=metrics.gpu_count,
            driver_version=metrics.driver_version,
            cuda_version=metrics.cuda_version,
            gpus=gpus,
            gpu_topology=gpu_topology,
        )

    if isinstance(metrics, BaselineMetrics):
        return SystemPerformanceProfile(
            gpu_usage=0.0,
            temp=metrics.cpu_temp if metrics.cpu_temp is not None else 0.0,
            sys_power=0.0,
            pcpu_usage=metrics.cpu_usage,
            ecpu_usage=0.0,
            ane_power=0.0,
        )

    # macOS/macmon metrics
    return SystemPerformanceProfile(
        gpu_usage=metrics.gpu_usage[1],
        temp=metrics.temp.gpu_temp_avg,
        sys_power=metrics.sys_power,
        pcpu_usage=metrics.pcpu_usage[1],
        ecpu_usage=metrics.ecpu_usage[1],
        ane_power=metrics.ane_power,
    )


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    poll_interval_s = 1.0
    while True:
        try:
            metrics = await get_metrics_async()
            if metrics is None:
                return

            network_interfaces = get_network_interfaces()
            # these awaits could be joined but realistically they should be cached
            model_id, chip_id = await get_model_and_chip()
            friendly_name = await get_friendly_name()

            # do the memory profile last to get a fresh reading to not conflict with the other memory profiling loop
            memory_profile = get_memory_profile()

            system_profile = _build_system_profile(metrics)

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=system_profile,
                )
            )

        except asyncio.TimeoutError:
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except (MacMonError, NvidiaMonitorError) as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)
