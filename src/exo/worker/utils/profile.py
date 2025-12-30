import asyncio
import os
import platform
from typing import Any, Callable, Coroutine, Union

import anyio
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
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
    NvidiaMetrics,
    NvidiaMonitorError,
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

# Union type for all possible metrics types
AnyMetrics = Union[Metrics, NvidiaMetrics]


async def get_metrics_async() -> AnyMetrics | None:
    """Return detailed Metrics on macOS or NVIDIA metrics on Linux, or None if unavailable."""
    system = platform.system().lower()

    if system == "darwin":
        return await macmon_get_metrics_async()
    elif system == "linux":
        # Try NVIDIA GPU monitoring on Linux
        if is_nvidia_available():
            try:
                return await nvidia_get_metrics_async()
            except NvidiaMonitorError as e:
                logger.debug(f"NVIDIA monitoring failed: {e}")
                return None
        return None
    else:
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


def _build_system_profile(metrics: AnyMetrics) -> SystemPerformanceProfile:
    """Build a SystemPerformanceProfile from either macOS or NVIDIA metrics."""
    if isinstance(metrics, NvidiaMetrics):
        # NVIDIA GPU metrics (Linux)
        return SystemPerformanceProfile(
            gpu_usage=metrics.gpu_usage,
            temp=metrics.gpu_temp,
            sys_power=metrics.gpu_power,
            pcpu_usage=0.0,  # Not available from NVML
            ecpu_usage=0.0,  # Not applicable (Apple-specific)
            ane_power=0.0,  # Not applicable (Apple-specific)
            # Extended NVIDIA fields
            gpu_memory_total_mb=metrics.gpu_memory_total_mb,
            gpu_memory_used_mb=metrics.gpu_memory_used_mb,
            gpu_count=metrics.gpu_count,
            driver_version=metrics.driver_version,
            cuda_version=metrics.cuda_version,
        )
    else:
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
