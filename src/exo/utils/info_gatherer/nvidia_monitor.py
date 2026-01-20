"""
NVIDIA GPU monitoring module for Linux systems with NVIDIA GPUs.

This module provides GPU metrics collection similar to macmon.py but for NVIDIA GPUs
using the pynvml library (NVIDIA Management Library Python bindings).
"""

import platform
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pynvml


class NvidiaMonitorError(Exception):
    """Exception raised for errors in the NVIDIA Monitor functions."""


class NvidiaGpuInfo(BaseModel):
    """Information about a single NVIDIA GPU."""

    index: int
    name: str
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    gpu_utilization: float  # percentage 0-100
    memory_utilization: float  # percentage 0-100
    temperature: float  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts

    model_config = ConfigDict(extra="ignore")


class NvidiaMetrics(BaseModel):
    """Complete set of metrics for all NVIDIA GPUs in the system."""

    gpu_count: int
    gpus: list[NvidiaGpuInfo]
    # Aggregate metrics (from primary GPU or average)
    gpu_usage: float  # percentage 0-100
    gpu_temp: float  # Celsius
    gpu_power: float  # Watts
    gpu_memory_total_mb: int
    gpu_memory_used_mb: int
    gpu_memory_free_mb: int
    driver_version: str
    cuda_version: str

    model_config = ConfigDict(extra="ignore")


_nvml_initialized: bool = False
_pynvml_module: "pynvml | None" = None


def _get_pynvml() -> "pynvml":
    """Lazily import and return the pynvml module."""
    global _pynvml_module
    if _pynvml_module is None:
        try:
            import pynvml as _pynvml

            _pynvml_module = _pynvml
        except ImportError as e:
            raise NvidiaMonitorError(
                "pynvml is not installed. Install it with: pip install pynvml"
            ) from e
    return _pynvml_module


def _ensure_nvml_initialized() -> None:
    """Initialize NVML if not already initialized."""
    global _nvml_initialized
    if not _nvml_initialized:
        pynvml = _get_pynvml()
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True

            # Log GPU discovery info
            device_count = pynvml.nvmlDeviceGetCount()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")

            logger.info(
                f"NVIDIA NVML initialized: {device_count} GPU(s) detected, driver {driver_version}"
            )

            # Log individual GPU info
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_gb = memory_info.total / (1024**3)
                    logger.info(f"  GPU {i}: {name} ({memory_gb:.1f} GB VRAM)")
                except pynvml.NVMLError as e:
                    logger.warning(f"  GPU {i}: Could not query info: {e}")

        except pynvml.NVMLError as e:
            raise NvidiaMonitorError(f"Failed to initialize NVML: {e}") from e


def is_nvidia_available() -> bool:
    """Check if NVIDIA GPU monitoring is available on this system."""
    # Only support Linux for now (could extend to Windows)
    system = platform.system().lower()
    if system not in ("linux", "windows"):
        logger.debug(
            f"NVIDIA monitoring not available on {system} (only Linux/Windows supported)"
        )
        return False

    try:
        _ensure_nvml_initialized()
        pynvml = _get_pynvml()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            logger.debug(f"NVIDIA monitoring available: {device_count} GPU(s)")
            return True
        else:
            logger.debug("NVIDIA driver loaded but no GPUs found")
            return False
    except NvidiaMonitorError as e:
        logger.debug(f"NVIDIA monitoring not available: {e}")
        return False
    except Exception as e:
        logger.debug(f"NVIDIA monitoring check failed: {e}")
        return False


def get_gpu_count() -> int:
    """Get the number of NVIDIA GPUs in the system."""
    _ensure_nvml_initialized()
    pynvml = _get_pynvml()
    return pynvml.nvmlDeviceGetCount()


def _get_gpu_info(index: int) -> NvidiaGpuInfo:
    """Get detailed information about a specific GPU."""
    _ensure_nvml_initialized()
    pynvml = _get_pynvml()

    handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    # Basic info
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("utf-8")

    # Memory info
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_total_mb = memory_info.total // (1024 * 1024)
    memory_used_mb = memory_info.used // (1024 * 1024)
    memory_free_mb = memory_info.free // (1024 * 1024)

    # Utilization
    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = float(utilization.gpu)
        memory_utilization = float(utilization.memory)
    except pynvml.NVMLError:
        gpu_utilization = 0.0
        memory_utilization = 0.0

    # Temperature
    try:
        temperature = float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
    except pynvml.NVMLError:
        temperature = 0.0

    # Power
    try:
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
    except pynvml.NVMLError:
        power_draw = 0.0

    try:
        power_limit = (
            pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        )  # mW to W
    except pynvml.NVMLError:
        power_limit = 0.0

    return NvidiaGpuInfo(
        index=index,
        name=name,
        uuid=uuid,
        memory_total_mb=memory_total_mb,
        memory_used_mb=memory_used_mb,
        memory_free_mb=memory_free_mb,
        gpu_utilization=gpu_utilization,
        memory_utilization=memory_utilization,
        temperature=temperature,
        power_draw=power_draw,
        power_limit=power_limit,
    )


def get_metrics() -> NvidiaMetrics:
    """
    Get comprehensive metrics for all NVIDIA GPUs in the system.

    Returns:
        NvidiaMetrics: Complete metrics including per-GPU details and aggregates.

    Raises:
        NvidiaMonitorError: If there's an error accessing GPU metrics.
    """
    _ensure_nvml_initialized()
    pynvml = _get_pynvml()

    try:
        gpu_count = pynvml.nvmlDeviceGetCount()

        if gpu_count == 0:
            raise NvidiaMonitorError("No NVIDIA GPUs found")

        # Get driver and CUDA versions
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        except pynvml.NVMLError:
            driver_version = "Unknown"

        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_major = cuda_version // 1000
            cuda_minor = (cuda_version % 1000) // 10
            cuda_version_str = f"{cuda_major}.{cuda_minor}"
        except pynvml.NVMLError:
            cuda_version_str = "Unknown"

        # Collect info for all GPUs
        gpus: list[NvidiaGpuInfo] = []
        for i in range(gpu_count):
            try:
                gpu_info = _get_gpu_info(i)
                gpus.append(gpu_info)
            except pynvml.NVMLError as e:
                # Skip GPUs that can't be queried
                continue

        if not gpus:
            raise NvidiaMonitorError("Could not query any NVIDIA GPUs")

        # Calculate aggregates from primary GPU (index 0)
        primary_gpu = gpus[0]

        # Calculate totals for memory
        total_memory = sum(g.memory_total_mb for g in gpus)
        used_memory = sum(g.memory_used_mb for g in gpus)
        free_memory = sum(g.memory_free_mb for g in gpus)

        # Average utilization and temperature across all GPUs
        avg_utilization = sum(g.gpu_utilization for g in gpus) / len(gpus)
        avg_temp = sum(g.temperature for g in gpus) / len(gpus)
        total_power = sum(g.power_draw for g in gpus)

        return NvidiaMetrics(
            gpu_count=len(gpus),
            gpus=gpus,
            gpu_usage=avg_utilization,
            gpu_temp=avg_temp,
            gpu_power=total_power,
            gpu_memory_total_mb=total_memory,
            gpu_memory_used_mb=used_memory,
            gpu_memory_free_mb=free_memory,
            driver_version=driver_version,
            cuda_version=cuda_version_str,
        )

    except pynvml.NVMLError as e:
        raise NvidiaMonitorError(f"Error getting NVIDIA metrics: {e}") from e


async def get_metrics_async() -> NvidiaMetrics:
    """
    Asynchronously get NVIDIA GPU metrics.

    This is a wrapper around get_metrics() for consistency with macmon API.
    NVML calls are generally fast enough to not require true async execution,
    but we could move this to a thread pool if needed.
    """
    return get_metrics()


def shutdown() -> None:
    """Shutdown NVML. Should be called when the application exits."""
    global _nvml_initialized
    if _nvml_initialized:
        try:
            pynvml = _get_pynvml()
            pynvml.nvmlShutdown()
            _nvml_initialized = False
        except Exception:
            pass
