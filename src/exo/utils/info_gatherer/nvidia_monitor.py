"""Optional NVML telemetry. Missing measurements are unknown, never idle GPUs."""

# ruff: noqa: N802
# Protocol method names must match NVIDIA's API.

import asyncio
import importlib
import platform
import threading
from typing import Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class NvidiaMonitorError(Exception):
    """GPU discovery or required telemetry could not be collected."""


class NvidiaGpuInfo(BaseModel):
    index: int
    name: str
    uuid: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    gpu_utilization: float | None
    memory_utilization: float | None
    temperature: float | None
    power_draw: float | None
    power_limit: float | None

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class NvidiaMetrics(BaseModel):
    # gpu_count and memory totals refer to readable devices only.
    gpu_count: int
    gpus: list[NvidiaGpuInfo]
    gpu_usage: float | None
    gpu_temp: float | None
    gpu_power: float | None
    gpu_memory_total_mb: int
    gpu_memory_used_mb: int
    gpu_memory_free_mb: int
    driver_version: str
    cuda_version: str
    detected_gpu_count: int | None = None
    unavailable_gpu_indices: list[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _MemoryInfo(Protocol):
    total: int
    used: int
    free: int


class _Utilization(Protocol):
    gpu: int
    memory: int


@runtime_checkable
class _Nvml(Protocol):
    """The subset of NVIDIA's Python API used by this optional integration."""

    NVMLError: type[Exception]
    NVML_TEMPERATURE_GPU: int

    def nvmlInit(self) -> None: ...
    def nvmlShutdown(self) -> None: ...
    def nvmlDeviceGetCount(self) -> int: ...
    def nvmlDeviceGetHandleByIndex(self, index: int) -> object: ...
    def nvmlDeviceGetName(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetUUID(self, handle: object) -> str | bytes: ...
    def nvmlDeviceGetMemoryInfo(self, handle: object) -> _MemoryInfo: ...
    def nvmlDeviceGetUtilizationRates(self, handle: object) -> _Utilization: ...
    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int: ...
    def nvmlDeviceGetPowerUsage(self, handle: object) -> int: ...
    def nvmlDeviceGetPowerManagementLimit(self, handle: object) -> int: ...
    def nvmlSystemGetDriverVersion(self) -> str | bytes: ...
    def nvmlSystemGetCudaDriverVersion_v2(self) -> int: ...


_nvml_initialized = False
_pynvml_module: _Nvml | None = None
_nvml_lock = threading.RLock()


def _get_pynvml() -> _Nvml:
    global _pynvml_module
    if _pynvml_module is None:
        try:
            module = importlib.import_module("pynvml")
        except ImportError as exc:
            raise NvidiaMonitorError(
                "NVIDIA bindings are not installed. Install the optional nvidia-ml-py package."
            ) from exc
        if not isinstance(module, _Nvml):
            raise NvidiaMonitorError(
                "Installed NVIDIA bindings do not expose the required NVML API"
            )
        _pynvml_module = module
    return _pynvml_module


def _ensure_nvml_initialized() -> None:
    global _nvml_initialized
    with _nvml_lock:
        if not _nvml_initialized:
            nvml = _get_pynvml()
            try:
                nvml.nvmlInit()
            except nvml.NVMLError as exc:
                raise NvidiaMonitorError(f"Failed to initialize NVML: {exc}") from exc
            _nvml_initialized = True


def is_nvidia_available() -> bool:
    if platform.system().lower() not in {"linux", "windows"}:
        return False
    try:
        return get_gpu_count() > 0
    except NvidiaMonitorError:
        return False


def get_gpu_count() -> int:
    with _nvml_lock:
        _ensure_nvml_initialized()
        nvml = _get_pynvml()
        try:
            return nvml.nvmlDeviceGetCount()
        except nvml.NVMLError as exc:
            raise NvidiaMonitorError(f"Could not count NVIDIA GPUs: {exc}") from exc


def _text(value: str | bytes) -> str:
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


def _get_gpu_info(index: int) -> NvidiaGpuInfo:
    _ensure_nvml_initialized()
    nvml = _get_pynvml()
    handle = nvml.nvmlDeviceGetHandleByIndex(index)
    name = _text(nvml.nvmlDeviceGetName(handle))
    uuid = _text(nvml.nvmlDeviceGetUUID(handle))
    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
    try:
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization, memory_utilization = (
            float(utilization.gpu),
            float(utilization.memory),
        )
    except nvml.NVMLError:
        gpu_utilization, memory_utilization = None, None
    try:
        temperature = float(
            nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        )
    except nvml.NVMLError:
        temperature = None
    try:
        power_draw = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except nvml.NVMLError:
        power_draw = None
    try:
        power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
    except nvml.NVMLError:
        power_limit = None
    return NvidiaGpuInfo(
        index=index,
        name=name,
        uuid=uuid,
        memory_total_mb=memory.total // (1024 * 1024),
        memory_used_mb=memory.used // (1024 * 1024),
        memory_free_mb=memory.free // (1024 * 1024),
        gpu_utilization=gpu_utilization,
        memory_utilization=memory_utilization,
        temperature=temperature,
        power_draw=power_draw,
        power_limit=power_limit,
    )


def _complete_sum(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def get_metrics() -> NvidiaMetrics:
    """Collect one serialized snapshot; shutdown cannot interrupt NVML calls."""
    with _nvml_lock:
        return _collect_metrics()


def _collect_metrics() -> NvidiaMetrics:
    _ensure_nvml_initialized()
    nvml = _get_pynvml()
    try:
        count = nvml.nvmlDeviceGetCount()
        if count == 0:
            raise NvidiaMonitorError("No NVIDIA GPUs found")
        try:
            driver = _text(nvml.nvmlSystemGetDriverVersion())
        except nvml.NVMLError:
            driver = "Unknown"
        try:
            version = nvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda = f"{version // 1000}.{(version % 1000) // 10}"
        except nvml.NVMLError:
            cuda = "Unknown"
        gpus: list[NvidiaGpuInfo] = []
        unavailable: list[int] = []
        for index in range(count):
            try:
                gpus.append(_get_gpu_info(index))
            except nvml.NVMLError:
                unavailable.append(index)
        if not gpus:
            raise NvidiaMonitorError("Could not query any NVIDIA GPUs")
        utilization = _complete_sum([gpu.gpu_utilization for gpu in gpus])
        temperature = _complete_sum([gpu.temperature for gpu in gpus])
        power = _complete_sum([gpu.power_draw for gpu in gpus])
        return NvidiaMetrics(
            gpu_count=len(gpus),
            detected_gpu_count=count,
            unavailable_gpu_indices=unavailable,
            gpus=gpus,
            gpu_usage=utilization / len(gpus)
            if utilization is not None and not unavailable
            else None,
            gpu_temp=temperature / len(gpus)
            if temperature is not None and not unavailable
            else None,
            gpu_power=power if not unavailable else None,
            gpu_memory_total_mb=sum(gpu.memory_total_mb for gpu in gpus),
            gpu_memory_used_mb=sum(gpu.memory_used_mb for gpu in gpus),
            gpu_memory_free_mb=sum(gpu.memory_free_mb for gpu in gpus),
            driver_version=driver,
            cuda_version=cuda,
        )
    except nvml.NVMLError as exc:
        raise NvidiaMonitorError(f"Error getting NVIDIA metrics: {exc}") from exc


async def get_metrics_async() -> NvidiaMetrics:
    """Keep blocking driver calls off the event loop."""
    return await asyncio.to_thread(get_metrics)


def shutdown() -> None:
    global _nvml_initialized
    with _nvml_lock:
        if _nvml_initialized:
            nvml = _get_pynvml()
            try:
                nvml.nvmlShutdown()
            except nvml.NVMLError:
                logger.warning(
                    "NVML shutdown failed; clearing local initialization state"
                )
            finally:
                _nvml_initialized = False
