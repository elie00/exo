"""Typed NVML doubles validate the collector without claiming GPU qualification."""

# ruff: noqa: N802
# Double methods intentionally match NVIDIA's API.

import asyncio
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import ModuleType

import pytest

from exo.utils.info_gatherer import nvidia_monitor as monitor
from exo.utils.info_gatherer.nvidia_monitor import (
    NvidiaMetrics,
    NvidiaMonitorError,
    get_gpu_count,
    get_metrics,
    get_metrics_async,
    is_nvidia_available,
    shutdown,
)


class NvmlError(Exception):
    pass


@dataclass
class MemoryInfo:
    total: int = 24576 * 1024**2
    used: int = 2048 * 1024**2
    free: int = 22528 * 1024**2


@dataclass
class Utilization:
    gpu: int = 50
    memory: int = 8


@dataclass
class FakeNvml:
    NVMLError = NvmlError
    NVML_TEMPERATURE_GPU = 0
    count: int = 2
    init_calls: int = 0
    shutdown_calls: int = 0
    init_failures: int = 0
    init_delay: float = 0
    shutdown_failure: bool = False
    device_failures: set[int] = field(default_factory=set)
    missing_sensors: set[str] = field(default_factory=set)
    power_failures: set[int] = field(default_factory=set)
    count_threads: list[int] = field(default_factory=list)

    def nvmlInit(self) -> None:
        self.init_calls += 1
        time.sleep(self.init_delay)
        if self.init_failures:
            self.init_failures -= 1
            raise NvmlError("driver unavailable")

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_failure:
            raise NvmlError("driver gone")

    def nvmlDeviceGetCount(self) -> int:
        self.count_threads.append(threading.get_ident())
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        if index in self.device_failures:
            raise NvmlError("GPU lost")
        return index

    def nvmlDeviceGetName(self, handle: object) -> bytes:
        return b"NVIDIA test"

    def nvmlDeviceGetUUID(self, handle: object) -> str:
        return f"GPU-{handle}"

    def nvmlDeviceGetMemoryInfo(self, handle: object) -> MemoryInfo:
        return MemoryInfo()

    def nvmlDeviceGetUtilizationRates(self, handle: object) -> Utilization:
        if "utilization" in self.missing_sensors:
            raise NvmlError("not supported")
        return Utilization()

    def nvmlDeviceGetTemperature(self, handle: object, sensor: int) -> int:
        if "temperature" in self.missing_sensors:
            raise NvmlError("not supported")
        return 55

    def nvmlDeviceGetPowerUsage(self, handle: object) -> int:
        if "power" in self.missing_sensors or handle in self.power_failures:
            raise NvmlError("not supported")
        return 200000

    def nvmlDeviceGetPowerManagementLimit(self, handle: object) -> int:
        if "power_limit" in self.missing_sensors:
            raise NvmlError("not supported")
        return 450000

    def nvmlSystemGetDriverVersion(self) -> bytes:
        if "driver_version" in self.missing_sensors:
            raise NvmlError("not supported")
        return b"535.154.05"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        if "cuda_version" in self.missing_sensors:
            raise NvmlError("not supported")
        return 12020


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> FakeNvml:
    fake = FakeNvml()
    monkeypatch.setattr(monitor, "_pynvml_module", fake)
    monkeypatch.setattr(monitor, "_nvml_initialized", False)
    return fake


def test_complete_snapshot_converts_units_and_decodes_names(nvml: FakeNvml) -> None:
    metrics = get_metrics()
    assert metrics.gpu_count == metrics.detected_gpu_count == nvml.count
    assert metrics.gpu_memory_total_mb == 49152
    assert metrics.gpu_memory_used_mb == 4096 and metrics.gpu_memory_free_mb == 45056
    assert metrics.gpu_usage == 50.0 and metrics.gpu_temp == 55.0
    assert metrics.gpu_power == 400.0
    assert metrics.gpus[0].power_limit == 450.0
    assert metrics.gpus[0].name == "NVIDIA test" and metrics.gpus[1].uuid == "GPU-1"
    assert metrics.driver_version == "535.154.05" and metrics.cuda_version == "12.2"
    assert metrics.unavailable_gpu_indices == []


@pytest.mark.parametrize(
    "sensor", ["utilization", "temperature", "power", "power_limit"]
)
def test_unsupported_sensors_remain_unknown(nvml: FakeNvml, sensor: str) -> None:
    nvml.missing_sensors.add(sensor)
    metrics = get_metrics()
    assert metrics.gpu_memory_total_mb == 49152
    if sensor == "utilization":
        assert metrics.gpu_usage is None and metrics.gpus[0].memory_utilization is None
    elif sensor == "temperature":
        assert metrics.gpu_temp is None
    elif sensor == "power":
        assert metrics.gpu_power is None and metrics.gpus[0].power_draw is None
        assert '"gpu_power":null' in metrics.model_dump_json()
    else:
        assert metrics.gpus[0].power_limit is None


def test_one_missing_sensor_cannot_understate_total_power(nvml: FakeNvml) -> None:
    nvml.power_failures.add(1)
    metrics = get_metrics()
    assert metrics.gpus[0].power_draw == 200.0 and metrics.gpus[1].power_draw is None
    assert metrics.gpu_power is None and metrics.gpu_usage == 50.0


def test_partial_failure_preserves_physical_indices(nvml: FakeNvml) -> None:
    nvml.device_failures.add(0)
    metrics = get_metrics()
    assert metrics.detected_gpu_count == 2 and metrics.gpu_count == 1
    assert metrics.unavailable_gpu_indices == [0] and metrics.gpus[0].index == 1
    assert metrics.gpu_memory_total_mb == 24576
    assert (
        metrics.gpu_usage is None
        and metrics.gpu_power is None
        and metrics.gpu_temp is None
    )


def test_all_device_failures_raise(nvml: FakeNvml) -> None:
    nvml.device_failures = {0, 1}
    with pytest.raises(NvidiaMonitorError, match="Could not query any"):
        get_metrics()


def test_no_gpus_raise(nvml: FakeNvml) -> None:
    nvml.count = 0
    assert get_gpu_count() == 0
    with pytest.raises(NvidiaMonitorError, match="No NVIDIA GPUs"):
        get_metrics()


@pytest.mark.parametrize(
    ("system", "available"), [("Darwin", False), ("Linux", True), ("Windows", True)]
)
def test_availability_is_platform_aware(
    nvml: FakeNvml, monkeypatch: pytest.MonkeyPatch, system: str, available: bool
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    assert is_nvidia_available() is available
    assert nvml.init_calls == int(available)


def test_availability_handles_initialization_failure(
    nvml: FakeNvml, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    nvml.init_failures = 1
    assert is_nvidia_available() is False
    assert is_nvidia_available() is True


def test_unknown_versions_do_not_discard_device_memory(nvml: FakeNvml) -> None:
    nvml.missing_sensors = {"driver_version", "cuda_version"}
    metrics = get_metrics()
    assert metrics.driver_version == metrics.cuda_version == "Unknown"
    assert metrics.gpu_memory_total_mb == 49152


def test_concurrent_polls_initialize_once(nvml: FakeNvml) -> None:
    nvml.init_delay = 0.01

    def poll(_: int) -> NvidiaMetrics:
        return get_metrics()

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(poll, range(16)))
    assert all(snapshot.gpu_count == 2 for snapshot in snapshots)
    assert nvml.init_calls == 1


def test_initialization_failure_can_be_retried(nvml: FakeNvml) -> None:
    nvml.init_failures = 1
    with pytest.raises(NvidiaMonitorError, match="initialize"):
        get_metrics()
    assert get_metrics().gpu_count == 2 and nvml.init_calls == 2


@pytest.mark.parametrize("failure", [False, True])
def test_shutdown_resets_state_even_on_failure(nvml: FakeNvml, failure: bool) -> None:
    shutdown()
    assert nvml.shutdown_calls == 0
    get_metrics()
    nvml.shutdown_failure = failure
    shutdown()
    shutdown()
    assert nvml.shutdown_calls == 1
    assert get_metrics().gpu_count == 2 and nvml.init_calls == 2


def test_async_poll_keeps_driver_calls_off_event_loop(nvml: FakeNvml) -> None:
    assert asyncio.run(get_metrics_async()).gpu_count == 2
    assert nvml.count_threads and all(
        ident != threading.get_ident() for ident in nvml.count_threads
    )


def test_missing_binding_reports_install_hint(
    nvml: FakeNvml, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor, "_pynvml_module", None)

    def missing(_name: str) -> ModuleType:
        raise ImportError("test fixture")

    monkeypatch.setattr(monitor.importlib, "import_module", missing)
    with pytest.raises(NvidiaMonitorError, match="nvidia-ml-py"):
        get_metrics()


def test_invalid_binding_reports_contract_failure(
    nvml: FakeNvml, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor, "_pynvml_module", None)

    def invalid(_name: str) -> ModuleType:
        return ModuleType("invalid")

    monkeypatch.setattr(monitor.importlib, "import_module", invalid)
    with pytest.raises(NvidiaMonitorError, match="required NVML API"):
        get_metrics()
