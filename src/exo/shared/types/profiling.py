from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryPerformanceProfile(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0

    # NVIDIA GPU specific fields (populated on Linux with NVIDIA GPUs)
    gpu_memory_total_mb: int | None = None
    gpu_memory_used_mb: int | None = None
    gpu_count: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    @property
    def gpu_memory_available_mb(self) -> int:
        """Calculate available GPU memory in MB."""
        if self.gpu_memory_total_mb is None or self.gpu_memory_used_mb is None:
            return 0
        return max(0, self.gpu_memory_total_mb - self.gpu_memory_used_mb)

    @property
    def has_gpu_memory(self) -> bool:
        """Check if GPU memory information is available."""
        return self.gpu_memory_total_mb is not None and self.gpu_memory_total_mb > 0


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str


class NodePerformanceProfile(CamelCaseModel):
    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = []
    system: SystemPerformanceProfile


class ConnectionProfile(CamelCaseModel):
    throughput: float
    latency: float
    jitter: float
