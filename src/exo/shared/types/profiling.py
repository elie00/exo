from collections.abc import Sequence
from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    # GPU VRAM fields for NVIDIA GPUs (populated on Linux with NVIDIA GPUs)
    gpu_vram_total: Memory | None = None
    gpu_vram_available: Memory | None = None
    gpu_vram_used: Memory | None = None

    @property
    def has_gpu_vram(self) -> bool:
        """Check if GPU VRAM information is available."""
        return self.gpu_vram_total is not None and self.gpu_vram_total.in_bytes > 0

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        gpu_vram_total: int | None = None,
        gpu_vram_available: int | None = None,
        gpu_vram_used: int | None = None,
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            gpu_vram_total=Memory.from_bytes(gpu_vram_total) if gpu_vram_total else None,
            gpu_vram_available=Memory.from_bytes(gpu_vram_available) if gpu_vram_available else None,
            gpu_vram_used=Memory.from_bytes(gpu_vram_used) if gpu_vram_used else None,
        )

    @classmethod
    def from_psutil(
        cls,
        *,
        override_memory: int | None,
        gpu_vram_total: int | None = None,
        gpu_vram_used: int | None = None,
    ) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        # Calculate available VRAM
        gpu_vram_available = None
        if gpu_vram_total is not None and gpu_vram_used is not None:
            gpu_vram_available = max(0, gpu_vram_total - gpu_vram_used)

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
            gpu_vram_total=gpu_vram_total,
            gpu_vram_available=gpu_vram_available,
            gpu_vram_used=gpu_vram_used,
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0

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


class NodeIdentity(CamelCaseModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"


class NodeNetworkInfo(CamelCaseModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(CamelCaseModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []
