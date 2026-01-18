from typing import Literal, Self

import psutil

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


GpuLinkType = Literal[
    "nvlink",
    "pcie",
    "pcie_switch",
    "system",
    "single_gpu",
    "unknown",
]
"""
Type of interconnect between two GPUs or within a single GPU.

- nvlink: Direct NVLink connection (high bandwidth, low latency)
- pcie: Direct PCIe peer-to-peer connection
- pcie_switch: Connection via PCIe switch
- system: Connection via CPU/system memory (slowest)
- single_gpu: Same GPU (no transfer needed)
- unknown: Could not determine link type
"""


class GpuDeviceInfo(CamelCaseModel):
    """Detailed information about a single GPU device."""

    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization: float  # percentage 0-100
    temperature: float  # Celsius
    power_draw: float  # Watts
    power_limit: float  # Watts
    pcie_gen: int | None = None  # PCIe generation (3, 4, 5, etc.)
    pcie_width: int | None = None  # PCIe link width (x16, x8, etc.)

    @property
    def memory_available_mb(self) -> int:
        """Calculate available GPU memory in MB."""
        return max(0, self.memory_total_mb - self.memory_used_mb)


class GpuTopologyLink(CamelCaseModel):
    """Represents a connection between two GPUs."""

    gpu_index_a: int
    gpu_index_b: int
    link_type: GpuLinkType
    nvlink_count: int = 0  # Number of NVLink connections (0 if not NVLink)
    p2p_supported: bool = False  # Whether peer-to-peer access is supported
    bandwidth_gbps: float | None = None  # Estimated bandwidth in GB/s


class GpuTopology(CamelCaseModel):
    """GPU topology information for a node with multiple GPUs."""

    gpu_count: int
    links: list[GpuTopologyLink] = []

    def get_link(self, gpu_a: int, gpu_b: int) -> GpuTopologyLink | None:
        """Get the link between two GPUs, or None if not found."""
        for link in self.links:
            if (link.gpu_index_a == gpu_a and link.gpu_index_b == gpu_b) or (
                link.gpu_index_a == gpu_b and link.gpu_index_b == gpu_a
            ):
                return link
        return None

    def has_nvlink(self) -> bool:
        """Check if any NVLink connections exist."""
        return any(link.link_type == "nvlink" for link in self.links)

    def all_p2p_supported(self) -> bool:
        """Check if all GPU pairs support P2P access."""
        return all(link.p2p_supported for link in self.links)


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
    """System performance profile including CPU and GPU metrics."""

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0

    # NVIDIA GPU aggregate fields (populated on Linux with NVIDIA GPUs)
    gpu_memory_total_mb: int | None = None
    gpu_memory_used_mb: int | None = None
    gpu_count: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    # Per-GPU details (populated when multiple GPUs are present)
    gpus: list[GpuDeviceInfo] = []
    gpu_topology: GpuTopology | None = None

    @property
    def gpu_memory_available_mb(self) -> int:
        """Calculate total available GPU memory in MB (aggregate across all GPUs)."""
        if self.gpu_memory_total_mb is None or self.gpu_memory_used_mb is None:
            return 0
        return max(0, self.gpu_memory_total_mb - self.gpu_memory_used_mb)

    @property
    def has_gpu_memory(self) -> bool:
        """Check if GPU memory information is available."""
        return self.gpu_memory_total_mb is not None and self.gpu_memory_total_mb > 0

    @property
    def is_multi_gpu(self) -> bool:
        """Check if this node has multiple GPUs."""
        return self.gpu_count is not None and self.gpu_count > 1

    def get_gpu(self, index: int) -> GpuDeviceInfo | None:
        """Get info for a specific GPU by index."""
        for gpu in self.gpus:
            if gpu.index == index:
                return gpu
        return None


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
