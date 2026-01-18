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
    pcie_gen: int | None = None  # PCIe generation (3, 4, 5)
    pcie_width: int | None = None  # PCIe link width (1, 2, 4, 8, 16)

    model_config = ConfigDict(extra="ignore")


class NvidiaGpuLink(BaseModel):
    """Information about a link between two GPUs."""

    gpu_index_a: int
    gpu_index_b: int
    link_type: str  # "nvlink", "pcie", "pcie_switch", "system", "unknown"
    nvlink_count: int = 0  # Number of NVLink bridges (0 if not NVLink)
    p2p_read_supported: bool = False
    p2p_write_supported: bool = False
    p2p_atomics_supported: bool = False

    model_config = ConfigDict(extra="ignore")

    @property
    def p2p_supported(self) -> bool:
        """Check if any P2P access is supported."""
        return self.p2p_read_supported or self.p2p_write_supported


class NvidiaTopology(BaseModel):
    """GPU topology information for multi-GPU systems."""

    gpu_count: int
    links: list[NvidiaGpuLink] = []

    model_config = ConfigDict(extra="ignore")

    def get_link(self, gpu_a: int, gpu_b: int) -> NvidiaGpuLink | None:
        """Get the link between two GPUs."""
        for link in self.links:
            if (link.gpu_index_a == gpu_a and link.gpu_index_b == gpu_b) or (
                link.gpu_index_a == gpu_b and link.gpu_index_b == gpu_a
            ):
                return link
        return None

    def has_nvlink(self) -> bool:
        """Check if any NVLink connections exist."""
        return any(link.link_type == "nvlink" for link in self.links)


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
    # Topology information (populated for multi-GPU systems)
    topology: NvidiaTopology | None = None

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

    # PCIe info
    pcie_gen: int | None = None
    pcie_width: int | None = None
    try:
        pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
    except (pynvml.NVMLError, AttributeError):
        pass
    try:
        pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    except (pynvml.NVMLError, AttributeError):
        pass

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
        pcie_gen=pcie_gen,
        pcie_width=pcie_width,
    )


def _get_gpu_link(index_a: int, index_b: int) -> NvidiaGpuLink:
    """Get link information between two GPUs."""
    _ensure_nvml_initialized()
    pynvml = _get_pynvml()

    handle_a = pynvml.nvmlDeviceGetHandleByIndex(index_a)
    handle_b = pynvml.nvmlDeviceGetHandleByIndex(index_b)

    # Determine link type using topology common ancestor
    link_type = "unknown"
    try:
        # NVML topology levels (from closest to farthest):
        # NVML_TOPOLOGY_INTERNAL = 0 (same GPU)
        # NVML_TOPOLOGY_SINGLE = 10 (NVLink)
        # NVML_TOPOLOGY_MULTIPLE = 20 (multiple NVLinks)
        # NVML_TOPOLOGY_HOSTBRIDGE = 30 (same PCIe host bridge)
        # NVML_TOPOLOGY_NODE = 40 (same NUMA node)
        # NVML_TOPOLOGY_SYSTEM = 50 (different NUMA nodes)
        topo_level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_a, handle_b)

        # Map topology level to link type
        if topo_level <= 20:  # SINGLE or MULTIPLE (NVLink)
            link_type = "nvlink"
        elif topo_level <= 30:  # HOSTBRIDGE (same PCIe switch/bridge)
            link_type = "pcie"
        elif topo_level <= 40:  # NODE (same NUMA, different PCIe)
            link_type = "pcie_switch"
        else:  # SYSTEM (different NUMA nodes)
            link_type = "system"
    except (pynvml.NVMLError, AttributeError):
        link_type = "unknown"

    # Count NVLink bridges
    nvlink_count = 0
    if link_type == "nvlink":
        try:
            # Check each possible NVLink (up to 18 for A100/H100)
            for link_idx in range(18):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle_a, link_idx)
                    if state:
                        # Check if this NVLink connects to the target GPU
                        try:
                            remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(
                                handle_a, link_idx
                            )
                            target_pci = pynvml.nvmlDeviceGetPciInfo_v3(handle_b)
                            if remote_pci.busId == target_pci.busId:
                                nvlink_count += 1
                        except (pynvml.NVMLError, AttributeError):
                            # If we can't verify, still count the active link
                            nvlink_count += 1
                except pynvml.NVMLError:
                    break  # No more NVLinks
        except (pynvml.NVMLError, AttributeError):
            pass

    # Check P2P capabilities
    p2p_read = False
    p2p_write = False
    p2p_atomics = False
    try:
        # NVML_P2P_CAPS_INDEX_READ = 0
        # NVML_P2P_CAPS_INDEX_WRITE = 1
        # NVML_P2P_CAPS_INDEX_NVLINK = 2
        # NVML_P2P_CAPS_INDEX_ATOMICS = 3
        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
            handle_a, handle_b, pynvml.NVML_P2P_CAPS_INDEX_READ
        )
        p2p_read = p2p_status == pynvml.NVML_P2P_STATUS_OK
    except (pynvml.NVMLError, AttributeError):
        pass

    try:
        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
            handle_a, handle_b, pynvml.NVML_P2P_CAPS_INDEX_WRITE
        )
        p2p_write = p2p_status == pynvml.NVML_P2P_STATUS_OK
    except (pynvml.NVMLError, AttributeError):
        pass

    try:
        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
            handle_a, handle_b, pynvml.NVML_P2P_CAPS_INDEX_ATOMICS
        )
        p2p_atomics = p2p_status == pynvml.NVML_P2P_STATUS_OK
    except (pynvml.NVMLError, AttributeError):
        pass

    return NvidiaGpuLink(
        gpu_index_a=index_a,
        gpu_index_b=index_b,
        link_type=link_type,
        nvlink_count=nvlink_count,
        p2p_read_supported=p2p_read,
        p2p_write_supported=p2p_write,
        p2p_atomics_supported=p2p_atomics,
    )


def get_topology() -> NvidiaTopology | None:
    """
    Get GPU topology information for multi-GPU systems.

    Returns None for single-GPU systems.
    """
    _ensure_nvml_initialized()
    pynvml = _get_pynvml()

    gpu_count = pynvml.nvmlDeviceGetCount()
    if gpu_count < 2:
        return None

    links: list[NvidiaGpuLink] = []
    for i in range(gpu_count):
        for j in range(i + 1, gpu_count):
            try:
                link = _get_gpu_link(i, j)
                links.append(link)
            except pynvml.NVMLError as e:
                logger.debug(f"Could not get link info for GPU {i} <-> {j}: {e}")

    topology = NvidiaTopology(gpu_count=gpu_count, links=links)

    # Log topology info
    if topology.has_nvlink():
        nvlink_links = [link for link in links if link.link_type == "nvlink"]
        logger.info(
            f"GPU topology: {gpu_count} GPUs with {len(nvlink_links)} NVLink connection(s)"
        )
    else:
        logger.info(f"GPU topology: {gpu_count} GPUs connected via PCIe")

    return topology


def get_metrics() -> NvidiaMetrics:
    """
    Get comprehensive metrics for all NVIDIA GPUs in the system.

    Returns:
        NvidiaMetrics: Complete metrics including per-GPU details, aggregates, and topology.

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
            except pynvml.NVMLError:
                continue

        if not gpus:
            raise NvidiaMonitorError("Could not query any NVIDIA GPUs")

        # Calculate totals for memory
        total_memory = sum(g.memory_total_mb for g in gpus)
        used_memory = sum(g.memory_used_mb for g in gpus)
        free_memory = sum(g.memory_free_mb for g in gpus)

        # Average utilization and temperature across all GPUs
        avg_utilization = sum(g.gpu_utilization for g in gpus) / len(gpus)
        avg_temp = sum(g.temperature for g in gpus) / len(gpus)
        total_power = sum(g.power_draw for g in gpus)

        # Get topology for multi-GPU systems
        topology = get_topology() if len(gpus) > 1 else None

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
            topology=topology,
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
