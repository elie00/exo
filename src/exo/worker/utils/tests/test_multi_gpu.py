"""Tests for multi-GPU support functionality."""

from unittest.mock import MagicMock, patch

import pytest

from exo.shared.types.profiling import (
    GpuDeviceInfo,
    GpuTopology,
    GpuTopologyLink,
    SystemPerformanceProfile,
)
from exo.worker.utils.nvidia_monitor import (
    NvidiaGpuInfo,
    NvidiaGpuLink,
    NvidiaMetrics,
    NvidiaTopology,
)
from exo.worker.utils.profile import (
    _convert_nvidia_gpu_info,
    _convert_nvidia_link,
    _convert_nvidia_topology,
)


class TestGpuDeviceInfo:
    """Tests for GpuDeviceInfo model."""

    def test_creation(self):
        """Test creating a GpuDeviceInfo instance."""
        gpu = GpuDeviceInfo(
            index=0,
            uuid="GPU-12345",
            name="RTX 4090",
            memory_total_mb=24576,
            memory_used_mb=1024,
            memory_free_mb=23552,
            utilization=50.0,
            temperature=65.0,
            power_draw=300.0,
            power_limit=450.0,
            pcie_gen=4,
            pcie_width=16,
        )
        assert gpu.index == 0
        assert gpu.name == "RTX 4090"
        assert gpu.memory_available_mb == 23552

    def test_memory_available_calculation(self):
        """Test the memory_available_mb property."""
        gpu = GpuDeviceInfo(
            index=0,
            uuid="GPU-1",
            name="Test GPU",
            memory_total_mb=8192,
            memory_used_mb=2048,
            memory_free_mb=6144,
            utilization=25.0,
            temperature=50.0,
            power_draw=100.0,
            power_limit=200.0,
        )
        assert gpu.memory_available_mb == 6144


class TestGpuTopologyLink:
    """Tests for GpuTopologyLink model."""

    def test_nvlink_creation(self):
        """Test creating an NVLink connection."""
        link = GpuTopologyLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="nvlink",
            nvlink_count=4,
            p2p_supported=True,
            bandwidth_gbps=600.0,
        )
        assert link.link_type == "nvlink"
        assert link.nvlink_count == 4
        assert link.p2p_supported is True

    def test_pcie_creation(self):
        """Test creating a PCIe connection."""
        link = GpuTopologyLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="pcie",
            nvlink_count=0,
            p2p_supported=True,
        )
        assert link.link_type == "pcie"
        assert link.nvlink_count == 0


class TestGpuTopology:
    """Tests for GpuTopology model."""

    def test_creation_with_nvlink(self):
        """Test creating topology with NVLink."""
        links = [
            GpuTopologyLink(
                gpu_index_a=0,
                gpu_index_b=1,
                link_type="nvlink",
                nvlink_count=4,
                p2p_supported=True,
            ),
        ]
        topology = GpuTopology(gpu_count=2, links=links)
        assert topology.gpu_count == 2
        assert topology.has_nvlink() is True
        assert topology.all_p2p_supported() is True

    def test_creation_with_pcie(self):
        """Test creating topology with PCIe only."""
        links = [
            GpuTopologyLink(
                gpu_index_a=0,
                gpu_index_b=1,
                link_type="pcie",
                nvlink_count=0,
                p2p_supported=True,
            ),
        ]
        topology = GpuTopology(gpu_count=2, links=links)
        assert topology.has_nvlink() is False

    def test_get_link(self):
        """Test getting a specific link."""
        link = GpuTopologyLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="nvlink",
            nvlink_count=4,
            p2p_supported=True,
        )
        topology = GpuTopology(gpu_count=2, links=[link])

        # Forward lookup
        found = topology.get_link(0, 1)
        assert found is not None
        assert found.link_type == "nvlink"

        # Reverse lookup
        found_reverse = topology.get_link(1, 0)
        assert found_reverse is not None

        # Non-existent
        not_found = topology.get_link(0, 2)
        assert not_found is None


class TestNvidiaGpuLink:
    """Tests for NvidiaGpuLink model."""

    def test_p2p_supported_property(self):
        """Test the p2p_supported property."""
        link_with_read = NvidiaGpuLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="pcie",
            p2p_read_supported=True,
            p2p_write_supported=False,
        )
        assert link_with_read.p2p_supported is True

        link_with_write = NvidiaGpuLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="pcie",
            p2p_read_supported=False,
            p2p_write_supported=True,
        )
        assert link_with_write.p2p_supported is True

        link_no_p2p = NvidiaGpuLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="system",
            p2p_read_supported=False,
            p2p_write_supported=False,
        )
        assert link_no_p2p.p2p_supported is False


class TestNvidiaTopology:
    """Tests for NvidiaTopology model."""

    def test_has_nvlink(self):
        """Test NVLink detection."""
        nvlink_topo = NvidiaTopology(
            gpu_count=2,
            links=[
                NvidiaGpuLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="nvlink",
                    nvlink_count=4,
                )
            ],
        )
        assert nvlink_topo.has_nvlink() is True

        pcie_topo = NvidiaTopology(
            gpu_count=2,
            links=[
                NvidiaGpuLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="pcie",
                )
            ],
        )
        assert pcie_topo.has_nvlink() is False


class TestConversionFunctions:
    """Tests for conversion functions from NVIDIA types to profiling types."""

    def test_convert_nvidia_gpu_info(self):
        """Test converting NvidiaGpuInfo to GpuDeviceInfo."""
        nvidia_gpu = NvidiaGpuInfo(
            index=0,
            name="RTX 4090",
            uuid="GPU-12345",
            memory_total_mb=24576,
            memory_used_mb=1024,
            memory_free_mb=23552,
            gpu_utilization=50.0,
            memory_utilization=4.0,
            temperature=65.0,
            power_draw=300.0,
            power_limit=450.0,
            pcie_gen=4,
            pcie_width=16,
        )

        converted = _convert_nvidia_gpu_info(nvidia_gpu)

        assert isinstance(converted, GpuDeviceInfo)
        assert converted.index == 0
        assert converted.name == "RTX 4090"
        assert converted.uuid == "GPU-12345"
        assert converted.memory_total_mb == 24576
        assert converted.utilization == 50.0
        assert converted.pcie_gen == 4
        assert converted.pcie_width == 16

    def test_convert_nvidia_link(self):
        """Test converting NvidiaGpuLink to GpuTopologyLink."""
        nvidia_link = NvidiaGpuLink(
            gpu_index_a=0,
            gpu_index_b=1,
            link_type="nvlink",
            nvlink_count=4,
            p2p_read_supported=True,
            p2p_write_supported=True,
        )

        converted = _convert_nvidia_link(nvidia_link)

        assert isinstance(converted, GpuTopologyLink)
        assert converted.gpu_index_a == 0
        assert converted.gpu_index_b == 1
        assert converted.link_type == "nvlink"
        assert converted.nvlink_count == 4
        assert converted.p2p_supported is True

    def test_convert_nvidia_topology(self):
        """Test converting NvidiaTopology to GpuTopology."""
        nvidia_topo = NvidiaTopology(
            gpu_count=2,
            links=[
                NvidiaGpuLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="nvlink",
                    nvlink_count=4,
                    p2p_read_supported=True,
                    p2p_write_supported=True,
                )
            ],
        )

        converted = _convert_nvidia_topology(nvidia_topo)

        assert isinstance(converted, GpuTopology)
        assert converted.gpu_count == 2
        assert len(converted.links) == 1
        assert converted.has_nvlink() is True


class TestSystemPerformanceProfileMultiGpu:
    """Tests for SystemPerformanceProfile with multi-GPU data."""

    def test_is_multi_gpu(self):
        """Test the is_multi_gpu property."""
        single_gpu = SystemPerformanceProfile(
            gpu_count=1,
            gpu_memory_total_mb=24576,
            gpu_memory_used_mb=1024,
        )
        assert single_gpu.is_multi_gpu is False

        multi_gpu = SystemPerformanceProfile(
            gpu_count=4,
            gpu_memory_total_mb=98304,
            gpu_memory_used_mb=4096,
        )
        assert multi_gpu.is_multi_gpu is True

        no_gpu = SystemPerformanceProfile()
        assert no_gpu.is_multi_gpu is False

    def test_get_gpu(self):
        """Test the get_gpu method."""
        gpus = [
            GpuDeviceInfo(
                index=0,
                uuid="GPU-0",
                name="RTX 4090",
                memory_total_mb=24576,
                memory_used_mb=1024,
                memory_free_mb=23552,
                utilization=50.0,
                temperature=65.0,
                power_draw=300.0,
                power_limit=450.0,
            ),
            GpuDeviceInfo(
                index=1,
                uuid="GPU-1",
                name="RTX 4090",
                memory_total_mb=24576,
                memory_used_mb=2048,
                memory_free_mb=22528,
                utilization=60.0,
                temperature=70.0,
                power_draw=350.0,
                power_limit=450.0,
            ),
        ]

        profile = SystemPerformanceProfile(
            gpu_count=2,
            gpu_memory_total_mb=49152,
            gpu_memory_used_mb=3072,
            gpus=gpus,
        )

        gpu0 = profile.get_gpu(0)
        assert gpu0 is not None
        assert gpu0.uuid == "GPU-0"

        gpu1 = profile.get_gpu(1)
        assert gpu1 is not None
        assert gpu1.uuid == "GPU-1"

        gpu2 = profile.get_gpu(2)
        assert gpu2 is None

    def test_profile_with_topology(self):
        """Test SystemPerformanceProfile with GPU topology."""
        topology = GpuTopology(
            gpu_count=2,
            links=[
                GpuTopologyLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="nvlink",
                    nvlink_count=4,
                    p2p_supported=True,
                )
            ],
        )

        profile = SystemPerformanceProfile(
            gpu_count=2,
            gpu_memory_total_mb=49152,
            gpu_memory_used_mb=3072,
            gpu_topology=topology,
        )

        assert profile.gpu_topology is not None
        assert profile.gpu_topology.has_nvlink() is True


class TestNvidiaGpuInfoPcie:
    """Tests for PCIe info in NvidiaGpuInfo."""

    def test_pcie_info(self):
        """Test PCIe generation and width fields."""
        gpu = NvidiaGpuInfo(
            index=0,
            name="RTX 4090",
            uuid="GPU-12345",
            memory_total_mb=24576,
            memory_used_mb=1024,
            memory_free_mb=23552,
            gpu_utilization=50.0,
            memory_utilization=4.0,
            temperature=65.0,
            power_draw=300.0,
            power_limit=450.0,
            pcie_gen=4,
            pcie_width=16,
        )
        assert gpu.pcie_gen == 4
        assert gpu.pcie_width == 16

    def test_pcie_info_optional(self):
        """Test that PCIe info is optional."""
        gpu = NvidiaGpuInfo(
            index=0,
            name="RTX 4090",
            uuid="GPU-12345",
            memory_total_mb=24576,
            memory_used_mb=1024,
            memory_free_mb=23552,
            gpu_utilization=50.0,
            memory_utilization=4.0,
            temperature=65.0,
            power_draw=300.0,
            power_limit=450.0,
        )
        assert gpu.pcie_gen is None
        assert gpu.pcie_width is None
