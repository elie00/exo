"""Tests for multi-GPU placement helper functions."""

import pytest

from exo.master.placement_utils import (
    NodeWithProfile,
    get_cycle_topology_score,
    get_node_gpu_summary,
    get_node_per_gpu_memory,
    get_node_topology_score,
    rank_cycles_by_topology,
    select_best_cycle_for_tensor_parallel,
)
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    GpuDeviceInfo,
    GpuTopology,
    GpuTopologyLink,
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import NodeInfo


def create_multi_gpu_node(
    ram_mb: int,
    gpus: list[GpuDeviceInfo],
    topology: GpuTopology | None = None,
) -> NodeWithProfile:
    """Create a node with multiple GPUs."""
    total_vram = sum(g.memory_total_mb for g in gpus)
    used_vram = sum(g.memory_used_mb for g in gpus)

    return NodeWithProfile(
        node_id=NodeId(),
        node_profile=NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="test",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=ram_mb * 1024 * 1024,
                ram_available=ram_mb * 1024 * 1024,
                swap_total=0,
                swap_available=0,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(
                gpu_memory_total_mb=total_vram,
                gpu_memory_used_mb=used_vram,
                gpu_count=len(gpus),
                driver_version="535.154.05",
                cuda_version="12.2",
                gpus=gpus,
                gpu_topology=topology,
            ),
        ),
    )


def create_single_gpu_node(
    ram_mb: int,
    gpu_name: str,
    vram_total_mb: int,
    vram_used_mb: int,
) -> NodeWithProfile:
    """Create a node with a single GPU."""
    gpu = GpuDeviceInfo(
        index=0,
        uuid="GPU-0",
        name=gpu_name,
        memory_total_mb=vram_total_mb,
        memory_used_mb=vram_used_mb,
        memory_free_mb=vram_total_mb - vram_used_mb,
        utilization=50.0,
        temperature=65.0,
        power_draw=300.0,
        power_limit=450.0,
    )
    return create_multi_gpu_node(ram_mb, [gpu])


def create_cpu_only_node(ram_mb: int) -> NodeWithProfile:
    """Create a CPU-only node (no GPU)."""
    return NodeWithProfile(
        node_id=NodeId(),
        node_profile=NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="test",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=ram_mb * 1024 * 1024,
                ram_available=ram_mb * 1024 * 1024,
                swap_total=0,
                swap_available=0,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
        ),
    )


class TestGetNodeGpuSummary:
    """Tests for get_node_gpu_summary function."""

    def test_no_gpu(self):
        """Test summary for a node without GPU."""
        node = create_cpu_only_node(ram_mb=64000)
        summary = get_node_gpu_summary(node)
        assert summary == "No GPU"

    def test_single_gpu(self):
        """Test summary for a node with a single GPU."""
        node = create_single_gpu_node(
            ram_mb=64000,
            gpu_name="RTX 4090",
            vram_total_mb=24576,
            vram_used_mb=1024,
        )
        summary = get_node_gpu_summary(node)
        assert "RTX 4090" in summary
        assert "24GB" in summary

    def test_multi_gpu_nvlink(self):
        """Test summary for a multi-GPU node with NVLink."""
        gpus = [
            GpuDeviceInfo(
                index=0,
                uuid="GPU-0",
                name="A100",
                memory_total_mb=81920,
                memory_used_mb=1024,
                memory_free_mb=80896,
                utilization=50.0,
                temperature=65.0,
                power_draw=300.0,
                power_limit=400.0,
            ),
            GpuDeviceInfo(
                index=1,
                uuid="GPU-1",
                name="A100",
                memory_total_mb=81920,
                memory_used_mb=2048,
                memory_free_mb=79872,
                utilization=60.0,
                temperature=70.0,
                power_draw=350.0,
                power_limit=400.0,
            ),
        ]
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
        node = create_multi_gpu_node(ram_mb=512000, gpus=gpus, topology=topology)
        summary = get_node_gpu_summary(node)

        assert "2x" in summary
        assert "A100" in summary
        assert "160GB" in summary  # 2 * 80GB
        assert "NVLink" in summary

    def test_multi_gpu_pcie(self):
        """Test summary for a multi-GPU node with PCIe only."""
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
        topology = GpuTopology(
            gpu_count=2,
            links=[
                GpuTopologyLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="pcie",
                    nvlink_count=0,
                    p2p_supported=True,
                )
            ],
        )
        node = create_multi_gpu_node(ram_mb=128000, gpus=gpus, topology=topology)
        summary = get_node_gpu_summary(node)

        assert "2x" in summary
        assert "RTX 4090" in summary
        assert "48GB" in summary  # 2 * 24GB
        assert "PCIe" in summary


class TestGetNodePerGpuMemory:
    """Tests for get_node_per_gpu_memory function."""

    def test_no_gpu_info(self):
        """Test when no per-GPU info is available."""
        node = create_cpu_only_node(ram_mb=64000)
        result = get_node_per_gpu_memory(node)
        assert result == []

    def test_single_gpu(self):
        """Test per-GPU memory for single GPU."""
        node = create_single_gpu_node(
            ram_mb=64000,
            gpu_name="RTX 4090",
            vram_total_mb=24576,
            vram_used_mb=1024,
        )
        result = get_node_per_gpu_memory(node)

        assert len(result) == 1
        gpu_idx, memory = result[0]
        assert gpu_idx == 0
        assert memory.in_mb == pytest.approx(23552, rel=0.01)

    def test_multi_gpu(self):
        """Test per-GPU memory for multiple GPUs."""
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
                memory_used_mb=4096,
                memory_free_mb=20480,
                utilization=60.0,
                temperature=70.0,
                power_draw=350.0,
                power_limit=450.0,
            ),
        ]
        node = create_multi_gpu_node(ram_mb=128000, gpus=gpus)
        result = get_node_per_gpu_memory(node)

        assert len(result) == 2

        # GPU 0 should have more available memory
        gpu0_idx, gpu0_mem = result[0]
        assert gpu0_idx == 0
        assert gpu0_mem.in_mb == pytest.approx(23552, rel=0.01)

        # GPU 1 should have less available memory
        gpu1_idx, gpu1_mem = result[1]
        assert gpu1_idx == 1
        assert gpu1_mem.in_mb == pytest.approx(20480, rel=0.01)

    def test_four_gpu_node(self):
        """Test per-GPU memory for a 4-GPU DGX-style node."""
        gpus = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="A100",
                memory_total_mb=81920,
                memory_used_mb=1024 * (i + 1),
                memory_free_mb=81920 - 1024 * (i + 1),
                utilization=25.0 * (i + 1),
                temperature=50.0 + i * 5,
                power_draw=200.0 + i * 50,
                power_limit=400.0,
            )
            for i in range(4)
        ]
        node = create_multi_gpu_node(ram_mb=1024000, gpus=gpus)
        result = get_node_per_gpu_memory(node)

        assert len(result) == 4

        # Verify each GPU has expected available memory
        for i, (gpu_idx, memory) in enumerate(result):
            assert gpu_idx == i
            expected_available = 81920 - 1024 * (i + 1)
            assert memory.in_mb == pytest.approx(expected_available, rel=0.01)


class TestGetNodeTopologyScore:
    """Tests for get_node_topology_score function."""

    def test_single_gpu_scores_perfect(self):
        """Single GPU nodes should score 1.0 (no interconnect needed)."""
        node = create_single_gpu_node(
            ram_mb=64000,
            gpu_name="RTX 4090",
            vram_total_mb=24576,
            vram_used_mb=1024,
        )
        score = get_node_topology_score(node)
        assert score == 1.0

    def test_cpu_only_scores_perfect(self):
        """CPU-only nodes should score 1.0."""
        node = create_cpu_only_node(ram_mb=64000)
        score = get_node_topology_score(node)
        assert score == 1.0

    def test_all_nvlink_scores_perfect(self):
        """All NVLink connections should score 1.0."""
        gpus = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="A100",
                memory_total_mb=81920,
                memory_used_mb=1024,
                memory_free_mb=80896,
                utilization=50.0,
                temperature=65.0,
                power_draw=300.0,
                power_limit=400.0,
            )
            for i in range(2)
        ]
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
        node = create_multi_gpu_node(ram_mb=512000, gpus=gpus, topology=topology)
        score = get_node_topology_score(node)
        assert score == 1.0

    def test_pcie_p2p_scores_medium(self):
        """PCIe with P2P should score around 0.5."""
        gpus = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="RTX 4090",
                memory_total_mb=24576,
                memory_used_mb=1024,
                memory_free_mb=23552,
                utilization=50.0,
                temperature=65.0,
                power_draw=300.0,
                power_limit=450.0,
            )
            for i in range(2)
        ]
        topology = GpuTopology(
            gpu_count=2,
            links=[
                GpuTopologyLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="pcie",
                    nvlink_count=0,
                    p2p_supported=True,
                )
            ],
        )
        node = create_multi_gpu_node(ram_mb=128000, gpus=gpus, topology=topology)
        score = get_node_topology_score(node)
        assert score == 0.5

    def test_no_p2p_scores_low(self):
        """No P2P support should score low."""
        gpus = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="RTX 3070",
                memory_total_mb=8192,
                memory_used_mb=512,
                memory_free_mb=7680,
                utilization=50.0,
                temperature=65.0,
                power_draw=200.0,
                power_limit=220.0,
            )
            for i in range(2)
        ]
        topology = GpuTopology(
            gpu_count=2,
            links=[
                GpuTopologyLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="system",
                    nvlink_count=0,
                    p2p_supported=False,
                )
            ],
        )
        node = create_multi_gpu_node(ram_mb=64000, gpus=gpus, topology=topology)
        score = get_node_topology_score(node)
        assert score == 0.2


class TestGetCycleTopologyScore:
    """Tests for get_cycle_topology_score function."""

    def test_empty_cycle(self):
        """Empty cycle should score 0.0."""
        score = get_cycle_topology_score([])
        assert score == 0.0

    def test_single_node_cycle(self):
        """Single node cycle inherits node score."""
        node = create_single_gpu_node(
            ram_mb=64000,
            gpu_name="RTX 4090",
            vram_total_mb=24576,
            vram_used_mb=1024,
        )
        score = get_cycle_topology_score([node])
        assert score == 1.0

    def test_mixed_topology_cycle(self):
        """Cycle with mixed topology takes minimum score."""
        # High-quality node (NVLink)
        gpus_nvlink = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="A100",
                memory_total_mb=81920,
                memory_used_mb=1024,
                memory_free_mb=80896,
                utilization=50.0,
                temperature=65.0,
                power_draw=300.0,
                power_limit=400.0,
            )
            for i in range(2)
        ]
        topo_nvlink = GpuTopology(
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
        node_nvlink = create_multi_gpu_node(ram_mb=512000, gpus=gpus_nvlink, topology=topo_nvlink)

        # Low-quality node (no P2P)
        gpus_no_p2p = [
            GpuDeviceInfo(
                index=i,
                uuid=f"GPU-{i}",
                name="RTX 3070",
                memory_total_mb=8192,
                memory_used_mb=512,
                memory_free_mb=7680,
                utilization=50.0,
                temperature=65.0,
                power_draw=200.0,
                power_limit=220.0,
            )
            for i in range(2)
        ]
        topo_no_p2p = GpuTopology(
            gpu_count=2,
            links=[
                GpuTopologyLink(
                    gpu_index_a=0,
                    gpu_index_b=1,
                    link_type="system",
                    nvlink_count=0,
                    p2p_supported=False,
                )
            ],
        )
        node_no_p2p = create_multi_gpu_node(ram_mb=64000, gpus=gpus_no_p2p, topology=topo_no_p2p)

        # Cycle score is the minimum
        score = get_cycle_topology_score([node_nvlink, node_no_p2p])
        assert score == 0.2  # Minimum of 1.0 and 0.2


class TestRankCyclesByTopology:
    """Tests for rank_cycles_by_topology function."""

    def test_ranks_by_score_descending(self):
        """Cycles should be ranked by score descending."""
        # Create nodes with different topology qualities
        node_nvlink = create_multi_gpu_node(
            ram_mb=512000,
            gpus=[
                GpuDeviceInfo(
                    index=i,
                    uuid=f"GPU-{i}",
                    name="A100",
                    memory_total_mb=81920,
                    memory_used_mb=1024,
                    memory_free_mb=80896,
                    utilization=50.0,
                    temperature=65.0,
                    power_draw=300.0,
                    power_limit=400.0,
                )
                for i in range(2)
            ],
            topology=GpuTopology(
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
            ),
        )

        node_pcie = create_multi_gpu_node(
            ram_mb=128000,
            gpus=[
                GpuDeviceInfo(
                    index=i,
                    uuid=f"GPU-{i}",
                    name="RTX 4090",
                    memory_total_mb=24576,
                    memory_used_mb=1024,
                    memory_free_mb=23552,
                    utilization=50.0,
                    temperature=65.0,
                    power_draw=300.0,
                    power_limit=450.0,
                )
                for i in range(2)
            ],
            topology=GpuTopology(
                gpu_count=2,
                links=[
                    GpuTopologyLink(
                        gpu_index_a=0,
                        gpu_index_b=1,
                        link_type="pcie",
                        nvlink_count=0,
                        p2p_supported=True,
                    )
                ],
            ),
        )

        # Convert to NodeInfo
        node_info_nvlink = NodeInfo(
            node_id=node_nvlink.node_id,
            node_profile=node_nvlink.node_profile,
        )
        node_info_pcie = NodeInfo(
            node_id=node_pcie.node_id,
            node_profile=node_pcie.node_profile,
        )

        cycles = [[node_info_pcie], [node_info_nvlink]]
        ranked = rank_cycles_by_topology(cycles)

        assert len(ranked) == 2
        # NVLink should be ranked first
        assert ranked[0][1] == 1.0  # NVLink score
        assert ranked[1][1] == 0.5  # PCIe score
