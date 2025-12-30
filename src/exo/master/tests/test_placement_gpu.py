"""
Tests for GPU-aware placement functionality.

These tests verify that the placement system correctly uses GPU VRAM
when available and falls back to RAM appropriately.
"""

from typing import Callable

import pytest

from exo.master.placement_utils import (
    _get_cycle_effective_memory,
    _get_node_effective_memory,
    filter_cycles_by_memory,
    get_shard_assignments,
)
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.worker.shards import Sharding


def create_node_with_gpu(
    ram_mb: int,
    gpu_memory_total_mb: int,
    gpu_memory_used_mb: int,
    node_id: NodeId | None = None,
) -> NodeInfo:
    """Create a node with both RAM and GPU memory."""
    if node_id is None:
        node_id = NodeId()
    return NodeInfo(
        node_id=node_id,
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
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_count=1,
                driver_version="535.154.05",
                cuda_version="12.2",
            ),
        ),
    )


def create_node_cpu_only(
    ram_mb: int,
    node_id: NodeId | None = None,
) -> NodeInfo:
    """Create a CPU-only node (no GPU memory)."""
    if node_id is None:
        node_id = NodeId()
    return NodeInfo(
        node_id=node_id,
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
            system=SystemPerformanceProfile(),  # No GPU fields
        ),
    )


class TestGetNodeEffectiveMemory:
    """Tests for _get_node_effective_memory function."""

    def test_gpu_node_prefer_gpu_returns_vram(self):
        """When prefer_gpu=True and node has GPU, return VRAM."""
        node = create_node_with_gpu(
            ram_mb=32000,  # 32GB RAM
            gpu_memory_total_mb=24000,  # 24GB VRAM
            gpu_memory_used_mb=4000,  # 4GB used
        )
        # Create a NodeWithProfile-like object
        from exo.master.placement_utils import NodeWithProfile

        node_with_profile = NodeWithProfile(
            node_id=node.node_id,
            node_profile=node.node_profile,  # type: ignore
        )

        memory = _get_node_effective_memory(node_with_profile, prefer_gpu=True)

        # Should return available VRAM (24000 - 4000 = 20000 MB)
        assert memory.in_mb == pytest.approx(20000, rel=0.01)

    def test_gpu_node_prefer_ram_returns_ram(self):
        """When prefer_gpu=False, return RAM even if GPU is available."""
        node = create_node_with_gpu(
            ram_mb=32000,  # 32GB RAM
            gpu_memory_total_mb=24000,  # 24GB VRAM
            gpu_memory_used_mb=4000,
        )
        from exo.master.placement_utils import NodeWithProfile

        node_with_profile = NodeWithProfile(
            node_id=node.node_id,
            node_profile=node.node_profile,  # type: ignore
        )

        memory = _get_node_effective_memory(node_with_profile, prefer_gpu=False)

        # Should return RAM
        assert memory.in_mb == pytest.approx(32000, rel=0.01)

    def test_cpu_only_node_returns_ram(self):
        """CPU-only nodes always return RAM."""
        node = create_node_cpu_only(ram_mb=64000)  # 64GB RAM
        from exo.master.placement_utils import NodeWithProfile

        node_with_profile = NodeWithProfile(
            node_id=node.node_id,
            node_profile=node.node_profile,  # type: ignore
        )

        memory = _get_node_effective_memory(node_with_profile, prefer_gpu=True)

        # Should return RAM since no GPU
        assert memory.in_mb == pytest.approx(64000, rel=0.01)


class TestGetCycleEffectiveMemory:
    """Tests for _get_cycle_effective_memory function."""

    def test_all_gpu_nodes_prefer_gpu_returns_total_vram(self):
        """When all nodes have GPU and prefer_gpu=True, return total VRAM."""
        from exo.master.placement_utils import NodeWithProfile

        nodes = [
            create_node_with_gpu(
                ram_mb=32000, gpu_memory_total_mb=24000, gpu_memory_used_mb=0
            ),
            create_node_with_gpu(
                ram_mb=32000, gpu_memory_total_mb=24000, gpu_memory_used_mb=0
            ),
        ]
        cycle = [
            NodeWithProfile(node_id=n.node_id, node_profile=n.node_profile)  # type: ignore
            for n in nodes
        ]

        memory = _get_cycle_effective_memory(cycle, prefer_gpu=True)

        # Should return total VRAM (24000 + 24000 = 48000 MB)
        assert memory.in_mb == pytest.approx(48000, rel=0.01)

    def test_mixed_gpu_cpu_nodes_returns_ram(self):
        """When nodes are mixed (some GPU, some CPU), return RAM."""
        from exo.master.placement_utils import NodeWithProfile

        nodes = [
            create_node_with_gpu(
                ram_mb=32000, gpu_memory_total_mb=24000, gpu_memory_used_mb=0
            ),
            create_node_cpu_only(ram_mb=64000),  # CPU only
        ]
        cycle = [
            NodeWithProfile(node_id=n.node_id, node_profile=n.node_profile)  # type: ignore
            for n in nodes
        ]

        memory = _get_cycle_effective_memory(cycle, prefer_gpu=True)

        # Should return total RAM (32000 + 64000 = 96000 MB)
        assert memory.in_mb == pytest.approx(96000, rel=0.01)

    def test_all_gpu_nodes_prefer_ram_returns_ram(self):
        """When prefer_gpu=False, return RAM even with all GPU nodes."""
        from exo.master.placement_utils import NodeWithProfile

        nodes = [
            create_node_with_gpu(
                ram_mb=32000, gpu_memory_total_mb=24000, gpu_memory_used_mb=0
            ),
            create_node_with_gpu(
                ram_mb=32000, gpu_memory_total_mb=24000, gpu_memory_used_mb=0
            ),
        ]
        cycle = [
            NodeWithProfile(node_id=n.node_id, node_profile=n.node_profile)  # type: ignore
            for n in nodes
        ]

        memory = _get_cycle_effective_memory(cycle, prefer_gpu=False)

        # Should return total RAM (32000 + 32000 = 64000 MB)
        assert memory.in_mb == pytest.approx(64000, rel=0.01)


class TestFilterCyclesByMemoryGpu:
    """Tests for filter_cycles_by_memory with GPU memory."""

    @pytest.fixture
    def topology(self) -> Topology:
        return Topology()

    def test_gpu_cycle_passes_vram_filter(
        self,
        topology: Topology,
        create_connection: Callable[[NodeId, NodeId], Connection],
    ):
        """A cycle with sufficient VRAM should pass when prefer_gpu=True."""
        node1_id = NodeId()
        node2_id = NodeId()

        # Each node has 24GB VRAM, 32GB RAM
        node1 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node1_id,
        )
        node2 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node2_id,
        )

        topology.add_node(node1)
        topology.add_node(node2)
        topology.add_connection(create_connection(node1_id, node2_id))
        topology.add_connection(create_connection(node2_id, node1_id))

        cycles = topology.get_cycles()

        # Require 40GB - should pass with VRAM (48GB total), fail with RAM if it were less
        filtered = filter_cycles_by_memory(
            cycles, Memory.from_mb(40000), prefer_gpu=True
        )

        assert len(filtered) == 1

    def test_gpu_cycle_fails_insufficient_vram(
        self,
        topology: Topology,
        create_connection: Callable[[NodeId, NodeId], Connection],
    ):
        """A cycle with insufficient VRAM should fail when prefer_gpu=True."""
        node1_id = NodeId()
        node2_id = NodeId()

        # Each node has 24GB VRAM (48GB total)
        node1 = create_node_with_gpu(
            ram_mb=128000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node1_id,
        )
        node2 = create_node_with_gpu(
            ram_mb=128000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node2_id,
        )

        topology.add_node(node1)
        topology.add_node(node2)
        topology.add_connection(create_connection(node1_id, node2_id))
        topology.add_connection(create_connection(node2_id, node1_id))

        cycles = topology.get_cycles()

        # Require 60GB - should fail (only 48GB VRAM)
        # Even though there's 256GB RAM, we use VRAM when prefer_gpu=True and all have GPU
        filtered = filter_cycles_by_memory(
            cycles, Memory.from_mb(60000), prefer_gpu=True
        )

        assert len(filtered) == 0

    def test_mixed_cycle_uses_ram(
        self,
        topology: Topology,
        create_connection: Callable[[NodeId, NodeId], Connection],
    ):
        """A mixed GPU/CPU cycle should use RAM even with prefer_gpu=True."""
        node1_id = NodeId()
        node2_id = NodeId()

        # Node 1 has GPU (24GB VRAM, 32GB RAM)
        node1 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node1_id,
        )
        # Node 2 is CPU only (64GB RAM)
        node2 = create_node_cpu_only(ram_mb=64000, node_id=node2_id)

        topology.add_node(node1)
        topology.add_node(node2)
        topology.add_connection(create_connection(node1_id, node2_id))
        topology.add_connection(create_connection(node2_id, node1_id))

        cycles = topology.get_cycles()

        # Require 90GB - should pass with RAM (96GB total)
        # Would fail if using VRAM (only 24GB from node1)
        filtered = filter_cycles_by_memory(
            cycles, Memory.from_mb(90000), prefer_gpu=True
        )

        assert len(filtered) == 1


class TestShardAssignmentsGpu:
    """Tests for GPU-aware shard assignments."""

    @pytest.fixture
    def topology(self) -> Topology:
        return Topology()

    def test_shard_distribution_by_vram(
        self,
        topology: Topology,
        create_connection: Callable[[NodeId, NodeId], Connection],
    ):
        """Shards should be distributed proportionally to VRAM when prefer_gpu=True."""
        node1_id = NodeId()
        node2_id = NodeId()

        # Node 1: 12GB VRAM, Node 2: 24GB VRAM (1:2 ratio)
        node1 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=12000,
            gpu_memory_used_mb=0,
            node_id=node1_id,
        )
        node2 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node2_id,
        )

        topology.add_node(node1)
        topology.add_node(node2)
        topology.add_connection(create_connection(node1_id, node2_id))
        topology.add_connection(create_connection(node2_id, node1_id))

        model_meta = ModelMetadata(
            model_id=ModelId("test-model"),
            pretty_name="Test Model",
            n_layers=12,
            storage_size=Memory.from_mb(30000),
            hidden_size=4096,
            supports_tensor=True,
        )

        cycles = topology.get_cycles()
        selected_cycle = cycles[0]

        shard_assignments = get_shard_assignments(
            model_meta, selected_cycle, Sharding.Pipeline, prefer_gpu=True
        )

        # Get layer counts for each node
        runner_id_1 = shard_assignments.node_to_runner[node1_id]
        runner_id_2 = shard_assignments.node_to_runner[node2_id]

        shard_1 = shard_assignments.runner_to_shard[runner_id_1]
        shard_2 = shard_assignments.runner_to_shard[runner_id_2]

        layers_1 = shard_1.end_layer - shard_1.start_layer
        layers_2 = shard_2.end_layer - shard_2.start_layer

        # With 1:2 ratio, expect ~4 layers for node1, ~8 layers for node2
        assert layers_1 + layers_2 == 12
        assert layers_1 == pytest.approx(4, abs=1)
        assert layers_2 == pytest.approx(8, abs=1)

    def test_shard_distribution_by_ram_when_prefer_gpu_false(
        self,
        topology: Topology,
        create_connection: Callable[[NodeId, NodeId], Connection],
    ):
        """Shards should be distributed by RAM when prefer_gpu=False."""
        node1_id = NodeId()
        node2_id = NodeId()

        # Node 1: 12GB VRAM, 32GB RAM; Node 2: 24GB VRAM, 64GB RAM
        # VRAM ratio 1:2, RAM ratio 1:2
        node1 = create_node_with_gpu(
            ram_mb=32000,
            gpu_memory_total_mb=48000,
            gpu_memory_used_mb=0,
            node_id=node1_id,
        )
        node2 = create_node_with_gpu(
            ram_mb=64000,
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=0,
            node_id=node2_id,
        )

        topology.add_node(node1)
        topology.add_node(node2)
        topology.add_connection(create_connection(node1_id, node2_id))
        topology.add_connection(create_connection(node2_id, node1_id))

        model_meta = ModelMetadata(
            model_id=ModelId("test-model"),
            pretty_name="Test Model",
            n_layers=12,
            storage_size=Memory.from_mb(80000),
            hidden_size=4096,
            supports_tensor=True,
        )

        cycles = topology.get_cycles()
        selected_cycle = cycles[0]

        shard_assignments = get_shard_assignments(
            model_meta, selected_cycle, Sharding.Pipeline, prefer_gpu=False
        )

        runner_id_1 = shard_assignments.node_to_runner[node1_id]
        runner_id_2 = shard_assignments.node_to_runner[node2_id]

        shard_1 = shard_assignments.runner_to_shard[runner_id_1]
        shard_2 = shard_assignments.runner_to_shard[runner_id_2]

        layers_1 = shard_1.end_layer - shard_1.start_layer
        layers_2 = shard_2.end_layer - shard_2.start_layer

        # With RAM ratio 1:2, expect ~4 layers for node1, ~8 layers for node2
        assert layers_1 + layers_2 == 12
        assert layers_1 == pytest.approx(4, abs=1)
        assert layers_2 == pytest.approx(8, abs=1)


class TestSystemPerformanceProfileGpu:
    """Tests for GPU-related properties in SystemPerformanceProfile."""

    def test_has_gpu_memory_true(self):
        """has_gpu_memory returns True when GPU memory is set."""
        profile = SystemPerformanceProfile(
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=4000,
        )
        assert profile.has_gpu_memory is True

    def test_has_gpu_memory_false_when_none(self):
        """has_gpu_memory returns False when GPU memory is None."""
        profile = SystemPerformanceProfile()
        assert profile.has_gpu_memory is False

    def test_has_gpu_memory_false_when_zero(self):
        """has_gpu_memory returns False when GPU memory is 0."""
        profile = SystemPerformanceProfile(
            gpu_memory_total_mb=0,
            gpu_memory_used_mb=0,
        )
        assert profile.has_gpu_memory is False

    def test_gpu_memory_available_mb(self):
        """gpu_memory_available_mb calculates correctly."""
        profile = SystemPerformanceProfile(
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=4000,
        )
        assert profile.gpu_memory_available_mb == 20000

    def test_gpu_memory_available_mb_zero_when_none(self):
        """gpu_memory_available_mb returns 0 when no GPU."""
        profile = SystemPerformanceProfile()
        assert profile.gpu_memory_available_mb == 0

    def test_gpu_memory_available_mb_never_negative(self):
        """gpu_memory_available_mb never returns negative."""
        profile = SystemPerformanceProfile(
            gpu_memory_total_mb=24000,
            gpu_memory_used_mb=30000,  # More used than total (edge case)
        )
        assert profile.gpu_memory_available_mb == 0
