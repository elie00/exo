"""GPU placement contracts on the post-merge granular node state schema."""

import pytest

from exo.master.placement_utils import (
    filter_cycles_by_memory,
    get_cycle_effective_memory,
    get_node_effective_memory,
    get_shard_assignments,
)
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.shared.types.topology import Cycle
from exo.shared.types.worker.shards import Sharding


def node_memory(
    ram_mb: int, gpu_total_mb: int | None = None, gpu_used_mb: int | None = None
) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=ram_mb * 1024**2,
        ram_available=ram_mb * 1024**2,
        swap_total=0,
        swap_available=0,
        gpu_vram_total=gpu_total_mb * 1024**2 if gpu_total_mb is not None else None,
        gpu_vram_used=gpu_used_mb * 1024**2 if gpu_used_mb is not None else None,
        gpu_vram_available=(
            max(0, gpu_total_mb - gpu_used_mb) * 1024**2
            if gpu_total_mb is not None and gpu_used_mb is not None
            else None
        ),
    )


@pytest.mark.parametrize(
    "memory,prefer_gpu,expected_mb",
    [
        (node_memory(32000, 24000, 4000), True, 20000),
        (node_memory(32000, 24000, 4000), False, 32000),
        (node_memory(64000), True, 64000),
        (node_memory(64000, 24000, 24000), True, 0),
        (node_memory(64000, 24000, None), True, 0),
    ],
)
def test_node_effective_memory(
    memory: MemoryUsage, prefer_gpu: bool, expected_mb: int
) -> None:
    result = get_node_effective_memory(memory, prefer_gpu=prefer_gpu)
    assert result.in_mb == expected_mb


@pytest.mark.parametrize(
    "memories,prefer_gpu,expected_mb",
    [
        ([node_memory(32000, 24000, 0), node_memory(32000, 24000, 0)], True, 48000),
        ([node_memory(32000, 24000, 0), node_memory(64000)], True, 96000),
        ([node_memory(32000, 24000, 0), node_memory(32000, 24000, 0)], False, 64000),
    ],
)
def test_cycle_memory_uses_one_consistent_domain(
    memories: list[MemoryUsage], prefer_gpu: bool, expected_mb: int
) -> None:
    memory_by_node = {NodeId(): memory for memory in memories}
    cycle = Cycle(node_ids=list(memory_by_node))
    result = get_cycle_effective_memory(cycle, memory_by_node, prefer_gpu=prefer_gpu)
    assert result.in_mb == expected_mb


@pytest.mark.parametrize(
    "memories,required_mb,accepted",
    [
        ([node_memory(32000, 24000, 0), node_memory(32000, 24000, 0)], 40000, True),
        ([node_memory(128000, 24000, 0), node_memory(128000, 24000, 0)], 60000, False),
        ([node_memory(32000, 24000, 0), node_memory(64000)], 90000, True),
    ],
)
def test_filter_cycles_gpu_memory(
    memories: list[MemoryUsage], required_mb: int, accepted: bool
) -> None:
    memory_by_node = {NodeId(): memory for memory in memories}
    cycle = Cycle(node_ids=list(memory_by_node))
    result = filter_cycles_by_memory(
        [cycle], memory_by_node, Memory.from_mb(required_mb), prefer_gpu=True
    )
    assert result == ([cycle] if accepted else [])


def test_filter_does_not_place_node_with_missing_telemetry() -> None:
    cycle = Cycle(node_ids=[NodeId(), NodeId()])
    assert (
        filter_cycles_by_memory(
            [cycle],
            {cycle.node_ids[0]: node_memory(64000)},
            Memory.from_mb(1),
            prefer_gpu=True,
        )
        == []
    )


@pytest.mark.parametrize("prefer_gpu", [True, False])
def test_shard_distribution_respects_selected_memory_domain(prefer_gpu: bool) -> None:
    nodes = [NodeId(), NodeId()]
    memory_by_node = {
        nodes[0]: node_memory(64000, 12000, 0),
        nodes[1]: node_memory(32000, 24000, 0),
    }
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(30000),
        n_layers=12,
        hidden_size=4096,
        supports_tensor=True,
    )
    assignments = get_shard_assignments(
        model_card,
        Cycle(node_ids=nodes),
        Sharding.Pipeline,
        memory_by_node,
        prefer_gpu=prefer_gpu,
    )
    shards = [
        assignments.runner_to_shard[assignments.node_to_runner[node]] for node in nodes
    ]
    assert [(shard.start_layer, shard.end_layer) for shard in shards] == (
        [(0, 4), (4, 12)] if prefer_gpu else [(0, 8), (8, 12)]
    )


def test_shards_reject_exhausted_gpu_even_when_host_has_memory() -> None:
    node = NodeId()
    model = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(100),
        n_layers=2,
        hidden_size=64,
        supports_tensor=False,
    )
    with pytest.raises(ValueError, match="available memory is 0"):
        get_shard_assignments(
            model,
            Cycle(node_ids=[node]),
            Sharding.Pipeline,
            {node: node_memory(64000, 24000, 24000)},
            prefer_gpu=True,
        )


def test_zero_vram_samples_survive_serialization() -> None:
    memory = node_memory(64000, 24000, 0)
    assert memory.gpu_vram_used == Memory()
    roundtrip = MemoryUsage.model_validate_json(memory.model_dump_json())
    assert roundtrip.gpu_vram_used == Memory()
    exhausted = node_memory(64000, 24000, 24000)
    assert exhausted.gpu_vram_available == Memory()
    assert (
        MemoryUsage.model_validate_json(exhausted.model_dump_json()).gpu_vram_available
        == Memory()
    )


@pytest.mark.parametrize(
    "total,used,has_memory,available",
    [
        (24000, 4000, True, 20000),
        (None, None, False, 0),
        (0, 0, False, 0),
        (24000, 30000, True, 0),
    ],
)
def test_system_performance_profile_gpu_properties(
    total: int | None, used: int | None, has_memory: bool, available: int
) -> None:
    profile = SystemPerformanceProfile(
        gpu_memory_total_mb=total, gpu_memory_used_mb=used
    )
    assert profile.has_gpu_memory is has_memory
    assert profile.gpu_memory_available_mb == available


def test_master_placement_respects_prefer_gpu_and_rejects_unwired_gguf() -> None:
    from exo.master.placement import place_instance
    from exo.master.tests.conftest import create_node_network
    from exo.shared.topology import Topology
    from exo.shared.types.commands import PlaceInstance
    from exo.shared.types.worker.instances import InstanceMeta

    node = NodeId()
    topology = Topology()
    topology.add_node(node)
    model = ModelCard(
        model_id=ModelId("test"),
        storage_size=Memory.from_mb(1000),
        n_layers=4,
        hidden_size=128,
        supports_tensor=False,
    )
    command = PlaceInstance(
        model_card=model,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        prefer_gpu=True,
    )
    memory = {node: node_memory(64000, 2000, 2000)}
    network = {node: create_node_network()}
    with pytest.raises(ValueError, match="sufficient memory"):
        place_instance(command, topology, {}, memory, network)
    command.prefer_gpu = False
    assert len(place_instance(command, topology, {}, memory, network)) == 1
    command.instance_meta = InstanceMeta.GGUFPipeline
    with pytest.raises(ValueError, match="GGUF pipeline placement"):
        place_instance(command, topology, {}, memory, network)
