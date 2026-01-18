from collections.abc import Generator
from typing import TypeGuard, cast

from loguru import logger
from pydantic import BaseModel

from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import GpuDeviceInfo, GpuTopology, NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


def narrow_all_nodes(nodes: list[NodeInfo]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)


def get_node_gpu_summary(node: NodeWithProfile) -> str:
    """
    Get a human-readable summary of a node's GPU configuration.

    Returns a string like "2x RTX 4090 (48GB total, NVLink)" or "No GPU".
    """
    system = node.node_profile.system
    if not system.has_gpu_memory:
        return "No GPU"

    gpu_count = system.gpu_count or 1
    total_gb = (system.gpu_memory_total_mb or 0) / 1024

    # Get GPU names if available
    gpu_names: list[str] = []
    if system.gpus:
        gpu_names = list({gpu.name for gpu in system.gpus})

    # Check topology for NVLink
    has_nvlink = False
    if system.gpu_topology is not None:
        has_nvlink = system.gpu_topology.has_nvlink()

    name_str = gpu_names[0] if len(gpu_names) == 1 else "Mixed GPUs"
    if gpu_count == 1:
        return f"{name_str} ({total_gb:.0f}GB)"

    topology_str = "NVLink" if has_nvlink else "PCIe"
    return f"{gpu_count}x {name_str} ({total_gb:.0f}GB total, {topology_str})"


def get_node_per_gpu_memory(node: NodeWithProfile) -> list[tuple[int, Memory]]:
    """
    Get per-GPU available memory for a node.

    Returns a list of (gpu_index, available_memory) tuples.
    Returns empty list if no per-GPU info is available.
    """
    system = node.node_profile.system
    if not system.gpus:
        return []

    return [
        (gpu.index, Memory.from_mb(gpu.memory_available_mb))
        for gpu in system.gpus
    ]


def log_cycle_gpu_summary(cycle: list[NodeWithProfile]) -> None:
    """Log a summary of GPU configuration for nodes in a cycle."""
    for node in cycle:
        summary = get_node_gpu_summary(node)
        logger.debug(f"  Node {node.node_id}: {summary}")


def get_node_topology_score(node: NodeWithProfile) -> float:
    """
    Compute a topology quality score for a node's internal GPU configuration.

    Higher scores indicate better GPU interconnects (NVLink > PCIe > system).
    Score is normalized between 0.0 and 1.0.

    Scoring:
    - Single GPU: 1.0 (no interconnect needed)
    - Multi-GPU with all NVLink: 1.0
    - Multi-GPU with partial NVLink: 0.7
    - Multi-GPU with PCIe P2P: 0.5
    - Multi-GPU without P2P: 0.2
    """
    system = node.node_profile.system
    gpu_count = system.gpu_count or 0

    if gpu_count <= 1:
        return 1.0  # Single GPU or no GPU - no interconnect concerns

    topology = system.gpu_topology
    if topology is None:
        return 0.3  # No topology info - assume basic PCIe

    if not topology.links:
        return 0.3  # No link info

    nvlink_count = sum(1 for link in topology.links if link.link_type == "nvlink")
    pcie_p2p_count = sum(
        1 for link in topology.links
        if link.link_type in ("pcie", "pcie_switch") and link.p2p_supported
    )
    total_links = len(topology.links)

    if total_links == 0:
        return 0.3

    # All links are NVLink
    if nvlink_count == total_links:
        return 1.0

    # Some NVLink connections
    if nvlink_count > 0:
        return 0.7 + 0.3 * (nvlink_count / total_links)

    # All PCIe with P2P
    if pcie_p2p_count == total_links:
        return 0.5

    # Some P2P support
    if pcie_p2p_count > 0:
        return 0.3 + 0.2 * (pcie_p2p_count / total_links)

    # No P2P support
    return 0.2


def get_cycle_topology_score(cycle: list[NodeWithProfile]) -> float:
    """
    Compute an aggregate topology quality score for a cycle of nodes.

    Returns the minimum topology score among all nodes in the cycle,
    since the weakest link determines overall performance.
    """
    if not cycle:
        return 0.0

    scores = [get_node_topology_score(node) for node in cycle]
    return min(scores)


def rank_cycles_by_topology(
    cycles: list[list[NodeInfo]],
) -> list[tuple[list[NodeInfo], float]]:
    """
    Rank cycles by their GPU topology quality.

    Returns a list of (cycle, score) tuples sorted by score descending.
    Cycles without complete profiles are ranked lowest.
    """
    scored_cycles: list[tuple[list[NodeInfo], float]] = []

    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            scored_cycles.append((cycle, 0.0))
            continue

        score = get_cycle_topology_score(cast(list[NodeWithProfile], cycle))
        scored_cycles.append((cycle, score))

    # Sort by score descending
    scored_cycles.sort(key=lambda x: x[1], reverse=True)
    return scored_cycles


def select_best_cycle_for_tensor_parallel(
    cycles: list[list[NodeInfo]],
    required_memory: Memory,
) -> list[NodeInfo] | None:
    """
    Select the best cycle for tensor parallelism based on topology.

    Tensor parallelism benefits most from high-bandwidth GPU interconnects,
    so this function prioritizes cycles with NVLink connectivity.

    Args:
        cycles: List of candidate cycles
        required_memory: Minimum memory required

    Returns:
        The best cycle for tensor parallelism, or None if no suitable cycle found.
    """
    # Filter to cycles with sufficient memory
    viable_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_vram = Memory()
        for node in cycle:
            if node.node_profile.system.has_gpu_memory:
                total_vram = total_vram + Memory.from_mb(
                    node.node_profile.system.gpu_memory_available_mb
                )

        if total_vram >= required_memory:
            viable_cycles.append(cycle)

    if not viable_cycles:
        return None

    # Rank by topology quality
    ranked = rank_cycles_by_topology(viable_cycles)

    if ranked:
        best_cycle, best_score = ranked[0]
        logger.info(f"Selected cycle with topology score {best_score:.2f} for tensor parallel")
        return best_cycle

    return None


def _get_node_effective_memory(
    node: NodeWithProfile, prefer_gpu: bool = True
) -> Memory:
    """
    Get the effective memory for a node (VRAM if GPU available and preferred, else RAM).

    Args:
        node: Node with performance profile
        prefer_gpu: If True, prefer GPU VRAM over RAM when available

    Returns:
        Memory object representing effective available memory
    """
    if prefer_gpu and node.node_profile.system.has_gpu_memory:
        return Memory.from_mb(node.node_profile.system.gpu_memory_available_mb)
    return node.node_profile.memory.ram_available


def _get_cycle_effective_memory(
    cycle: list[NodeWithProfile], prefer_gpu: bool = True
) -> Memory:
    """
    Get the total effective memory for a cycle.

    If prefer_gpu is True and ALL nodes have GPU memory, uses VRAM.
    Otherwise uses RAM.

    Args:
        cycle: List of nodes with performance profiles
        prefer_gpu: If True, prefer GPU VRAM over RAM when available

    Returns:
        Total Memory available in the cycle
    """
    # Check if all nodes have GPU memory
    all_have_gpu = all(node.node_profile.system.has_gpu_memory for node in cycle)

    if prefer_gpu and all_have_gpu:
        return sum(
            (_get_node_effective_memory(node, prefer_gpu=True) for node in cycle),
            start=Memory(),
        )
    else:
        return sum(
            (node.node_profile.memory.ram_available for node in cycle),
            start=Memory(),
        )


def filter_cycles_by_memory(
    cycles: list[list[NodeInfo]],
    required_memory: Memory,
    prefer_gpu: bool = True,
) -> list[list[NodeInfo]]:
    """
    Filter cycles by available memory (RAM or VRAM).

    Args:
        cycles: List of node cycles to filter
        required_memory: Minimum memory required for the model
        prefer_gpu: If True, prefer GPU VRAM over RAM when available

    Returns:
        List of cycles that have sufficient memory
    """
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        # Calculate total RAM available
        total_ram = sum(
            (node.node_profile.memory.ram_available for node in cycle), start=Memory()
        )

        # Calculate total VRAM available (for nodes with GPU)
        total_vram = Memory()
        gpu_node_count = 0
        total_gpu_devices = 0
        has_nvlink = False

        for node in cycle:
            if node.node_profile.system.has_gpu_memory:
                vram_available_mb = node.node_profile.system.gpu_memory_available_mb
                total_vram = total_vram + Memory.from_mb(vram_available_mb)
                gpu_node_count += 1
                total_gpu_devices += node.node_profile.system.gpu_count or 1

                # Check for NVLink in multi-GPU nodes
                if node.node_profile.system.gpu_topology is not None:
                    if node.node_profile.system.gpu_topology.has_nvlink():
                        has_nvlink = True

        # Decide which memory pool to use
        # If prefer_gpu and ALL nodes have GPU memory, use VRAM
        # Otherwise, fall back to RAM
        if prefer_gpu and gpu_node_count == len(cycle) and total_vram > Memory():
            effective_memory = total_vram
            nvlink_str = " (NVLink)" if has_nvlink else ""
            logger.debug(
                f"Cycle: {len(cycle)} nodes, {total_gpu_devices} GPUs, "
                f"{total_vram.in_gb:.1f}GB VRAM{nvlink_str}"
            )
        else:
            effective_memory = total_ram
            if prefer_gpu and gpu_node_count > 0 and gpu_node_count < len(cycle):
                logger.debug(
                    f"Mixed GPU/CPU cycle ({gpu_node_count}/{len(cycle)} GPU nodes), using RAM"
                )

        if effective_memory >= required_memory:
            filtered_cycles.append(cast(list[NodeInfo], cycle))

    return filtered_cycles


def get_smallest_cycles(cycles: list[list[NodeInfo]]) -> list[list[NodeInfo]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def get_shard_assignments_for_pipeline_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
    prefer_gpu: bool = True,
):
    cycle_memory = _get_cycle_effective_memory(selected_cycle, prefer_gpu=prefer_gpu)
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_memory = _get_node_effective_memory(node, prefer_gpu=prefer_gpu)
            node_layers = round(
                total_layers * (node_memory.in_bytes / cycle_memory.in_bytes)
            )
            node_layers = max(1, node_layers)

        runner_id = RunnerId()

        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments_for_tensor_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
):
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node in enumerate(selected_cycle):
        shard = TensorShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    sharding: Sharding,
    prefer_gpu: bool = True,
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
                prefer_gpu=prefer_gpu,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    cycles = cycle_digraph.get_cycles()
    expected_length = len(list(cycle_digraph.list_nodes()))
    cycles = [cycle for cycle in cycles if len(cycle) == expected_length]
    if not cycles:
        if expected_length > 1:
            logger.warning(
                f"No cycles of length {expected_length} found even though chosen subgraph contained {expected_length} nodes"
            )
        return []

    get_thunderbolt = False
    if cycle_digraph.is_thunderbolt_cycle(cycles[0]):
        get_thunderbolt = True

    logger.info(f"Using thunderbolt cycle: {get_thunderbolt}")

    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        for connection in cycle_digraph.list_connections():
            if (
                connection.local_node_id == current_node.node_id
                and connection.send_back_node_id == next_node.node_id
            ):
                if get_thunderbolt and not connection.is_thunderbolt():
                    continue
                assert connection.send_back_multiaddr is not None
                host = Host(
                    ip=connection.send_back_multiaddr.ip_address,
                    port=connection.send_back_multiaddr.port,
                )
                hosts.append(host)
                break

    return hosts


def get_mlx_ibv_devices_matrix(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            # Find the IP J uses to talk to I
            for connection_ip, _ in _find_connection_ip(node_j, node_i, cycle_digraph):
                # This is a local IP on I, which is attached to an interface: find that interface
                if interface_name := _find_rdma_interface_name_for_ip(
                    connection_ip, node_i
                ):
                    matrix[i][j] = interface_name
                    logger.info(
                        f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                    )
                    break
            else:
                logger.warning(
                    f"Failed to find interface name between {node_i.node_id} and {node_j.node_id}"
                )
                raise ValueError(
                    "Current ibv backend requires all-to-all rdma connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeInfo,
    node_j: NodeInfo,
    cycle_digraph: Topology,
) -> Generator[tuple[str, bool]]:
    """Find all IP addresses that connect node i to node j, with thunderbolt flag."""
    for connection in cycle_digraph.list_connections():
        if (
            connection.local_node_id == node_i.node_id
            and connection.send_back_node_id == node_j.node_id
        ):
            yield connection.send_back_multiaddr.ip_address, connection.is_thunderbolt()


def _find_rdma_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if node_info.node_profile is None:
        return None

    logger.info(f"Searching {node_info.node_id} for ip {ip_address}:")
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in ["en2", "en3", "en4", "en5", "en6", "en7"]:
            continue
        logger.info(f" | {interface.name}: {interface.ip_address}")
        if interface.ip_address != ip_address:
            continue

        logger.info("Found")
        return f"rdma_{interface.name}"

    return None


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    """Find the interface name for an IP address on a node (any interface)."""
    if node_info.node_profile is None:
        return None

    for interface in node_info.node_profile.network_interfaces:
        if interface.ip_address == ip_address:
            return interface.name

    return None


def _find_ip_prioritised(
    node: NodeInfo, other_node: NodeInfo, cycle_digraph: Topology
) -> str | None:
    # TODO: Actually prioritize in the correct Ethernet > Wifi > Non-TB > TB order.
    """Find an IP address between nodes with prioritization.

    Priority order:
    1. en0 (Ethernet on Mac Studio, WiFi on MacBook)
    2. en1 (WiFi on Mac Studio, Ethernet on MacBook)
    3. Non-Thunderbolt connections
    4. Any other IP address
    """
    ips = list(_find_connection_ip(node, other_node, cycle_digraph))
    # We expect a unique iface -> ip mapping
    iface_map = {_find_interface_name_for_ip(ip, other_node): ip for ip, _ in ips}

    en0_ip = iface_map.get("en0")
    if en0_ip:
        return en0_ip

    en1_ip = iface_map.get("en1")
    if en1_ip:
        return en1_ip

    non_thunderbolt_ip = next(
        (ip for (ip, is_thunderbolt) in ips if not is_thunderbolt), None
    )

    if non_thunderbolt_ip:
        return non_thunderbolt_ip

    if ips:
        return ips[0][0]

    return None


def get_mlx_ring_hosts_by_node(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
    ephemeral_port: int,
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node in enumerate(selected_cycle):
        node_id = node.node_id
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(node, other_node, cycle_digraph)
            if connection_ip is None:
                logger.warning(
                    f"Failed to find prioritised connection IP between {node_id} and {other_node.node_id}"
                )
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    selected_cycle: list[NodeInfo],
    coordinator_port: int,
    cycle_digraph: Topology,
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX Jaccl (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    rank_0_node = selected_cycle[0]
    logger.info(f"Selecting coordinator from rank 0 node: {rank_0_node.node_id}")

    def get_ip_for_node(n: NodeInfo) -> str:
        if n.node_id == rank_0_node.node_id:
            return "0.0.0.0"

        for ip, _ in _find_connection_ip(n, rank_0_node, cycle_digraph):
            return ip

        logger.warning(
            f"Failed to find directly connected ip between {n.node_id} and {rank_0_node.node_id}"
        )
        raise ValueError("Current ibv backend requires all-to-all rdma connections")

    return {
        n.node_id: f"{get_ip_for_node(n)}:{coordinator_port}" for n in selected_cycle
    }
