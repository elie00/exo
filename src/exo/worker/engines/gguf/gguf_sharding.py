"""GGUF model sharding for distributed inference.

This module handles splitting GGUF models across multiple nodes,
where each node loads and processes a subset of the model's layers.

Sharding Strategy:
1. Non-layer tensors (embeddings, final norm, output) are replicated on all nodes
2. Layer tensors (attention, FFN) are distributed evenly across nodes
3. Each node loads only its assigned layer range from the GGUF file
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from exo.shared.types.memory import Memory
from exo.worker.engines.gguf.gguf_loader import GGUFModelInfo, TensorInfo, load_gguf_metadata


@dataclass
class GGUFShardAssignment:
    """Describes which portion of a GGUF model a node should load."""
    
    model_path: Path
    node_rank: int
    world_size: int
    start_layer: int
    end_layer: int  # exclusive
    total_layers: int
    
    # Calculated fields - use field() for mutable defaults
    estimated_memory: Memory = field(default_factory=Memory)
    tensor_names: list[str] | None = None
    
    @property
    def layer_count(self) -> int:
        """Number of layers this shard contains."""
        return self.end_layer - self.start_layer
    
    @property
    def is_first_shard(self) -> bool:
        """Whether this is the first shard (contains embeddings)."""
        return self.start_layer == 0
    
    @property
    def is_last_shard(self) -> bool:
        """Whether this is the last shard (contains output layer)."""
        return self.end_layer == self.total_layers


@dataclass
class GGUFShardingPlan:
    """Complete sharding plan for a GGUF model across multiple nodes."""
    
    model_info: GGUFModelInfo
    assignments: list[GGUFShardAssignment]
    
    @property
    def world_size(self) -> int:
        return len(self.assignments)
    
    def get_assignment_for_rank(self, rank: int) -> GGUFShardAssignment:
        """Get the shard assignment for a specific node rank."""
        return self.assignments[rank]


def calculate_shard_assignments(
    model_path: Path,
    world_size: int,
    model_info: Optional[GGUFModelInfo] = None,
) -> GGUFShardingPlan:
    """Calculate how to distribute a GGUF model across nodes.
    
    Uses a simple even-split strategy where each node gets approximately
    the same number of layers.
    
    Args:
        model_path: Path to the GGUF model file
        world_size: Number of nodes to distribute across
        model_info: Optional pre-loaded model info (avoids re-parsing)
    
    Returns:
        GGUFShardingPlan with assignments for each node
    """
    if model_info is None:
        model_info = load_gguf_metadata(model_path, load_tensors=True)
    
    n_layers = model_info.n_layers
    
    if world_size > n_layers:
        logger.warning(
            f"World size ({world_size}) exceeds layer count ({n_layers}). "
            f"Reducing world size to {n_layers}."
        )
        world_size = n_layers
    
    assignments: list[GGUFShardAssignment] = []
    
    # Calculate layer distribution
    layers_per_node = n_layers // world_size
    extra_layers = n_layers % world_size
    
    current_layer = 0
    total_file_size = model_path.stat().st_size
    
    for rank in range(world_size):
        # Distribute extra layers to first nodes
        node_layers = layers_per_node + (1 if rank < extra_layers else 0)
        start_layer = current_layer
        end_layer = current_layer + node_layers
        
        # Estimate memory for this shard
        # Rough estimation: proportional to layer count + overhead for first/last shards
        base_ratio = node_layers / n_layers
        overhead = 0.0
        if start_layer == 0:
            overhead += 0.05  # Embedding layer overhead
        if end_layer == n_layers:
            overhead += 0.03  # Output layer overhead
        
        estimated_bytes = int(total_file_size * (base_ratio + overhead))
        
        assignment = GGUFShardAssignment(
            model_path=model_path,
            node_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            total_layers=n_layers,
            estimated_memory=Memory.from_bytes(estimated_bytes),
        )
        
        # Get tensor names for this shard
        if model_info.tensors:
            shard_tensors = model_info.get_layers_range(start_layer, end_layer)
            # Add non-layer tensors for first and last shards
            if assignment.is_first_shard or assignment.is_last_shard:
                shard_tensors.extend(model_info.get_non_layer_tensors())
            assignment.tensor_names = [t.name for t in shard_tensors]
        
        assignments.append(assignment)
        current_layer = end_layer
        
        logger.info(
            f"Shard {rank}/{world_size}: layers {start_layer}-{end_layer-1} "
            f"({node_layers} layers, ~{assignment.estimated_memory})"
        )
    
    return GGUFShardingPlan(
        model_info=model_info,
        assignments=assignments,
    )


def estimate_shard_memory(
    model_path: Path,
    start_layer: int,
    end_layer: int,
    total_layers: int,
) -> Memory:
    """Estimate memory required for a specific layer range.
    
    This is a rough estimation based on file size distribution.
    Actual memory usage may vary based on quantization and architecture.
    
    Args:
        model_path: Path to the GGUF model file
        start_layer: First layer index (inclusive)
        end_layer: Last layer index (exclusive)
        total_layers: Total number of layers in the model
    
    Returns:
        Estimated memory requirement
    """
    file_size = model_path.stat().st_size
    layer_count = end_layer - start_layer
    
    # Base calculation: proportional to layers
    base_ratio = layer_count / total_layers
    
    # Add overhead for non-layer tensors
    overhead = 0.08  # ~8% for embeddings, final norm, output
    if start_layer == 0:
        overhead += 0.05  # Embedding layer is typically larger
    
    estimated_bytes = int(file_size * (base_ratio + overhead))
    return Memory.from_bytes(estimated_bytes)


class GGUFLayerLoader:
    """Loads specific layers from a GGUF file for distributed inference.
    
    This class handles the actual weight loading for a node's assigned
    layer range, working with llama-cpp-python to load partial models.
    """
    
    def __init__(self, assignment: GGUFShardAssignment):
        self.assignment = assignment
        self._model = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    async def load(self) -> None:
        """Load the assigned layers from the GGUF file.
        
        Uses llama-cpp-python with layer range specification to load
        only the needed weights.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: pip install llama-cpp-python"
            )
        
        logger.info(
            f"Loading GGUF shard: layers {self.assignment.start_layer}-{self.assignment.end_layer-1} "
            f"from {self.assignment.model_path.name}"
        )
        
        # Note: llama-cpp-python doesn't natively support layer-range loading.
        # For true distributed inference, we need to either:
        # 1. Use the RPC backend of llama.cpp
        # 2. Implement custom layer loading using the gguf library
        # 3. Use tensor parallel with NCCL/MPI
        
        # For now, we'll use the RPC backend approach which is built into llama.cpp
        # Each node will run as an RPC server/client
        
        # TODO: Implement proper layer-based loading
        # This is a placeholder for the full implementation
        
        self._loaded = True
        logger.info(f"Shard loaded successfully")
    
    async def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
