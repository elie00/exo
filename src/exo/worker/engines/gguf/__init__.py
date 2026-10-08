"""GGUF/Ollama backend for EXO distributed inference.

This module provides local GGUF discovery, metadata, inference adapters and
experimental networking scaffolding. True weight sharding is not implemented.

Key Features:
- Automatic discovery of local Ollama models
- GGUF file parsing for model architecture and tensor info
- Layer-based sharding calculation for multi-node distribution
- Experimental TCP token relay with a versioned numeric payload format
- Model-card compatibility with EXO; master GGUF placement remains unsupported

Quick Start:
    # Discover available models
    from exo.worker.engines.gguf import discover_ollama_models
    models = discover_ollama_models()
    
    # Get sharding plan
    from exo.worker.engines.gguf import calculate_shard_assignments
    plan = calculate_shard_assignments(model.model_path, world_size=2)
    
    # Run distributed inference
    from exo.worker.engines.gguf.launcher import main as run_launcher
    run_launcher()

CLI Usage:
    # Discover models
    uv run python -m exo.worker.engines.gguf.cli discover
    
    # Show sharding plan
    uv run python -m exo.worker.engines.gguf.cli shard qwen3-coder:30b -n 2
    
    # Run distributed inference
    uv run python -m exo.worker.engines.gguf.launcher \\
        --model qwen3-coder:30b --rank 0 --world-size 2 \\
        --nodes "localhost:50100,192.168.1.11:50100"
"""

# Model discovery
# Distributed pipeline
from exo.worker.engines.gguf.distributed_pipeline import (
    ActivationPacket,
    ActivationTransport,
    DistributedGGUFPipeline,
    GGUFLayerExtractor,
    NodeRole,
    PipelineNodeConfig,
    TCPActivationTransport,
    create_distributed_pipeline,
)

# EXO integration
from exo.worker.engines.gguf.exo_integration import (
    OllamaModelProvider,
    discover_and_register_ollama_models,
    get_ollama_model_cards,
    get_ollama_provider,
    ollama_model_to_metadata,
)

# GGUF file parsing
from exo.worker.engines.gguf.gguf_loader import (
    GGMLType,
    GGUFModelInfo,
    GGUFValueType,
    TensorInfo,
    get_layer_count,
    load_gguf_metadata,
    load_gguf_metadata_async,
)

# Runner for EXO
from exo.worker.engines.gguf.gguf_runner import (
    LLAMA_CPP_AVAILABLE,
    GGUFRunner,
    gguf_runner_main,
)

# Sharding calculation
from exo.worker.engines.gguf.gguf_sharding import (
    GGUFLayerLoader,
    GGUFShardAssignment,
    GGUFShardingPlan,
    calculate_shard_assignments,
    estimate_shard_memory,
)
from exo.worker.engines.gguf.ollama_discovery import (
    OllamaModel,
    discover_ollama_models,
    discover_ollama_models_async,
    get_ollama_home,
    get_ollama_model_path,
    get_ollama_models_dir,
)

__all__ = [
    # Discovery
    "OllamaModel",
    "discover_ollama_models",
    "discover_ollama_models_async",
    "get_ollama_model_path",
    "get_ollama_home",
    "get_ollama_models_dir",
    # GGUF Parsing
    "GGUFModelInfo",
    "TensorInfo",
    "GGMLType",
    "GGUFValueType",
    "load_gguf_metadata",
    "load_gguf_metadata_async",
    "get_layer_count",
    # Sharding
    "GGUFShardAssignment",
    "GGUFShardingPlan",
    "calculate_shard_assignments",
    "estimate_shard_memory",
    "GGUFLayerLoader",
    # Distributed Pipeline
    "NodeRole",
    "PipelineNodeConfig",
    "ActivationPacket",
    "ActivationTransport",
    "TCPActivationTransport",
    "GGUFLayerExtractor",
    "DistributedGGUFPipeline",
    "create_distributed_pipeline",
    # EXO Integration
    "ollama_model_to_metadata",
    "discover_and_register_ollama_models",
    "get_ollama_model_cards",
    "OllamaModelProvider",
    "get_ollama_provider",
    # Runner
    "GGUFRunner",
    "gguf_runner_main",
    "LLAMA_CPP_AVAILABLE",
]

# Version
__version__ = "0.1.0"
