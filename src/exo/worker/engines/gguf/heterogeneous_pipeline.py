"""Heterogeneous Pipeline Parallelism for GGUF models.

This module enables distributed inference across different hardware backends
(Metal on Mac, CUDA on NVIDIA) using llama-cpp-python.

Architecture:
    ┌─────────────────┐         ┌─────────────────┐
    │   Mac (Metal)   │◄───────►│  Dell (CUDA)    │
    │   llama.cpp     │  TCP    │  llama.cpp      │
    │   Layers 0-N/2  │ tensors │  Layers N/2-N   │
    └─────────────────┘         └─────────────────┘

Key Components:
    1. GGUFShardLoader: Loads a subset of layers from a GGUF model
    2. ActivationExchange: TCP-based tensor exchange between nodes
    3. HeterogeneousPipeline: Coordinates distributed inference

Limitations:
    - llama.cpp doesn't natively support layer-by-layer execution
    - Current implementation uses model replication with request routing
    - True pipeline parallelism requires llama.cpp modifications

Future Work:
    - Implement activation extraction at layer boundaries
    - Add support for speculative decoding across nodes
    - Optimize tensor serialization for low latency
"""

from __future__ import annotations

import asyncio
import json
import platform
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from llama_cpp import Llama


class BackendType(str, Enum):
    """Supported inference backends."""
    METAL = "metal"      # Apple Silicon
    CUDA = "cuda"        # NVIDIA GPU
    CPU = "cpu"          # CPU fallback


@dataclass(frozen=True)
class NodeCapabilities:
    """Hardware capabilities of a node."""
    node_id: str
    backend: BackendType
    memory_mb: int
    gpu_name: str | None = None
    max_layers: int | None = None
    
    @classmethod
    def detect(cls, node_id: str) -> "NodeCapabilities":
        """Detect capabilities of the current node."""
        system = platform.system().lower()
        
        if system == "darwin":
            # macOS - check for Apple Silicon
            import subprocess
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                is_apple_silicon = "Apple" in result.stdout
            except Exception:
                is_apple_silicon = platform.machine() == "arm64"
            
            if is_apple_silicon:
                # Get unified memory
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    memory_mb = int(result.stdout.strip()) // (1024 * 1024)
                except Exception:
                    memory_mb = 8192  # Default 8GB
                
                return cls(
                    node_id=node_id,
                    backend=BackendType.METAL,
                    memory_mb=memory_mb,
                    gpu_name="Apple Silicon",
                )
        
        elif system in ("linux", "windows"):
            # Check for NVIDIA GPU
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode()
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_mb = mem_info.total // (1024 * 1024)
                    pynvml.nvmlShutdown()
                    
                    return cls(
                        node_id=node_id,
                        backend=BackendType.CUDA,
                        memory_mb=memory_mb,
                        gpu_name=name,
                    )
            except Exception:
                pass
        
        # Fallback to CPU
        import psutil
        memory_mb = psutil.virtual_memory().total // (1024 * 1024)
        
        return cls(
            node_id=node_id,
            backend=BackendType.CPU,
            memory_mb=memory_mb,
        )


class TensorMessage(BaseModel):
    """Message containing tensor data for inter-node exchange."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    sequence_id: int
    layer_idx: int
    shape: tuple[int, ...]
    dtype: str
    data_base64: str  # Base64 encoded numpy array
    
    @classmethod
    def from_numpy(cls, sequence_id: int, layer_idx: int, arr: np.ndarray) -> "TensorMessage":
        """Create message from numpy array."""
        import base64
        return cls(
            sequence_id=sequence_id,
            layer_idx=layer_idx,
            shape=arr.shape,
            dtype=str(arr.dtype),
            data_base64=base64.b64encode(arr.tobytes()).decode("ascii"),
        )
    
    def to_numpy(self) -> np.ndarray:
        """Convert message back to numpy array."""
        import base64
        data = base64.b64decode(self.data_base64)
        return np.frombuffer(data, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass
class GGUFModelConfig:
    """Configuration for loading a GGUF model."""
    model_path: Path
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int | None = None
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    use_mmap: bool = True
    use_mlock: bool = False
    verbose: bool = False


class GGUFInferenceNode:
    """A single inference node running llama-cpp-python.
    
    This class wraps llama-cpp-python and provides:
    - Automatic backend detection (Metal/CUDA/CPU)
    - Model loading with proper GPU offloading
    - Inference with streaming support
    """
    
    def __init__(
        self,
        node_id: str,
        config: GGUFModelConfig,
    ):
        self.node_id = node_id
        self.config = config
        self.capabilities = NodeCapabilities.detect(node_id)
        self._model: "Llama | None" = None
        self._loaded = False
    
    @property
    def backend(self) -> BackendType:
        return self.capabilities.backend
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None
    
    async def load_model(self) -> None:
        """Load the GGUF model with appropriate backend settings."""
        if self._loaded:
            return
        
        logger.info(f"Loading model on {self.node_id} ({self.backend.value})")
        logger.info(f"  Model: {self.config.model_path}")
        logger.info(f"  Memory: {self.capabilities.memory_mb} MB")
        
        # Import here to avoid loading llama_cpp until needed
        from llama_cpp import Llama
        
        # Configure GPU layers based on backend
        n_gpu_layers = self.config.n_gpu_layers
        if self.backend == BackendType.CPU:
            n_gpu_layers = 0
        
        # Load model
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: Llama(
                model_path=str(self.config.model_path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=self.config.n_threads,
                n_gpu_layers=n_gpu_layers,
                use_mmap=self.config.use_mmap,
                use_mlock=self.config.use_mlock,
                verbose=self.config.verbose,
            )
        )
        
        self._loaded = True
        logger.info(f"Model loaded on {self.node_id}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Generate text from prompt."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        assert self._model is not None
        
        loop = asyncio.get_event_loop()
        
        if stream:
            # Streaming generation
            def _generate():
                return self._model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
            
            generator = await loop.run_in_executor(None, _generate)
            
            for output in generator:
                text = output["choices"][0].get("text", "")
                if text:
                    yield text
        else:
            # Non-streaming generation
            def _generate():
                return self._model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
            
            result = await loop.run_in_executor(None, _generate)
            yield result["choices"][0]["text"]
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """Chat completion."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        assert self._model is not None
        
        loop = asyncio.get_event_loop()
        
        if stream:
            def _chat():
                return self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
            
            generator = await loop.run_in_executor(None, _chat)
            
            for output in generator:
                delta = output["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
        else:
            def _chat():
                return self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
            
            result = await loop.run_in_executor(None, _chat)
            yield result["choices"][0]["message"]["content"]
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info(f"Model unloaded from {self.node_id}")


@dataclass
class HeterogeneousCluster:
    """Manages a cluster of heterogeneous inference nodes.
    
    This class coordinates inference across nodes with different backends
    (Metal, CUDA, CPU) using a round-robin load balancing strategy.
    
    For true pipeline parallelism (layer sharding), see the TODO comments
    below for the required llama.cpp modifications.
    """
    
    nodes: list[GGUFInferenceNode] = field(default_factory=list)
    current_node_idx: int = 0
    
    def add_node(self, node: GGUFInferenceNode) -> None:
        """Add a node to the cluster."""
        self.nodes.append(node)
        logger.info(f"Added node {node.node_id} ({node.backend.value}) to cluster")
    
    async def initialize(self) -> None:
        """Initialize all nodes and load models."""
        logger.info(f"Initializing heterogeneous cluster with {len(self.nodes)} nodes")
        
        # Load models in parallel
        await asyncio.gather(*[node.load_model() for node in self.nodes])
        
        logger.info("Cluster initialized:")
        for node in self.nodes:
            logger.info(f"  {node.node_id}: {node.backend.value} ({node.capabilities.memory_mb} MB)")
    
    def get_nodes_by_backend(self, backend: BackendType) -> list[GGUFInferenceNode]:
        """Get all nodes with a specific backend."""
        return [n for n in self.nodes if n.backend == backend]
    
    def _select_node(self) -> GGUFInferenceNode:
        """Select the next node using round-robin."""
        node = self.nodes[self.current_node_idx]
        self.current_node_idx = (self.current_node_idx + 1) % len(self.nodes)
        return node
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate text using a node from the cluster."""
        node = self._select_node()
        logger.debug(f"Routing request to {node.node_id} ({node.backend.value})")
        
        async for token in node.generate(prompt, max_tokens, temperature):
            yield token
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Chat completion using a node from the cluster."""
        node = self._select_node()
        logger.debug(f"Routing chat to {node.node_id} ({node.backend.value})")
        
        async for token in node.chat(messages, max_tokens, temperature):
            yield token
    
    async def parallel_generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate responses for multiple prompts in parallel."""
        async def _generate_one(node: GGUFInferenceNode, prompt: str) -> str:
            tokens = []
            async for token in node.generate(prompt, max_tokens, temperature):
                tokens.append(token)
            return "".join(tokens)
        
        tasks = []
        for i, prompt in enumerate(prompts):
            node = self.nodes[i % len(self.nodes)]
            tasks.append(_generate_one(node, prompt))
        
        return await asyncio.gather(*tasks)
    
    def shutdown(self) -> None:
        """Shutdown all nodes."""
        for node in self.nodes:
            node.unload()


# =============================================================================
# Pipeline Parallelism (Future Implementation)
# =============================================================================
# 
# True pipeline parallelism requires modifications to llama.cpp to:
# 1. Export intermediate activations at layer boundaries
# 2. Accept activations as input instead of tokens
# 3. Run only a subset of layers
#
# The architecture would be:
#
# Node 1 (Mac/Metal):
#   - Tokenize input
#   - Run layers 0 to N/2
#   - Send activation tensor to Node 2
#
# Node 2 (Dell/CUDA):
#   - Receive activation tensor from Node 1
#   - Run layers N/2 to N
#   - Decode output tokens
#   - Send tokens back
#
# This requires a custom fork of llama.cpp with these capabilities.
# See: https://github.com/ggerganov/llama.cpp/discussions/
#
# For now, we use model replication with request-level parallelism.
# =============================================================================


async def test_heterogeneous_inference():
    """Test heterogeneous inference on the local node."""
    # Detect capabilities
    caps = NodeCapabilities.detect("local")
    print(f"Node capabilities: {caps}")
    
    # This would need a real GGUF model path
    # config = GGUFModelConfig(model_path=Path("model.gguf"))
    # node = GGUFInferenceNode("local", config)
    # await node.load_model()
    # 
    # async for token in node.generate("Hello, "):
    #     print(token, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(test_heterogeneous_inference())
