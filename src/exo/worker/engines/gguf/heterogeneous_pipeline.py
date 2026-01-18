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


# =============================================================================
# Network-Based Distributed Inference
# =============================================================================


@dataclass
class InferenceRequest:
    """Request for distributed inference."""
    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = True


@dataclass
class InferenceResponse:
    """Response from distributed inference."""
    request_id: str
    token: str
    done: bool = False
    error: str | None = None


class GGUFInferenceServer:
    """TCP server for remote GGUF inference.
    
    This server runs on a node and accepts inference requests from other nodes.
    It allows heterogeneous nodes (Mac/Dell) to participate in distributed inference.
    """
    
    def __init__(
        self,
        node: GGUFInferenceNode,
        host: str = "0.0.0.0",
        port: int = 50100,
    ):
        self.node = node
        self.host = host
        self.port = port
        self._server: asyncio.Server | None = None
        self._running = False
    
    async def start(self) -> None:
        """Start the inference server."""
        if not self.node.is_loaded:
            await self.node.load_model()
        
        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )
        self._running = True
        
        logger.info(f"GGUF Inference Server started on {self.host}:{self.port}")
        logger.info(f"  Node: {self.node.node_id} ({self.node.backend.value})")
    
    async def serve_forever(self) -> None:
        """Serve requests until stopped."""
        if self._server is None:
            await self.start()
        
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        addr = writer.get_extra_info("peername")
        logger.debug(f"Client connected from {addr}")
        
        try:
            while True:
                # Read request length (4 bytes, big-endian)
                length_bytes = await reader.readexactly(4)
                length = struct.unpack(">I", length_bytes)[0]
                
                # Read request JSON
                request_bytes = await reader.readexactly(length)
                request_data = json.loads(request_bytes.decode("utf-8"))
                
                request = InferenceRequest(
                    request_id=request_data["request_id"],
                    prompt=request_data["prompt"],
                    max_tokens=request_data.get("max_tokens", 256),
                    temperature=request_data.get("temperature", 0.7),
                    stream=request_data.get("stream", True),
                )
                
                logger.debug(f"Received request {request.request_id}")
                
                # Generate response
                try:
                    async for token in self.node.generate(
                        request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ):
                        response = InferenceResponse(
                            request_id=request.request_id,
                            token=token,
                            done=False,
                        )
                        await self._send_response(writer, response)
                    
                    # Send final response
                    response = InferenceResponse(
                        request_id=request.request_id,
                        token="",
                        done=True,
                    )
                    await self._send_response(writer, response)
                    
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    response = InferenceResponse(
                        request_id=request.request_id,
                        token="",
                        done=True,
                        error=str(e),
                    )
                    await self._send_response(writer, response)
        
        except asyncio.IncompleteReadError:
            logger.debug(f"Client {addr} disconnected")
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        response: InferenceResponse,
    ) -> None:
        """Send a response to the client."""
        response_data = {
            "request_id": response.request_id,
            "token": response.token,
            "done": response.done,
            "error": response.error,
        }
        response_bytes = json.dumps(response_data).encode("utf-8")
        
        # Send length + data
        writer.write(struct.pack(">I", len(response_bytes)))
        writer.write(response_bytes)
        await writer.drain()


class GGUFInferenceClient:
    """TCP client for remote GGUF inference.
    
    This client connects to a remote GGUFInferenceServer and sends
    inference requests.
    """
    
    def __init__(self, host: str, port: int = 50100):
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._request_counter = 0
    
    async def connect(self) -> None:
        """Connect to the remote server."""
        self._reader, self._writer = await asyncio.open_connection(
            self.host, self.port
        )
        self._connected = True
        logger.info(f"Connected to GGUF server at {self.host}:{self.port}")
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate text using the remote server."""
        if not self._connected:
            await self.connect()
        
        assert self._reader is not None
        assert self._writer is not None
        
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        
        # Send request
        request_data = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        request_bytes = json.dumps(request_data).encode("utf-8")
        
        self._writer.write(struct.pack(">I", len(request_bytes)))
        self._writer.write(request_bytes)
        await self._writer.drain()
        
        # Read responses
        while True:
            length_bytes = await self._reader.readexactly(4)
            length = struct.unpack(">I", length_bytes)[0]
            
            response_bytes = await self._reader.readexactly(length)
            response_data = json.loads(response_bytes.decode("utf-8"))
            
            if response_data.get("error"):
                raise RuntimeError(response_data["error"])
            
            if response_data["token"]:
                yield response_data["token"]
            
            if response_data["done"]:
                break


@dataclass
class RemoteNode:
    """A remote inference node accessed via TCP."""
    node_id: str
    host: str
    port: int = 50100
    client: GGUFInferenceClient | None = None
    
    async def connect(self) -> None:
        """Connect to the remote node."""
        self.client = GGUFInferenceClient(self.host, self.port)
        await self.client.connect()
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate text using the remote node."""
        if self.client is None:
            await self.connect()
        assert self.client is not None
        
        async for token in self.client.generate(prompt, max_tokens, temperature):
            yield token
    
    async def disconnect(self) -> None:
        """Disconnect from the remote node."""
        if self.client:
            await self.client.disconnect()


@dataclass
class NetworkedHeterogeneousCluster:
    """A heterogeneous cluster with both local and remote nodes.
    
    This class coordinates inference across:
    - Local nodes (direct llama-cpp-python access)
    - Remote nodes (accessed via TCP)
    
    Example:
        # Create cluster
        cluster = NetworkedHeterogeneousCluster()
        
        # Add local node (Mac with Metal)
        local_node = GGUFInferenceNode("mac", config)
        await local_node.load_model()
        cluster.add_local_node(local_node)
        
        # Add remote node (Dell with CUDA)
        cluster.add_remote_node("dell", "100.101.73.105", 50100)
        await cluster.connect_remote_nodes()
        
        # Generate using best available node
        async for token in cluster.generate("Hello!"):
            print(token, end="")
    """
    
    local_nodes: list[GGUFInferenceNode] = field(default_factory=list)
    remote_nodes: list[RemoteNode] = field(default_factory=list)
    current_idx: int = 0
    
    def add_local_node(self, node: GGUFInferenceNode) -> None:
        """Add a local inference node."""
        self.local_nodes.append(node)
        logger.info(f"Added local node: {node.node_id} ({node.backend.value})")
    
    def add_remote_node(self, node_id: str, host: str, port: int = 50100) -> None:
        """Add a remote inference node."""
        self.remote_nodes.append(RemoteNode(node_id, host, port))
        logger.info(f"Added remote node: {node_id} at {host}:{port}")
    
    async def connect_remote_nodes(self) -> None:
        """Connect to all remote nodes."""
        for node in self.remote_nodes:
            try:
                await node.connect()
            except Exception as e:
                logger.warning(f"Failed to connect to {node.node_id}: {e}")
    
    @property
    def total_nodes(self) -> int:
        return len(self.local_nodes) + len(self.remote_nodes)
    
    def _get_next_node(self) -> GGUFInferenceNode | RemoteNode:
        """Get the next node using round-robin."""
        all_nodes: list[GGUFInferenceNode | RemoteNode] = [
            *self.local_nodes,
            *self.remote_nodes,
        ]
        node = all_nodes[self.current_idx % len(all_nodes)]
        self.current_idx += 1
        return node
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate using load-balanced node selection."""
        node = self._get_next_node()
        
        if isinstance(node, GGUFInferenceNode):
            logger.debug(f"Using local node: {node.node_id}")
            async for token in node.generate(prompt, max_tokens, temperature):
                yield token
        else:
            logger.debug(f"Using remote node: {node.node_id}")
            async for token in node.generate(prompt, max_tokens, temperature):
                yield token
    
    async def shutdown(self) -> None:
        """Shutdown all nodes."""
        for node in self.local_nodes:
            node.unload()
        for node in self.remote_nodes:
            await node.disconnect()


async def test_heterogeneous_inference():
    """Test heterogeneous inference on the local node."""
    caps = NodeCapabilities.detect("local")
    print(f"Node capabilities: {caps}")


def run_inference_server():
    """CLI entry point for running an inference server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GGUF inference server")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=50100, help="Port to bind")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1=all)")
    
    args = parser.parse_args()
    
    async def main():
        config = GGUFModelConfig(
            model_path=Path(args.model),
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )
        
        caps = NodeCapabilities.detect("server")
        node = GGUFInferenceNode(f"server-{caps.backend.value}", config)
        server = GGUFInferenceServer(node, args.host, args.port)
        
        print(f"Starting GGUF Inference Server")
        print(f"  Model: {args.model}")
        print(f"  Backend: {caps.backend.value}")
        print(f"  Address: {args.host}:{args.port}")
        
        await server.serve_forever()
    
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(test_heterogeneous_inference())
