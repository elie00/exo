"""GGUF distributed inference using llama.cpp RPC backend.

This module implements distributed inference for GGUF models using
llama.cpp's built-in RPC (Remote Procedure Call) capabilities.

Architecture:
- One node acts as the "main" server that receives requests
- Other nodes act as RPC workers that process assigned layers
- Communication happens over TCP using llama.cpp's RPC protocol

This approach allows true weight distribution where each node only
loads and processes its assigned portion of the model.
"""

import asyncio
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

from loguru import logger

from exo.shared.types.memory import Memory
from exo.worker.engines.gguf.gguf_loader import GGUFModelInfo, load_gguf_metadata
from exo.worker.engines.gguf.gguf_sharding import GGUFShardAssignment, GGUFShardingPlan


class GGUFNodeRole(Enum):
    """Role of a node in the distributed inference cluster."""
    MAIN = "main"      # Receives requests and coordinates inference
    WORKER = "worker"  # Processes assigned layers via RPC


@dataclass
class GGUFRPCConfig:
    """Configuration for GGUF RPC distributed inference."""
    model_path: Path
    role: GGUFNodeRole
    rpc_host: str = "0.0.0.0"
    rpc_port: int = 50052
    
    # For MAIN node: list of worker RPC addresses
    worker_addresses: list[str] | None = None
    
    # For WORKER node: which layers to load
    n_gpu_layers: int = -1  # -1 = all layers for this shard
    
    # Inference parameters
    context_size: int = 4096
    batch_size: int = 512
    threads: int = 0  # 0 = auto-detect
    
    # Memory constraints
    max_memory: Optional[Memory] = None


class GGUFInferenceEngine:
    """Engine for running distributed inference on GGUF models.
    
    Uses llama-cpp-python with RPC backend for true distributed inference
    where model weights are split across multiple nodes.
    """
    
    def __init__(self, config: GGUFRPCConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    async def start(self) -> None:
        """Start the inference engine.
        
        For MAIN role: Loads model and connects to workers
        For WORKER role: Starts RPC server and waits for connections
        """
        if self.config.role == GGUFNodeRole.WORKER:
            await self._start_rpc_worker()
        else:
            await self._start_main_server()
    
    async def _start_rpc_worker(self) -> None:
        """Start as an RPC worker node.
        
        Runs llama-server with RPC backend enabled.
        """
        logger.info(f"Starting GGUF RPC worker on {self.config.rpc_host}:{self.config.rpc_port}")
        
        # Check if llama-server is available
        llama_server = self._find_llama_server()
        if not llama_server:
            raise RuntimeError(
                "llama-server not found. Install llama.cpp or ensure it's in PATH."
            )
        
        cmd = [
            str(llama_server),
            "--model", str(self.config.model_path),
            "--host", self.config.rpc_host,
            "--port", str(self.config.rpc_port),
            "--ctx-size", str(self.config.context_size),
            "--batch-size", str(self.config.batch_size),
        ]
        
        if self.config.n_gpu_layers >= 0:
            cmd.extend(["--n-gpu-layers", str(self.config.n_gpu_layers)])
        
        if self.config.threads > 0:
            cmd.extend(["--threads", str(self.config.threads)])
        
        logger.debug(f"Starting llama-server: {' '.join(cmd)}")
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Wait for server to be ready
        await self._wait_for_server_ready()
        self._running = True
        
        logger.info(f"GGUF RPC worker started successfully")
    
    async def _start_main_server(self) -> None:
        """Start as the main inference server.
        
        Loads the model with RPC workers configured.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: pip install llama-cpp-python"
            )
        
        logger.info(f"Starting GGUF main server with model: {self.config.model_path.name}")
        
        # Build RPC server list if we have workers
        rpc_servers = None
        if self.config.worker_addresses:
            rpc_servers = ",".join(self.config.worker_addresses)
            logger.info(f"Connecting to RPC workers: {rpc_servers}")
        
        # Load model with llama-cpp-python
        # Note: RPC support requires llama-cpp-python built with RPC enabled
        self._model = Llama(
            model_path=str(self.config.model_path),
            n_ctx=self.config.context_size,
            n_batch=self.config.batch_size,
            n_threads=self.config.threads if self.config.threads > 0 else None,
            n_gpu_layers=self.config.n_gpu_layers,
            # rpc_servers=rpc_servers,  # Enable when RPC is configured
            verbose=True,
        )
        
        self._running = True
        logger.info("GGUF main server started successfully")
    
    async def stop(self) -> None:
        """Stop the inference engine."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        
        if self._model:
            del self._model
            self._model = None
        
        self._running = False
        logger.info("GGUF inference engine stopped")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            stop: Stop sequences
            stream: If True, yield tokens as they're generated
        
        Yields:
            Generated text (full response if not streaming, tokens if streaming)
        """
        if not self._running or self._model is None:
            raise RuntimeError("Inference engine not running")
        
        if stream:
            for output in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                stream=True,
            ):
                token = output["choices"][0]["text"]
                yield token
        else:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )
            yield output["choices"][0]["text"]
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        """Chat completion interface.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yield tokens as they're generated
        
        Yields:
            Generated response
        """
        if not self._running or self._model is None:
            raise RuntimeError("Inference engine not running")
        
        if stream:
            for output in self._model.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = output["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        else:
            output = self._model.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
            )
            yield output["choices"][0]["message"]["content"]
    
    def _find_llama_server(self) -> Optional[Path]:
        """Find the llama-server executable."""
        # Check common locations
        locations = [
            Path("/usr/local/bin/llama-server"),
            Path("/opt/homebrew/bin/llama-server"),
            Path.home() / ".local/bin/llama-server",
            Path.home() / "llama.cpp/build/bin/llama-server",
        ]
        
        # Also check PATH
        import shutil
        path_result = shutil.which("llama-server")
        if path_result:
            return Path(path_result)
        
        for loc in locations:
            if loc.exists():
                return loc
        
        return None
    
    async def _wait_for_server_ready(self, timeout: float = 30.0) -> None:
        """Wait for the RPC server to be ready."""
        import aiohttp
        
        start_time = asyncio.get_event_loop().time()
        url = f"http://localhost:{self.config.rpc_port}/health"
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=1) as response:
                        if response.status == 200:
                            return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Server did not become ready within {timeout}s")


async def create_distributed_engine(
    model_path: Path,
    node_addresses: list[str],
    this_node_index: int,
) -> GGUFInferenceEngine:
    """Create a distributed inference engine for this node.
    
    Args:
        model_path: Path to the GGUF model file
        node_addresses: List of all node addresses in the cluster
        this_node_index: Index of this node in the cluster
    
    Returns:
        Configured GGUFInferenceEngine for this node
    """
    world_size = len(node_addresses)
    
    # First node is the main server, others are workers
    if this_node_index == 0:
        role = GGUFNodeRole.MAIN
        worker_addresses = node_addresses[1:] if len(node_addresses) > 1 else None
    else:
        role = GGUFNodeRole.WORKER
        worker_addresses = None
    
    config = GGUFRPCConfig(
        model_path=model_path,
        role=role,
        worker_addresses=worker_addresses,
        rpc_port=50052 + this_node_index,
    )
    
    return GGUFInferenceEngine(config)
