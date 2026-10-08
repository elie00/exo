"""Experimental GGUF local inference and RPC worker process scaffolding.

Local completion calls use llama-cpp-python. The main process does not yet wire
RPC worker addresses into that backend; requesting workers fails explicitly.
No distributed weight loading or NVIDIA inference qualification is claimed.
"""

import asyncio
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, TypedDict

from loguru import logger

from exo.shared.types.memory import Memory
from exo.worker.engines.gguf.gguf_loader import GGUFModelInfo, load_gguf_metadata
from exo.worker.engines.gguf.llama_backend import LlamaModel, load_llama_model


class GGUFNodeRole(Enum):
    """Role of a node in the distributed inference cluster."""

    MAIN = "main"  # Receives requests and coordinates inference
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


class GPUInfo(TypedDict):
    available: bool
    nvidia_available: bool
    device_name: str | None
    vram_total_mb: int
    vram_free_mb: int
    recommended_gpu_layers: int
    cuda_version: str | None
    driver_version: str | None


class GGUFInferenceEngine:
    """Local llama.cpp inference adapter with unqualified RPC worker scaffolding.

    NVML can inform requested GPU-layer counts; it does not establish backend
    CUDA support or qualify inference on a physical NVIDIA device.
    """

    def __init__(self, config: GGUFRPCConfig):
        self.config = config
        self._model: LlamaModel | None = None
        self._tokenizer = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._running = False
        self._gpu_info = self._detect_gpu()

    def _detect_gpu(self) -> GPUInfo:
        """Detect available GPU and return configuration info."""
        gpu_info: GPUInfo = {
            "available": False,
            "nvidia_available": False,
            "device_name": None,
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "recommended_gpu_layers": -1,
            "cuda_version": None,
            "driver_version": None,
        }

        try:
            from exo.utils.info_gatherer.nvidia_monitor import (
                get_metrics,
                is_nvidia_available,
            )

            if is_nvidia_available():
                metrics = get_metrics()
                gpu_info["available"] = True
                gpu_info["nvidia_available"] = True
                gpu_info["vram_total_mb"] = metrics.gpu_memory_total_mb
                gpu_info["vram_free_mb"] = metrics.gpu_memory_free_mb
                gpu_info["cuda_version"] = metrics.cuda_version
                gpu_info["driver_version"] = metrics.driver_version

                if metrics.gpus:
                    gpu_info["device_name"] = metrics.gpus[0].name

                # Calculate recommended GPU layers based on available VRAM
                # Rough estimate: ~100MB per layer for quantized models
                vram_for_layers = gpu_info["vram_free_mb"] - 500  # Reserve 500MB
                gpu_info["recommended_gpu_layers"] = max(0, vram_for_layers // 100)

                logger.info(
                    f"NVIDIA device detected: {gpu_info['device_name']} "
                    f"({gpu_info['vram_free_mb']}MB free VRAM, "
                    f"recommended layers: {gpu_info['recommended_gpu_layers']})"
                )
        except ImportError:
            logger.debug("pynvml not available for GPU detection")
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")

        return gpu_info

    def _get_optimal_gpu_layers(
        self, model_info: Optional[GGUFModelInfo] = None
    ) -> int:
        """Calculate optimal number of GPU layers based on model size and VRAM."""
        if not self._gpu_info["nvidia_available"]:
            return 0

        # If config specifies layers, use that
        if self.config.n_gpu_layers >= 0:
            return self.config.n_gpu_layers

        # If we have model info, calculate based on model size
        if model_info and self._gpu_info["vram_free_mb"] > 0:
            # Estimate memory per layer
            model_size_mb = model_info.size_bytes / (1024 * 1024)
            n_layers = model_info.n_layers
            memory_per_layer = model_size_mb / n_layers if n_layers > 0 else 100

            available_vram = self._gpu_info["vram_free_mb"] - 500  # Reserve 500MB
            max_layers = (
                max(0, int(available_vram / memory_per_layer))
                if memory_per_layer > 0
                else 0
            )

            logger.info(
                f"Model: {n_layers} layers, ~{memory_per_layer:.0f}MB/layer. "
                f"VRAM: {self._gpu_info['vram_free_mb']}MB. Offloading {min(max_layers, n_layers)} layers to GPU."
            )
            return min(max_layers, n_layers)

        # Fallback to recommended layers
        return self._gpu_info.get("recommended_gpu_layers", -1)

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
        logger.info(
            f"Starting GGUF RPC worker on {self.config.rpc_host}:{self.config.rpc_port}"
        )

        # Check if llama-server is available
        llama_server = self._find_llama_server()
        if not llama_server:
            raise RuntimeError(
                "llama-server not found. Install llama.cpp or ensure it's in PATH."
            )

        cmd = [
            str(llama_server),
            "--model",
            str(self.config.model_path),
            "--host",
            self.config.rpc_host,
            "--port",
            str(self.config.rpc_port),
            "--ctx-size",
            str(self.config.context_size),
            "--batch-size",
            str(self.config.batch_size),
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

        logger.info("GGUF RPC worker started successfully")

    async def _start_main_server(self) -> None:
        """Start as the main inference server.

        Loads the model with RPC workers configured and optimal GPU offloading.
        """

        if self.config.worker_addresses:
            raise NotImplementedError(
                "GGUF RPC workers are not connected to the main backend yet"
            )

        logger.info(
            f"Starting GGUF main server with model: {self.config.model_path.name}"
        )

        # Build RPC server list if we have workers
        rpc_servers = None
        if self.config.worker_addresses:
            rpc_servers = ",".join(self.config.worker_addresses)
            logger.info(f"Connecting to RPC workers: {rpc_servers}")

        # Load model metadata to calculate optimal GPU layers
        model_info: Optional[GGUFModelInfo] = None
        try:
            model_info = load_gguf_metadata(self.config.model_path)
            logger.info(
                f"Model info: {model_info.architecture}, "
                f"{model_info.n_layers} layers, "
                f"{model_info.size_bytes / (1024**3):.1f}GB"
            )
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")

        # Calculate optimal GPU layers
        n_gpu_layers = self._get_optimal_gpu_layers(model_info)

        # Log GPU configuration
        if self._gpu_info["nvidia_available"]:
            logger.info(
                f"Requesting {n_gpu_layers} GPU layers on "
                f"{self._gpu_info['device_name']} "
                f"(CUDA {self._gpu_info.get('cuda_version', 'N/A')})"
            )
        else:
            logger.info("No NVIDIA telemetry available; requesting CPU inference")

        # Load model with llama-cpp-python
        # Note: RPC support requires llama-cpp-python built with RPC enabled
        self._model = load_llama_model(
            model_path=str(self.config.model_path),
            n_ctx=self.config.context_size,
            n_batch=self.config.batch_size,
            n_threads=self.config.threads if self.config.threads > 0 else None,
            n_gpu_layers=n_gpu_layers,
            # rpc_servers=rpc_servers,  # Enable when RPC is configured
            verbose=True,
        )

        self._running = True
        logger.info(
            f"GGUF main server started successfully "
            f"(GPU layers: {n_gpu_layers}, context: {self.config.context_size})"
        )

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
            self._model.close()
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
            for output in self._model.complete_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            ):
                token = output.choices[0].text
                yield token
        else:
            output = self._model.complete(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )
            yield output.choices[0].text

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
            for output in self._model.chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                delta = output.choices[0].delta
                if delta.content is not None:
                    yield delta.content
        else:
            output = self._model.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            yield output.choices[0].message.content or ""

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
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(
                        url, timeout=aiohttp.ClientTimeout(total=1)
                    ) as response,
                ):
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
