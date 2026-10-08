"""Experimental TCP token relay for GGUF models.

The head currently runs complete llama.cpp generation and sends token text in
numeric payloads. Other nodes relay this output; they do not execute assigned
transformer layers. Layer extraction selects metadata only. This is scaffolding
for future distributed inference, not a working weight-sharded backend.
"""

import asyncio
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from loguru import logger
from pydantic import JsonValue

from exo.worker.engines.gguf.activation_wire import (
    MAX_PACKET_BYTES,
    ActivationEnvelope,
    ArrayPayload,
)


class NodeRole(Enum):
    """Role of a node in the pipeline."""

    HEAD = "head"  # First node: handles embeddings
    MIDDLE = "middle"  # Middle nodes: process layers
    TAIL = "tail"  # Last node: handles output + generation


@dataclass
class PipelineNodeConfig:
    """Configuration for a node in the distributed pipeline."""

    node_id: str
    rank: int
    world_size: int

    # Model info
    model_path: Path
    start_layer: int
    end_layer: int
    total_layers: int

    # Network config
    listen_host: str = "0.0.0.0"
    listen_port: int = 50100

    # Previous node (for receiving activations)
    prev_node_host: Optional[str] = None
    prev_node_port: Optional[int] = None

    # Next node (for sending activations)
    next_node_host: Optional[str] = None
    next_node_port: Optional[int] = None

    # Inference params
    context_size: int = 4096
    batch_size: int = 512

    @property
    def role(self) -> NodeRole:
        if self.rank == 0:
            return NodeRole.HEAD
        elif self.rank == self.world_size - 1:
            return NodeRole.TAIL
        return NodeRole.MIDDLE

    @property
    def is_head(self) -> bool:
        return self.role == NodeRole.HEAD

    @property
    def is_tail(self) -> bool:
        return self.role == NodeRole.TAIL


@dataclass
class ActivationPacket:
    """Packet containing hidden states passed between nodes."""

    request_id: str
    sequence_position: int
    hidden_states: np.ndarray
    attention_mask: Optional[np.ndarray] = None
    position_ids: Optional[np.ndarray] = None
    is_final: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def serialize(self) -> bytes:
        """Serialize the packet for network transmission."""
        envelope = ActivationEnvelope(
            request_id=self.request_id,
            sequence_position=self.sequence_position,
            hidden_states=ArrayPayload.from_array(self.hidden_states),
            attention_mask=ArrayPayload.from_array(self.attention_mask)
            if self.attention_mask is not None
            else None,
            position_ids=ArrayPayload.from_array(self.position_ids)
            if self.position_ids is not None
            else None,
            is_final=self.is_final,
            metadata=self.metadata,
        )
        data = envelope.model_dump_json().encode("utf-8")
        if len(data) > MAX_PACKET_BYTES:
            raise ValueError("Activation packet exceeds size limit")
        return struct.pack(">I", len(data)) + data

    @classmethod
    def deserialize(cls, data: bytes) -> "ActivationPacket":
        if len(data) > MAX_PACKET_BYTES:
            raise ValueError("Activation packet exceeds size limit")
        envelope = ActivationEnvelope.model_validate_json(data)
        return cls(
            request_id=envelope.request_id,
            sequence_position=envelope.sequence_position,
            hidden_states=envelope.hidden_states.to_array(),
            attention_mask=envelope.attention_mask.to_array()
            if envelope.attention_mask is not None
            else None,
            position_ids=envelope.position_ids.to_array()
            if envelope.position_ids is not None
            else None,
            is_final=envelope.is_final,
            metadata=envelope.metadata,
        )


class ActivationTransport(ABC):
    """Abstract transport layer for activation communication."""

    @abstractmethod
    async def send(self, packet: ActivationPacket) -> None:
        """Send an activation packet to the next node."""

    @abstractmethod
    async def receive(self) -> ActivationPacket:
        """Receive an activation packet from the previous node."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport layer."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport layer."""


class TCPActivationTransport(ActivationTransport):
    """TCP-based transport for activation communication.

    Uses persistent connections for low-latency communication.
    """

    def __init__(self, config: PipelineNodeConfig):
        self.config = config
        self._server: Optional[asyncio.Server] = None
        self._send_writer: Optional[asyncio.StreamWriter] = None
        self._recv_reader: Optional[asyncio.StreamReader] = None
        self._recv_queue: asyncio.Queue[ActivationPacket] = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        """Start the TCP transport."""
        self._running = True

        # Start server to receive activations (if not head node)
        if not self.config.is_head:
            self._server = await asyncio.start_server(
                self._handle_connection,
                self.config.listen_host,
                self.config.listen_port,
            )
            logger.info(
                f"Activation server listening on {self.config.listen_host}:{self.config.listen_port}"
            )

        # Connect to next node (if not tail node)
        if not self.config.is_tail and self.config.next_node_host:
            await self._connect_to_next_node()

    async def _connect_to_next_node(self) -> None:
        """Establish connection to the next node in the pipeline."""
        max_retries = 300  # 5 minutes timeout
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                _, writer = await asyncio.open_connection(
                    self.config.next_node_host,
                    self.config.next_node_port,
                )
                self._send_writer = writer
                logger.info(
                    f"Connected to next node: {self.config.next_node_host}:{self.config.next_node_port}"
                )
                return
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Waiting for next node... (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect to next node after {max_retries} attempts: {e}"
                    ) from e

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle incoming connection from previous node."""
        logger.info("Received connection from previous node")
        self._recv_reader = reader

        try:
            while self._running:
                # Read length prefix
                length_data = await reader.readexactly(4)
                length = int.from_bytes(length_data, byteorder="big")
                if length > MAX_PACKET_BYTES:
                    raise ValueError("Activation packet exceeds size limit")

                # Read packet data
                data = await reader.readexactly(length)
                packet = ActivationPacket.deserialize(data)

                await self._recv_queue.put(packet)
        except asyncio.IncompleteReadError:
            logger.info("Connection from previous node closed")
        except Exception as e:
            if self._running:
                logger.error(f"Error receiving activation: {e}")

    async def send(self, packet: ActivationPacket) -> None:
        """Send an activation packet to the next node."""
        if self._send_writer is None:
            raise RuntimeError("Not connected to next node")

        data = packet.serialize()
        self._send_writer.write(data)
        await self._send_writer.drain()

    async def receive(self) -> ActivationPacket:
        """Receive an activation packet from the previous node."""
        return await self._recv_queue.get()

    async def stop(self) -> None:
        """Stop the TCP transport."""
        self._running = False

        if self._send_writer:
            self._send_writer.close()
            await self._send_writer.wait_closed()

        if self._server:
            self._server.close()
            await self._server.wait_closed()


class GGUFLayerExtractor:
    """Selects tensor metadata for a prospective layer range; no weights loaded."""

    def __init__(self, model_path: Path, start_layer: int, end_layer: int):
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._weights: dict[str, np.ndarray] = {}
        self._config: dict[str, Any] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load only the required layers from the GGUF file."""
        import asyncio

        # Run in executor since GGUF loading is sync
        await asyncio.get_event_loop().run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous layer loading."""
        try:
            from exo.worker.engines.gguf.gguf_loader import load_gguf_metadata

            # Load metadata to understand structure
            info = load_gguf_metadata(self.model_path, load_tensors=True)
            self._config = {
                "architecture": info.architecture,
                "n_layers": info.n_layers,
                "hidden_size": info.hidden_size,
                "n_heads": info.n_heads,
                "n_kv_heads": info.n_kv_heads,
                "vocab_size": info.vocab_size,
            }

            # Get tensor names for our layers
            layer_tensors = info.get_layers_range(self.start_layer, self.end_layer)

            # For head node, also get embedding tensors
            if self.start_layer == 0:
                layer_tensors.extend(
                    [
                        t
                        for t in info.get_non_layer_tensors()
                        if "embed" in t.name.lower() or "token" in t.name.lower()
                    ]
                )

            # For tail node, also get output tensors
            if self.end_layer == info.n_layers:
                layer_tensors.extend(
                    [
                        t
                        for t in info.get_non_layer_tensors()
                        if "output" in t.name.lower() or "lm_head" in t.name.lower()
                    ]
                )

            logger.info(
                f"Selected metadata for {len(layer_tensors)} tensors in layers "
                f"{self.start_layer}-{self.end_layer - 1}"
            )

            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load layer weights: {e}")
            raise

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class DistributedGGUFPipeline:
    """Experimental relay: full generation on the head, token forwarding elsewhere."""

    def __init__(self, config: PipelineNodeConfig):
        self.config = config
        self.transport = TCPActivationTransport(config)
        self.layer_extractor = GGUFLayerExtractor(
            config.model_path,
            config.start_layer,
            config.end_layer,
        )
        self._model = None
        self._running = False
        self._generation_callback: Optional[Callable[[str], Awaitable[None]]] = None

    async def initialize(self) -> None:
        """Initialize the pipeline node."""
        logger.info(
            f"Initializing pipeline node {self.config.rank}/{self.config.world_size}"
        )
        logger.info(f"Role: {self.config.role.value}")
        logger.info(f"Layers: {self.config.start_layer} - {self.config.end_layer - 1}")

        # Load model weights for our layers
        await self.layer_extractor.load()

        # For now, load full model but only use our layers
        # True partial loading requires deeper llama.cpp integration
        try:
            from exo.worker.engines.gguf.llama_backend import load_llama_model

            # Load with GPU layers corresponding to our shard
            n_gpu_layers = self.config.end_layer - self.config.start_layer

            self._model = load_llama_model(
                model_path=str(self.config.model_path),
                n_ctx=self.config.context_size,
                n_batch=self.config.batch_size,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Start transport layer
        await self.transport.start()

        self._running = True
        logger.info(f"Pipeline node {self.config.rank} initialized")

    async def shutdown(self) -> None:
        """Shutdown the pipeline node."""
        self._running = False
        await self.transport.stop()

        if self._model:
            self._model.close()
            self._model = None

        logger.info(f"Pipeline node {self.config.rank} shutdown")

    def set_generation_callback(
        self, callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """Set callback for generated tokens (only used by TAIL node)."""
        self._generation_callback = callback

    async def run_inference(self, prompt: str, max_tokens: int = 100) -> None:
        """Run inference on this node.

        For HEAD node: tokenize and start pipeline
        For MIDDLE/TAIL: receive and process activations
        """
        if self.config.is_head:
            await self._run_head_inference(prompt, max_tokens)
        else:
            await self._run_worker_inference()

    async def _run_head_inference(self, prompt: str, max_tokens: int) -> None:
        """HEAD node: Start the inference pipeline."""
        if self._model is None:
            raise RuntimeError("Model not loaded")

        import uuid

        request_id = str(uuid.uuid4())

        logger.info(f"HEAD: Starting inference for request {request_id}")

        # Use llama.cpp for full generation since we have the model
        # In a true partial-loading scenario, we'd only do embedding + first layers
        for output in self._model.complete_stream(
            prompt,
            max_tokens=max_tokens,
        ):
            token = output.choices[0].text
            finish_reason = output.choices[0].finish_reason

            # Create activation packet (in real impl, this would be hidden states)
            packet = ActivationPacket(
                request_id=request_id,
                sequence_position=0,
                hidden_states=np.array([ord(c) for c in token], dtype=np.float32),
                is_final=finish_reason is not None,
                metadata={"token": token, "finish_reason": finish_reason},
            )

            # For now, if we're the only node, just output directly
            if self.config.world_size == 1 and self._generation_callback:
                await self._generation_callback(token)
            elif not self.config.is_tail:
                await self.transport.send(packet)

    async def _run_worker_inference(self) -> None:
        """MIDDLE/TAIL node: Process received activations."""
        logger.info(f"Worker {self.config.rank}: Waiting for activations...")

        while self._running:
            try:
                packet = await asyncio.wait_for(self.transport.receive(), timeout=1.0)

                logger.debug(f"Received activation: {packet.request_id}")

                # Process through our layers (simplified for now)
                # In real impl, we'd apply our transformer layers to hidden_states

                if self.config.is_tail:
                    # TAIL: Output generated token
                    if self._generation_callback:
                        token = packet.metadata.get("token", "")
                        if not isinstance(token, str):
                            raise ValueError("Relayed token must be text")
                        await self._generation_callback(token)
                else:
                    # MIDDLE: Forward to next node
                    await self.transport.send(packet)

                if packet.is_final:
                    logger.info(f"Request {packet.request_id} complete")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error processing activation: {e}")


async def create_distributed_pipeline(
    model_path: Path,
    node_addresses: list[tuple[str, int]],  # [(host, port), ...]
    this_node_index: int,
) -> DistributedGGUFPipeline:
    """Create a distributed pipeline for this node.

    Args:
        model_path: Path to the GGUF model file
        node_addresses: List of (host, port) for all nodes in order
        this_node_index: Index of this node in the list

    Returns:
        Configured DistributedGGUFPipeline ready to initialize
    """
    from exo.worker.engines.gguf.gguf_loader import load_gguf_metadata
    from exo.worker.engines.gguf.gguf_sharding import calculate_shard_assignments

    world_size = len(node_addresses)

    # Get model info and sharding plan
    info = load_gguf_metadata(model_path, load_tensors=False)
    plan = calculate_shard_assignments(model_path, world_size, info)

    assignment = plan.get_assignment_for_rank(this_node_index)

    # Build config
    _, this_port = node_addresses[this_node_index]

    prev_host, prev_port = None, None
    if this_node_index > 0:
        prev_host, prev_port = node_addresses[this_node_index - 1]

    next_host, next_port = None, None
    if this_node_index < world_size - 1:
        next_host, next_port = node_addresses[this_node_index + 1]

    config = PipelineNodeConfig(
        node_id=f"node-{this_node_index}",
        rank=this_node_index,
        world_size=world_size,
        model_path=model_path,
        start_layer=assignment.start_layer,
        end_layer=assignment.end_layer,
        total_layers=assignment.total_layers,
        listen_host="0.0.0.0",
        listen_port=this_port,
        prev_node_host=prev_host,
        prev_node_port=prev_port,
        next_node_host=next_host,
        next_node_port=next_port,
    )

    return DistributedGGUFPipeline(config)


# CLI for testing distributed pipeline
async def _test_single_node():
    """Test single-node pipeline inference."""
    from exo.worker.engines.gguf.ollama_discovery import discover_ollama_models

    models = discover_ollama_models()
    if not models:
        print("No Ollama models found!")
        return

    model = models[0]
    print(f"Testing with: {model.full_name}")

    config = PipelineNodeConfig(
        node_id="test-node",
        rank=0,
        world_size=1,
        model_path=model.model_path,
        start_layer=0,
        end_layer=48,  # Will be overridden
        total_layers=48,
    )

    pipeline = DistributedGGUFPipeline(config)

    async def on_token(token: str):
        print(token, end="", flush=True)

    pipeline.set_generation_callback(on_token)

    await pipeline.initialize()

    print("\nPrompt: Hello, how are you?\n")
    print("Response: ", end="")
    await pipeline.run_inference("Hello, how are you?", max_tokens=50)
    print("\n")

    await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(_test_single_node())
