import time
from collections.abc import AsyncGenerator
from typing import cast

import anyio
from anyio import create_task_group
from anyio.abc import TaskGroup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework
from loguru import logger
from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from exo.master.placement import place_instance as get_instance_placements
from exo.shared.apply import apply
from exo.shared.election import ElectionMessage
from exo.shared.logging import InterceptLogger
from exo.shared.models.model_cards import MODEL_CARDS
from exo.shared.models.model_meta import get_model_meta
from exo.shared.types.api import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    CreateInstanceParams,
    CreateInstanceResponse,
    DeleteInstanceResponse,
    FinishReason,
    ModelList,
    ModelListModel,
    PlaceInstanceParams,
    PlacementPreview,
    PlacementPreviewResponse,
    StreamingChoiceResponse,
)
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import (
    ChatCompletion,
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    PlaceInstance,
    TaskFinished,
)
from exo.shared.types.common import CommandId, NodeId, SessionId
from exo.shared.types.events import ChunkGenerated, Event, ForwarderEvent, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.state import State
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.banner import print_startup_banner
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.dashboard_path import find_dashboard
from exo.utils.event_buffer import OrderedBuffer

# GGUF Integration imports
from exo.worker.engines.gguf.exo_integration import get_ollama_model_cards
from exo.worker.engines.gguf.launcher import run_distributed_node, parse_node_addresses
import subprocess
import asyncio


encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def chunk_to_response(
    chunk: TokenChunk, command_id: CommandId
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=command_id,
        created=int(time.time()),
        model=chunk.model,
        choices=[
            StreamingChoiceResponse(
                index=0,
                delta=ChatCompletionMessage(role="assistant", content=chunk.text),
                finish_reason=chunk.finish_reason,
            )
        ],
    )


async def resolve_model_meta(model_id: str) -> ModelMetadata:
    if model_id in MODEL_CARDS:
        model_card = MODEL_CARDS[model_id]
        return model_card.metadata
    
    # Check if it's an Ollama model
    if model_id.startswith("ollama-") or "ollama" in model_id:
        try:
            cards = get_ollama_model_cards()
            if model_id in cards:
                return cards[model_id].metadata
            # Try to find by partial match if needed
            for card in cards.values():
                if str(card.model_id) == model_id:
                    return card.metadata
        except Exception as e:
            logger.error(f"Error resolving Ollama metadata: {e}")

    return await get_model_meta(model_id)


class API:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        port: int,
        # Ideally this would be a MasterForwarderEvent but type system says no :(
        global_event_receiver: Receiver[ForwarderEvent],
        command_sender: Sender[ForwarderCommand],
        # This lets us pause the API if an election is running
        election_receiver: Receiver[ElectionMessage],
    ) -> None:
        self.state = State()
        self._event_log: list[Event] = []
        self.command_sender = command_sender
        self.global_event_receiver = global_event_receiver
        self.election_receiver = election_receiver
        self.event_buffer: OrderedBuffer[Event] = OrderedBuffer[Event]()
        self.node_id: NodeId = node_id
        self.session_id: SessionId = session_id
        self.last_completed_election: int = 0
        self.port = port

        self.paused: bool = False
        self.paused_ev: anyio.Event = anyio.Event()

        self.app = FastAPI()
        self._setup_cors()
        self._setup_routes()

        self.app.mount(
            "/",
            StaticFiles(
                directory=find_dashboard(),
                html=True,
            ),
            name="dashboard",
        )

        self._chat_completion_queues: dict[CommandId, Sender[TokenChunk]] = {}
        self._tg: TaskGroup | None = None
        
        # Ollama cluster nodes: [(host, port), ...]
        # Always includes localhost:11434 as the first node
        self._ollama_nodes: list[tuple[str, int]] = [("localhost", 11434)]

    def reset(self, new_session_id: SessionId, result_clock: int):
        logger.info("Resetting API State")
        self.state = State()
        self.session_id = new_session_id
        self.event_buffer = OrderedBuffer[Event]()
        self._chat_completion_queues = {}
        self.unpause(result_clock)

    def unpause(self, result_clock: int):
        logger.info("Unpausing API")
        self.last_completed_election = result_clock
        self.paused = False
        self.paused_ev.set()
        self.paused_ev = anyio.Event()

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        self.app.get("/node_id")(lambda: self.node_id)
        self.app.post("/instance")(self.create_instance)
        self.app.post("/place_instance")(self.place_instance)
        self.app.get("/instance/placement")(self.get_placement)
        self.app.get("/instance/previews")(self.get_placement_previews)
        self.app.get("/instance/{instance_id}")(self.get_instance)
        self.app.delete("/instance/{instance_id}")(self.delete_instance)
        self.app.get("/models")(self.get_models)
        self.app.get("/v1/models")(self.get_models)
        self.app.post("/v1/chat/completions", response_model=None)(
            self.chat_completions
        )
        self.app.get("/state")(lambda: self.state)
        self.app.get("/events")(lambda: self._event_log)
        
        # Ollama cluster management routes
        self.app.get("/ollama/nodes")(self.get_ollama_nodes)
        self.app.post("/ollama/nodes")(self.add_ollama_node)
        self.app.delete("/ollama/nodes/{host}")(self.remove_ollama_node)
        self.app.post("/ollama/generate")(self.generate_ollama)


    async def place_instance(self, payload: PlaceInstanceParams):
        command = PlaceInstance(
            model_meta=await resolve_model_meta(payload.model_id),
            sharding=payload.sharding,
            instance_meta=payload.instance_meta,
            min_nodes=payload.min_nodes,
            prefer_gpu=payload.prefer_gpu,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=command.model_meta,
        )

    async def create_instance(
        self, payload: CreateInstanceParams
    ) -> CreateInstanceResponse:
        instance = payload.instance
        model_meta = await resolve_model_meta(instance.shard_assignments.model_id)
        required_memory = model_meta.storage_size
        available_memory = self._calculate_total_available_memory()

        if required_memory > available_memory:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient memory to create instance. Required: {required_memory.in_gb:.1f}GB, Available: {available_memory.in_gb:.1f}GB",
            )

        command = CreateInstance(
            instance=instance,
        )
        await self._send(command)

        return CreateInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            model_meta=model_meta,
        )

    async def get_placement(
        self,
        model_id: str,
        sharding: Sharding = Sharding.Pipeline,
        instance_meta: InstanceMeta = InstanceMeta.MlxRing,
        min_nodes: int = 1,
        prefer_gpu: bool = True,
    ) -> Instance:
        model_meta = await resolve_model_meta(model_id)

        try:
            placements = get_instance_placements(
                PlaceInstance(
                    model_meta=model_meta,
                    sharding=sharding,
                    instance_meta=instance_meta,
                    min_nodes=min_nodes,
                    prefer_gpu=prefer_gpu,
                ),
                topology=self.state.topology,
                current_instances=self.state.instances,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        current_ids = set(self.state.instances.keys())
        new_ids = [
            instance_id for instance_id in placements if instance_id not in current_ids
        ]
        if len(new_ids) != 1:
            raise HTTPException(
                status_code=500,
                detail="Expected exactly one new instance from placement",
            )

        return placements[new_ids[0]]

    async def get_placement_previews(
        self, model_id: ModelId, prefer_gpu: bool = True
    ) -> PlacementPreviewResponse:
        seen: set[tuple[ModelId, Sharding, InstanceMeta, int]] = set()
        previews: list[PlacementPreview] = []
        if len(list(self.state.topology.list_nodes())) == 0:
            return PlacementPreviewResponse(previews=[])

        cards = [card for card in MODEL_CARDS.values() if card.short_id == model_id]
        if not cards:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        instance_combinations: list[tuple[Sharding, InstanceMeta, int]] = []
        for sharding in (Sharding.Pipeline, Sharding.Tensor):
            for instance_meta in (InstanceMeta.MlxRing, InstanceMeta.MlxJaccl):
                instance_combinations.extend(
                    [
                        (sharding, instance_meta, i)
                        for i in range(
                            1, len(list(self.state.topology.list_nodes())) + 1
                        )
                    ]
                )
        # TODO: PDD
        # instance_combinations.append((Sharding.PrefillDecodeDisaggregation, InstanceMeta.MlxRing, 1))

        for card in cards:
            model_meta = card.metadata
            for sharding, instance_meta, min_nodes in instance_combinations:
                try:
                    placements = get_instance_placements(
                        PlaceInstance(
                            model_meta=model_meta,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            min_nodes=min_nodes,
                            prefer_gpu=prefer_gpu,
                        ),
                        topology=self.state.topology,
                        current_instances=self.state.instances,
                    )
                except ValueError as exc:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error=str(exc),
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                current_ids = set(self.state.instances.keys())
                new_instances = [
                    instance
                    for instance_id, instance in placements.items()
                    if instance_id not in current_ids
                ]

                if len(new_instances) != 1:
                    if (card.model_id, sharding, instance_meta, 0) not in seen:
                        previews.append(
                            PlacementPreview(
                                model_id=card.model_id,
                                sharding=sharding,
                                instance_meta=instance_meta,
                                instance=None,
                                error="Expected exactly one new instance from placement",
                            )
                        )
                    seen.add((card.model_id, sharding, instance_meta, 0))
                    continue

                instance = new_instances[0]
                shard_assignments = instance.shard_assignments
                node_ids = list(shard_assignments.node_to_runner.keys())

                memory_delta_by_node: dict[str, int] = {}
                if node_ids:
                    total_bytes = model_meta.storage_size.in_bytes
                    per_node = total_bytes // len(node_ids)
                    remainder = total_bytes % len(node_ids)
                    for index, node_id in enumerate(sorted(node_ids, key=str)):
                        extra = 1 if index < remainder else 0
                        memory_delta_by_node[str(node_id)] = per_node + extra

                if (
                    card.model_id,
                    sharding,
                    instance_meta,
                    len(node_ids),
                ) not in seen:
                    previews.append(
                        PlacementPreview(
                            model_id=card.model_id,
                            sharding=sharding,
                            instance_meta=instance_meta,
                            instance=instance,
                            memory_delta_by_node=memory_delta_by_node or None,
                            error=None,
                        )
                    )
                seen.add((card.model_id, sharding, instance_meta, len(node_ids)))

        return PlacementPreviewResponse(previews=previews)

    def get_instance(self, instance_id: InstanceId) -> Instance:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")
        return self.state.instances[instance_id]

    async def delete_instance(self, instance_id: InstanceId) -> DeleteInstanceResponse:
        if instance_id not in self.state.instances:
            raise HTTPException(status_code=404, detail="Instance not found")

        command = DeleteInstance(
            instance_id=instance_id,
        )
        await self._send(command)
        return DeleteInstanceResponse(
            message="Command received.",
            command_id=command.command_id,
            instance_id=instance_id,
        )

    async def _process_gpt_oss(self, token_chunks: Receiver[TokenChunk]):
        stream = StreamableParser(encoding, role=Role.ASSISTANT)
        thinking = False

        async for chunk in token_chunks:
            stream.process(chunk.token_id)

            delta = stream.last_content_delta
            ch = stream.current_channel

            if ch == "analysis" and not thinking:
                thinking = True
                yield chunk.model_copy(update={"text": "<think>"})

            if ch != "analysis" and thinking:
                thinking = False
                yield chunk.model_copy(update={"text": "</think>"})

            if delta:
                yield chunk.model_copy(update={"text": delta})

            if chunk.finish_reason is not None:
                if thinking:
                    yield chunk.model_copy(update={"text": "</think>"})
                yield chunk
                break

    async def _chat_chunk_stream(
        self, command_id: CommandId, parse_gpt_oss: bool
    ) -> AsyncGenerator[TokenChunk, None]:
        """Yield `TokenChunk`s for a given command until completion."""

        try:
            self._chat_completion_queues[command_id], recv = channel[TokenChunk]()

            with recv as token_chunks:
                if parse_gpt_oss:
                    async for chunk in self._process_gpt_oss(token_chunks):
                        yield chunk
                        if chunk.finish_reason is not None:
                            break
                else:
                    async for chunk in token_chunks:
                        yield chunk
                        if chunk.finish_reason is not None:
                            break

        except anyio.get_cancelled_exc_class():
            # TODO: TaskCancelled
            """
            self.command_sender.send_nowait(
                ForwarderCommand(origin=self.node_id, command=command)
            )
            """
            raise
        finally:
            command = TaskFinished(finished_command_id=command_id)
            await self._send(command)
            del self._chat_completion_queues[command_id]

    async def _generate_chat_stream(
        self, command_id: CommandId, parse_gpt_oss: bool
    ) -> AsyncGenerator[str, None]:
        """Generate chat completion stream as JSON strings."""

        async for chunk in self._chat_chunk_stream(command_id, parse_gpt_oss):
            chunk_response: ChatCompletionResponse = chunk_to_response(
                chunk, command_id
            )
            logger.debug(f"chunk_response: {chunk_response}")

            yield f"data: {chunk_response.model_dump_json()}\n\n"

            if chunk.finish_reason is not None:
                yield "data: [DONE]\n\n"

    async def _collect_chat_completion(
        self, command_id: CommandId, parse_gpt_oss: bool
    ) -> ChatCompletionResponse:
        """Collect all token chunks for a chat completion and return a single response."""

        text_parts: list[str] = []
        model: str | None = None
        finish_reason: FinishReason | None = None

        async for chunk in self._chat_chunk_stream(command_id, parse_gpt_oss):
            if model is None:
                model = chunk.model

            text_parts.append(chunk.text)

            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        combined_text = "".join(text_parts)
        assert model is not None

        return ChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=combined_text,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    async def _trigger_notify_user_to_download_model(self, model_id: str) -> None:
        logger.warning(
            "TODO: we should send a notification to the user to download the model"
        )

    async def _run_ollama_inference(self, payload: ChatCompletionTaskParams) -> AsyncGenerator[str, None]:
        """Run inference using GGUF/Ollama backend."""
        # Detect model path
        ollama_cards = get_ollama_model_cards()
        target_card = None
        for card in ollama_cards.values():
            if card.short_id == payload.model or str(card.model_id) == payload.model:
                target_card = card
                break
        
        if not target_card:
            yield f"data: {{\"error\": \"Model {payload.model} not found\"}}\n\n"
            return
            
        model_path = target_card.gguf_path
        logger.info(f"Running GGUF inference on {model_path}")

        # Construct basic local GGUF pipeline for now (single node or auto-detect)
        # For simplicity in this integration, we run a subprocess that prints tokens
        # Ideally we would use the DistributedGGUFPipeline class directly if async loop permits
        
        # We'll use a wrapper around llama-cpp-python here for immediate feedback
        # This is a temporary "local" execution mode to satisfy the frontend integration
        from llama_cpp import Llama
        
        # Run in executor to avoid blocking main loop
        def run_inference_sync():
            try:
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=2048,
                    n_gpu_layers=-1, # Try to use GPU
                    verbose=False
                )
                
                messages = [{"role": "user", "content": payload.messages[-1].content}] # Simplified
                if payload.messages:
                    if payload.messages[-1].role == "user":
                         prompt = payload.messages[-1].content
                    else:
                         prompt = "Hello"
                
                stream = llm(
                    prompt,
                    max_tokens=payload.max_tokens or 256,
                    stop=["User:", "\n\n"],
                    stream=True
                )
                
                for output in stream:
                    text = output['choices'][0]['text']
                    yield text
            except Exception as e:
                logger.error(f"GGUF Inference error: {e}")
                return

        # Bridge the sync generator to async
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def producer():
            try:
                for token in run_inference_sync():
                     asyncio.run_coroutine_threadsafe(queue.put(token), loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), loop) # Sentinel
            except Exception as e:
                logger.error(f"Producer error: {e}")
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        import threading
        t = threading.Thread(target=producer)
        t.start()
        
        import time
        created = int(time.time())
        
        while True:
            token = await queue.get()
            if token is None:
                break
                
            # Yield in OpenAI format
            chunk_resp = ChatCompletionResponse(
                id="gguf-stream",
                created=created,
                model=payload.model,
                choices=[
                    StreamingChoiceResponse(
                        index=0,
                        delta=ChatCompletionMessage(role="assistant", content=token),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk_resp.model_dump_json()}\n\n"
            
        # Final done
        yield "data: [DONE]\n\n"



    async def chat_completions(
        self, payload: ChatCompletionTaskParams
    ) -> ChatCompletionResponse | StreamingResponse:
        """Handle chat completions, supporting both streaming and non-streaming responses."""
        
        # Check if this is an Ollama model - route directly without instance
        if payload.model.startswith("ollama-") or "ollama/" in payload.model:
            logger.info(f"Routing to Ollama cluster for model: {payload.model}")
            
            # Convert model ID to Ollama format (e.g., "ollama-qwen3-coder-30b" -> "qwen3-coder:30b")
            ollama_model = payload.model
            if ollama_model.startswith("ollama-"):
                ollama_model = ollama_model[7:]  # Remove "ollama-" prefix
            if ollama_model.startswith("ollama/"):
                ollama_model = ollama_model[7:]  # Remove "ollama/" prefix
            # Convert hyphens back to colons for size suffix (e.g., "qwen3-coder-30b" -> "qwen3-coder:30b")
            parts = ollama_model.rsplit("-", 1)
            if len(parts) == 2 and parts[1].endswith("b"):
                ollama_model = f"{parts[0]}:{parts[1]}"
            
            # Build prompt from messages
            prompt = ""
            for msg in payload.messages:
                if msg.role == "user":
                    prompt = msg.content  # Use last user message
            
            if payload.stream:
                async def ollama_stream():
                    import aiohttp
                    # Try each configured Ollama node
                    for host, port in self._ollama_nodes:
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    f"http://{host}:{port}/api/generate",
                                    json={
                                        "model": ollama_model,
                                        "prompt": prompt,
                                        "stream": True,
                                        "options": {"num_predict": payload.max_tokens or 1024}
                                    },
                                    timeout=aiohttp.ClientTimeout(total=300)
                                ) as resp:
                                    if resp.status == 200:
                                        async for line in resp.content:
                                            if line:
                                                try:
                                                    import json
                                                    data = json.loads(line)
                                                    token = data.get("response", "")
                                                    if token:
                                                        chunk = ChatCompletionResponse(
                                                            id="ollama-chat",
                                                            created=int(time.time()),
                                                            model=payload.model,
                                                            choices=[
                                                                StreamingChoiceResponse(
                                                                    index=0,
                                                                    delta=ChatCompletionMessage(role="assistant", content=token),
                                                                    finish_reason=None if not data.get("done") else FinishReason.Stop,
                                                                )
                                                            ],
                                                        )
                                                        yield f"data: {chunk.model_dump_json()}\n\n"
                                                    if data.get("done"):
                                                        yield "data: [DONE]\n\n"
                                                        return
                                                except Exception:
                                                    continue
                                        return
                        except Exception as e:
                            logger.debug(f"Failed to connect to {host}:{port}: {e}")
                            continue
                    yield 'data: {"error": "No Ollama node available"}\n\n'
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    ollama_stream(),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming
                import aiohttp
                for host, port in self._ollama_nodes:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                f"http://{host}:{port}/api/generate",
                                json={
                                    "model": ollama_model,
                                    "prompt": prompt,
                                    "stream": False,
                                    "options": {"num_predict": payload.max_tokens or 1024}
                                },
                                timeout=aiohttp.ClientTimeout(total=300)
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    return ChatCompletionResponse(
                                        id="ollama-chat",
                                        created=int(time.time()),
                                        model=payload.model,
                                        choices=[
                                            ChatCompletionChoice(
                                                index=0,
                                                message=ChatCompletionMessage(
                                                    role="assistant",
                                                    content=data.get("response", ""),
                                                ),
                                                finish_reason=FinishReason.Stop,
                                            )
                                        ],
                                    )
                    except Exception as e:
                        logger.debug(f"Failed to connect to {host}:{port}: {e}")
                        continue
                raise HTTPException(status_code=503, detail="No Ollama node available")
        
        # Regular MLX model path - requires instance
        model_meta = await resolve_model_meta(payload.model)
        payload.model = model_meta.model_id
        parse_gpt_oss = "gpt-oss" in model_meta.model_id.lower()
        logger.info(f"{parse_gpt_oss=}")

        if not any(
            instance.shard_assignments.model_id == payload.model
            for instance in self.state.instances.values()
        ):
            await self._trigger_notify_user_to_download_model(payload.model)
            raise HTTPException(
                status_code=404, detail=f"No instance found for model {payload.model}"
            )

        command = ChatCompletion(
            request_params=payload,
        )
        await self._send(command)
        if payload.stream:
            return StreamingResponse(
                self._generate_chat_stream(command.command_id, parse_gpt_oss),
                media_type="text/event-stream",
            )

        return await self._collect_chat_completion(command.command_id, parse_gpt_oss)

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for node in self.state.topology.list_nodes():
            if node.node_profile is not None:
                total_available += node.node_profile.memory.ram_available

        return total_available

    async def get_models(self) -> ModelList:
        """Returns list of available models."""
        return ModelList(
            data=[
                ModelListModel(
                    id=card.short_id,
                    hugging_face_id=card.model_id,
                    name=card.name,
                    description=card.description,
                    tags=card.tags,
                    storage_size_megabytes=int(card.metadata.storage_size.in_mb),
                    supports_tensor=card.metadata.supports_tensor,
                )
                for card in MODEL_CARDS.values()
            ] + [
                ModelListModel(
                    id=card.short_id,
                    hugging_face_id=str(card.model_id),
                    name=card.name,
                    description=card.description,
                    tags=card.tags,
                    storage_size_megabytes=int(card.metadata.storage_size.in_mb),
                    supports_tensor=False,
                )
                for card in get_ollama_model_cards().values()
            ]
        )

    # ==================== Ollama Cluster Management ====================
    
    async def get_ollama_nodes(self):
        """Get list of configured Ollama nodes with their status."""
        import aiohttp
        
        nodes = []
        for host, port in self._ollama_nodes:
            node_info = {
                "host": host,
                "port": port,
                "available": False,
                "models": [],
                "model_count": 0,
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://{host}:{port}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = [m["name"] for m in data.get("models", [])]
                            node_info["available"] = True
                            node_info["models"] = models
                            node_info["model_count"] = len(models)
            except Exception as e:
                logger.debug(f"Failed to check Ollama node {host}:{port}: {e}")
            
            nodes.append(node_info)
        
        return {"nodes": nodes}
    
    async def add_ollama_node(self, payload: dict):
        """Add a new Ollama node to the cluster."""
        host = payload.get("host", "").strip()
        port = int(payload.get("port", 11434))
        
        if not host:
            raise HTTPException(status_code=400, detail="Host is required")
        
        # Check if already exists
        for h, p in self._ollama_nodes:
            if h == host and p == port:
                raise HTTPException(status_code=400, detail=f"Node {host}:{port} already exists")
        
        # Verify node is reachable before adding
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail=f"Node {host}:{port} is not reachable")
                    data = await resp.json()
                    model_count = len(data.get("models", []))
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=400, detail=f"Cannot connect to {host}:{port}: {str(e)}")
        
        self._ollama_nodes.append((host, port))
        logger.info(f"Added Ollama node {host}:{port} with {model_count} models")
        
        return {"message": f"Node {host}:{port} added successfully", "models": model_count}
    
    async def remove_ollama_node(self, host: str):
        """Remove an Ollama node from the cluster."""
        # Don't allow removing localhost
        if host == "localhost" or host == "127.0.0.1":
            raise HTTPException(status_code=400, detail="Cannot remove local Ollama node")
        
        # Find and remove the node
        for i, (h, p) in enumerate(self._ollama_nodes):
            if h == host:
                self._ollama_nodes.pop(i)
                logger.info(f"Removed Ollama node {host}:{p}")
                return {"message": f"Node {host} removed successfully"}
        
        raise HTTPException(status_code=404, detail=f"Node {host} not found")
    
    async def generate_ollama(self, payload: dict):
        """Generate text using distributed Ollama cluster."""
        from exo.worker.engines.gguf.ollama_distributed import DistributedOllamaCluster, OllamaNode
        
        model = payload.get("model", "")
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.7))
        stream = payload.get("stream", False)
        
        if not model:
            raise HTTPException(status_code=400, detail="Model is required")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Create distributed cluster
        cluster = DistributedOllamaCluster(model_name=model)
        for host, port in self._ollama_nodes:
            cluster.add_node(host, port)
        
        await cluster.initialize()
        
        if stream:
            async def stream_generator():
                async for token in cluster.generate(prompt, max_tokens, temperature):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Collect all tokens
            tokens = []
            async for token in cluster.generate(prompt, max_tokens, temperature):
                tokens.append(token)
            
            return {
                "model": model,
                "response": "".join(tokens),
                "nodes_used": len(self._ollama_nodes)
            }


    async def run(self):
        cfg = Config()
        cfg.bind = f"0.0.0.0:{self.port}"
        # nb: shared.logging needs updating if any of this changes
        cfg.accesslog = None
        cfg.errorlog = "-"
        cfg.logger_class = InterceptLogger

        async with create_task_group() as tg:
            self._tg = tg
            logger.info("Starting API")
            tg.start_soon(self._apply_state)
            tg.start_soon(self._pause_on_new_election)
            print_startup_banner(self.port)
            await serve(
                cast(ASGIFramework, self.app),
                cfg,
                shutdown_trigger=lambda: anyio.sleep_forever(),
            )

        self.command_sender.close()
        self.global_event_receiver.close()

    async def _apply_state(self):
        with self.global_event_receiver as events:
            async for f_event in events:
                if f_event.origin != self.session_id.master_node_id:
                    continue
                self.event_buffer.ingest(f_event.origin_idx, f_event.event)
                for idx, event in self.event_buffer.drain_indexed():
                    self._event_log.append(event)
                    self.state = apply(self.state, IndexedEvent(event=event, idx=idx))
                    if (
                        isinstance(event, ChunkGenerated)
                        and event.command_id in self._chat_completion_queues
                    ):
                        assert isinstance(event.chunk, TokenChunk)
                        await self._chat_completion_queues[event.command_id].send(
                            event.chunk
                        )

    async def _pause_on_new_election(self):
        with self.election_receiver as ems:
            async for message in ems:
                if message.clock > self.last_completed_election:
                    self.paused = True

    async def _send(self, command: Command):
        while self.paused:
            await self.paused_ev.wait()
        await self.command_sender.send(
            ForwarderCommand(origin=self.node_id, command=command)
        )
