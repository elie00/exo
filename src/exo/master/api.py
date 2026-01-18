import time
import re
import json as json_module
from collections.abc import AsyncGenerator
from typing import cast, Any

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


# ============================================================================
# Tool Calling Support Helper Functions
# ============================================================================

def _format_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """
    Format tools into a system prompt that instructs the LLM how to call functions.
    
    Args:
        tools: List of tool definitions following OpenAI format
        
    Returns:
        A formatted string to inject into the system prompt
    """
    if not tools:
        return ""
    
    tool_descriptions: list[str] = []
    example_calls: list[str] = []
    
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        description = func.get("description", "No description")
        parameters = func.get("parameters", {})
        
        # Format parameters
        params_str = ""
        example_args = {}
        if parameters.get("properties"):
            param_list = []
            for param_name, param_info in parameters["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                required = param_name in parameters.get("required", [])
                req_marker = " [REQUIRED]" if required else ""
                param_list.append(f"  • {param_name} ({param_type}){req_marker}: {param_desc}")
                # Generate example value
                if param_type == "string":
                    example_args[param_name] = f"example_{param_name}"
                elif param_type == "number" or param_type == "integer":
                    example_args[param_name] = 42
                elif param_type == "boolean":
                    example_args[param_name] = True
                else:
                    example_args[param_name] = f"value"
            params_str = "\n" + "\n".join(param_list)
        
        tool_descriptions.append(f"• {name}: {description}{params_str}")
        example_calls.append(f'{{"name": "{name}", "arguments": {json_module.dumps(example_args)}}}')
    
    if not tool_descriptions:
        return ""
    
    # Build a more forceful, explicit prompt
    tools_list = "\n".join(tool_descriptions)
    first_example = example_calls[0] if example_calls else '{"name": "function_name", "arguments": {}}'
    
    return f"""[TOOL USE INSTRUCTIONS]
You have access to these functions:

{tools_list}

IMPORTANT: When the user's request requires using a function, you MUST respond with ONLY the following format - no other text before or after:

<tool_call>
{first_example}
</tool_call>

RULES:
1. If you need to call a function, respond with ONLY the <tool_call> block
2. Do NOT explain what you're doing - just output the tool_call
3. Do NOT add any text before or after the <tool_call> tags
4. The JSON inside must be valid with "name" and "arguments" fields
5. If you don't need a function, respond normally without tool_call tags

Example - if asked "What's the weather in Paris?", respond EXACTLY like this:
<tool_call>
{{"name": "get_weather", "arguments": {{"city": "Paris"}}}}
</tool_call>
"""


def _extract_tool_calls(response_text: str) -> tuple[str, list[dict[str, Any]] | None]:
    """
    Extract tool calls from LLM response and return clean content.
    
    Args:
        response_text: Raw response text from the LLM
        
    Returns:
        Tuple of (cleaned_content, tool_calls or None)
        - cleaned_content: Response text with tool_call tags removed
        - tool_calls: List of tool call objects in OpenAI format, or None if no calls found
    """
    # Pattern to match <tool_call>...</tool_call>
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if not matches:
        return response_text, None
    
    tool_calls: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        try:
            # Parse the JSON inside the tool_call tags
            call_data = json_module.loads(match.strip())
            
            # Convert to OpenAI tool_call format
            tool_call = {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": call_data.get("name", ""),
                    "arguments": json_module.dumps(call_data.get("arguments", {}))
                }
            }
            tool_calls.append(tool_call)
        except json_module.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            continue
    
    # Remove tool_call tags from content
    cleaned_content = re.sub(pattern, '', response_text, flags=re.DOTALL).strip()
    
    if not tool_calls:
        return response_text, None
    
    return cleaned_content, tool_calls


def _inject_tools_into_messages(
    messages: list[ChatCompletionMessage],
    tools: list[dict[str, Any]] | None
) -> list[ChatCompletionMessage]:
    """
    Inject tool descriptions into the messages as a system message.
    
    Args:
        messages: Original list of messages
        tools: Tool definitions to inject
        
    Returns:
        Modified messages list with tool instructions prepended
    """
    if not tools:
        return messages
    
    tools_prompt = _format_tools_prompt(tools)
    if not tools_prompt:
        return messages
    
    # Create a new system message with tool instructions
    tool_system_msg = ChatCompletionMessage(
        role="system",
        content=tools_prompt
    )
    
    # Prepend to messages (or merge with existing system message)
    new_messages = list(messages)
    if new_messages and new_messages[0].role == "system":
        # Append to existing system message
        existing_content = new_messages[0].content or ""
        new_messages[0] = ChatCompletionMessage(
            role="system",
            content=f"{existing_content}\n\n{tools_prompt}"
        )
    else:
        # Prepend new system message
        new_messages.insert(0, tool_system_msg)
    
    return new_messages


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
        
        # Track active requests for cancellation support
        self._active_requests: dict[CommandId, anyio.CancelScope] = {}
        
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
        
        # GPU information endpoint
        self.app.get("/gpu/info")(self.get_gpu_info)
        
        # Ollama cluster management routes
        self.app.get("/ollama/nodes")(self.get_ollama_nodes)
        self.app.post("/ollama/nodes")(self.add_ollama_node)
        self.app.delete("/ollama/nodes/{host}")(self.remove_ollama_node)
        self.app.post("/ollama/generate")(self.generate_ollama)
        
        # Request cancellation endpoint
        self.app.delete("/v1/chat/completions/{command_id}")(self.cancel_completion)
        self.app.get("/v1/chat/completions/active")(self.list_active_requests)
        
        # Network profiling endpoints
        self.app.get("/network/profile")(self.get_network_profile)
        self.app.post("/network/bandwidth_test")(self.bandwidth_test_endpoint)
        
        # Image generation endpoint
        self.app.post("/v1/images/generate")(self.generate_image)
        self.app.post("/v1/images/generations")(self.generate_image)  # OpenAI compat


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
        self, command_id: CommandId, parse_gpt_oss: bool, has_tools: bool = False
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

        # Extract tool calls if tools were provided
        tool_calls = None
        content = combined_text
        if has_tools:
            content, tool_calls = _extract_tool_calls(combined_text)
            if tool_calls:
                # When there are tool calls, finish_reason should be "tool_calls"
                finish_reason = "tool_calls"
                logger.info(f"Extracted {len(tool_calls)} tool call(s) from response")

        return ChatCompletionResponse(
            id=command_id,
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content if content else None,
                        tool_calls=tool_calls,
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
        
        # Tool calling support: inject tools into messages
        has_tools = bool(payload.tools)
        if has_tools:
            logger.info(f"Tool calling enabled with {len(payload.tools)} tool(s)")
            # Create a modified payload with tools injected into messages
            modified_messages = _inject_tools_into_messages(payload.messages, payload.tools)
            payload = ChatCompletionTaskParams(
                model=payload.model,
                messages=modified_messages,
                frequency_penalty=payload.frequency_penalty,
                logit_bias=payload.logit_bias,
                logprobs=payload.logprobs,
                top_logprobs=payload.top_logprobs,
                max_tokens=payload.max_tokens,
                n=payload.n,
                presence_penalty=payload.presence_penalty,
                response_format=payload.response_format,
                seed=payload.seed,
                stop=payload.stop,
                stream=payload.stream,
                temperature=payload.temperature,
                top_p=payload.top_p,
                tools=payload.tools,
                tool_choice=payload.tool_choice,
                parallel_tool_calls=payload.parallel_tool_calls,
                user=payload.user,
            )

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

        return await self._collect_chat_completion(command.command_id, parse_gpt_oss, has_tools)

    def _calculate_total_available_memory(self) -> Memory:
        """Calculate total available memory across all nodes in bytes."""
        total_available = Memory()

        for node in self.state.topology.list_nodes():
            if node.node_profile is not None:
                total_available += node.node_profile.memory.ram_available

        return total_available

    async def get_gpu_info(self):
        """Get GPU information for all nodes in the cluster."""
        nodes_gpu_info = []
        total_vram_mb = 0
        total_vram_available_mb = 0
        gpu_node_count = 0

        for node in self.state.topology.list_nodes():
            node_info = {
                "node_id": str(node.node_id),
                "has_gpu": False,
                "gpu_count": 0,
                "device_name": None,
                "vram_total_mb": 0,
                "vram_available_mb": 0,
                "vram_used_mb": 0,
                "driver_version": None,
                "cuda_version": None,
                "gpu_usage_percent": 0.0,
                "gpu_temp_celsius": 0.0,
            }

            if node.node_profile is not None:
                system = node.node_profile.system
                memory = node.node_profile.memory

                # Check SystemPerformanceProfile for GPU info
                if system.has_gpu_memory:
                    node_info["has_gpu"] = True
                    node_info["gpu_count"] = system.gpu_count or 1
                    node_info["vram_total_mb"] = system.gpu_memory_total_mb or 0
                    node_info["vram_available_mb"] = system.gpu_memory_available_mb
                    node_info["vram_used_mb"] = system.gpu_memory_used_mb or 0
                    node_info["driver_version"] = system.driver_version
                    node_info["cuda_version"] = system.cuda_version
                    node_info["gpu_usage_percent"] = system.gpu_usage
                    node_info["gpu_temp_celsius"] = system.temp

                    total_vram_mb += node_info["vram_total_mb"]
                    total_vram_available_mb += node_info["vram_available_mb"]
                    gpu_node_count += 1

                # Also check MemoryPerformanceProfile for VRAM (new fields)
                if memory.has_gpu_vram:
                    node_info["has_gpu"] = True
                    if memory.gpu_vram_total:
                        node_info["vram_total_mb"] = int(memory.gpu_vram_total.in_mb)
                    if memory.gpu_vram_available:
                        node_info["vram_available_mb"] = int(memory.gpu_vram_available.in_mb)
                    if memory.gpu_vram_used:
                        node_info["vram_used_mb"] = int(memory.gpu_vram_used.in_mb)

            nodes_gpu_info.append(node_info)

        return {
            "cluster_summary": {
                "total_nodes": len(nodes_gpu_info),
                "gpu_nodes": gpu_node_count,
                "total_vram_mb": total_vram_mb,
                "total_vram_available_mb": total_vram_available_mb,
                "gpu_aware_placement_enabled": gpu_node_count > 0,
            },
            "nodes": nodes_gpu_info,
        }

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

    # ==================== Request Cancellation ====================
    
    async def cancel_completion(self, command_id: str):
        """Cancel an active chat completion request."""
        if command_id in self._active_requests:
            scope = self._active_requests[command_id]
            scope.cancel()
            logger.info(f"Cancelled request {command_id}")
            return {
                "status": "cancelled",
                "command_id": command_id,
                "message": "Request cancelled successfully"
            }
        
        # Check if it's a known completed request (in queues)
        if command_id in self._chat_completion_queues:
            return {
                "status": "pending",
                "command_id": command_id,
                "message": "Request found in queue but not yet actively processing"
            }
        
        raise HTTPException(
            status_code=404,
            detail=f"Request {command_id} not found. It may have already completed or never existed."
        )
    
    async def list_active_requests(self):
        """List all active chat completion requests that can be cancelled."""
        active = []
        for command_id, scope in self._active_requests.items():
            active.append({
                "command_id": command_id,
                "cancelled": scope.cancel_called,
            })
        
        pending = []
        for command_id in self._chat_completion_queues.keys():
            if command_id not in self._active_requests:
                pending.append({
                    "command_id": command_id,
                    "status": "pending"
                })
        
        return {
            "active_count": len(active),
            "pending_count": len(pending),
            "active_requests": active,
            "pending_requests": pending,
        }

    # ==================== Network Profiling ====================
    
    async def get_network_profile(self):
        """Get network latency and connection info for all cluster nodes."""
        from exo.worker.utils.net_profiler import get_network_profiler, ConnectionType
        
        profiler = get_network_profiler(self.port)
        
        # Build node address map from topology
        node_addresses: dict[str, str] = {}
        for node in self.state.topology.list_nodes():
            # Try to get IP from connections
            for conn in self.state.topology.connections:
                if conn.local_node_id == node.node_id:
                    if conn.send_back_multiaddr:
                        ip = conn.send_back_multiaddr.ip_address
                        if ip:
                            node_addresses[str(node.node_id)] = ip
                            break
        
        # Profile connections
        from exo.shared.types.common import NodeId
        metrics = await profiler.profile_all_connections(
            self.node_id,
            {NodeId(k): v for k, v in node_addresses.items()}
        )
        
        results = []
        for m in metrics:
            results.append({
                "target_node": str(m.target_node),
                "target_ip": m.target_ip,
                "latency_ms": round(m.latency_ms, 2),
                "latency_min_ms": round(m.latency_min_ms, 2),
                "latency_max_ms": round(m.latency_max_ms, 2),
                "latency_stddev_ms": round(m.latency_stddev_ms, 2),
                "connection_type": m.connection_type.value,
                "interface": m.interface_name,
                "measured_at": m.measured_at,
            })
        
        return {
            "source_node": str(self.node_id),
            "profiles": results,
            "summary": {
                "nodes_profiled": len(results),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 2) if results else 0,
            }
        }
    
    async def bandwidth_test_endpoint(self, request):
        """Endpoint for bandwidth testing - echoes back received data."""
        from fastapi import Request
        body = await request.body()
        return {"received_bytes": len(body), "echo": True}

    # ==================== Image Generation ====================
    
    async def generate_image(self, request: dict):
        """Generate images using distributed diffusion models."""
        from exo.worker.engines.image.image_generator import (
            get_image_generator,
            ImageGenerationRequest,
        )
        
        try:
            # Parse request
            gen_request = ImageGenerationRequest(
                model=request.get("model", "flux-schnell"),
                prompt=request.get("prompt", ""),
                negative_prompt=request.get("negative_prompt"),
                size=request.get("size", "1024x1024"),
                n=request.get("n", 1),
                quality=request.get("quality", "standard"),
                response_format=request.get("response_format", "b64_json"),
                guidance_scale=request.get("guidance_scale", 7.5),
                num_inference_steps=request.get("num_inference_steps", 50),
                seed=request.get("seed"),
            )
            
            # Generate
            generator = get_image_generator()
            response = await generator.generate(gen_request)
            
            return response.model_dump()
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Image generation failed: {str(e)}"
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
