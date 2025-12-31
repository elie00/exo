"""GGUF Runner for EXO distributed inference.

This module provides a runner implementation for GGUF/Ollama models,
following the same pattern as the MLX runner but using llama-cpp-python
for inference.
"""

import time
from pathlib import Path
from typing import Optional

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.runner.bootstrap import logger

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class GGUFRunner:
    """Runner for GGUF models using llama-cpp-python.
    
    Handles loading, inference, and distributed communication for
    GGUF format models (Ollama, etc.).
    """
    
    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,  # -1 = all layers to GPU
        n_threads: Optional[int] = None,
        rpc_servers: Optional[str] = None,
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF inference. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.rpc_servers = rpc_servers
        
        self._model: Optional[Llama] = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def load(self) -> None:
        """Load the GGUF model."""
        logger.info(f"Loading GGUF model: {self.model_path.name}")
        
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=True,
        )
        
        self._loaded = True
        logger.info(f"GGUF model loaded successfully")
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
        stream: bool = True,
    ):
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
            GenerationResponse for each generated token
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")
        
        token_idx = 0
        
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
                finish_reason = output["choices"][0].get("finish_reason")
                
                yield GenerationResponse(
                    token=token_idx,
                    text=token,
                    finish_reason=finish_reason,
                )
                token_idx += 1
        else:
            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            )
            text = output["choices"][0]["text"]
            finish_reason = output["choices"][0].get("finish_reason", "stop")
            
            yield GenerationResponse(
                token=0,
                text=text,
                finish_reason=finish_reason,
            )
    
    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ):
        """Chat completion interface.
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yield tokens as they're generated
        
        Yields:
            GenerationResponse for each generated token
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")
        
        token_idx = 0
        
        if stream:
            for output in self._model.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = output["choices"][0].get("delta", {})
                finish_reason = output["choices"][0].get("finish_reason")
                
                if "content" in delta:
                    yield GenerationResponse(
                        token=token_idx,
                        text=delta["content"],
                        finish_reason=finish_reason,
                    )
                    token_idx += 1
                elif finish_reason:
                    yield GenerationResponse(
                        token=token_idx,
                        text="",
                        finish_reason=finish_reason,
                    )
        else:
            output = self._model.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = output["choices"][0]["message"]["content"]
            finish_reason = output["choices"][0].get("finish_reason", "stop")
            
            yield GenerationResponse(
                token=0,
                text=text,
                finish_reason=finish_reason,
            )


def gguf_runner_main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    gguf_path: Path,
):
    """Main entry point for GGUF runner process.
    
    This follows the same pattern as the MLX runner but uses
    llama-cpp-python for GGUF model inference.
    
    Args:
        bound_instance: The bound instance configuration
        event_sender: Channel to send events to the supervisor
        task_receiver: Channel to receive tasks from the supervisor
        gguf_path: Path to the GGUF model file
    """
    runner_id = bound_instance.bound_runner_id
    shard_metadata = bound_instance.bound_shard
    
    try:
        logger.info("GGUF runner starting")
        
        runner: Optional[GGUFRunner] = None
        current_status: RunnerStatus = RunnerIdle()
        
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
        )
        
        with task_receiver as tasks:
            for task in tasks:
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Running
                    )
                )
                event_sender.send(TaskAcknowledged(task_id=task.task_id))
                
                match task:
                    case ConnectToGroup() if isinstance(
                        current_status, (RunnerIdle, RunnerFailed)
                    ):
                        # GGUF uses RPC for distributed, no group init needed for now
                        logger.info("GGUF runner connecting (no-op for single node)")
                        current_status = RunnerConnecting()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        current_status = RunnerConnected()
                    
                    case LoadModel() if isinstance(current_status, (RunnerConnected, RunnerIdle)):
                        current_status = RunnerLoading()
                        logger.info("GGUF runner loading model")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        
                        runner = GGUFRunner(
                            model_path=gguf_path,
                            n_ctx=4096,
                            n_gpu_layers=-1,  # Use all GPU layers
                        )
                        runner.load()
                        
                        current_status = RunnerLoaded()
                        logger.info("GGUF runner loaded")
                    
                    case StartWarmup() if isinstance(current_status, RunnerLoaded):
                        assert runner is not None
                        current_status = RunnerWarmingUp()
                        logger.info("GGUF runner warming up")
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        
                        # Warmup with a simple generation
                        for _ in runner.generate("Hello", max_tokens=1, stream=False):
                            pass
                        
                        current_status = RunnerReady()
                        logger.info("GGUF runner ready")
                    
                    case ChatCompletion(
                        task_params=task_params, command_id=command_id
                    ) if isinstance(current_status, RunnerReady):
                        assert runner is not None
                        logger.info(f"GGUF runner received chat request")
                        
                        current_status = RunnerRunning()
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id, runner_status=current_status
                            )
                        )
                        
                        # Convert task_params messages to llama-cpp format
                        messages = [
                            {"role": msg.role, "content": str(msg.content)}
                            for msg in task_params.messages
                            if msg.content is not None
                        ]
                        
                        for response in runner.chat(
                            messages=messages,
                            max_tokens=task_params.max_tokens,
                            temperature=task_params.temperature,
                            stream=True,
                        ):
                            if shard_metadata.device_rank == 0:
                                event_sender.send(
                                    ChunkGenerated(
                                        command_id=command_id,
                                        chunk=TokenChunk(
                                            idx=response.token,
                                            model=shard_metadata.model_meta.model_id,
                                            text=response.text,
                                            token_id=response.token,
                                            finish_reason=response.finish_reason,
                                        ),
                                    )
                                )
                        
                        current_status = RunnerReady()
                        logger.info("GGUF runner ready")
                    
                    case Shutdown():
                        logger.info("GGUF runner shutting down")
                        if runner is not None:
                            runner.unload()
                        event_sender.send(
                            TaskStatusUpdated(
                                task_id=task.task_id, task_status=TaskStatus.Complete
                            )
                        )
                        break
                    
                    case _:
                        raise ValueError(
                            f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                        )
                
                event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                event_sender.send(
                    RunnerStatusUpdated(
                        runner_id=runner_id, runner_status=current_status
                    )
                )
        
        event_sender.send(
            RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown())
        )
    
    except ClosedResourceError:
        logger.warning("GGUF runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"GGUF Runner {runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        event_sender.close()
        task_receiver.close()
        event_sender.join()
        task_receiver.join()
        logger.info("GGUF runner finished")
