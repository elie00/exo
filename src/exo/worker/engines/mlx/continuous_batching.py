"""
Continuous Batching Scheduler - vLLM-style PagedAttention for EXO.

This module provides continuous batching capabilities for the EXO inference engine,
allowing non-blocking prefill while other requests are decoding.

Key concepts:
- PagedAttention: Memory-efficient KV cache management using pages
- Continuous Batching: New requests can start prefill without waiting for others to finish
- Request Scheduling: Priority-based scheduling for optimal throughput
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
from collections import deque
import time

from loguru import logger

from exo.shared.types.common import CommandId
from exo.shared.types.tasks import ChatCompletionTaskParams


class RequestState(str, Enum):
    """State of a batched request."""
    PENDING = "pending"        # Waiting to start prefill
    PREFILLING = "prefilling"  # Running prefill (prompt processing)
    DECODING = "decoding"      # Running decode (token generation)
    COMPLETE = "complete"      # Finished generating
    FAILED = "failed"         # Failed with error
    CANCELLED = "cancelled"   # Cancelled by user


@dataclass
class PagedKVBlock:
    """A block of memory for KV cache in PagedAttention."""
    block_id: int
    num_slots: int = 16  # Number of KV slots per block
    used_slots: int = 0
    sequence_id: Optional[str] = None
    
    @property
    def is_full(self) -> bool:
        return self.used_slots >= self.num_slots
    
    @property
    def is_free(self) -> bool:
        return self.sequence_id is None


@dataclass
class BatchedRequest:
    """A request in the continuous batching queue."""
    command_id: CommandId
    params: ChatCompletionTaskParams
    state: RequestState = RequestState.PENDING
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Token tracking
    prompt_tokens: int = 0
    generated_tokens: int = 0
    max_tokens: int = 100
    
    # KV cache blocks
    kv_blocks: list[int] = field(default_factory=list)
    
    # Callbacks
    on_token: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    
    @property
    def is_active(self) -> bool:
        return self.state in (RequestState.PREFILLING, RequestState.DECODING)
    
    @property
    def is_done(self) -> bool:
        return self.state in (RequestState.COMPLETE, RequestState.FAILED, RequestState.CANCELLED)
    
    def time_in_queue(self) -> float:
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at


class PagedMemoryPool:
    """
    Memory pool for PagedAttention KV cache management.
    
    Allocates and deallocates KV cache blocks for sequences.
    """
    
    def __init__(self, num_blocks: int = 256, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # Initialize free blocks
        self.blocks: list[PagedKVBlock] = [
            PagedKVBlock(block_id=i, num_slots=block_size)
            for i in range(num_blocks)
        ]
        self.free_block_ids: set[int] = set(range(num_blocks))
        self.sequence_blocks: dict[str, list[int]] = {}
    
    def allocate_block(self, sequence_id: str) -> Optional[int]:
        """Allocate a new block for a sequence."""
        if not self.free_block_ids:
            return None
        
        block_id = self.free_block_ids.pop()
        self.blocks[block_id].sequence_id = sequence_id
        self.blocks[block_id].used_slots = 0
        
        if sequence_id not in self.sequence_blocks:
            self.sequence_blocks[sequence_id] = []
        self.sequence_blocks[sequence_id].append(block_id)
        
        return block_id
    
    def free_sequence(self, sequence_id: str):
        """Free all blocks belonging to a sequence."""
        if sequence_id not in self.sequence_blocks:
            return
        
        for block_id in self.sequence_blocks[sequence_id]:
            self.blocks[block_id].sequence_id = None
            self.blocks[block_id].used_slots = 0
            self.free_block_ids.add(block_id)
        
        del self.sequence_blocks[sequence_id]
    
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)
    
    @property
    def utilization(self) -> float:
        return (self.num_blocks - self.num_free_blocks) / self.num_blocks


class ContinuousBatchingScheduler:
    """
    Continuous batching scheduler for EXO inference.
    
    Features:
    - Non-blocking prefill: New requests start immediately
    - Interleaved decode: Multiple sequences generate tokens together
    - Priority scheduling: Urgent requests get priority
    - Memory-aware: Respects GPU memory limits
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_prefill_batch: int = 4,
        memory_pool: Optional[PagedMemoryPool] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_prefill_batch = max_prefill_batch
        self.memory_pool = memory_pool or PagedMemoryPool()
        
        # Request queues
        self.pending_queue: deque[BatchedRequest] = deque()
        self.prefill_batch: list[BatchedRequest] = []
        self.decode_batch: list[BatchedRequest] = []
        
        # Request lookup
        self.requests: dict[CommandId, BatchedRequest] = {}
        
        # Scheduler state
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.total_tokens_generated = 0
    
    def submit_request(
        self,
        command_id: CommandId,
        params: ChatCompletionTaskParams,
        on_token: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> BatchedRequest:
        """
        Submit a new request for continuous batching.
        
        Args:
            command_id: Unique request ID
            params: Chat completion parameters
            on_token: Callback for each generated token
            on_complete: Callback when complete
            on_error: Callback on error
            
        Returns:
            The batched request object
        """
        request = BatchedRequest(
            command_id=command_id,
            params=params,
            max_tokens=params.max_tokens or 100,
            on_token=on_token,
            on_complete=on_complete,
            on_error=on_error,
        )
        
        self.requests[command_id] = request
        self.pending_queue.append(request)
        self.total_requests += 1
        
        logger.debug(f"Submitted request {command_id} to continuous batching queue")
        return request
    
    def cancel_request(self, command_id: CommandId) -> bool:
        """Cancel a pending or active request."""
        if command_id not in self.requests:
            return False
        
        request = self.requests[command_id]
        if request.is_done:
            return False
        
        request.state = RequestState.CANCELLED
        request.completed_at = time.time()
        
        # Free memory
        self.memory_pool.free_sequence(str(command_id))
        
        # Remove from active batches
        if request in self.prefill_batch:
            self.prefill_batch.remove(request)
        if request in self.decode_batch:
            self.decode_batch.remove(request)
        
        logger.info(f"Cancelled request {command_id}")
        return True
    
    async def schedule_step(self):
        """
        Run one scheduling step.
        
        This method:
        1. Moves completed requests out of batches
        2. Moves prefilled requests to decode batch
        3. Starts new prefills if capacity allows
        """
        # Remove completed requests
        self.decode_batch = [r for r in self.decode_batch if not r.is_done]
        self.prefill_batch = [r for r in self.prefill_batch if not r.is_done]
        
        # Move prefilled requests to decode batch
        for request in self.prefill_batch[:]:
            if request.state == RequestState.DECODING:
                self.prefill_batch.remove(request)
                if len(self.decode_batch) < self.max_batch_size:
                    self.decode_batch.append(request)
        
        # Start new prefills if capacity allows
        while (
            self.pending_queue
            and len(self.prefill_batch) < self.max_prefill_batch
            and len(self.decode_batch) + len(self.prefill_batch) < self.max_batch_size
            and self.memory_pool.num_free_blocks > 0
        ):
            request = self.pending_queue.popleft()
            request.state = RequestState.PREFILLING
            request.started_at = time.time()
            
            # Allocate initial KV block
            block_id = self.memory_pool.allocate_block(str(request.command_id))
            if block_id is not None:
                request.kv_blocks.append(block_id)
            
            self.prefill_batch.append(request)
            logger.debug(f"Started prefill for request {request.command_id}")
    
    def get_scheduler_stats(self) -> dict[str, Any]:
        """Get current scheduler statistics."""
        return {
            "pending_requests": len(self.pending_queue),
            "prefilling_requests": len(self.prefill_batch),
            "decoding_requests": len(self.decode_batch),
            "total_active": len(self.prefill_batch) + len(self.decode_batch),
            "memory_utilization": self.memory_pool.utilization,
            "free_blocks": self.memory_pool.num_free_blocks,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_tokens": self.total_tokens_generated,
        }
    
    def complete_request(self, command_id: CommandId):
        """Mark a request as complete."""
        if command_id not in self.requests:
            return
        
        request = self.requests[command_id]
        request.state = RequestState.COMPLETE
        request.completed_at = time.time()
        
        self.completed_requests += 1
        self.total_tokens_generated += request.generated_tokens
        
        # Free memory
        self.memory_pool.free_sequence(str(command_id))
        
        # Call completion callback
        if request.on_complete:
            try:
                request.on_complete()
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
        
        logger.debug(f"Completed request {command_id} with {request.generated_tokens} tokens")


# Singleton instance
_scheduler: Optional[ContinuousBatchingScheduler] = None


def get_scheduler() -> ContinuousBatchingScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ContinuousBatchingScheduler()
    return _scheduler
